//! Channel-owned packet continuation, object bindings, shadow registers and MME.
use crate::{
    Class, Registers, Result, Source,
    mme::{self, Mme},
    packet::{Packet, Word},
};

pub trait FrontendHost {
    fn state(&mut self, source: Source, dirty: crate::DirtyFlags) -> Result<()>;
    fn method(&mut self, source: Source, writes: crate::MemoryWrites) -> Result<()>;
    /// Drain prior methods, including their memory effects, before coherent reads.
    fn barrier(&mut self) -> Result<()>;
    fn read(&mut self, address: u64, bytes: usize) -> Result<Vec<u8>>;
    fn stopped(&self) -> bool;
}
struct Engine {
    regs: Registers,
    shadow: Box<[u32; 0xe00]>,
    macro_call: Option<u32>,
    parameters: mme::Parameters,
    upload_writes: crate::MemoryWrites,
}
impl Engine {
    fn new(class: Class) -> Self {
        Self {
            regs: Registers::new(class),
            shadow: Box::new([0; 0xe00]),
            macro_call: None,
            parameters: mme::Parameters::default(),
            upload_writes: crate::MemoryWrites::None,
        }
    }
}
pub struct Channel {
    packet: Packet,
    bindings: [Option<Class>; 8],
    engines: [Engine; 5],
    graphics_mme: Mme,
    mme_jit: bool,
    compute_mme: Option<Mme>,
    dram: Box<[u32]>,
    semaphore: [u32; 3],
    acquire: Option<SemaphoreAcquire>,
}

#[derive(Clone, Copy, Debug)]
pub struct SemaphoreAcquire {
    pub source: Source,
    pub address: u64,
    pub value: u32,
    pub operation: u32,
}
impl SemaphoreAcquire {
    pub fn matches(self, current: u32) -> bool {
        match self.operation {
            1 => current == self.value,
            4 => current >= self.value,
            8 => current & self.value != 0,
            _ => unreachable!("validated semaphore acquire"),
        }
    }
}
pub struct Progress {
    pub words: usize,
    pub acquire: Option<SemaphoreAcquire>,
}
impl Channel {
    pub fn new(mme_jit: bool) -> Self {
        Self {
            packet: Packet::default(),
            bindings: [None; 8],
            engines: [
                Class::ThreeD,
                Class::Compute,
                Class::Inline,
                Class::TwoD,
                Class::Copy,
            ]
            .map(Engine::new),
            graphics_mme: Mme::new(mme_jit),
            mme_jit,
            compute_mme: None,
            dram: vec![0; mme::DRAM_WORDS].into_boxed_slice(),
            semaphore: [0; 3],
            acquire: None,
        }
    }
}
impl Channel {
    pub fn execute(
        &mut self,
        mut source: Source,
        bytes: &[u8],
        host: &mut impl FrontendHost,
    ) -> Result<Progress> {
        if !bytes.len().is_multiple_of(4) {
            return Err(source.error("partial command word"));
        }
        let first_word = source.offset;
        for (offset, word) in bytes.chunks_exact(4).enumerate() {
            if host.stopped() {
                return Err(source.error("command execution cancelled"));
            }
            source.offset = first_word + offset as u32;
            let word = u32::from_le_bytes(word.try_into().unwrap());
            match self.packet.word(word).map_err(|e| source.error(e))? {
                Word::Header => (),
                Word::End => break,
                Word::Method {
                    method,
                    subchannel,
                    argument,
                    last,
                } => {
                    source.method = method;
                    source.subchannel = subchannel;
                    source.argument = argument;
                    source.class = self.bindings[subchannel as usize].map_or(0, |c| c as u32);
                    self.method(source, last, host)
                        .map_err(|e| source.error(e))?;
                    if let Some(acquire) = self.acquire.take() {
                        return Ok(Progress {
                            words: offset + 1,
                            acquire: Some(acquire),
                        });
                    }
                }
            }
        }
        Ok(Progress {
            words: bytes.len() / 4,
            acquire: None,
        })
    }
    fn method(
        &mut self,
        mut source: Source,
        last: bool,
        host: &mut impl FrontendHost,
    ) -> Result<()> {
        let m = source.method;
        let value = source.argument;
        if m == 0 {
            let class = Class::from_id(value)?;
            self.bindings[source.subchannel as usize] = Some(class);
            source.class = value;
            return host.method(source, crate::MemoryWrites::None);
        }
        if m < 0x100 {
            if matches!(m, 0x10 | 0x14 | 0x18) {
                self.semaphore[((m - 0x10) / 4) as usize] = value;
            }
            if m == 0x1c && matches!(value & 31, 1 | 4 | 8) {
                // ACQUIRE_SWITCH permits another channel to run while waiting.
                // Acquires stop command fetch, not just downstream preparation.
                // https://github.com/NVIDIA/open-gpu-kernel-modules/blob/main/src/common/sdk/nvidia/inc/class/clc56f.h
                if value & !0x101f != 0
                    || self.semaphore[0] & !255 != 0
                    || self.semaphore[1] & 3 != 0
                {
                    return Err("unsupported semaphore acquire flags/address".into());
                }
                self.acquire = Some(SemaphoreAcquire {
                    source,
                    address: (u64::from(self.semaphore[0]) << 32) | u64::from(self.semaphore[1]),
                    value: self.semaphore[2],
                    operation: value & 31,
                });
                return Ok(());
            }
            let writes = if matches!(m, 0x34 | 0x78) {
                crate::MemoryWrites::Sync
            } else if m == 0x1c {
                crate::MemoryWrites::Range {
                    address: (u64::from(self.semaphore[0]) << 32) | u64::from(self.semaphore[1]),
                    bytes: 16,
                }
            } else {
                crate::MemoryWrites::None
            };
            return host.method(source, writes);
        }
        let class = self.bindings[source.subchannel as usize].ok_or("method has no bound class")?;
        let engine = &mut self.engines[class.index()];
        if m >= 0x3800 {
            if !class.mme() || m >= 0x4000 {
                return Err("macro call on unsupported class or slot".into());
            }
            if engine.macro_call.is_none() {
                if m & 4 != 0 {
                    return Err("MME call starts on parameter method".into());
                }
                engine.macro_call = Some(m);
            }
            engine.parameters.push(value);
            if !last {
                return Ok(());
            }
            let slot = ((engine.macro_call.take().unwrap() - 0x3800) / 8) as usize;
            // @SPECULATIVE: compute initially shares graphics code/bindings and
            // data RAM, until it uploads its own code. This matches native SDK
            // streams and Lichii's NativeMacroContext, not a public HW contract.
            let mme = if class == Class::Compute {
                self.compute_mme.as_mut().unwrap_or(&mut self.graphics_mme)
            } else {
                &mut self.graphics_mme
            };
            let Engine {
                regs,
                shadow,
                parameters,
                upload_writes,
                ..
            } = engine;
            let mut callback = MacroHost {
                regs,
                shadow,
                upload_writes,
                source,
                host,
            };
            mme.execute(slot, parameters, &mut self.dram, &mut callback)?;
            return Ok(());
        }
        if engine.macro_call.is_some() {
            return Err("ordinary method interrupts an unfinished macro".into());
        }
        if class.mme() && matches!(m, 0x114..=0x120) {
            if class == Class::Compute && self.compute_mme.is_none() {
                self.compute_mme = Some(Mme::new(self.mme_jit));
            }
            let mme = if class == Class::Compute {
                self.compute_mme.as_mut().unwrap()
            } else {
                &mut self.graphics_mme
            };
            engine.regs.set(m, value);
            match m {
                0x114 => mme.upload_address(value)?,
                0x118 => mme.upload(value)?,
                0x11c => mme.bind_address(value)?,
                0x120 => {
                    mme.bind(value)?;
                    engine.regs.set(0x11c, engine.regs.get(0x11c) + 1);
                }
                _ => unreachable!(),
            }
            return Ok(());
        }
        let mut callback = MacroHost {
            regs: &mut engine.regs,
            shadow: &mut engine.shadow,
            upload_writes: &mut engine.upload_writes,
            source,
            host,
        };
        callback.emit(m, value, &mut self.dram, &mut engine.parameters)
    }
}
struct MacroHost<'a, H> {
    regs: &'a mut Registers,
    shadow: &'a mut [u32; 0xe00],
    source: Source,
    upload_writes: &'a mut crate::MemoryWrites,
    host: &'a mut H,
}
impl<H: FrontendHost> mme::Host for MacroHost<'_, H> {
    #[cfg(target_arch = "x86_64")]
    fn bank(&mut self) -> (*mut u32, *const u32) {
        self.regs.mme_bank()
    }
    fn state(&mut self, method: u32) -> Result<u32> {
        if self.regs.method(method).is_none() {
            return Err(format!("MME STATE reads unknown method {method:#x}").into());
        }
        Ok(self.regs.get(method))
    }
    fn stopped(&self) -> bool {
        self.host.stopped()
    }
    fn emit(
        &mut self,
        m: u32,
        mut value: u32,
        dram: &mut [u32],
        fifo: &mut mme::Parameters,
    ) -> Result<()> {
        let class = self.regs.class;
        let info = self.regs.method(m).ok_or_else(|| {
            format!("unknown method {m:#x}, emitted argument={value:#x}").into_boxed_str()
        })?;
        if class.mme() && m == 0x124 {
            if value > 3 {
                return Err("invalid MME shadow control".into());
            }
        } else if class.mme() {
            match self.regs.get(0x124) {
                0 | 1 => self.shadow[(m / 4) as usize] = value,
                2 => (),
                3 => value = self.shadow[(m / 4) as usize],
                _ => unreachable!(),
            }
        }
        let changed = self.regs.set(m, value);
        if class.mme() && matches!(m, 0x114..=0x120) {
            return Err("MME self-modifying code upload unsupported".into());
        }
        if info.ordinary() {
            return if changed {
                self.host.state(
                    Source {
                        method: m,
                        argument: value,
                        ..self.source
                    },
                    info.dirty(),
                )
            } else {
                Ok(())
            };
        }
        if class.mme() && matches!(m, 0x55c | 0x560) {
            let count = value as usize;
            let start = self.regs.get(0x558) as usize;
            if count > mme::DRAM_WORDS
                || (m == 0x55c && start.checked_add(count).is_none_or(|end| end > dram.len()))
            {
                return Err("MME DMA read exceeds data RAM/FIFO".into());
            }
            if count != 0 {
                let data = self.host.read(self.regs.address(0x550, 0x554), count * 4)?;
                for (i, b) in data.chunks_exact(4).enumerate() {
                    let word = u32::from_le_bytes(b.try_into().unwrap());
                    if m == 0x55c {
                        dram[start + i] = word;
                    } else {
                        fifo.push(word);
                    }
                }
            }
            return Ok(());
        }
        if class == Class::ThreeD && m == 0x2310 {
            super::engines::validate_firmware_write(value)?;
            // @SPECULATIVE: the SDK's Falcon-4 mailbox uses scratch[1]/[2]
            // as value/mask, then polls scratch[0] for 1. Lichii acknowledges
            // every call; retain the identified configuration writes instead.
            // No DMA, shader or guest fence is completed by this mailbox.
            self.host.method(
                Source {
                    method: m,
                    argument: value,
                    ..self.source
                },
                crate::MemoryWrites::None,
            )?;
            self.regs.set(0x3400, 1);
            return Ok(());
        }
        if m == 0x1b0 && matches!(class, Class::ThreeD | Class::Compute | Class::Inline) {
            *self.upload_writes = crate::writes::method(self.regs, 0x1b4);
        }
        let writes =
            if m == 0x1b4 && matches!(class, Class::ThreeD | Class::Compute | Class::Inline) {
                *self.upload_writes
            } else {
                crate::writes::method(self.regs, m)
            };
        if class == Class::ThreeD && (0x2390..=0x23cc).contains(&m) {
            let offset = self.regs.get(0x238c);
            self.regs.set(
                0x238c,
                offset
                    .checked_add(4)
                    .ok_or("constant-buffer offset overflow")?,
            );
        }

        let source = Source {
            method: m,
            argument: value,
            ..self.source
        };
        self.host.method(source, writes)
    }
}
use mme::Host as _;
