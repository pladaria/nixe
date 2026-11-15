//! Channel-owned packet continuation, object bindings, shadow registers and MME.
mod ownership;
use crate::{
    Class, Registers, Result, Source,
    mme::{self, Mme},
    packet::{Packet, Word},
};

pub trait FrontendHost {
    fn install_macro(&mut self, source: Source, program: crate::MacroProgram) -> Result<()>;
    fn run_macro(&mut self, source: Source, start: u32, parameters: &[u8]) -> Result<u64>;
    fn wait_until(&mut self, position: u64) -> Result<()>;
    fn reconcile(&mut self, source: Source) -> Result<()>;
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
    ownership: Option<ownership::Ownership>,
    macro_call: Option<u32>,
    parameters: mme::Parameters,
    upload_writes: crate::MemoryWrites,
}
impl Engine {
    fn new(class: Class) -> Self {
        Self {
            regs: Registers::new(class),
            shadow: Box::new([0; 0xe00]),
            ownership: None,
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
    pub fn new(mme_jit: bool) -> (Self, crate::BackState) {
        let mut channel = Self {
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
        };
        let back = crate::BackState::new(mme_jit);
        channel.engines[0].ownership = Some(ownership::Ownership::new(&back));
        (channel, back)
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
        let mut offset = 0;
        while offset < bytes.len() / 4 {
            if host.stopped() {
                return Err(source.error("command execution cancelled"));
            }
            if let Some(run) = self.packet.macro_run(bytes.len() / 4 - offset) {
                let arguments = &bytes[offset * 4..(offset + run.count) * 4];
                offset += run.count;
                source.offset = first_word + offset as u32 - 1;
                source.method = run.last_method;
                source.subchannel = run.subchannel;
                source.argument =
                    u32::from_le_bytes(arguments[arguments.len() - 4..].try_into().unwrap());
                source.class = self.bindings[run.subchannel as usize].map_or(0, |c| c as u32);
                self.macro_parameters(source, run.first, run.last, arguments, host)
                    .map_err(|e| source.error(e))?;
                continue;
            }
            source.offset = first_word + offset as u32;
            let word = u32::from_le_bytes(bytes[offset * 4..offset * 4 + 4].try_into().unwrap());
            offset += 1;
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
                            words: offset,
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
    fn macro_parameters(
        &mut self,
        source: Source,
        first: u32,
        last: bool,
        bytes: &[u8],
        host: &mut impl FrontendHost,
    ) -> Result<()> {
        let class = self.bindings[source.subchannel as usize].ok_or("method has no bound class")?;
        if !class.mme() || first >= 0x4000 {
            return Err("macro call on unsupported class or slot".into());
        }
        let engine = &mut self.engines[class.index()];
        if engine.macro_call.is_none() {
            if first & 4 != 0 {
                return Err("MME call starts on parameter method".into());
            }
            engine.macro_call = Some(first);
        }
        if !last {
            engine.parameters.append(bytes);
            return Ok(());
        }
        let mut fifo = engine.parameters.invocation(bytes);
        let slot = ((engine.macro_call.take().unwrap() - 0x3800) / 8) as usize;
        // @SPECULATIVE: compute initially shares graphics code/bindings and
        // data RAM, until it uploads its own code. This matches native SDK
        // streams and Lichii's NativeMacroContext, not a public HW contract.
        let mme = if class == Class::Compute {
            self.compute_mme.as_mut().unwrap_or(&mut self.graphics_mme)
        } else {
            &mut self.graphics_mme
        };
        let entry = mme.resolve(slot, class == Class::ThreeD && fifo.len() <= 0x1000)?;
        let start = match entry {
            mme::Entry::Back(start) => {
                let (effects, install) = mme.back_program(start);
                if let Some(program) = install {
                    host.install_macro(source, program)?;
                }
                let position = host.run_macro(source, start, fifo.remaining())?;
                engine.ownership.as_mut().unwrap().produce(
                    &effects.writes,
                    position,
                    engine.regs.get(0x124) <= 1,
                );
                fifo.clear();
                return Ok(());
            }
            mme::Entry::Front(start) => start,
            _ => unreachable!("resolved macro entry"),
        };
        let Engine {
            regs,
            shadow,
            ownership,
            upload_writes,
            ..
        } = engine;
        let mut callback = MacroHost {
            regs,
            shadow,
            ownership: ownership.as_mut(),
            upload_writes,
            source,
            host,
        };
        mme.execute_start(start, &mut fifo, &mut self.dram, &mut callback)?;
        Ok(())
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
        if m >= 0x3800 {
            return self.macro_parameters(source, m, last, &value.to_le_bytes(), host);
        }
        let class = self.bindings[source.subchannel as usize].ok_or("method has no bound class")?;
        let engine = &mut self.engines[class.index()];
        if engine.macro_call.is_some() {
            return Err("ordinary method interrupts an unfinished macro".into());
        }
        if class.mme() && matches!(m, 0x114..=0x120) {
            let mut value = value;
            match engine.regs.get(0x124) {
                0 | 1 => engine.shadow[m as usize / 4] = value,
                2 => (),
                3 => value = engine.shadow[m as usize / 4],
                _ => unreachable!(),
            }
            source.argument = value;
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
            // Backend STATE reads must observe uploads and the start-table
            // pointer's implicit increment, even though code ownership stays
            // on the frontend until an immutable generation is installed.
            return if m == 0x120 {
                host.method(source, crate::MemoryWrites::None)
            } else {
                host.state(source, crate::DirtyFlags::default())
            };
        }
        let mut callback = MacroHost {
            regs: &mut engine.regs,
            shadow: &mut engine.shadow,
            ownership: engine.ownership.as_mut(),
            upload_writes: &mut engine.upload_writes,
            source,
            host,
        };
        if callback.emit(m, value)? == mme::Emission::Dma {
            callback.dma(m, &mut self.dram, &mut engine.parameters.invocation(&[]))?;
        }
        Ok(())
    }
}
struct MacroHost<'a, H> {
    regs: &'a mut Registers,
    shadow: &'a mut [u32; 0xe00],
    ownership: Option<&'a mut ownership::Ownership>,
    source: Source,
    upload_writes: &'a mut crate::MemoryWrites,
    host: &'a mut H,
}
impl<H: FrontendHost> mme::Host for MacroHost<'_, H> {
    const TRACK_OWNERSHIP: bool = true;
    #[cfg(target_arch = "x86_64")]
    fn bank(&self) -> (*const u32, *const u32, *const u64) {
        let (bank, valid) = self.regs.mme_bank();
        (
            bank,
            valid,
            self.ownership
                .as_ref()
                .map_or(ownership::UNOWNED.as_ptr(), |o| o.pointer()),
        )
    }
    fn state(&mut self, method: u32) -> Result<u32> {
        if self.regs.method(method).is_none() {
            return Err(format!("MME STATE reads unknown method {method:#x}").into());
        }
        if let Some(ownership) = self.ownership.as_mut() {
            ownership.read(method, self.regs, self.host)?;
        }
        Ok(self.regs.get(method))
    }
    fn stopped(&self) -> bool {
        self.host.stopped()
    }
    fn emit(&mut self, m: u32, mut value: u32) -> Result<mme::Emission> {
        let class = self.regs.class;
        let info = self.regs.method(m).ok_or_else(|| {
            format!("unknown method {m:#x}, emitted argument={value:#x}").into_boxed_str()
        })?;
        let mut force = false;
        let mut shadow_changed = false;
        if let Some(ownership) = self.ownership.as_mut() {
            force = ownership.take_over(m, self.regs.get(0x124), self.shadow, self.host)?;
        }
        if class.mme() && m == 0x124 {
            if value > 3 {
                return Err("invalid MME shadow control".into());
            }
        } else if class.mme() {
            match self.regs.get(0x124) {
                0 | 1 => {
                    shadow_changed = self.shadow[(m / 4) as usize] != value;
                    self.shadow[(m / 4) as usize] = value;
                }
                2 => (),
                3 => value = self.shadow[(m / 4) as usize],
                _ => unreachable!(),
            }
        }
        let changed = self.regs.set(m, value);
        force |= shadow_changed && !changed;
        if class.mme() && matches!(m, 0x114..=0x120) {
            return Err("MME self-modifying code upload unsupported".into());
        }
        if info.ordinary() {
            return if force {
                self.host.reconcile(Source {
                    method: m,
                    argument: value,
                    ..self.source
                })
            } else if changed {
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
            }
            .map(|()| mme::Emission::Complete);
        }
        if class.mme() && matches!(m, 0x55c | 0x560) {
            return Ok(mme::Emission::Dma);
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
            return Ok(mme::Emission::Complete);
        }
        if m == 0x1b0 && matches!(class, Class::ThreeD | Class::Compute | Class::Inline) {
            *self.upload_writes = crate::writes::method(self.regs, 0x1b4);
        }
        let writes = if class == Class::ThreeD
            && m == 0x1558
            && matches!(value, 2..=4)
            && (self.regs.get(0x1944) == 0
                || self
                    .ownership
                    .as_ref()
                    .is_some_and(|owner| owner.produced(0x1944)))
        {
            // A coherent condition read may publish dirty cache contents. Its
            // completion must precede subsequent frontend memory reads, even
            // when an asynchronously produced override is not yet known here.
            crate::MemoryWrites::Any
        } else if m == 0x1b4 && matches!(class, Class::ThreeD | Class::Compute | Class::Inline) {
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
        self.host
            .method(source, writes)
            .map(|()| mme::Emission::Complete)
    }
    fn dma(&mut self, m: u32, dram: &mut [u32], fifo: &mut mme::Fifo<'_>) -> Result<()> {
        let value = self.regs.get(m);
        let count = value as usize;
        let start = if m == 0x55c {
            self.state(0x558)? as usize
        } else {
            0
        };
        if count > mme::DRAM_WORDS
            || (m == 0x55c && start.checked_add(count).is_none_or(|end| end > dram.len()))
        {
            return Err("MME DMA read exceeds data RAM/FIFO".into());
        }
        if count != 0 {
            let data = {
                let address = (u64::from(self.state(0x550)?) << 32) | u64::from(self.state(0x554)?);
                self.host.read(address, count * 4)?
            };
            if m == 0x55c {
                for (i, b) in data.chunks_exact(4).enumerate() {
                    dram[start + i] = u32::from_le_bytes(b.try_into().unwrap());
                }
            } else {
                fifo.append(&data);
            }
        }
        // The DMA already ran here. Publish only its resulting register
        // and shadow value for later backend STATE/replay, not another DMA.
        self.host.state(
            Source {
                method: m,
                argument: value,
                ..self.source
            },
            crate::DirtyFlags::default(),
        )
    }
}
use mme::Host as _;
