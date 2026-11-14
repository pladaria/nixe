//! Preparation consumes resolved methods in order, retaining canonical native
//! state and performing memory effects through the coherent owner.
use crate::{Class, Registers, Result, Source, compute, registers};

pub struct GraphicsState {
    pub preparation_dirty: bool,
    pub vertex_dirty: u32,
    pub index_dirty: bool,
    pub uniform_dirty: [u32; 5],
    pub targets_dirty: u16,
    /// (defined bits, values) of retained privileged configuration writes.
    /// Unwritten bits have no invented reset value.
    pub privileged: std::collections::BTreeMap<u32, (u32, u32)>,
    pub roots: [[u32; 64]; 8],
    pub constants: [[Option<(u64, u32)>; 32]; 5],
}
impl Default for GraphicsState {
    fn default() -> Self {
        Self {
            targets_dirty: 0x1ff,
            preparation_dirty: true,
            vertex_dirty: u32::MAX,
            index_dirty: true,
            uniform_dirty: [u32::MAX; 5],
            privileged: Default::default(),
            roots: [[0; 64]; 8],
            constants: [[None; 32]; 5],
        }
    }
}
/// A completed native primitive, with a consecutive range of hardware instance IDs.
#[derive(Clone, Copy, Debug)]
pub struct Draw {
    pub source: Source,
    pub first_instance: u32,
    pub instances: u32,
}
pub trait PreparationHost {
    fn read(&mut self, address: u64, bytes: usize) -> Result<Vec<u8>>;
    fn write(&mut self, address: u64, bytes: Vec<u8>) -> Result<()>;
    fn report(&mut self, address: u64, bytes: Vec<u8>) -> Result<()>;
    fn reduce_report(
        &mut self,
        address: u64,
        payload: u32,
        operation: u32,
        signed: bool,
    ) -> Result<()>;
    fn reset_samples_passed(&mut self) -> Result<()>;
    fn report_samples_passed(&mut self, address: u64, short: bool) -> Result<()>;
    fn barrier(&mut self) -> Result<()>;
    fn fragment_barrier(&mut self, before_depth: bool) -> Result<()>;
    fn invalidate_texture_data(&mut self) -> Result<()>;
    fn timestamp(&self) -> u64;
    fn stopped(&self) -> bool;
    fn blit(&mut self, source: Source, operation: &crate::blit::Blit) -> Result<()>;
    fn dispatch(&mut self, launch: &compute::Launch, registers: &Registers) -> Result<()>;
    fn draw(&mut self, draw: Draw, registers: &Registers, state: &mut GraphicsState) -> Result<()>;
    fn clear(
        &mut self,
        source: Source,
        registers: &Registers,
        state: &mut GraphicsState,
    ) -> Result<()>;
}
#[derive(Default)]
struct Upload {
    bytes: Vec<u8>,
    expected: usize,
    address: u64,
    row_bytes: u32,
    rows: u32,
    pitch: u32,
    flags: u32,
}
struct Engine {
    regs: Registers,
    upload: Upload,
    graphics: Option<Box<GraphicsState>>,
    root_table: usize,
    root_offset: usize,
    primitive: bool,
    instance: u32,
    count: Option<Source>,
    pending: Option<Draw>,
}
impl Engine {
    fn new(class: Class) -> Self {
        Self {
            regs: Registers::new(class),
            upload: Upload::default(),
            graphics: (class == Class::ThreeD).then(|| Box::new(GraphicsState::default())),
            root_table: 0,
            root_offset: 0,
            primitive: false,
            instance: 0,
            count: None,
            pending: None,
        }
    }
    fn flush(&mut self, host: &mut impl PreparationHost) -> Result<()> {
        if let Some(draw) = self.pending.take() {
            host.draw(draw, &self.regs, self.graphics.as_mut().unwrap())
                .map_err(|error| draw.source.error(error))?;
        }
        Ok(())
    }
    fn end(&mut self, host: &mut impl PreparationHost) -> Result<()> {
        let Some(source) = self.count.take() else {
            return Err("primitive END without a supported vertex/index count".into());
        };
        if let Some(pending) = self.pending.as_mut()
            && pending.source.method == source.method
            && pending.source.argument == source.argument
            && pending.first_instance.checked_add(pending.instances) == Some(self.instance)
            && pending.instances != u32::MAX
        {
            pending.instances += 1;
            return Ok(());
        }
        self.flush(host)?;
        self.pending = Some(Draw {
            source,
            first_instance: self.instance,
            instances: 1,
        });
        Ok(())
    }
}
pub struct Preparation {
    engines: [Engine; 5],
    host: [u32; 64],
}
impl Default for Preparation {
    fn default() -> Self {
        Self {
            engines: [
                Class::ThreeD,
                Class::Compute,
                Class::Inline,
                Class::TwoD,
                Class::Copy,
            ]
            .map(Engine::new),
            host: [0; 64],
        }
    }
}
impl Preparation {
    /// Drain before coherent frontend dependencies, channel switches and host access.
    /// BEGIN/END state and the instance counter survive a split command stream.
    pub fn flush(&mut self, host: &mut impl PreparationHost) -> Result<()> {
        self.engines[Class::ThreeD.index()].flush(host)
    }
    pub fn execute(
        &mut self,
        source: Source,
        scheduler: &mut compute::Scheduler,
        host: &mut impl PreparationHost,
    ) -> Result<()> {
        let m = source.method;
        let value = source.argument;
        if m == 0 {
            return Ok(());
        }
        if m < 0x100 {
            self.flush(host)?;
            return self.host_method(m, value, host);
        }
        let class = Class::from_id(source.class)?;
        if class != Class::ThreeD {
            self.flush(host)?;
        }
        let engine = &mut self.engines[class.index()];
        let (name, plain) = registers::method(class, m).ok_or("unknown prepared method")?;
        let previous = engine.regs.get(m);
        if class == Class::ThreeD {
            let continuing = match m {
                0x1618 => (value >> 26) & 3 != 0 && (previous ^ value) & 0xffff == 0,
                0x1614 | 0xd78 | 0x17e0 => true,
                _ => plain && previous == value,
            };
            if !continuing {
                engine.flush(host)?;
            }
        }
        if engine.regs.set(m, value)
            && let Some(graphics) = engine.graphics.as_mut()
        {
            match m {
                0x600..=0x6fc => graphics.vertex_dirty |= 1 << ((m - 0x600) / 8),
                0x1c00..=0x1dfc => {
                    graphics.vertex_dirty |= 1 << ((m - 0x1c00) / 16);
                    if (m - 0x1c00).is_multiple_of(16) || (m - 0x1c00) % 16 == 12 {
                        graphics.preparation_dirty = true;
                    }
                }
                0x17c8..=0x17d8 => graphics.index_dirty = true,
                // Draw arguments and resource contents do not change fixed state.
                0x17dc..=0x17f8 | 0xd74..=0xd78 | 0x2380..=0x2490 => (),
                0x1618 => graphics.preparation_dirty |= (previous ^ value) & 0xffff != 0,
                _ if plain => graphics.preparation_dirty = true,
                _ => (),
            }
            graphics.targets_dirty |= match m {
                0x800..=0x9e4 => 1 << ((m - 0x800) / 0x40),
                0xfe0..=0xff0 | 0x1228..=0x1230 | 0x1538 => 0x100,
                0xfb8 | 0x121c => 0x1ff,
                _ => 0,
            };
        }
        let regs = &mut engine.regs;
        if class == Class::TwoD && m == 0x8dc {
            if let Some(operation) = crate::blit::decode(regs)? {
                host.blit(source, &operation)?;
            }
            return Ok(());
        }
        if class == Class::Compute {
            match m {
                0x1b0c => {
                    // C7C0's report fields differ from C797: no graphics query
                    // selection. Release completes preceding compute effects;
                    // reductions, traps and interrupt delivery need consumers.
                    // https://github.com/NVIDIA/open-gpu-kernel-modules/blob/main/src/common/sdk/nvidia/inc/class/clc7c0.h
                    if value & !0x1000_0004 != 0 {
                        return Err("unsupported compute semaphore operation/flags".into());
                    }
                    host.barrier()?;
                    let address = regs.address(0x1b00, 0x1b04);
                    if address == 0 {
                        // @SPECULATIVE: the native SDK also emits null-target
                        // compute releases. Preserve ordering without writing
                        // a nonexistent destination, as the reference query
                        // cache does for an unmapped report.
                        // https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/video_core/query_cache/query_cache.h
                        return Ok(());
                    }
                    let payload = regs.get(0x1b08);
                    let bytes = if value & (1 << 28) != 0 {
                        payload.to_le_bytes().to_vec()
                    } else {
                        let mut bytes = u64::from(payload).to_le_bytes().to_vec();
                        bytes.extend_from_slice(&host.timestamp().to_le_bytes());
                        bytes
                    };
                    return host.report(address, bytes);
                }
                0x41c => {
                    let address = regs
                        .address(0x318, 0x31c)
                        .checked_mul(256)
                        .ok_or("inline QMD address overflow")?;
                    let bytes = (0x320..=0x41c)
                        .step_by(4)
                        .flat_map(|m| regs.get(m).to_le_bytes())
                        .collect();
                    host.write(address, bytes)?;
                    if regs.get(0x29c) & (1 << 24) != 0 || regs.get(0x330) & (1 << 12) == 0 {
                        scheduler.signal(address, 3, regs, host)?;
                    }
                    return Ok(());
                }
                0x2b8 => return Err("non-signaling PCAS queue operation not implemented".into()),
                0x2bc => {
                    if value & !3 != 0 {
                        return Err("invalid signaling PCAS flags".into());
                    }
                    return scheduler.signal(
                        u64::from(regs.get(0x2b4)) << 8,
                        if value & 2 != 0 {
                            if value & 1 != 0 { 3 } else { 2 }
                        } else {
                            value
                        },
                        regs,
                        host,
                    );
                }
                0x2c0 => {
                    return scheduler.signal(u64::from(regs.get(0x2b4)) << 8, value, regs, host);
                }
                _ => (),
            }
        }
        if class == Class::Copy && m == 0x300 {
            return copy(regs, value, host);
        }
        if class == Class::ThreeD {
            let graphics = engine.graphics.as_mut().unwrap();
            match m {
                0xde0 => {
                    // https://github.com/NVIDIA/open-gpu-doc/blob/master/classes/3d/clc797.h
                    // SYSMEMBAR requires a separate system-visibility contract.
                    if value & !2 != 0 {
                        return Err("unsupported pixel barrier system-visibility/flags".into());
                    }
                    return host.fragment_barrier(value & 2 != 0);
                }
                0xf74 => {
                    // @SPECULATIVE: only the observed zero operand has a
                    // reference-backed invalidation/order interpretation.
                    if value != 0 {
                        return Err("unsupported texture-data invalidation operand".into());
                    }
                    return host.invalidate_texture_data();
                }
                0x12c8 => {
                    // This resets hierarchical Z-cull metadata, not the depth/
                    // stencil surface. The host owns its own depth tests and
                    // keeps no guest Z-cull cache, so valid selectors require
                    // no memory write, clear, barrier or pipeline invalidation.
                    // https://gitlab.freedesktop.org/mesa/mesa/-/blob/main/src/nouveau/headers/nvidia/classes/clc797.h
                    // https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/video_core/engines/maxwell_3d.cpp
                    if value & !0x001f_ffff != 0 {
                        return Err("CLEAR_ZCULL_REGION contains reserved bits".into());
                    }
                    return Ok(());
                }
                0x1b0c => {
                    // https://github.com/NVIDIA/open-gpu-kernel-modules/blob/main/src/common/sdk/nvidia/inc/class/clc797.h
                    // C797 report semaphore: literal, timestamp and sample-count
                    // results have separate publication paths. Acquire/reduction
                    // modes require their own consumers.
                    if value & !0x1f80_f017 != 0 || !matches!(value & 3, 0 | 2) {
                        return Err("unsupported graphics semaphore operation/flags".into());
                    }
                    let report = (value >> 23) & 31;
                    if report == 21 {
                        if value & !0x1a80_f007 != 0 || value & 3 != 2 || (value >> 12) & 15 != 15 {
                            return Err("unsupported sample-count report operation/location".into());
                        }
                        // DisableFlush changes cache maintenance, not the query
                        // result or ordering. The shared query path already
                        // establishes the result-buffer dependency explicitly.
                        return host.report_samples_passed(
                            regs.address(0x1b00, 0x1b04),
                            value & (1 << 28) != 0,
                        );
                    }
                    if report != 0 && report != 20 {
                        return Err("graphics query report not implemented".into());
                    }
                    host.barrier()?;
                    let address = regs.address(0x1b00, 0x1b04);
                    if address == 0 && value & 3 == 0 && report == 0 {
                        // @SPECULATIVE: native SDK streams use a null-target
                        // release to drain preceding work. The reference drops
                        // the unmapped report; retain the dependency explicitly,
                        // without inventing writable backing or a counter signal.
                        return Ok(());
                    }
                    let payload = if report == 20 {
                        host.timestamp()
                    } else {
                        u64::from(regs.get(0x1b08))
                    };
                    let bytes = if value & (1 << 28) != 0 {
                        (payload as u32).to_le_bytes().to_vec()
                    } else {
                        let mut bytes = payload.to_le_bytes().to_vec();
                        bytes.extend_from_slice(&host.timestamp().to_le_bytes());
                        bytes
                    };
                    return host.report(address, bytes);
                }
                0x1530 => {
                    // Clear the actual occlusion accumulator. Z-cull metadata
                    // has no separate host representation to clear.
                    // https://github.com/NVIDIA/open-gpu-doc/blob/master/classes/3d/clc797.h
                    return match value {
                        1 => host.reset_samples_passed(),
                        2 => Ok(()),
                        _ => Err(format!("unsupported graphics counter reset {value:#x}").into()),
                    };
                }
                0x2310 => {
                    validate_firmware_write(value)?;
                    let data = regs.get(0x3404);
                    let mask = regs.get(0x3408);
                    let (defined, bits) = graphics.privileged.entry(value).or_default();
                    *defined |= mask;
                    *bits = (*bits & !mask) | (data & mask);
                    regs.set(0x3400, 1);
                    return Ok(());
                }
                0x16a8 => {
                    // Version 4 is the Volta/Ampere shader program header.
                    if value >> 16 > 4 || value & 0xffff < 4 {
                        return Err("unsupported shader program header version range".into());
                    }
                    return Ok(());
                }
                0x2410..=0x2490 if (m - 0x2410).is_multiple_of(32) => {
                    if value & !0x1f1 != 0 {
                        return Err("invalid constant-buffer binding".into());
                    }
                    graphics.uniform_dirty[((m - 0x2410) / 32) as usize] |=
                        1 << ((value >> 4) & 31);
                    graphics.constants[((m - 0x2410) / 32) as usize]
                        [((value >> 4) & 31) as usize] =
                        (value & 1 != 0).then(|| (regs.address(0x2384, 0x2388), regs.get(0x2380)));
                    return Ok(());
                }
                0x504 => {
                    if value & !0xff07 != 0 || value & 0x300 != 0 {
                        return Err("invalid root table selector".into());
                    }
                    engine.root_table = (value & 7) as usize;
                    engine.root_offset = ((value >> 8) & 255) as usize;
                    return Ok(());
                }
                0x508 => {
                    let slot = graphics.roots[engine.root_table]
                        .get_mut(engine.root_offset / 4)
                        .ok_or("root table upload overflow")?;
                    *slot = value;
                    engine.root_offset += 4;
                    return Ok(());
                }
                0x1618 => {
                    if engine.primitive {
                        return Err("nested primitive BEGIN".into());
                    }
                    // Instance selection is stateful across BEGIN/END pairs;
                    // each draw consumes the current index plus the global base.
                    // https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/video_core/engines/draw_manager.cpp#L233-L246
                    if value & !0x0c00ffff != 0 {
                        return Err(
                            format!("unsupported primitive-ID/split BEGIN {value:#x}").into()
                        );
                    }
                    engine.instance = match (value >> 26) & 3 {
                        0 => 0,
                        1 => engine.instance.wrapping_add(1),
                        2 => engine.instance,
                        _ => return Err("reserved primitive instance mode".into()),
                    };
                    engine.primitive = true;
                    return Ok(());
                }
                0x1614 => {
                    if !engine.primitive {
                        return Err("END without BEGIN".into());
                    }
                    if value > 1 {
                        return Err("invalid primitive END argument".into());
                    }
                    engine.primitive = false;
                    return engine.end(host);
                }
                0xd78 | 0x17e0 => {
                    if !engine.primitive {
                        return Err("draw without BEGIN".into());
                    }
                    engine.count = Some(source);
                    return Ok(());
                }
                0x19d0 => return host.clear(source, regs, graphics),
                0x26c | 0x274 | 0x304 | 0x1214 | 0x1218 | 0x123c | 0x17e4..=0x17f8 => {
                    return Err(format!("native draw method {m:#x} is not implemented").into());
                }
                0x2390..=0x23cc => {
                    let offset = regs.get(0x238c);
                    let size = regs.get(0x2380);
                    if u64::from(offset) + 4 > u64::from(size) {
                        return Err("constant-buffer upload outside selected buffer".into());
                    }
                    host.write(
                        regs.address(0x2384, 0x2388)
                            .checked_add(u64::from(offset))
                            .ok_or("constant-buffer address overflow")?,
                        value.to_le_bytes().to_vec(),
                    )?;
                    regs.set(0x238c, offset + 4);
                    return Ok(());
                }
                _ => (),
            }
        }
        match m {
            0x100 | 0x1a2c => Ok(()),
            0x110 | 0x13c | 0x1144 | 0x11f4 | 0x56c | 0x570 | 0xdec => host.barrier(),
            0x148 | 0x14c => Ok(()), // One ordered owner; no intra-channel interleaving.
            0x21c | 0x1288 | 0x1330 | 0x1334 | 0x1338 | 0x1424 | 0x1428 | 0x244
                if class.mme() && !plain =>
            {
                host.barrier()
            }
            0xda4 if class == Class::ThreeD => host.barrier(),
            0x1698 if class == Class::Compute => host.barrier(),
            0x1b0 if matches!(class, Class::ThreeD | Class::Compute | Class::Inline) => {
                if engine.upload.expected != 0 {
                    return Err("new inline upload before previous data completed".into());
                }
                if value & !0x1071 != 0 || value & 1 == 0 || value & 0x30 == 0x30 {
                    return Err("unsupported inline DMA layout/reduction/interrupt".into());
                }
                let row_bytes = regs.get(0x180);
                let rows = regs.get(0x184);
                let expected = u64::from(row_bytes) * u64::from(rows);
                if expected == 0 || expected > 64 * 1024 * 1024 {
                    return Err("invalid inline DMA size".into());
                }
                engine.upload = Upload {
                    bytes: Vec::with_capacity(expected as usize),
                    expected: expected as usize,
                    address: regs.address(0x188, 0x18c),
                    row_bytes,
                    rows,
                    pitch: regs.get(0x190),
                    flags: value,
                };
                Ok(())
            }
            0x1b4 if matches!(class, Class::ThreeD | Class::Compute | Class::Inline) => {
                let upload = &mut engine.upload;
                if upload.expected == 0 {
                    return Err("inline data without LAUNCH_DMA".into());
                }
                let remaining = upload.expected - upload.bytes.len();
                upload
                    .bytes
                    .extend_from_slice(&value.to_le_bytes()[..remaining.min(4)]);
                if upload.bytes.len() == upload.expected {
                    if upload.rows == 1 || upload.pitch == upload.row_bytes {
                        if host.stopped() {
                            return Err("inline upload cancelled".into());
                        }
                        // Transfer ownership of contiguous data directly to
                        // coherent publication, without copying it row by row.
                        host.write(upload.address, std::mem::take(&mut upload.bytes))?;
                    } else {
                        for row in 0..upload.rows {
                            if host.stopped() {
                                return Err("inline upload cancelled".into());
                            }
                            let at = row as usize * upload.row_bytes as usize;
                            host.write(
                                upload
                                    .address
                                    .checked_add(u64::from(row) * u64::from(upload.pitch))
                                    .ok_or("inline DMA address overflow")?,
                                upload.bytes[at..at + upload.row_bytes as usize].to_vec(),
                            )?;
                        }
                    }
                    if upload.flags & 0x30 != 0 {
                        host.barrier()?;
                    }
                    if upload.flags & 0x30 == 0x20 {
                        let bytes = if upload.flags & 0x1000 != 0 {
                            regs.get(0x1e4).to_le_bytes().to_vec()
                        } else {
                            let mut b = u64::from(regs.get(0x1e4)).to_le_bytes().to_vec();
                            b.extend_from_slice(&host.timestamp().to_le_bytes());
                            b
                        };
                        host.report(regs.address(0x1dc, 0x1e0), bytes)?;
                    }
                    upload.expected = 0;
                    upload.bytes.clear();
                }
                Ok(())
            }
            _ if plain => Ok(()),
            _ => Err(format!("unsupported {class:?} effect {name}").into()),
        }
    }
    fn host_method(&mut self, m: u32, value: u32, host: &mut impl PreparationHost) -> Result<()> {
        self.host[(m / 4) as usize] = value;
        match m {
            8 | 0x10 | 0x14 | 0x18 | 0x28 | 0x2c | 0x30 | 0x50 => Ok(()),
            0x34 => match value >> 27 {
                5 | 14 | 16 | 21 => host.barrier(),
                _ => Err(format!("unsupported channel memory operation {:#x}", value >> 27).into()),
            },
            0x78 if value <= 1 => host.barrier(),
            0x1c if value & 0x1f == 2 && value & !0x0110_001f == 0 => {
                host.barrier()?;
                let address = (u64::from(self.host[4]) << 32) | u64::from(self.host[5]);
                let payload = self.host[6];
                let bytes = if value & (1 << 24) != 0 {
                    payload.to_le_bytes().to_vec()
                } else {
                    let mut b = u64::from(payload).to_le_bytes().to_vec();
                    b.extend_from_slice(&host.timestamp().to_le_bytes());
                    b
                };
                host.report(address, bytes)
            }
            _ => Err(format!("unsupported channel method {m:#x}, argument={value:#x}").into()),
        }
    }
}
fn copy(regs: &Registers, flags: u32, host: &mut impl PreparationHost) -> Result<()> {
    if flags & !0x0200_039f != 0 || flags & 3 == 3 || flags & 0x18 > 0x10 {
        return Err("unsupported DMA launch flags".into());
    }
    if flags & 3 != 0 {
        if flags & 0x180 != 0x180 {
            return Err("block-linear DMA copy not implemented".into());
        }
        let source = regs.address(0x400, 0x404);
        let destination = regs.address(0x408, 0x40c);
        let width = regs.get(0x418) as usize;
        let rows = if flags & 0x200 != 0 {
            regs.get(0x41c)
        } else {
            1
        };
        if width > 64 * 1024 * 1024 {
            return Err("DMA row exceeds transfer limit".into());
        }
        for row in 0..rows {
            if host.stopped() {
                return Err("DMA copy cancelled".into());
            }
            let data = host.read(
                source
                    .checked_add(u64::from(row) * u64::from(regs.get(0x410)))
                    .ok_or("DMA source address overflow")?,
                width,
            )?;
            host.write(
                destination
                    .checked_add(u64::from(row) * u64::from(regs.get(0x414)))
                    .ok_or("DMA destination address overflow")?,
                data,
            )?;
        }
    }
    host.barrier()?;
    if flags & 0x18 != 0 {
        let mut bytes = regs.get(0x248).to_le_bytes().to_vec();
        if flags & 0x18 == 0x10 {
            bytes.extend_from_slice(&[0; 4]);
            bytes.extend_from_slice(&host.timestamp().to_le_bytes());
        }
        host.report(regs.address(0x240, 0x244), bytes)?;
    }
    Ok(())
}

/// @SPECULATIVE: Falcon-4 masked configuration protocol, inferred from the
/// native SDK macros and the reference's completion mailbox. Restrict it to the
/// observed SETUP/SWDX configuration registers, rather than acknowledging an
/// arbitrary firmware request. These values remain part of graphics state.
/// The first address is independently documented as GR_PRI_GPCS_SETUP_DEBUG:
/// https://github.com/OE4T/linux-nvgpu/blob/patches-rel-36/drivers/gpu/nvgpu/include/nvgpu/hw/ga10b/hw_gr_ga10b.h
pub(crate) fn validate_firmware_write(address: u32) -> Result<()> {
    if matches!(
        address,
        0x418800 | 0x419a04 | 0x419a08 | 0x419ba4 | 0x418e40 | 0x418e44 | 0x418e58..=0x418e6c
    ) && address & 3 == 0
    {
        Ok(())
    } else {
        Err(format!("unsupported Falcon-4 configuration address {address:#x}").into())
    }
}
