//! QMD 3.0 layout and PCAS dependency scheduling. The table belongs to an address
//! space, so another channel can satisfy a scheduled descriptor's dependencies.
//! https://github.com/NVIDIA/open-gpu-doc/blob/master/classes/compute/clc6c0qmd.h
use crate::{PreparationHost, Registers, Result};
use std::collections::{BTreeMap, VecDeque};

#[derive(Clone, Debug)]
pub struct Launch {
    pub address: u64,
    pub words: [u32; 64],
    pub program: u64,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub shared_bytes: u32,
    pub constants: [Option<(u64, u32)>; 8],
}
impl Launch {
    pub fn read(address: u64, host: &mut impl PreparationHost) -> Result<Self> {
        let bytes = host.read(address, 256)?;
        let mut w = [0; 64];
        for (dst, b) in w.iter_mut().zip(bytes.chunks_exact(4)) {
            *dst = u32::from_le_bytes(b.try_into().unwrap());
        }
        if w[18] & 255 != 0x30 || w[4] & 0x100 != 0 {
            return Err(format!(
                "unsupported QMD version/queue at {address:#x}: version={:#x} control={:#x}",
                w[18] & 255,
                w[4]
            )
            .into());
        }
        let mut constants = [None; 8];
        for (i, constant) in constants.iter_mut().enumerate() {
            if w[20] & (1 << i) != 0 {
                let h = w[33 + i * 2];
                *constant = Some((
                    (u64::from(h & 0x1ffff) << 32) | u64::from(w[32 + i * 2]),
                    (h >> 19) << 4,
                ));
            }
        }
        Ok(Self {
            address,
            program: (u64::from(w[49] & 0x1ffff) << 32) | u64::from(w[48]),
            grid: [w[12], w[13] & 0xffff, w[14] & 0xffff],
            block: [w[18] >> 16, w[19] & 0xffff, w[19] >> 16],
            shared_bytes: w[17] & 0x3ffff,
            constants,
            words: w,
        })
    }
}
#[derive(Default)]
struct State {
    dependencies: i64,
    deferred: i64,
    loaded: bool,
    scheduled: bool,
    completed: bool,
}
#[derive(Default)]
pub struct Scheduler {
    states: BTreeMap<u64, State>,
}
impl Scheduler {
    pub fn invalidate(&mut self, address: u64, bytes: u64) -> Result<()> {
        let end = address
            .checked_add(bytes)
            .ok_or("QMD invalidation range overflow")?;
        if self
            .states
            .range(..end)
            .any(|(&a, s)| a + 256 > address && s.scheduled && !s.completed)
        {
            return Err("memory retirement overlaps an active QMD".into());
        }
        self.states.retain(|&a, _| a >= end || a + 256 <= address);
        Ok(())
    }

    pub fn signal(
        &mut self,
        address: u64,
        argument: u32,
        regs: &Registers,
        host: &mut impl PreparationHost,
    ) -> Result<()> {
        if argument & !0x00ff_ffff != 0 || argument & 0x3ff0 != 0 {
            return Err("unsupported PCAS selector/reserved bits".into());
        }
        let mut pending = VecDeque::from([(
            address,
            argument & 15,
            i64::from(((argument >> 14) & 0x3ff) + 1),
        )]);
        while let Some((address, action, count)) = pending.pop_front() {
            if host.stopped() {
                return Err("compute scheduling cancelled".into());
            }
            if action == 0 {
                continue;
            }
            if self.states.len() >= 4096 && !self.states.contains_key(&address) {
                self.states.retain(|_, s| !s.completed || s.deferred != 0);
                if self.states.len() >= 4096 {
                    return Err("QMD dependency table exhausted".into());
                }
            }
            let state = self.states.entry(address).or_default();
            match action {
                0 => continue,
                1 => {
                    if state.scheduled && !state.completed {
                        return Err("invalidate of active QMD".into());
                    }
                    state.loaded = false;
                    continue;
                }
                2 | 3 => {
                    if action == 3 || !state.loaded {
                        let qmd = Launch::read(address, host)?;
                        state.dependencies = i64::from((qmd.words[4] >> 14) & 0xffff)
                            .checked_add(state.deferred)
                            .ok_or("QMD dependency overflow")?;
                        state.deferred = 0;
                        state.loaded = true;
                        state.completed = false;
                    }
                    state.scheduled = true;
                }
                7 | 12 => {
                    let counter = if !state.loaded || state.completed {
                        &mut state.deferred
                    } else {
                        &mut state.dependencies
                    };
                    *counter = counter
                        .checked_add(if action == 7 { -count } else { count })
                        .ok_or("QMD dependency overflow")?;
                    if !state.loaded || state.completed {
                        continue;
                    }
                }
                _ => return Err(format!("unsupported PCAS action {action}").into()),
            }
            if !state.loaded || !state.scheduled || state.completed || state.dependencies > 0 {
                continue;
            }
            if state.dependencies < 0 {
                return Err("QMD dependency underflow".into());
            }
            // Release words are mutable while the QMD waits on dependencies.
            // Read them coherently at execution, just as the descriptor itself.
            let qmd = Launch::read(address, host)?;
            if qmd.grid.iter().all(|&n| n != 0) {
                if qmd.block.contains(&0) {
                    return Err("QMD has an empty thread block".into());
                }
                host.dispatch(&qmd, regs)?;
            }
            host.barrier()?;
            // Never publish releases after a failed/unsupported dispatch.
            for base in [24, 28, 52] {
                let flags = qmd.words[base + 1];
                if flags & (1 << 23) == 0 {
                    continue;
                }
                let destination = (u64::from(flags & 255) << 32) | u64::from(qmd.words[base]);
                let payload = u64::from(qmd.words[base + 2])
                    | if flags & (1 << 29) != 0 {
                        u64::from(qmd.words[base + 3]) << 32
                    } else {
                        0
                    };
                if flags & (3 << 27) != 0 {
                    return Err(format!(
                        "QMD interrupt not implemented: qmd={address:#x} slot={base} flags={flags:#010x} destination={destination:#x} payload={payload:#x}"
                    ).into());
                }
                if flags & (1 << 26) != 0 {
                    if flags >> 30 != 1 || flags & (1 << 29) != 0 || (flags >> 24) & 3 > 1 {
                        return Err(
                            format!("unsupported QMD reduction format: {flags:#010x}").into()
                        );
                    }
                    host.reduce_report(
                        destination,
                        payload as u32,
                        (flags >> 20) & 7,
                        flags & (1 << 24) != 0,
                    )?;
                    continue;
                }
                let bytes = match flags >> 30 {
                    0 => {
                        let mut b = payload.to_le_bytes().to_vec();
                        b.extend_from_slice(&host.timestamp().to_le_bytes());
                        b
                    }
                    1 => (payload as u32).to_le_bytes().to_vec(),
                    2 => payload.to_le_bytes().to_vec(),
                    _ => return Err("invalid QMD release structure size".into()),
                };
                host.report(destination, bytes)?;
            }
            state.completed = true;
            state.scheduled = false;
            if qmd.words[16] & (1 << 5) != 0 {
                return Err("second dependent QMD pointer not implemented".into());
            }
            for (enable, pointer, shift) in [(0, 15, 1)] {
                if qmd.words[16] & (1 << enable) == 0 {
                    continue;
                }
                let action = match (qmd.words[16] >> shift) & 7 {
                    1 => 2,
                    3 => 3,
                    4 => 7,
                    _ => return Err("unsupported dependent QMD action".into()),
                };
                pending.push_back((u64::from(qmd.words[pointer]) << 8, action, 1));
            }
        }
        Ok(())
    }
}
