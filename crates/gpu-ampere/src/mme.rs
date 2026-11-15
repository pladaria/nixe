//! TU104 MME: two ALUs and two method outputs per 96-bit instruction.
//! ISA reference: https://gitlab.freedesktop.org/mesa/mesa/-/blob/main/src/nouveau/mme/mme_tu104_sim.c
//! Independent native-SDK interpretation: https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/video_core/macro_turing.h
// Instruction semantics derived from Mesa (Copyright 2022 Collabora Ltd., MIT).
// See MESA-LICENSE for the original permission notice.
use crate::Result;
use std::sync::Arc;
#[cfg(feature = "performance-counters")]
mod diagnostics;
mod effects;
pub(crate) use effects::Effects;
#[cfg(target_arch = "x86_64")]
mod jit;

mod parameters;
pub(crate) use parameters::{Fifo, Parameters};

const INSTRUCTIONS: usize = 0x1000;
pub(crate) const DRAM_WORDS: usize = 0xc00;
const PREDICATES: [&[u8; 4]; 16] = [
    b"UUUU", b"TTTT", b"FFFF", b"TTUU", b"FFUU", b"TFUU", b"TUUU", b"FUUU", b"UUTT", b"UUTF",
    b"UUTU", b"UUFT", b"UUFF", b"UUFU", b"UUUT", b"UUUF",
];

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Alu {
    op: u8,
    dst: u8,
    x: u8,
    y: u8,
    imm: u16,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Instruction {
    end: bool,
    predicate: u8,
    predicate_reg: u8,
    alu: [Alu; 2],
    output: [[u8; 2]; 2],
    loads: [bool; 2],
}
impl Instruction {
    fn decode(words: [u32; 3]) -> Self {
        let bits =
            (u128::from(words[0]) << 64) | (u128::from(words[1]) << 32) | u128::from(words[2]);
        let field = |bit: u32, width: u32| ((bits >> bit) & ((1_u128 << width) - 1_u128)) as u32;
        let mut d = Self {
            end: field(0, 1) != 0,
            predicate: field(1, 4) as u8,
            predicate_reg: field(5, 5) as u8,
            ..Self::default()
        };
        for (i, at) in [10, 46].into_iter().enumerate() {
            d.alu[i] = Alu {
                op: field(at, 5) as u8,
                dst: field(at + 5, 5) as u8,
                x: field(at + 10, 5) as u8,
                y: field(at + 15, 5) as u8,
                imm: field(at + 20, 16) as u16,
            };
        }
        d.output = [
            [field(82, 3) as u8, field(85, 4) as u8],
            [field(89, 3) as u8, field(92, 4) as u8],
        ];
        for i in 0..2 {
            d.loads[i] = d.predicate_reg == 28 + i as u8
                || d.alu
                    .iter()
                    .any(|a| a.x == 28 + i as u8 || a.y == 28 + i as u8)
                || d.output.iter().flatten().any(|&v| v == 3 + i as u8);
        }
        d
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Emission {
    Complete,
    Dma,
}

pub(crate) trait Host {
    const TRACK_OWNERSHIP: bool = false;
    fn state(&mut self, method: u32) -> Result<u32>;
    /// Only DMA may borrow/modify the parameter FIFO and data RAM. Ordinary
    /// state and other effects retain the invocation's stable backing pointers.
    fn emit(&mut self, method: u32, value: u32) -> Result<Emission>;
    fn dma(&mut self, method: u32, dram: &mut [u32], fifo: &mut Fifo<'_>) -> Result<()>;
    fn stopped(&self) -> bool;
    /// Stable shared cell storage for the entire invocation. Methods may update
    /// cells, but cannot replace these retained banks or ownership arrays.
    #[cfg(target_arch = "x86_64")]
    fn bank(&self) -> (*const u32, *const u32, *const u64);
}

#[derive(Clone)]
pub(crate) struct Code {
    words: Box<[[u32; 3]]>,
    uploaded: Box<[u8]>,
    decoded: Box<[Instruction]>,
}
/// Immutable code and its entry proof may cross the ordered boundary; executable
/// storage and its allocator remain confined to their execution thread.
#[derive(Clone)]
pub struct MacroProgram {
    pub(crate) code: Arc<Code>,
    pub(crate) effects: Arc<Effects>,
    pub(crate) start: u32,
}
struct Proof {
    effects: Arc<Effects>,
    installed: bool,
}
#[derive(Clone, Copy)]
pub(crate) enum Entry {
    Unbound,
    Unclassified(u32),
    Front(u32),
    Back(u32),
}
impl Entry {
    fn start(self) -> Option<u32> {
        match self {
            Self::Unbound => None,
            Self::Unclassified(s) | Self::Front(s) | Self::Back(s) => Some(s),
        }
    }
}
pub(crate) struct Mme {
    context: Context,
    proofs: Vec<Option<Proof>>,
    entries: [Entry; 256],
    upload_word: usize,
    bind: usize,
}
pub(crate) struct Context {
    code: Arc<Code>,
    #[cfg(feature = "performance-counters")]
    statistics: diagnostics::Statistics,
    #[cfg(target_arch = "x86_64")]
    compiled: Option<jit::Cache>,
}
impl Context {
    pub fn new(code: Arc<Code>, mme_jit: bool) -> Self {
        #[cfg(not(target_arch = "x86_64"))]
        let _ = mme_jit;
        Self {
            code,
            #[cfg(feature = "performance-counters")]
            statistics: diagnostics::Statistics {
                owner: "frontend",
                ..Default::default()
            },
            #[cfg(target_arch = "x86_64")]
            compiled: mme_jit.then(jit::Cache::default),
        }
    }
    pub fn set_entry(&mut self, start: u32, enabled: bool) {
        #[cfg(target_arch = "x86_64")]
        if let Some(cache) = &mut self.compiled {
            cache.set_entry(start, enabled);
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = (start, enabled);
    }
    pub fn install(&mut self, code: Arc<Code>) -> bool {
        #[cfg(feature = "performance-counters")]
        {
            self.statistics.owner = "preparation";
        }
        if Arc::ptr_eq(&self.code, &code) {
            return false;
        }
        self.code = code;
        #[cfg(target_arch = "x86_64")]
        if let Some(cache) = &mut self.compiled {
            cache.clear_entries();
        }
        true
    }
}
impl Mme {
    pub fn new(mme_jit: bool) -> Self {
        Self {
            context: Context::new(
                Arc::new(Code {
                    words: vec![[0; 3]; INSTRUCTIONS].into_boxed_slice(),
                    uploaded: vec![0; INSTRUCTIONS].into_boxed_slice(),
                    decoded: vec![Instruction::default(); INSTRUCTIONS].into_boxed_slice(),
                }),
                mme_jit,
            ),
            proofs: Vec::new(),
            entries: [Entry::Unbound; 256],
            upload_word: 0,
            bind: 0,
        }
    }
}
impl Mme {
    pub fn upload_address(&mut self, value: u32) -> Result<()> {
        if value as usize >= INSTRUCTIONS {
            return Err("MME instruction pointer exceeds RAM".into());
        }
        self.upload_word = value as usize * 3;
        Ok(())
    }
    pub fn upload(&mut self, value: u32) -> Result<()> {
        let index = self.upload_word / 3;
        let word = self.upload_word % 3;
        let old = self
            .context
            .code
            .words
            .get(index)
            .ok_or("MME instruction RAM overflow")?;
        if old[word] != value || self.context.code.uploaded[index] & (1 << word) == 0 {
            // Only a changed generation with queued readers needs a copy.
            #[cfg(feature = "performance-counters")]
            if Arc::strong_count(&self.context.code) > 1 {
                self.context.statistics.code_copies += 1;
            }
            let code = Arc::make_mut(&mut self.context.code);
            code.words[index][word] = value;
            code.uploaded[index] |= 1 << word;
            if code.uploaded[index] == 7 {
                code.decoded[index] = Instruction::decode(code.words[index]);
            }
            if !self.proofs.is_empty() {
                self.proofs.clear();
                for entry in &mut self.entries {
                    if let Some(start) = entry.start() {
                        *entry = Entry::Unclassified(start);
                    }
                }
            }
            #[cfg(target_arch = "x86_64")]
            if let Some(cache) = &mut self.context.compiled {
                cache.invalidate();
            }
        }
        self.upload_word += 1;
        Ok(())
    }
    pub fn bind_address(&mut self, value: u32) -> Result<()> {
        if value as usize >= self.entries.len() {
            return Err("MME start table overflow".into());
        }
        self.bind = value as usize;
        Ok(())
    }
    pub fn bind(&mut self, value: u32) -> Result<()> {
        if value as usize >= INSTRUCTIONS {
            return Err("MME start exceeds code RAM".into());
        }
        let entry = self
            .entries
            .get_mut(self.bind)
            .ok_or("MME start table overflow")?;
        let previous = entry.start();
        *entry = Entry::Unclassified(value);
        if let Some(old) = previous
            && old != value
            && !self.entries.iter().any(|e| e.start() == Some(old))
        {
            self.context.set_entry(old, false);
        }
        self.context.set_entry(value, true);
        self.bind += 1;
        Ok(())
    }
    /// Binding changes and code uploads invalidate this slot's disposition.
    /// The common frontend call reads one slot and never touches the proof table.
    #[inline]
    pub fn resolve(&mut self, slot: usize, allow_back: bool) -> Result<Entry> {
        let entry = *self.entries.get(slot).ok_or("unbound MME macro")?;
        let start = entry.start().ok_or("unbound MME macro")?;
        if !allow_back {
            return Ok(Entry::Front(start));
        }
        if matches!(entry, Entry::Unclassified(_)) {
            return Ok(self.classify(slot, start));
        }
        Ok(entry)
    }
    #[cold]
    fn classify(&mut self, slot: usize, start: u32) -> Entry {
        if start as usize >= self.proofs.len() {
            self.proofs.resize_with(start as usize + 1, || None);
        }
        let proof = self.proofs[start as usize].get_or_insert_with(|| {
            let effects = Arc::new(effects::analyze(&self.context.code, start));
            #[cfg(feature = "performance-counters")]
            {
                self.context.statistics.analyses += 1;
                self.context.statistics.eligible_entries += u64::from(effects.eligible);
                log::debug!(
                    "MME effect proof start={start:#x} eligible={} writes={:x?}",
                    effects.eligible,
                    effects.writes
                );
            }
            Proof {
                effects,
                installed: false,
            }
        });
        let entry = if proof.effects.eligible {
            Entry::Back(start)
        } else {
            Entry::Front(start)
        };
        self.entries[slot] = entry;
        entry
    }
    pub fn execute_start(
        &mut self,
        start: u32,
        fifo: &mut Fifo<'_>,
        dram: &mut [u32],
        host: &mut impl Host,
    ) -> Result<()> {
        self.context.execute_start(start, fifo, dram, host)
    }
    pub(crate) fn back_program(&mut self, start: u32) -> (&Effects, Option<MacroProgram>) {
        let proof = self.proofs[start as usize]
            .as_mut()
            .expect("resolved backend macro");
        #[cfg(feature = "performance-counters")]
        {
            self.context.statistics.scheduled_calls += 1;
        }
        let install = if proof.effects.eligible && !proof.installed {
            proof.installed = true;
            Some(MacroProgram {
                code: self.context.code.clone(),
                effects: proof.effects.clone(),
                start,
            })
        } else {
            None
        };
        (&proof.effects, install)
    }
}
impl Context {
    pub(crate) fn execute_start(
        &mut self,
        start: u32,
        fifo: &mut Fifo<'_>,
        dram: &mut [u32],
        host: &mut impl Host,
    ) -> Result<()> {
        #[cfg(feature = "performance-counters")]
        {
            let begin = std::time::Instant::now();
            let mut meter = diagnostics::Meter {
                host,
                callbacks: Default::default(),
                state_callbacks: 0,
            };
            self.statistics.parameter_bytes += std::mem::size_of_val(fifo.remaining()) as u64;
            if fifo.borrowed() {
                self.statistics.borrowed_parameter_bytes += fifo.remaining().len() as u64;
            }
            let result = self.execute_code(start, fifo, dram, &mut meter);
            self.statistics.materialized_parameter_bytes += fifo.materialized_bytes;
            self.statistics.calls += 1;
            self.statistics.execution += begin.elapsed();
            self.statistics.callbacks += meter.callbacks;
            self.statistics.state_callbacks += meter.state_callbacks;
            result
        }
        #[cfg(not(feature = "performance-counters"))]
        self.execute_code(start, fifo, dram, host)
    }
    fn execute_code(
        &mut self,
        start: u32,
        fifo: &mut Fifo<'_>,
        dram: &mut [u32],
        host: &mut impl Host,
    ) -> Result<()> {
        #[cfg(target_arch = "x86_64")]
        if let Some(cache) = &mut self.compiled {
            cache.execute(
                &self.code.decoded,
                &self.code.uploaded,
                start,
                fifo,
                dram,
                host,
            )?;
            fifo.clear();
            return Ok(());
        }
        let mut run = Run {
            pc: start as usize,
            ..Run::default()
        };
        loop {
            if host.stopped() {
                return Err("MME execution cancelled".into());
            }
            let pc = run.pc;
            let result: Result<bool> = (|| {
                if self.code.uploaded.get(pc) != Some(&7) {
                    return Err("MME fetch outside uploaded instruction RAM".into());
                }
                let d = self.code.decoded[pc];
                let mut loads = [0; 2];
                for (i, load) in loads.iter_mut().enumerate() {
                    if d.loads[i] {
                        *load = fifo.pop().ok_or("MME parameter FIFO underflow")?;
                    }
                }
                let mask = if d.predicate == 0 {
                    15
                } else {
                    if matches!(d.predicate_reg, 25 | 26) {
                        return Err("MME predicate uses lane-dependent immediate".into());
                    }
                    let value = run.reg(d.predicate_reg, 0, d, loads)?;
                    PREDICATES[d.predicate as usize]
                        .iter()
                        .enumerate()
                        .fold(0, |mask, (i, &p)| {
                            mask | (u8::from(p != if value != 0 { b'T' } else { b'F' }) << i)
                        })
                };
                // @SPECULATIVE: native SDK macros require simultaneous operand
                // reads (Lichii); Mesa's software model commits ALU0 first.
                // Carry chaining still proceeds from lane 0 to lane 1.
                let mut operands = [[0; 2]; 2];
                for (i, operand) in operands.iter_mut().enumerate() {
                    if mask & (1 << i) != 0 {
                        *operand = [
                            run.reg(d.alu[i].x, i, d, loads)?,
                            run.reg(d.alu[i].y, i, d, loads)?,
                        ];
                    }
                }
                let mut next = pc + 1;
                let mut values = [0; 2];
                let mut stop = false;
                let mut cancel_end = false;
                for i in 0..2 {
                    if mask & (1 << i) == 0 {
                        continue;
                    }
                    let a = d.alu[i];
                    let [x, y] = operands[i];
                    let value = match a.op {
                        0 => {
                            let (v, c) = x.overflowing_add(y);
                            run.carry = u32::from(c);
                            v
                        }
                        1 => {
                            if i != 1 || d.alu[0].op != 0 {
                                return Err("MME ADDC without ADD pair".into());
                            }
                            x.wrapping_add(y).wrapping_add(run.carry)
                        }
                        2 => {
                            let (v, c) = x.overflowing_sub(y);
                            run.carry = u32::from(c);
                            v
                        }
                        3 => {
                            if i != 1 || d.alu[0].op != 2 {
                                return Err("MME SUBB without SUB pair".into());
                            }
                            x.wrapping_sub(y).wrapping_sub(run.carry)
                        }
                        4 | 6 => {
                            let v = if a.op == 4 {
                                (x as i32 as i64 as u64).wrapping_mul(y as i32 as i64 as u64)
                            } else {
                                u64::from(x) * u64::from(y)
                            };
                            run.carry = (v >> 32) as u32;
                            v as u32
                        }
                        5 => {
                            if a.x != 24 || a.y != 24 {
                                return Err("MME MULH operands must be zero".into());
                            }
                            run.carry
                        }
                        7 => {
                            if x != 0x1000 || y != 1 {
                                return Err("unsupported MME extended operation".into());
                            }
                            0
                        }
                        8 => x.leading_zeros(),
                        9 => x << (y & 31),
                        10 => x >> (y & 31),
                        11 => ((x as i32) >> (y & 31)) as u32,
                        12 => x & y,
                        13 => !(x & y),
                        14 => x | y,
                        15 => x ^ y,
                        16 => {
                            let dst = (a.imm >> 10) & 63;
                            let count = (a.imm >> 5) & 31;
                            let src = a.imm & 31;
                            if dst >= 32 {
                                return Err("MME MERGE destination outside word".into());
                            }
                            let mask = (1_u32 << count) - 1;
                            (x & !(mask << dst)) | (((y >> src) & mask) << dst)
                        }
                        17..=21 => {
                            if compare(a.op, x, y) {
                                u32::MAX
                            } else {
                                0
                            }
                        }
                        22 => host.state((x.wrapping_add(y) & 0xffff) * 4)?,
                        23 => {
                            if run.loops != 0 || a.imm < 2 || a.dst != 24 || a.y != 24 {
                                return Err("invalid MME LOOP".into());
                            }
                            // @SPECULATIVE: native SDK zero-count loops skip the
                            // body, unlike Mesa's MAX(1, count) model. Lichii
                            // documents this distinction in its TU104 interpreter.
                            if x == 0 {
                                next = pc + a.imm as usize;
                            } else {
                                run.loops = x - 1;
                                run.loop_start = pc + 1;
                                run.loop_end = pc + a.imm as usize - 1;
                            }
                            0
                        }
                        24 => {
                            if a.dst != 24 || a.x != 24 || a.y != 24 || a.imm & 0x8000 == 0 {
                                return Err("unsupported MME JAL encoding".into());
                            }
                            next = pc + (a.imm & 0x7fff) as usize;
                            0
                        }
                        25..=29 => {
                            if a.dst != 24 {
                                return Err("MME branch writes a register".into());
                            }
                            if compare(a.op - 8, x, y) == (a.imm & 0x8000 != 0) {
                                let offset = ((i32::from(a.imm) << 19) >> 19) as isize;
                                if offset == -4096 {
                                    stop = true;
                                } else {
                                    next = pc
                                        .checked_add_signed(offset)
                                        .ok_or("MME branch before code RAM")?;
                                    // @SPECULATIVE: bit 14 cancels this instruction's
                                    // END, with the last taken conditional branch
                                    // deciding cancellation. A previously pending END
                                    // still retires after this instruction's outputs.
                                    // Mesa does not model bit 14; follow the native
                                    // reference until hardware evidence is available.
                                    // https://github.com/zortos293/Lichii/blob/646b5d17135a2b2701ccc7a959afd0bbbf2f6160/src/video_core/macro_turing.h#L203
                                    cancel_end = a.imm & 0x4000 != 0;
                                }
                            }
                            0
                        }
                        30 => {
                            if a.y != 24 {
                                return Err("MME DREAD second operand must be zero".into());
                            }
                            *dram.get(x as usize).ok_or("MME DREAD outside data RAM")?
                        }
                        31 => {
                            if a.dst != 24 {
                                return Err("MME DWRITE destination must be zero".into());
                            }
                            *dram
                                .get_mut(x as usize)
                                .ok_or("MME DWRITE outside data RAM")? = y;
                            0
                        }
                        _ => unreachable!(),
                    };
                    if a.dst < 24 {
                        run.regs[a.dst as usize] = value;
                        run.initialized |= 1 << a.dst;
                    } else if a.dst != 24 {
                        return Err("invalid MME destination register".into());
                    }
                    values[i] = value;
                }
                for (i, output) in d.output.into_iter().enumerate() {
                    if mask & (4 << i) == 0 {
                        continue;
                    }
                    if output[0] != 0 {
                        let value = out(output[0], d, loads, values)?;
                        run.method = (value & 0xfff) * 4;
                        run.increment = ((value >> 12) & 15) * 4;
                    }
                    if output[1] != 0 {
                        if host.emit(run.method, out(output[1], d, loads, values)?)?
                            == Emission::Dma
                        {
                            host.dma(run.method, dram, fifo)?;
                        }
                        run.method += run.increment;
                    }
                }
                if stop || run.end {
                    return Ok(true);
                }
                run.end = d.end && !cancel_end;
                if run.loops != 0 && pc == run.loop_end {
                    run.loops -= 1;
                    next = run.loop_start;
                }
                run.pc = next;
                Ok(false)
            })();
            match result {
                Ok(true) => break,
                Ok(false) => (),
                Err(e) => {
                    return Err(format!(
                        "MME start={start:#x} pc={pc:#x} remaining_parameters={}: {e}",
                        fifo.len()
                    )
                    .into());
                }
            }
        }
        // The FIFO belongs to this invocation. A branch may finish without
        // consuming every supplied parameter; those words are not arguments
        // of the next call. Underflow is still rejected at the actual load.
        // https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/video_core/macro.cpp
        fifo.clear();
        Ok(())
    }
}
#[derive(Default)]
struct Run {
    regs: [u32; 24],
    initialized: u32,
    carry: u32,
    pc: usize,
    loops: u32,
    loop_start: usize,
    loop_end: usize,
    end: bool,
    method: u32,
    increment: u32,
}
impl Run {
    fn reg(&self, reg: u8, lane: usize, d: Instruction, loads: [u32; 2]) -> Result<u32> {
        match reg {
            0..=23 if self.initialized & (1 << reg) != 0 => Ok(self.regs[reg as usize]),
            0..=23 => Err(format!("uninitialized MME register r{reg}").into()),
            24 => Ok(0),
            25 => Ok(d.alu[lane].imm as i16 as i32 as u32),
            26 => Ok(d.alu[1 - lane].imm as i16 as i32 as u32),
            27 => Ok((u32::from(d.alu[0].imm) << 16) | u32::from(d.alu[1].imm)),
            28..=29 => Ok(loads[(reg - 28) as usize]),
            _ => Err("invalid MME source register".into()),
        }
    }
}
fn compare(op: u8, x: u32, y: u32) -> bool {
    match op {
        17 => (x as i32) < y as i32,
        18 => x < y,
        19 => (x as i32) <= y as i32,
        20 => x <= y,
        _ => x == y,
    }
}
fn out(selector: u8, d: Instruction, loads: [u32; 2], alu: [u32; 2]) -> Result<u32> {
    match selector {
        1..=2 => Ok(alu[(selector - 1) as usize]),
        3..=4 => Ok(loads[(selector - 3) as usize]),
        5..=6 => Ok(u32::from(d.alu[(selector - 5) as usize].imm)),
        8..=9 => Ok(u32::from(d.alu[(selector - 8) as usize].imm >> 12)),
        10 => Ok((u32::from(d.alu[0].imm) << 16) | u32::from(d.alu[1].imm)),
        _ => Err("invalid MME output selector".into()),
    }
}
