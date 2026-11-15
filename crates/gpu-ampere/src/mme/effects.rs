//! Finite constant propagation over (PC, pending END). Joins only lose facts;
//! unknown predicates and loop counts retain every possible successor.
//! https://github.com/zortos293/Lichii/blob/646b5d17135a2b2701ccc7a959afd0bbbf2f6160/src/video_core/macro_turing.h
use super::{Code, Instruction, PREDICATES};
use crate::{Class, registers};
type Value = Option<u32>;
#[derive(Clone, Copy, PartialEq, Eq)]
struct State {
    regs: [Value; 24],
    carry: Value,
    method: Value,
    increment: Value,
}
impl State {
    fn operand(self, d: Instruction, reg: u8, lane: usize) -> Value {
        match reg {
            0..=23 => self.regs[reg as usize],
            24 => Some(0),
            25 => Some(d.alu[lane].imm as i16 as i32 as u32),
            26 => Some(d.alu[1 - lane].imm as i16 as i32 as u32),
            27 => Some((u32::from(d.alu[0].imm) << 16) | u32::from(d.alu[1].imm)),
            _ => None,
        }
    }
}
#[derive(Clone, Copy)]
enum Flow {
    Next,
    Jump(usize),
    Branch(usize, bool),
    Stop,
}
pub(crate) struct Effects {
    /// Byte-method addresses, sorted and unique.
    pub writes: Vec<u32>,
    pub eligible: bool,
    write_set: Box<[u32; 0xe00 / 32]>,
}
impl Effects {
    pub fn contains(&self, method: u32) -> bool {
        method & 3 == 0
            && method < 0x3800
            && self.write_set[method as usize / 128] & (1 << ((method / 4) & 31)) != 0
    }
}
fn back_writable(m: u32) -> bool {
    // State read by frontend effect classification and registers with implicit
    // frontend writers cannot become backend-owned.
    if matches!(m, 0x114..=0x124 | 0x180..=0x1ac | 0x2380..=0x238c | 0x1b00..=0x1b0c | 0x3400 | 0x19c4 | 0x55c)
    {
        return false;
    }
    match m {
        0x504 | 0x508 | 0x110 | 0x560 | 0x1558 | 0x1530 | 0xde0 | 0xf74 | 0x11f4 => true,
        0x2410..=0x2490 if (m - 0x2410).is_multiple_of(32) => true,
        _ => registers::describe(Class::ThreeD, m).is_some_and(|(_, ordinary)| ordinary),
    }
}
struct Graph {
    states: Vec<Option<State>>,
    queued: Vec<bool>,
    work: Vec<usize>,
    repeats: Vec<Vec<usize>>,
}
impl Graph {
    fn enqueue(&mut self, index: usize) {
        if !self.queued[index] {
            self.queued[index] = true;
            self.work.push(index);
        }
    }
    fn reach(&mut self, pc: usize, last: bool, incoming: State) {
        if pc >= self.repeats.len() {
            return;
        } // Execution faults before effects.
        let index = pc * 2 + usize::from(last);
        if let Some(old) = &mut self.states[index] {
            let previous = *old;
            let join = |a: &mut Value, b: Value| {
                if *a != b {
                    *a = None;
                }
            };
            for (a, b) in old.regs.iter_mut().zip(incoming.regs) {
                join(a, b);
            }
            join(&mut old.carry, incoming.carry);
            join(&mut old.method, incoming.method);
            join(&mut old.increment, incoming.increment);
            if previous == *old {
                return;
            }
        } else {
            self.states[index] = Some(incoming);
        }
        self.enqueue(index);
    }
}
pub(super) fn analyze(code: &Code, start: u32) -> Effects {
    let count = code
        .uploaded
        .iter()
        .rposition(|&m| m != 0)
        .map_or(0, |i| i + 1);
    let mut graph = Graph {
        states: vec![None; count * 2],
        queued: vec![false; count * 2],
        work: Vec::new(),
        repeats: vec![Vec::new(); count],
    };
    graph.reach(
        start as usize,
        false,
        State {
            // Match the deterministic initial register file of both executors.
            regs: [Some(0); 24],
            carry: Some(0),
            method: Some(0),
            increment: Some(0),
        },
    );
    let mut known = true;
    let mut data_ram = false;
    let mut writes = std::collections::BTreeSet::new();
    while let Some(index) = graph.work.pop() {
        graph.queued[index] = false;
        let pc = index / 2;
        if code.uploaded[pc] != 7 {
            continue;
        }
        let before = graph.states[index].unwrap();
        let d = code.decoded[pc];
        let predicate = if d.predicate == 0 {
            Some(0)
        } else {
            before.operand(d, d.predicate_reg, 0)
        };
        for truth in [false, true] {
            if predicate.is_some_and(|p| (p != 0) != truth) {
                continue;
            }
            let mask = if d.predicate == 0 {
                15
            } else {
                PREDICATES[d.predicate as usize]
                    .iter()
                    .enumerate()
                    .fold(0, |mask, (i, &p)| {
                        mask | (u8::from(p != if truth { b'T' } else { b'F' }) << i)
                    })
            };
            let mut state = before;
            let mut alu = [Some(0); 2];
            let mut flows = [[Flow::Next; 2]; 2];
            let mut counts = [1; 2];
            for lane in 0..2 {
                if mask & (1 << lane) == 0 {
                    continue;
                }
                let a = d.alu[lane];
                let imm = u32::from(a.imm);
                let x = before.operand(d, a.x, lane);
                let y = before.operand(d, a.y, lane);
                let both = x.zip(y);
                let cmp = |x: u32, y: u32| match a.op {
                    17 | 25 => (x as i32) < (y as i32),
                    18 | 26 => x < y,
                    19 | 27 => (x as i32) <= (y as i32),
                    20 | 28 => x <= y,
                    _ => x == y,
                };
                let value = match a.op {
                    0 | 2 => {
                        let r = both.map(|(x, y)| {
                            if a.op == 0 {
                                x.overflowing_add(y)
                            } else {
                                x.overflowing_sub(y)
                            }
                        });
                        state.carry = r.map(|(_, c)| u32::from(c));
                        r.map(|(v, _)| v)
                    }
                    1 | 3 => both.zip(state.carry).map(|((x, y), c)| {
                        if a.op == 1 {
                            x.wrapping_add(y).wrapping_add(c)
                        } else {
                            x.wrapping_sub(y).wrapping_sub(c)
                        }
                    }),
                    4 | 6 => {
                        let p = both.map(|(x, y)| {
                            if a.op == 4 {
                                (x as i32 as i64).wrapping_mul(y as i32 as i64) as u64
                            } else {
                                u64::from(x) * u64::from(y)
                            }
                        });
                        state.carry = p.map(|p| (p >> 32) as u32);
                        p.map(|p| p as u32)
                    }
                    5 => state.carry,
                    7 => Some(0),
                    8 => x.map(u32::leading_zeros),
                    9 => both.map(|(x, y)| x.wrapping_shl(y)),
                    10 => both.map(|(x, y)| x.wrapping_shr(y)),
                    11 => both.map(|(x, y)| (x as i32).wrapping_shr(y) as u32),
                    12 => both.map(|(x, y)| x & y),
                    13 => both.map(|(x, y)| !(x & y)),
                    14 => both.map(|(x, y)| x | y),
                    15 => both.map(|(x, y)| x ^ y),
                    16 => both.and_then(|(x, y)| {
                        let dest = (imm >> 10) & 63;
                        let mask = (1u32 << ((imm >> 5) & 31)) - 1;
                        (dest < 32)
                            .then(|| (x & !(mask << dest)) | (((y >> (imm & 31)) & mask) << dest))
                    }),
                    17..=21 => both.map(|(x, y)| if cmp(x, y) { u32::MAX } else { 0 }),
                    22 => None,
                    23 => {
                        if imm < 2 {
                            known = false;
                            continue;
                        }
                        let end = pc + imm as usize - 1;
                        if end < count && !graph.repeats[end].contains(&(pc + 1)) {
                            graph.repeats[end].push(pc + 1);
                            for i in [end * 2, end * 2 + 1] {
                                if graph.states[i].is_some() {
                                    graph.enqueue(i);
                                }
                            }
                        }
                        counts[lane] = 0;
                        if x != Some(0) {
                            flows[lane][counts[lane]] = Flow::Next;
                            counts[lane] += 1;
                        }
                        if x.is_none_or(|v| v == 0) {
                            flows[lane][counts[lane]] = Flow::Jump(pc + imm as usize);
                            counts[lane] += 1;
                        }
                        Some(0)
                    }
                    24 => {
                        flows[lane][0] = Flow::Jump(pc + (imm & 0x7fff) as usize);
                        Some(0)
                    }
                    25..=29 => {
                        let offset = (imm as i32) << 19 >> 19;
                        let taken = both.map(|(x, y)| cmp(x, y) == (imm & 0x8000 != 0));
                        counts[lane] = 0;
                        if taken != Some(false) {
                            flows[lane][counts[lane]] = if offset == -4096 {
                                Flow::Stop
                            } else {
                                Flow::Branch(
                                    (pc as i64 + i64::from(offset)) as usize,
                                    imm & 0x4000 != 0,
                                )
                            };
                            counts[lane] += 1;
                        }
                        if taken != Some(true) {
                            flows[lane][counts[lane]] = Flow::Next;
                            counts[lane] += 1;
                        }
                        Some(0)
                    }
                    30 | 31 => {
                        data_ram = true;
                        None
                    }
                    _ => unreachable!(),
                };
                alu[lane] = value;
                if a.dst < 24 {
                    state.regs[a.dst as usize] = value;
                }
            }
            let output = |selector: u8| match selector {
                1 => alu[0],
                2 => alu[1],
                5 => Some(u32::from(d.alu[0].imm)),
                6 => Some(u32::from(d.alu[1].imm)),
                8 => Some(u32::from(d.alu[0].imm) >> 12),
                9 => Some(u32::from(d.alu[1].imm) >> 12),
                10 => Some((u32::from(d.alu[0].imm) << 16) | u32::from(d.alu[1].imm)),
                _ => None,
            };
            for lane in 0..2 {
                if mask & (4 << lane) == 0 {
                    continue;
                }
                if d.output[lane][0] != 0 {
                    let address = output(d.output[lane][0]);
                    state.method = address.map(|v| v & 0xfff);
                    state.increment = address.map(|v| (v >> 12) & 15);
                }
                if d.output[lane][1] != 0 {
                    if let Some(m) = state.method {
                        if let Some(byte) = m.checked_mul(4) {
                            writes.insert(byte);
                        } else {
                            known = false;
                        }
                    } else {
                        known = false;
                    }
                    state.method = state
                        .method
                        .zip(state.increment)
                        .map(|(m, i)| m.wrapping_add(i));
                }
            }
            if index & 1 != 0 {
                continue;
            }
            for first in &flows[0][..counts[0]] {
                for second in &flows[1][..counts[1]] {
                    let mut next = pc + 1;
                    let mut cancel = false;
                    let mut any_cancel = false;
                    let mut stopped = false;
                    for f in [first, second] {
                        match *f {
                            Flow::Next => (),
                            Flow::Jump(p) => next = p,
                            Flow::Branch(p, c) => {
                                next = p;
                                cancel = c;
                                any_cancel |= c;
                            }
                            Flow::Stop => stopped = true,
                        }
                    }
                    if stopped {
                        continue;
                    }
                    // @SPECULATIVE: execution uses the last taken branch's END
                    // cancellation. For two branches, hardware evidence does not
                    // rule out accumulating cancellation instead. Keep both
                    // successors when they differ so this proof is valid for
                    // either interpretation, including the delay instruction.
                    let end = d.end && !cancel;
                    let alternate_end = d.end && !any_cancel;
                    for pending in [end, alternate_end]
                        .into_iter()
                        .take(if end == alternate_end { 1 } else { 2 })
                    {
                        graph.reach(next, pending, state);
                        for i in 0..graph.repeats[pc].len() {
                            let repeat = graph.repeats[pc][i];
                            graph.reach(repeat, pending, state);
                        }
                    }
                }
            }
        }
    }
    let eligible =
        known && !data_ram && writes.contains(&0x560) && writes.iter().all(|&m| back_writable(m));
    let mut write_set = Box::new([0; 0xe00 / 32]);
    for &method in &writes {
        if method < 0x3800 {
            write_set[method as usize / 128] |= 1 << ((method / 4) & 31);
        }
    }
    Effects {
        writes: writes.into_iter().collect(),
        eligible,
        write_set,
    }
}
