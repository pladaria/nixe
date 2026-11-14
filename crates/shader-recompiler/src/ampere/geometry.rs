//! Verified native geometry exit-handler installation.
use super::instruction::{Instruction, unsupported};
use crate::{
    Environment, Result,
    types::{ProgramOffset, Stage},
};

const SETUP_BYTES: u32 = 5 * 16;

pub(super) struct ExitHandler {
    pub body: ProgramOffset,
    pub handler: ProgramOffset,
}

fn exact(i: Instruction, low: u32, middle: u32, high: u32) -> bool {
    i.bits(0, 32) == low
        && i.bits(32, 32) == middle
        && i.bits(64, 32) == high
        && i.bits(96, 32) & 0xc00001ff == 0
}

pub(super) fn analyze(env: &mut impl Environment) -> Result<Option<ExitHandler>> {
    let entry = env.metadata().entry_address()?;
    let first = Instruction::fetch(env, entry)?;
    if first.opcode() != 0x34e {
        return Ok(None);
    }
    if env.metadata().stage != Stage::Geometry {
        return Err(unsupported(
            "Native exit-handler installation outside geometry",
        ));
    }
    // @SPECULATIVE: the native geometry ABI installs an output finalizer using
    // LEPC/RPCMOV. Only the verified installation and clearing sequence is lifted
    // to control-flow edges; reject any program that observes its temporary state.
    // https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/shader_recompiler/frontend/ampere/geometry_exit.h
    let fetch = |env: &mut _, offset| {
        let pc = entry
            .0
            .checked_add(offset)
            .ok_or_else(|| unsupported("Native geometry handler address overflow"))?;
        Instruction::fetch(env, ProgramOffset(pc))
    };
    let offset = fetch(env, 16)?;
    let target = offset.bits(32, 32);
    let high = fetch(env, 48)?;
    if !exact(first, 0x734e, 0, 0)
        || !exact(offset, 0x37802, target, 0xf00)
        || !exact(fetch(env, 32)?, 0x7210, 3, 0x07f1e0ff)
        || !(exact(high, 0x01017210, 0xff, 0x007fe4ff) || exact(high, 0xff017224, 0xff, 0x000e0401))
        || !exact(fetch(env, 64)?, 0x7357, 0, 0)
        || target & 15 != 0
        || target <= SETUP_BYTES + 16
        || target >= 65536 * 16
    {
        return Err(unsupported("Native geometry exit-handler installation"));
    }
    if !exact(fetch(env, target)?, 0x00ff7324, 0xff, 0)
        || !exact(fetch(env, target + 16)?, 0x7957, 0, 0)
        || !exact(fetch(env, target + 32)?, 0x794d, 0, 0x03800000)
    {
        return Err(unsupported("Native geometry exit-handler body"));
    }
    let preceding = fetch(env, target - 16)?;
    let ret = preceding.opcode() == 0x950
        && preceding.bits(0, 24) == 0x7950
        && preceding.bits(24, 8) < 249
        && preceding.branch_target(ProgramOffset(entry.0 + target - 16))? == entry;
    if !exact(preceding, 0x794d, 0, 0x03800000) && !ret {
        return Err(unsupported("Native exit handler reached by fallthrough"));
    }
    prove_setup_dead(env, entry, target)?;
    Ok(Some(ExitHandler {
        body: ProgramOffset(entry.0 + SETUP_BYTES),
        handler: ProgramOffset(entry.0 + target),
    }))
}

fn prove_setup_dead(env: &mut impl Environment, entry: ProgramOffset, handler: u32) -> Result<()> {
    let mut registers = 0b1011u32;
    let mut predicates = 1u32;
    for offset in (SETUP_BYTES..handler).step_by(16) {
        if registers == 0 && predicates == 0 {
            return Ok(());
        }
        let i = Instruction::fetch(env, ProgramOffset(entry.0 + offset))?;
        let op = i.opcode();
        let family = op & 0x1ff;
        let form = i.bits(9, 3);
        let guard = i.bits(12, 4);
        let bit = |register: u32| if register < 32 { 1 << register } else { 0 };
        let pred = |index: u32| if index < 7 { 1 << index } else { 0 };
        let mut reads = 0;
        let mut writes = 0;
        let mut read_predicates = pred(guard & 7);
        let mut write_predicates = 0;
        let sources = |has_c| {
            bit(i.bits(24, 8))
                | if form == 1 { bit(i.bits(32, 8)) } else { 0 }
                | if matches!(form, 2 | 3 | 7) || (has_c && matches!(form, 1 | 4 | 5 | 6)) {
                    bit(i.bits(64, 8))
                } else {
                    0
                }
        };
        if op == 0x94d {
            if guard == 7 && registers & 1 == 0 && read_predicates & predicates == 0 {
                return Ok(());
            }
            return Err(unsupported(
                "Native geometry exits with live handler setup state",
            ));
        }
        if matches!(op, 0x919 | 0x923) {
            if op == 0x923 {
                reads |= bit(i.bits(24, 8));
            }
            writes |= bit(i.bits(16, 8));
        } else if matches!(op, 0x805 | 0xab9 | 0x945) {
            // No read or write of the installation's R0/R1/R3 and P0 state.
        } else if matches!(op, 0x381 | 0x981) {
            if op == 0x381 {
                read_predicates |= pred(7 - i.bits(64, 3));
            }
            let address = i.bits(24, 8);
            if address != 255 {
                reads |= bit(address) | bit(address + 1);
            }
            let count = match i.bits(73, 3) {
                0..=4 => 1,
                5 => 2,
                6 => 4,
                _ => 0,
            };
            for n in 0..count {
                writes |= bit(i.bits(16, 8) + n);
            }
        } else if family == 2 && matches!(form, 1 | 4 | 5 | 6) {
            if form == 1 {
                reads |= bit(i.bits(32, 8));
            }
            writes |= bit(i.bits(16, 8));
        } else if matches!(family, 0x24 | 0x16 | 0x11 | 0x12) && form != 0 {
            reads |= sources(true);
            if matches!(family, 0x24 | 0x11) && i.bits(74, 1) != 0 {
                read_predicates |= pred(i.bits(87, 3));
            }
            writes |= bit(i.bits(16, 8));
            if matches!(family, 0x11 | 0x12) {
                write_predicates |= pred(i.bits(81, 3));
            }
        } else if family == 0x105 && matches!(form, 1 | 5 | 6) {
            if form == 1 {
                reads |= bit(i.bits(32, 8));
            }
            writes |= bit(i.bits(16, 8));
        } else if op == 0xb82 {
            reads |= bit(i.bits(24, 8));
            writes |= bit(i.bits(16, 8));
            if i.bits(73, 3) == 5 {
                writes |= bit(i.bits(16, 8) + 1);
            }
        } else if op == 0x321 {
            reads |= bit(i.bits(24, 8)) | bit(i.bits(32, 8));
            for n in 0..=i.bits(74, 2) {
                writes |= bit(i.bits(16, 8) + n);
            }
        } else if matches!(op, 0x324 | 0x924) {
            reads |= sources(false);
            writes |= bit(i.bits(16, 8));
        } else if op == 0x322 {
            for n in 0..=i.bits(74, 2) {
                reads |= bit(i.bits(32, 8) + n);
            }
            reads |= bit(i.bits(64, 8));
        } else if family == 0xc && form != 0 && i.bits(72, 1) == 0 {
            reads |= sources(false);
            read_predicates |= pred(i.bits(87, 3));
            write_predicates |= pred(i.bits(81, 3)) | pred(i.bits(84, 3));
        } else {
            return Err(unsupported("Unproven native geometry exit setup liveness"));
        }
        if reads & registers != 0 || read_predicates & predicates != 0 {
            return Err(unsupported("Native geometry reads exit setup state"));
        }
        if guard == 7 {
            registers &= !writes;
            predicates &= !write_predicates;
        }
    }
    Err(unsupported(
        "Native geometry finalizer observes exit setup state",
    ))
}
