//! Frontend-visible destinations. Shader and tiled writes retain an unknown
//! destination; their execution must precede any subsequent dependent read.
use crate::{Class, Registers};

#[derive(Clone, Copy, Default)]
pub enum MemoryWrites {
    #[default]
    None,
    Range {
        address: u64,
        bytes: u64,
    },
    Any,
    Draw,
    Sync,
}

pub(crate) fn method(r: &Registers, method: u32) -> MemoryWrites {
    use MemoryWrites::*;
    match (r.class, method) {
        (Class::ThreeD, 0x2390..=0x23cc) => Range {
            address: r
                .address(0x2384, 0x2388)
                .wrapping_add(u64::from(r.get(0x238c))),
            bytes: 4,
        },
        (Class::ThreeD | Class::Compute, 0x1b0c) => Range {
            address: r.address(0x1b00, 0x1b04),
            bytes: 16,
        },
        (Class::ThreeD, 0x1614 | 0xd78 | 0x17e0) => Draw,
        (
            Class::ThreeD,
            0x19d0 | 0x26c | 0x274 | 0x304 | 0x1214 | 0x1218 | 0x123c | 0x17e4..=0x17f8,
        ) => Any,
        (Class::ThreeD | Class::Compute | Class::Inline, 0x1b4) => {
            // Include pitch gaps and the optional completion report. Tiled
            // destinations cannot be bounded by the linear upload registers.
            let rows = u64::from(r.get(0x184));
            let length = u64::from(r.get(0x180));
            let flags = r.get(0x1b0);
            if flags & 1 == 0 || flags & 0x30 != 0 {
                Any
            } else if rows == 0 || length == 0 {
                None
            } else {
                Range {
                    address: r.address(0x188, 0x18c),
                    bytes: u64::from(r.get(0x190)) * (rows - 1) + length,
                }
            }
        }
        (Class::Compute, 0x2b8..=0x2c0 | 0x41c)
        | (Class::Copy, 0x300)
        | (Class::TwoD, 0x860 | 0x8dc | 0x918) => Any,
        _ => None,
    }
}
