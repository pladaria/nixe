use super::{Operands, Result, Value, Visitor, unsupported};
use crate::{
    ir::{FmzMode, FpControl, FpRounding, Reg, Type},
    maxwell::bits::extract_signed,
};

impl Visitor<'_> {
    pub(super) fn global_atomic(&mut self, reduction: bool) -> Result<()> {
        let address = self.global_address(true)?;
        let (size, op, source, dest) = if reduction {
            (
                self.field::<20, 3>(),
                self.field::<23, 3>(),
                self.instruction.reg0(),
                Reg::RZ,
            )
        } else {
            (
                self.field::<49, 3>(),
                self.field::<52, 4>(),
                self.instruction.reg20(),
                self.instruction.reg0(),
            )
        };
        let load_only = match size {
            0..=2 => matches!(op, 3 | 4),
            3 => op != 0,
            4 => op > 2,
            5 => matches!(op, 0 | 3 | 4),
            _ => return Err(unsupported("Global atomic size")),
        };
        let result = if load_only {
            if matches!(size, 2 | 5) {
                let pair = self.emit.load_global64(address)?;
                self.emit.pack_uint2x32(pair)?
            } else {
                self.emit.load_global32(address)?
            }
        } else if matches!(size, 3 | 4) {
            let bits = self.emit.get_reg(source)?;
            let value = if size == 3 {
                self.emit.bit_cast(Type::F32, bits)?
            } else {
                self.emit.unpack_float2x16(bits)?
            };
            let control = FpControl {
                no_contraction: false,
                rounding: FpRounding::RN,
                fmz_mode: if size == 3 {
                    FmzMode::FTZ
                } else {
                    FmzMode::DontCare
                },
            };
            match (size, op) {
                (3, 0) => self.emit.global_atomic_f32_add(address, value, control)?,
                (4, 0) => self.emit.global_atomic_f16x2_add(address, value, control)?,
                (4, 1) => self.emit.global_atomic_f16x2_min(address, value, control)?,
                (4, 2) => self.emit.global_atomic_f16x2_max(address, value, control)?,
                _ => return Err(unsupported("Floating global atomic operation")),
            }
        } else {
            let value = if matches!(size, 2 | 5) {
                Operands {
                    emit: &mut self.emit,
                    instruction: self.instruction,
                }
                .read_u64(source)?
            } else {
                self.emit.get_reg(source)?
            };
            let signed = matches!(size, 1 | 5);
            match op {
                0 => self.emit.global_atomic_i_add(address, value)?,
                1 => self.emit.global_atomic_i_min(address, value, signed)?,
                2 => self.emit.global_atomic_i_max(address, value, signed)?,
                5 => self.emit.global_atomic_and(address, value)?,
                6 => self.emit.global_atomic_or(address, value)?,
                7 => self.emit.global_atomic_xor(address, value)?,
                8 => self.emit.global_atomic_exchange(address, value)?,
                _ => return Err(unsupported("Integer global atomic operation")),
            }
        };
        let mut operands = Operands {
            emit: &mut self.emit,
            instruction: self.instruction,
        };
        match size {
            2 | 5 => operands.write_pair(dest, result),
            3 => {
                // A load-only floating atomic does not produce a floating result.
                if load_only {
                    return Err(crate::CompileError::new(
                        crate::diagnostics::FailureCategory::InvalidArgument,
                        "Floating atomic load result has integer type",
                    ));
                }
                operands.write_f32(dest, result)
            }
            _ => operands.write_u32(dest, result),
        }
    }
    pub(super) fn shared_atomic(&mut self) -> Result<()> {
        let size = self.field::<28, 2>();
        let op = self.field::<52, 4>();
        if size == 2 && op != 8 {
            return Err(unsupported(
                "64-bit shared atomic operation other than exchange",
            ));
        }
        let address = if self.instruction.reg8() == Reg::RZ {
            Value::U32(self.field::<30, 22>() << 2)
        } else {
            let base = self.a()?;
            self.emit.i_add(
                base,
                Value::U32((extract_signed::<30, 22>(self.instruction.0) as u32).wrapping_shl(2)),
            )?
        };
        let source = self.instruction.reg20();
        let value = if size == 2 {
            Operands {
                emit: &mut self.emit,
                instruction: self.instruction,
            }
            .read_u64(source)?
        } else {
            self.emit.get_reg(source)?
        };
        let result = match op {
            0 => self.emit.shared_atomic_i_add(address, value)?,
            1 => self.emit.shared_atomic_i_min(address, value, size == 1)?,
            2 => self.emit.shared_atomic_i_max(address, value, size == 1)?,
            3 => self.emit.shared_atomic_inc(address, value)?,
            4 => self.emit.shared_atomic_dec(address, value)?,
            5 => self.emit.shared_atomic_and(address, value)?,
            6 => self.emit.shared_atomic_or(address, value)?,
            7 => self.emit.shared_atomic_xor(address, value)?,
            8 => self.emit.shared_atomic_exchange(address, value)?,
            _ => return Err(unsupported("Shared atomic operation")),
        };
        match size {
            0 | 1 => self.write(result),
            2 => Operands {
                emit: &mut self.emit,
                instruction: self.instruction,
            }
            .write_pair(self.instruction.reg0(), result),
            // The reserved size still performs its 32-bit operation without a result write.
            _ => Ok(()),
        }
    }
}
