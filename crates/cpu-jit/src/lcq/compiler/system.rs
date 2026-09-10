use super::*;
use nixe_cpu::decode::a64::system::Instruction;
use nixe_cpu::platform::TargetPlatform;
use nixe_cpu::semantics::a64::{RuntimeRegisterRead, runtime_register_read};

impl Translator<'_> {
    pub(super) fn system_value(&self, guest: GuestValue) -> Result<ir::Value, Error> {
        self.system_values[system_index(guest)].ok_or_else(|| {
            Error::internal(format!("LCQ input {guest:?} missing from shared liveness"))
        })
    }

    // Shared register identities/profile constants, same as the existing
    // compiler. NZCV consumers use the shared lazy recipe lowering.
    // https://developer.arm.com/documentation/ddi0601/2025-12/AArch64-Registers/NZCV--Condition-Flags
    // https://developer.arm.com/documentation/ddi0601/2025-12/AArch64-Registers/TPIDR-EL0--EL0-Read-Write-Software-Thread-ID-Register
    pub(super) fn system(
        &mut self,
        pc: GuestVirtualAddress,
        platform: TargetPlatform,
        instruction: Instruction,
        flags: &mut LazyFlags<ir::Value>,
    ) -> Result<bool, Error> {
        if !crate::analysis::system_instruction_supported(platform, instruction) {
            self.constant_exit(
                pc,
                pc,
                EdgeKind::Unsupported,
                NativeExitReason::Unsupported,
                flags,
            )?;
            return Ok(true);
        }
        if let Some(operation) = crate::lcq::system::fp_boundary(instruction) {
            // PRE-instruction state: the gateway merges and restores the old
            // host FP segment before cold code observes/replaces FPCR/FPSR.
            // No FP control/status write may precede that completion.
            self.constant_exit(
                pc,
                pc,
                EdgeKind::FpSystem(operation),
                NativeExitReason::Architectural,
                flags,
            )?;
            return Ok(true);
        }
        if let Some(operation) = crate::lcq::system::runtime_boundary(platform, instruction) {
            self.constant_exit(
                pc,
                pc,
                EdgeKind::RuntimeSystem(operation),
                NativeExitReason::Architectural,
                flags,
            )?;
            return Ok(true);
        }
        let f = instruction.operands();
        match instruction {
            Instruction::Hint(_) if crate::lcq::system::is_inline(platform, instruction) => {}
            Instruction::ReadRegister(_) => {
                let value = match f.system_key {
                    0xd53b_4200 => {
                        let packed = self.packed_flags(flags);
                        self.builder.ins().uextend(types::I64, packed)
                    }
                    0xd53b_4400 => {
                        let value = self.system_value(GuestValue::Fpcr)?;
                        self.builder.ins().uextend(types::I64, value)
                    }
                    0xd53b_d040 => self.system_value(GuestValue::TpidrEl0)?,
                    0xd53b_d060 => self.system_value(GuestValue::TpidrroEl0)?,
                    key => {
                        let Some(RuntimeRegisterRead::Constant(value)) =
                            runtime_register_read(platform, key)
                        else {
                            return Err(Error::internal(
                                "runtime system read has no native helper boundary",
                            ));
                        };
                        self.builder.ins().iconst(types::I64, value as i64)
                    }
                };
                self.write_register(f.rt, value)?;
            }
            Instruction::WriteRegister(_) => {
                let value = self.read_register(f.rt, false)?;
                match f.system_key {
                    0xd51b_4200 => {
                        *flags = self.nzcv_from_register(value);
                    }
                    0xd51b_d040 => {
                        self.system_values[system_index(GuestValue::TpidrEl0)] = Some(value);
                        self.dirty.tpidr_el0 = true;
                    }
                    _ => return Err(Error::internal("system write has no native boundary")),
                }
            }
            _ => {
                return Err(Error::internal(
                    "system operation has no native helper boundary",
                ));
            }
        }
        Ok(false)
    }
}
