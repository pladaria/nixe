//! Synchronous LCQ front end: one demanded straight-line image, never a CFG.

use crate::abi::BlockKey;
use crate::analysis::system_instruction_supported;
use nixe_cpu::decode::a64::{A64Instruction, control, system as a64_system};
use nixe_cpu::decode::{self, DecodeResult};
use nixe_cpu::location::LocationDescriptor;
use nixe_cpu::memory::{ExecutableMemory, InstructionImage};
use nixe_memory::GuestVirtualAddress;
use std::num::NonZeroU16;

const MAX_INSTRUCTIONS: NonZeroU16 = NonZeroU16::new(512).unwrap();
pub(crate) mod compiler;
pub(crate) mod exclusive;
pub(crate) mod fault;
pub(crate) mod fp;
pub(crate) mod invocation;
pub(crate) mod system;

/// Cold input held by the winning vCPU until native output is published or
/// abandoned. Its claim owns cancellation; no memory lock survives capture.
pub(crate) struct Compilation<'a> {
    pub claim: crate::lifetime::compile::Claim<'a>,
    pub fragment: Fragment,
    pub identity: crate::lifetime::unit::EmissionIdentity,
}

impl<'a> Compilation<'a> {
    pub(crate) fn capture(
        claim: crate::lifetime::compile::Claim<'a>,
        memory: &impl ExecutableMemory,
    ) -> Result<Self, crate::lifetime::Error> {
        claim.validate()?;
        let fragment =
            Fragment::capture(memory, claim.key()).map_err(crate::lifetime::Error::InvalidUnit)?;
        let claim = claim.after_capture()?;
        if !memory.image_is_current(&fragment.image) {
            return Err(crate::lifetime::Error::StalePublication);
        }
        let identity = claim.begin_unit()?;
        Ok(Self {
            claim,
            fragment,
            identity,
        })
    }
}

pub(crate) struct Fragment {
    pub key: BlockKey,
    pub image: InstructionImage,
    /// Existing decoder output, including precise unsupported/invalid identity.
    pub instructions: Box<[DecodeResult]>,
    // Capture-stop diagnostics; native exits are derived from decoded input.
    #[cfg(test)]
    pub end: End,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum End {
    Control,
    Architectural,
    FpMode,
    Unsupported,
    Invalid,
    #[cfg(test)]
    FetchFault,
    #[cfg(test)]
    Limit {
        continuation: GuestVirtualAddress,
    },
}

impl Fragment {
    pub(crate) fn capture(
        memory: &impl ExecutableMemory,
        key: BlockKey,
    ) -> Result<Self, &'static str> {
        if key.pc.get() & 3 != 0 || key.profile != key.platform.profile_id() {
            return Err("LCQ key has an unaligned PC or inconsistent platform/profile");
        }
        let decode = |pc, bits: u32| {
            decode::decode(
                key.platform,
                LocationDescriptor::new(pc, key.profile),
                bits.into(),
            )
        };
        // Classification only: callback execution is bounded, cannot touch
        // memory, and is restartable for tracking or device reconciliation.
        let image = memory.capture_instructions(
            key.address_space,
            key.pc,
            MAX_INSTRUCTIONS,
            &|pc, bits| boundary(&decode(pc, bits), key).is_some(),
        );
        let instructions: Box<[_]> = image
            .words()
            .iter()
            .enumerate()
            .map(|(index, word)| {
                decode(
                    GuestVirtualAddress::new(key.pc.get().wrapping_add(index as u64 * 4)),
                    word.bits,
                )
            })
            .collect();
        #[cfg(test)]
        let end = if image.fault().is_some() {
            End::FetchFault
        } else {
            instructions
                .last()
                .and_then(|decoded| boundary(decoded, key))
                .unwrap_or(End::Limit {
                    continuation: GuestVirtualAddress::new(
                        key.pc.get().wrapping_add(instructions.len() as u64 * 4),
                    ),
                })
        };
        Ok(Self {
            key,
            image,
            instructions,
            #[cfg(test)]
            end,
        })
    }
}

pub(crate) fn boundary(decoded: &DecodeResult, key: BlockKey) -> Option<End> {
    let decoded = match decoded {
        DecodeResult::Decoded(decoded) => decoded,
        DecodeResult::RecognizedUnimplemented(_) => return Some(End::Unsupported),
        DecodeResult::Unallocated { .. } | DecodeResult::Reserved { .. } => {
            return Some(End::Invalid);
        }
    };
    match decode::a64::normalize(&decoded.instruction, decoded.encoding) {
        A64Instruction::Control(control::Instruction::Nop(_)) => None,
        A64Instruction::Control(
            control::Instruction::SupervisorCall(_) | control::Instruction::Breakpoint(_),
        ) => Some(End::Architectural),
        A64Instruction::Control(_) => Some(End::Control),
        A64Instruction::FpSimd(instruction)
            if crate::fp_policy::fp_lowering_disposition(instruction).is_exact() =>
        {
            Some(End::Architectural)
        }
        A64Instruction::System(instruction)
            if !system_instruction_supported(key.platform, instruction) =>
        {
            Some(End::Unsupported)
        }
        A64Instruction::System(a64_system::Instruction::WriteRegister(fields))
            if fields.system_key == 0xd51b_4400 =>
        {
            Some(End::FpMode)
        }
        A64Instruction::System(instruction) if self::system::fp_boundary(instruction).is_some() => {
            Some(End::Architectural)
        }
        A64Instruction::System(instruction)
            if self::system::runtime_boundary(key.platform, instruction).is_some()
                && !self::system::is_cache_probe(key.platform, instruction) =>
        {
            Some(End::Architectural)
        }
        A64Instruction::Integer(_)
        | A64Instruction::Memory(_)
        | A64Instruction::System(_)
        | A64Instruction::FpSimd(_) => None,
        _ => Some(End::Unsupported),
    }
}

#[cfg(test)]
mod tests;
