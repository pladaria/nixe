//! Cold reconstruction after native escape, while the invocation still protects
//! code, physical maps and initialized frame spills. No guest access is replayed.

use crate::abi::{
    GuestValue, HostAbi, LazyFlags, NativeFrame, NzcvLocation, RegisterClass, ValueLocation,
};
use crate::lifetime::Fault;
use nixe_cpu::semantics::{
    arithmetic::{add_with_carry, subtract_with_carry},
    bits::BitWidth,
};
use nixe_cpu::state::a64::Nzcv;
use nixe_cpu_direct_memory::CapturedFault;

pub(crate) mod access;
pub(crate) mod cold;

pub(crate) struct Reconstructed {
    /// Still uncommitted pair-read bits, for typed cold completion only.
    pub completed_read: Option<u128>,
    /// The caller reconciles progress according to its resolution outcome.
    pub poll_remaining: i64,
}

/// Reconstruct the architectural prefix named by a published fault record.
///
/// # Safety
/// Native execution has escaped (never call this before returning Retry).
/// `fault` belongs to the captured PC under the still-active Invocation epoch;
/// `frame` is that invocation's unmoved frame with its initialized spill slots.
/// The worker's landing leaf already restored caller FP. On error the owner
/// must take the internal-fault path, never resume partially reconstructed code.
pub(crate) unsafe fn reconstruct(
    frame: &mut NativeFrame<'_>,
    captured: &CapturedFault<'_>,
    fault: &Fault<'_>,
) -> Result<Reconstructed, &'static str> {
    let map = captured_state(frame, captured, fault)?;
    let abi = map.abi;
    // Static maps permit inherited FP; the invocation decides whether the
    // captured image contains guest status or merely the caller's environment.
    let fp_status = if frame.host_fp.active != 0 {
        crate::fp_env::guest_status_from_host(
            abi,
            captured.fp().ok_or("missing captured FP state")?[1],
        )
    } else {
        0
    };
    let read =
        |location, bytes| unsafe { read_location(&frame.spill, captured, abi, location, bytes) };
    let flags = match &map.nzcv {
        NzcvLocation::Canonical => unsafe { (*frame.canonical.nzcv).bits() },
        NzcvLocation::Packed(location) => read(*location, 4)? as u32,
        NzcvLocation::Deferred(recipe) => recipe_flags(recipe, &read)?,
        NzcvLocation::Host { carry_inverted } => {
            let host = captured.flags();
            let mut bits = if abi == HostAbi::Aarch64 {
                host as u32 & 0xf000_0000
            } else {
                (((host >> 7) & 1) << 31
                    | ((host >> 6) & 1) << 30
                    | (host & 1) << 29
                    | ((host >> 11) & 1) << 28) as u32
            };
            if *carry_inverted {
                bits ^= 1 << 29;
            }
            bits
        }
    };
    let completed_read = fault
        .record
        .completed_read
        .map(|location| read(location, fault.record.bytes))
        .transpose()?;
    let poll_remaining = captured
        .integer(abi.reserved().poll)
        .ok_or("missing captured poll register")? as i64;
    for binding in map.bindings.iter() {
        if map
            .dirty_live
            .intersection(binding.value.state().unwrap())
            .is_empty()
        {
            continue;
        }
        let value = read(binding.location, binding.value.bytes())?;
        unsafe {
            match binding.value {
                GuestValue::General(index) => {
                    *frame.canonical.x.add(usize::from(index)) = value as u64
                }
                GuestValue::Sp => *frame.canonical.sp = value as u64,
                GuestValue::Vector(index) => {
                    *frame.canonical.vector.add(usize::from(index)) = value
                }
                GuestValue::Fpcr => *frame.canonical.fpcr = value as u32,
                GuestValue::Fpsr => *frame.canonical.fpsr = value as u32,
                GuestValue::TpidrEl0 => *frame.canonical.tpidr_el0 = value as u64,
                GuestValue::TpidrroEl0 => *frame.canonical.tpidrro_el0 = value as u64,
            }
        }
    }
    unsafe {
        let mask = u32::from(map.dirty_live.nzcv) << 28;
        *frame.canonical.nzcv =
            Nzcv::from_bits(((*frame.canonical.nzcv).bits() & !mask) | (flags & mask));
        *frame.canonical.fpsr |= fp_status; // after mapped software FPSR writeback
        *frame.canonical.pc = fault.instruction().key.block_key().pc.get();
    }
    // Guest sticky bits came from the saved image, NOT the dispatcher's live
    // host state. Prevent finish/drop from collecting host flags as guest FPSR.
    frame.host_fp.active = 0;
    frame.host_fp.suspended = 0;
    unsafe { frame.finish_fp() };
    frame.gateway_exit = 0;
    Ok(Reconstructed {
        completed_read,
        poll_remaining,
    })
}

fn captured_state<'a>(
    frame: &NativeFrame<'_>,
    captured: &CapturedFault<'_>,
    fault: &'a Fault<'_>,
) -> Result<&'a crate::abi::ExitStateMap, &'static str> {
    let map = &fault.unit.states[fault.record.state_map as usize].state;
    let abi = if cfg!(target_arch = "x86_64") {
        HostAbi::X86_64
    } else {
        HostAbi::Aarch64
    };
    if map.abi != abi
        || captured.integer(abi.reserved().frame) != Some(std::ptr::from_ref(frame).addr() as u64)
        || captured.native_pc()
            != fault.unit.code.allocation.address() + fault.record.native_start as usize
    {
        return Err("captured PC/frame does not match the published fault map");
    }
    map.validate()?;
    if frame.host_fp.saved == 0 || (frame.host_fp.active != 0 && !map.host_fpsr_pending) {
        return Err("captured fault map disagrees with invocation FP ownership");
    }
    Ok(map)
}

unsafe fn read_location(
    spill: &[std::mem::MaybeUninit<u8>],
    captured: &CapturedFault<'_>,
    abi: HostAbi,
    location: ValueLocation,
    bytes: u8,
) -> Result<u128, &'static str> {
    if !location.valid(abi, bytes) {
        return Err("invalid captured value location");
    }
    let value = match location {
        ValueLocation::Constant(value) => value.get(),
        ValueLocation::Register {
            class: RegisterClass::Integer,
            index,
        } => u128::from(
            captured
                .integer(index)
                .ok_or("missing captured integer register")?,
        ),
        ValueLocation::Register {
            class: RegisterClass::Vector,
            index,
        } => captured
            .vector(index)
            .ok_or("missing captured vector register")?,
        ValueLocation::Spill { offset, .. } => {
            let mut value = [0u8; 16];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    spill.as_ptr().cast::<u8>().add(offset as usize),
                    value.as_mut_ptr(),
                    usize::from(bytes),
                )
            };
            u128::from_le_bytes(value)
        }
    };
    Ok(if bytes == 16 {
        value
    } else {
        value & ((1u128 << (bytes * 8)) - 1)
    })
}

// Reuse the reference engine's Arm AddWithCarry arithmetic; never reconstruct
// guest flags from incidental host arithmetic in the dispatcher.
// https://developer.arm.com/documentation/ddi0602/2025-12/Shared-Pseudocode
fn recipe_flags(
    recipe: &LazyFlags<ValueLocation>,
    read: &impl Fn(ValueLocation, u8) -> Result<u128, &'static str>,
) -> Result<u32, &'static str> {
    use LazyFlags::*;
    let (lhs, rhs, carry, result, width, subtract) = match recipe {
        Canonical(value) | Packed(value) => return Ok(read(*value, 4)? as u32),
        Conditional {
            predicate,
            when_true,
            when_false,
        } => {
            return if read(*predicate, 1)? != 0 {
                recipe_flags(when_true, read)
            } else {
                Ok(when_false << 28)
            };
        }
        Logical { result, width } => return Ok(nz(read(*result, width / 8)?, *width)),
        Add {
            lhs,
            rhs,
            result,
            width,
        } => (lhs, rhs, false, result, width, false),
        Subtract {
            lhs,
            rhs,
            result,
            width,
        } => (lhs, rhs, true, result, width, true),
        AddCarry {
            lhs,
            rhs,
            carry,
            result,
            width,
        } => (lhs, rhs, read(*carry, 1)? != 0, result, width, false),
        SubtractCarry {
            lhs,
            rhs,
            carry,
            result,
            width,
        } => (lhs, rhs, read(*carry, 1)? != 0, result, width, true),
    };
    let bits = BitWidth::new(*width).map_err(|_| "invalid lazy flag width")?;
    let operands = (read(*lhs, width / 8)?, read(*rhs, width / 8)?);
    let computed = if subtract {
        subtract_with_carry(operands.0, operands.1, carry, bits)
    } else {
        add_with_carry(operands.0, operands.1, carry, bits)
    };
    Ok(nz(read(*result, width / 8)?, *width)
        | (u32::from(computed.carry_out) << 29)
        | (u32::from(computed.overflow) << 28))
}

fn nz(result: u128, width: u8) -> u32 {
    (((result >> (width - 1)) as u32 & 1) << 31) | (u32::from(result == 0) << 30)
}
