//! Typed runtime boundaries used by direct native code.

use std::panic::{AssertUnwindSafe, catch_unwind};

use nixe_cpu::decode::a64::fp_simd::{
    FloatAddOperation, FloatConversion, FloatFusedMultiplyOperation, FloatMultiplyOperation,
    FloatRoundOperation, FloatToIntegerRounding,
};
use nixe_cpu::execution::SchedulerRequest;
use nixe_cpu::memory::{
    AtomicRmwKind, BarrierAccess, BarrierDomain, BarrierOperation, CacheMaintenanceKind,
    DataAccessFault, MemoryAccess, MemoryAccessClass, MemoryAccessSize, MemoryAlignment,
    MemoryOrdering, MemoryValue,
};
use nixe_cpu::semantics::{
    a64_fp_simd::{self, ExactCompareOutcome, ExactFpOutcome, ExactIntegerOutcome},
    conditions::{Condition, evaluate_a64},
    floating_point::FpStatus,
};
use nixe_memory::{AddressSpaceId, GuestVirtualAddress};

use super::NativeContext;

pub(super) type ReadFn = unsafe extern "C" fn(*mut NativeContext, u64);
pub(super) type WriteFn = unsafe extern "C" fn(*mut NativeContext, u64, u64);
pub(super) type Write128Fn = unsafe extern "C" fn(*mut NativeContext, u64, u64, u64);
pub(super) type AtomicRmwFn = unsafe extern "C" fn(*mut NativeContext, u64, u64);
pub(super) type CompareExchangeFn = unsafe extern "C" fn(*mut NativeContext, u64, u64, u64);
pub(super) type CompareExchangePairFn =
    unsafe extern "C" fn(*mut NativeContext, u64, u64, u64, u64, u64);
pub(super) type ExclusiveLoadFn = unsafe extern "C" fn(*mut NativeContext, u64);
pub(super) type ExclusiveStoreFn = unsafe extern "C" fn(*mut NativeContext, u64, u64);
pub(super) type ExclusiveStorePairFn = unsafe extern "C" fn(*mut NativeContext, u64, u64, u64);
pub(super) type ContextFn = unsafe extern "C" fn(*mut NativeContext);
pub(super) type AddressFn = unsafe extern "C" fn(*mut NativeContext, u64);

const STATUS_OK: u32 = 0;
const STATUS_DATA_FAULT: u32 = 1;
const STATUS_INTERNAL: u32 = 2;
pub(super) const STATUS_FP_TRAP: u32 = 3;

pub(super) fn read(size: MemoryAccessSize, ordering: MemoryOrdering) -> ReadFn {
    match (size, ordering) {
        (MemoryAccessSize::Byte, MemoryOrdering::Relaxed) => read_relaxed::<1>,
        (MemoryAccessSize::Halfword, MemoryOrdering::Relaxed) => read_relaxed::<2>,
        (MemoryAccessSize::Word, MemoryOrdering::Relaxed) => read_relaxed::<4>,
        (MemoryAccessSize::Doubleword, MemoryOrdering::Relaxed) => read_relaxed::<8>,
        (MemoryAccessSize::Quadword, MemoryOrdering::Relaxed) => read_relaxed::<16>,
        (MemoryAccessSize::Byte, MemoryOrdering::Acquire) => read_acquire::<1>,
        (MemoryAccessSize::Halfword, MemoryOrdering::Acquire) => read_acquire::<2>,
        (MemoryAccessSize::Word, MemoryOrdering::Acquire) => read_acquire::<4>,
        (MemoryAccessSize::Doubleword, MemoryOrdering::Acquire) => read_acquire::<8>,
        _ => unreachable!("A64 scalar read has a supported size and ordering"),
    }
}

pub(super) const fn write128() -> Write128Fn {
    write_relaxed_128
}

pub(super) fn write(size: MemoryAccessSize, ordering: MemoryOrdering) -> WriteFn {
    match (size, ordering) {
        (MemoryAccessSize::Byte, MemoryOrdering::Relaxed) => write_relaxed::<1>,
        (MemoryAccessSize::Halfword, MemoryOrdering::Relaxed) => write_relaxed::<2>,
        (MemoryAccessSize::Word, MemoryOrdering::Relaxed) => write_relaxed::<4>,
        (MemoryAccessSize::Doubleword, MemoryOrdering::Relaxed) => write_relaxed::<8>,
        (MemoryAccessSize::Byte, MemoryOrdering::Release) => write_release::<1>,
        (MemoryAccessSize::Halfword, MemoryOrdering::Release) => write_release::<2>,
        (MemoryAccessSize::Word, MemoryOrdering::Release) => write_release::<4>,
        (MemoryAccessSize::Doubleword, MemoryOrdering::Release) => write_release::<8>,
        _ => unreachable!("A64 scalar write has a supported size and ordering"),
    }
}

pub(super) fn atomic_rmw(
    size: MemoryAccessSize,
    ordering: MemoryOrdering,
    kind: AtomicRmwKind,
) -> AtomicRmwFn {
    atomic_rmw_size(size, ordering, kind)
}

pub(super) fn compare_exchange(
    size: MemoryAccessSize,
    ordering: MemoryOrdering,
) -> CompareExchangeFn {
    compare_exchange_size(size, ordering)
}

pub(super) fn compare_exchange_pair(
    size: MemoryAccessSize,
    ordering: MemoryOrdering,
) -> CompareExchangePairFn {
    match (size, ordering) {
        (MemoryAccessSize::Doubleword, MemoryOrdering::Relaxed) => compare_pair::<8, 0>,
        (MemoryAccessSize::Doubleword, MemoryOrdering::Acquire) => compare_pair::<8, 1>,
        (MemoryAccessSize::Doubleword, MemoryOrdering::Release) => compare_pair::<8, 2>,
        (MemoryAccessSize::Doubleword, MemoryOrdering::AcquireRelease) => compare_pair::<8, 3>,
        (MemoryAccessSize::Quadword, MemoryOrdering::Relaxed) => compare_pair::<16, 0>,
        (MemoryAccessSize::Quadword, MemoryOrdering::Acquire) => compare_pair::<16, 1>,
        (MemoryAccessSize::Quadword, MemoryOrdering::Release) => compare_pair::<16, 2>,
        (MemoryAccessSize::Quadword, MemoryOrdering::AcquireRelease) => compare_pair::<16, 3>,
        _ => unreachable!("A64 CASP has a supported size and ordering"),
    }
}

pub(super) fn exclusive_load(size: MemoryAccessSize, acquire: bool) -> ExclusiveLoadFn {
    match (size, acquire) {
        (MemoryAccessSize::Byte, false) => load_exclusive::<1, 0>,
        (MemoryAccessSize::Halfword, false) => load_exclusive::<2, 0>,
        (MemoryAccessSize::Word, false) => load_exclusive::<4, 0>,
        (MemoryAccessSize::Doubleword, false) => load_exclusive::<8, 0>,
        (MemoryAccessSize::Quadword, false) => load_exclusive::<16, 0>,
        (MemoryAccessSize::Byte, true) => load_exclusive::<1, 1>,
        (MemoryAccessSize::Halfword, true) => load_exclusive::<2, 1>,
        (MemoryAccessSize::Word, true) => load_exclusive::<4, 1>,
        (MemoryAccessSize::Doubleword, true) => load_exclusive::<8, 1>,
        (MemoryAccessSize::Quadword, true) => load_exclusive::<16, 1>,
    }
}

pub(super) fn exclusive_store(size: MemoryAccessSize, release: bool) -> ExclusiveStoreFn {
    match (size, release) {
        (MemoryAccessSize::Byte, false) => store_exclusive::<1, 0>,
        (MemoryAccessSize::Halfword, false) => store_exclusive::<2, 0>,
        (MemoryAccessSize::Word, false) => store_exclusive::<4, 0>,
        (MemoryAccessSize::Doubleword, false) => store_exclusive::<8, 0>,
        (MemoryAccessSize::Byte, true) => store_exclusive::<1, 2>,
        (MemoryAccessSize::Halfword, true) => store_exclusive::<2, 2>,
        (MemoryAccessSize::Word, true) => store_exclusive::<4, 2>,
        (MemoryAccessSize::Doubleword, true) => store_exclusive::<8, 2>,
        _ => unreachable!("A64 exclusive store has a scalar size"),
    }
}

pub(super) fn exclusive_store_pair(size: MemoryAccessSize, release: bool) -> ExclusiveStorePairFn {
    match (size, release) {
        (MemoryAccessSize::Doubleword, false) => store_exclusive_pair::<8, 0>,
        (MemoryAccessSize::Doubleword, true) => store_exclusive_pair::<8, 2>,
        (MemoryAccessSize::Quadword, false) => store_exclusive_pair::<16, 0>,
        (MemoryAccessSize::Quadword, true) => store_exclusive_pair::<16, 2>,
        _ => unreachable!("A64 exclusive pair store has a pair access size"),
    }
}

pub(super) fn barrier(operation: BarrierOperation) -> ContextFn {
    match operation {
        BarrierOperation::InstructionSynchronization => barrier_isb,
        BarrierOperation::DataMemory { domain, access } => barrier_data_memory(domain, access),
        BarrierOperation::DataSynchronization { domain, access } => {
            barrier_data_synchronization(domain, access)
        }
    }
}

pub(super) unsafe extern "C" fn clear_exclusive(context: *mut NativeContext) {
    contain(context, |context| {
        unsafe { &mut *context.exclusive }.clear();
        Ok(())
    });
}

pub(super) const fn cache_all() -> ContextFn {
    cache_instruction_all
}

pub(super) fn cache_address(kind: CacheMaintenanceKind) -> AddressFn {
    match kind {
        CacheMaintenanceKind::InstructionInvalidate => cache_instruction_address,
        CacheMaintenanceKind::DataInvalidate => cache_data_invalidate,
        CacheMaintenanceKind::DataClean => cache_data_clean,
        CacheMaintenanceKind::DataCleanAndInvalidate => cache_data_clean_invalidate,
        _ => unreachable!("catalogued A64 cache operation has an exact slow path"),
    }
}

pub(super) const fn scheduler_detail(request: SchedulerRequest) -> u32 {
    match request {
        SchedulerRequest::Yield => 0,
        SchedulerRequest::WaitForEvent => 1,
        SchedulerRequest::WaitForInterrupt => 2,
        SchedulerRequest::SendEvent => 3,
    }
}

pub(super) unsafe extern "C" fn hint_wait_for_event(context: *mut NativeContext) {
    contain(context, |context| {
        let events = unsafe { &*context.events };
        context.slow_result_low = u64::from(!events.consume_event());
        Ok(())
    });
}

pub(super) unsafe extern "C" fn hint_wait_for_interrupt(context: *mut NativeContext) {
    contain(context, |context| {
        let events = unsafe { &*context.events };
        context.slow_result_low = u64::from(!events.interrupts_pending());
        Ok(())
    });
}

pub(super) unsafe extern "C" fn hint_send_event_local(context: *mut NativeContext) {
    contain(context, |context| {
        unsafe { &*context.events }.signal_event();
        Ok(())
    });
}

pub(super) unsafe extern "C" fn timer_frequency(context: *mut NativeContext) {
    contain(context, |context| {
        context.slow_result_low = unsafe { &*context.timer }.snapshot().frequency;
        Ok(())
    });
}

pub(super) unsafe extern "C" fn timer_counter(context: *mut NativeContext) {
    contain(context, |context| {
        context.slow_result_low = unsafe { &*context.timer }.snapshot().counter;
        Ok(())
    });
}

unsafe extern "C" fn read_relaxed<const SIZE: u8>(context: *mut NativeContext, address: u64) {
    read_memory::<SIZE, 0>(context, address, false);
}

unsafe extern "C" fn read_acquire<const SIZE: u8>(context: *mut NativeContext, address: u64) {
    read_memory::<SIZE, 1>(context, address, true);
}

fn read_memory<const SIZE: u8, const ORDER: u8>(
    context: *mut NativeContext,
    address: u64,
    aligned: bool,
) {
    contain(context, |context| {
        let access = access::<SIZE, ORDER>(MemoryAccessClass::Normal, aligned);
        let memory = unsafe { &*context.memory };
        let result = memory.read(
            AddressSpaceId::new(context.address_space),
            GuestVirtualAddress::new(address),
            access,
        )?;
        context.slow_result_low = result.value.bits() as u64;
        context.slow_result_high = (result.value.bits() >> 64) as u64;
        Ok(())
    });
}

unsafe extern "C" fn write_relaxed_128(
    context: *mut NativeContext,
    address: u64,
    low: u64,
    high: u64,
) {
    contain(context, |context| {
        let memory = unsafe { &*context.memory };
        memory.write(
            AddressSpaceId::new(context.address_space),
            GuestVirtualAddress::new(address),
            access::<16, 0>(MemoryAccessClass::Normal, false),
            MemoryValue::from_bits(
                MemoryAccessSize::Quadword,
                u128::from(low) | (u128::from(high) << 64),
            ),
        )?;
        Ok(())
    });
}

unsafe extern "C" fn write_relaxed<const SIZE: u8>(
    context: *mut NativeContext,
    address: u64,
    value: u64,
) {
    write_memory::<SIZE, 0>(context, address, value, false);
}

unsafe extern "C" fn write_release<const SIZE: u8>(
    context: *mut NativeContext,
    address: u64,
    value: u64,
) {
    write_memory::<SIZE, 2>(context, address, value, true);
}

fn write_memory<const SIZE: u8, const ORDER: u8>(
    context: *mut NativeContext,
    address: u64,
    value: u64,
    aligned: bool,
) {
    contain(context, |context| {
        let access = access::<SIZE, ORDER>(MemoryAccessClass::Normal, aligned);
        let memory = unsafe { &*context.memory };
        memory.write(
            AddressSpaceId::new(context.address_space),
            GuestVirtualAddress::new(address),
            access,
            MemoryValue::from_bits(size::<SIZE>(), u128::from(value)),
        )?;
        Ok(())
    });
}

fn atomic_rmw_size(
    size: MemoryAccessSize,
    ordering: MemoryOrdering,
    kind: AtomicRmwKind,
) -> AtomicRmwFn {
    macro_rules! select_kind {
        ($size:literal, $order:literal) => {
            match kind {
                AtomicRmwKind::Add => atomic_rmw_call::<$size, $order, 0>,
                AtomicRmwKind::Clear => atomic_rmw_call::<$size, $order, 1>,
                AtomicRmwKind::Xor => atomic_rmw_call::<$size, $order, 2>,
                AtomicRmwKind::Set => atomic_rmw_call::<$size, $order, 3>,
                AtomicRmwKind::SignedMaximum => atomic_rmw_call::<$size, $order, 4>,
                AtomicRmwKind::SignedMinimum => atomic_rmw_call::<$size, $order, 5>,
                AtomicRmwKind::UnsignedMaximum => atomic_rmw_call::<$size, $order, 6>,
                AtomicRmwKind::UnsignedMinimum => atomic_rmw_call::<$size, $order, 7>,
                AtomicRmwKind::Swap => atomic_rmw_call::<$size, $order, 8>,
            }
        };
    }
    match (size, ordering) {
        (MemoryAccessSize::Byte, MemoryOrdering::Relaxed) => select_kind!(1, 0),
        (MemoryAccessSize::Halfword, MemoryOrdering::Relaxed) => select_kind!(2, 0),
        (MemoryAccessSize::Word, MemoryOrdering::Relaxed) => select_kind!(4, 0),
        (MemoryAccessSize::Doubleword, MemoryOrdering::Relaxed) => select_kind!(8, 0),
        (MemoryAccessSize::Byte, MemoryOrdering::Acquire) => select_kind!(1, 1),
        (MemoryAccessSize::Halfword, MemoryOrdering::Acquire) => select_kind!(2, 1),
        (MemoryAccessSize::Word, MemoryOrdering::Acquire) => select_kind!(4, 1),
        (MemoryAccessSize::Doubleword, MemoryOrdering::Acquire) => select_kind!(8, 1),
        (MemoryAccessSize::Byte, MemoryOrdering::Release) => select_kind!(1, 2),
        (MemoryAccessSize::Halfword, MemoryOrdering::Release) => select_kind!(2, 2),
        (MemoryAccessSize::Word, MemoryOrdering::Release) => select_kind!(4, 2),
        (MemoryAccessSize::Doubleword, MemoryOrdering::Release) => select_kind!(8, 2),
        (MemoryAccessSize::Byte, MemoryOrdering::AcquireRelease) => select_kind!(1, 3),
        (MemoryAccessSize::Halfword, MemoryOrdering::AcquireRelease) => select_kind!(2, 3),
        (MemoryAccessSize::Word, MemoryOrdering::AcquireRelease) => select_kind!(4, 3),
        (MemoryAccessSize::Doubleword, MemoryOrdering::AcquireRelease) => select_kind!(8, 3),
        _ => unreachable!("A64 LSE RMW has a supported size and ordering"),
    }
}

unsafe extern "C" fn atomic_rmw_call<const SIZE: u8, const ORDER: u8, const KIND: u8>(
    context: *mut NativeContext,
    address: u64,
    operand: u64,
) {
    contain(context, |context| {
        let transaction = unsafe { &*context.memory }.atomic_read_modify_write(
            AddressSpaceId::new(context.address_space),
            GuestVirtualAddress::new(address),
            access::<SIZE, ORDER>(MemoryAccessClass::Atomic, true),
            atomic_kind::<KIND>(),
            MemoryValue::from_bits(size::<SIZE>(), u128::from(operand)),
        )?;
        context.slow_result_low = transaction.previous.bits() as u64;
        Ok(())
    });
}

fn compare_exchange_size(size: MemoryAccessSize, ordering: MemoryOrdering) -> CompareExchangeFn {
    match (size, ordering) {
        (MemoryAccessSize::Byte, MemoryOrdering::Relaxed) => compare_exchange_call::<1, 0>,
        (MemoryAccessSize::Halfword, MemoryOrdering::Relaxed) => compare_exchange_call::<2, 0>,
        (MemoryAccessSize::Word, MemoryOrdering::Relaxed) => compare_exchange_call::<4, 0>,
        (MemoryAccessSize::Doubleword, MemoryOrdering::Relaxed) => compare_exchange_call::<8, 0>,
        (MemoryAccessSize::Byte, MemoryOrdering::Acquire) => compare_exchange_call::<1, 1>,
        (MemoryAccessSize::Halfword, MemoryOrdering::Acquire) => compare_exchange_call::<2, 1>,
        (MemoryAccessSize::Word, MemoryOrdering::Acquire) => compare_exchange_call::<4, 1>,
        (MemoryAccessSize::Doubleword, MemoryOrdering::Acquire) => compare_exchange_call::<8, 1>,
        (MemoryAccessSize::Byte, MemoryOrdering::Release) => compare_exchange_call::<1, 2>,
        (MemoryAccessSize::Halfword, MemoryOrdering::Release) => compare_exchange_call::<2, 2>,
        (MemoryAccessSize::Word, MemoryOrdering::Release) => compare_exchange_call::<4, 2>,
        (MemoryAccessSize::Doubleword, MemoryOrdering::Release) => compare_exchange_call::<8, 2>,
        (MemoryAccessSize::Byte, MemoryOrdering::AcquireRelease) => compare_exchange_call::<1, 3>,
        (MemoryAccessSize::Halfword, MemoryOrdering::AcquireRelease) => {
            compare_exchange_call::<2, 3>
        }
        (MemoryAccessSize::Word, MemoryOrdering::AcquireRelease) => compare_exchange_call::<4, 3>,
        (MemoryAccessSize::Doubleword, MemoryOrdering::AcquireRelease) => {
            compare_exchange_call::<8, 3>
        }
        _ => unreachable!("A64 CAS has a supported size and ordering"),
    }
}

unsafe extern "C" fn compare_exchange_call<const SIZE: u8, const ORDER: u8>(
    context: *mut NativeContext,
    address: u64,
    expected: u64,
    replacement: u64,
) {
    contain(context, |context| {
        let transaction = unsafe { &*context.memory }.atomic_compare_exchange(
            AddressSpaceId::new(context.address_space),
            GuestVirtualAddress::new(address),
            access::<SIZE, ORDER>(MemoryAccessClass::Atomic, true),
            MemoryValue::from_bits(size::<SIZE>(), u128::from(expected)),
            MemoryValue::from_bits(size::<SIZE>(), u128::from(replacement)),
        )?;
        context.slow_result_low = transaction.previous.bits() as u64;
        Ok(())
    });
}

unsafe extern "C" fn compare_pair<const SIZE: u8, const ORDER: u8>(
    context: *mut NativeContext,
    address: u64,
    expected_low: u64,
    expected_high: u64,
    replacement_low: u64,
    replacement_high: u64,
) {
    contain(context, |context| {
        let element_bits = u32::from(SIZE) * 4;
        let expected = u128::from(expected_low) | (u128::from(expected_high) << element_bits);
        let replacement =
            u128::from(replacement_low) | (u128::from(replacement_high) << element_bits);
        let transaction = unsafe { &*context.memory }.atomic_compare_exchange(
            AddressSpaceId::new(context.address_space),
            GuestVirtualAddress::new(address),
            access::<SIZE, ORDER>(MemoryAccessClass::Atomic, true),
            MemoryValue::from_bits(size::<SIZE>(), expected),
            MemoryValue::from_bits(size::<SIZE>(), replacement),
        )?;
        let previous = transaction.previous.bits();
        context.slow_result_low = previous as u64;
        context.slow_result_high = (previous >> element_bits) as u64;
        Ok(())
    });
}

unsafe extern "C" fn load_exclusive<const SIZE: u8, const ORDER: u8>(
    context: *mut NativeContext,
    address: u64,
) {
    contain(context, |context| {
        let (read, reservation) = unsafe { &*context.memory }.load_exclusive(
            AddressSpaceId::new(context.address_space),
            GuestVirtualAddress::new(address),
            access::<SIZE, ORDER>(MemoryAccessClass::Exclusive, true),
        )?;
        unsafe { &mut *context.exclusive }.reserve(reservation);
        let bits = read.value.bits();
        context.slow_result_low = bits as u64;
        context.slow_result_high = (bits >> 64) as u64;
        Ok(())
    });
}

unsafe extern "C" fn store_exclusive<const SIZE: u8, const ORDER: u8>(
    context: *mut NativeContext,
    address: u64,
    value: u64,
) {
    contain(context, |context| {
        let monitor = unsafe { &mut *context.exclusive };
        let reservation = monitor.reservation();
        monitor.clear();
        let stored = if let Some(reservation) = reservation {
            unsafe { &*context.memory }
                .store_exclusive(
                    AddressSpaceId::new(context.address_space),
                    GuestVirtualAddress::new(address),
                    access::<SIZE, ORDER>(MemoryAccessClass::Exclusive, true),
                    MemoryValue::from_bits(size::<SIZE>(), u128::from(value)),
                    reservation,
                )?
                .1
        } else {
            false
        };
        context.slow_result_low = u64::from(!stored);
        Ok(())
    });
}

unsafe extern "C" fn store_exclusive_pair<const SIZE: u8, const ORDER: u8>(
    context: *mut NativeContext,
    address: u64,
    value_low: u64,
    value_high: u64,
) {
    contain(context, |context| {
        let monitor = unsafe { &mut *context.exclusive };
        let reservation = monitor.reservation();
        monitor.clear();
        let element_bits = u32::from(SIZE) * 4;
        let value = u128::from(value_low) | (u128::from(value_high) << element_bits);
        let stored = if let Some(reservation) = reservation {
            unsafe { &*context.memory }
                .store_exclusive(
                    AddressSpaceId::new(context.address_space),
                    GuestVirtualAddress::new(address),
                    access::<SIZE, ORDER>(MemoryAccessClass::Exclusive, true),
                    MemoryValue::from_bits(size::<SIZE>(), value),
                    reservation,
                )?
                .1
        } else {
            false
        };
        context.slow_result_low = u64::from(!stored);
        Ok(())
    });
}

fn barrier_data_memory(domain: BarrierDomain, access: BarrierAccess) -> ContextFn {
    barrier_selector::<0>(domain, access)
}

fn barrier_data_synchronization(domain: BarrierDomain, access: BarrierAccess) -> ContextFn {
    barrier_selector::<1>(domain, access)
}

fn barrier_selector<const SYNC: u8>(domain: BarrierDomain, access: BarrierAccess) -> ContextFn {
    macro_rules! select_access {
        ($domain:literal) => {
            match access {
                BarrierAccess::Reads => barrier_call::<SYNC, $domain, 0>,
                BarrierAccess::Writes => barrier_call::<SYNC, $domain, 1>,
                BarrierAccess::ReadsAndWrites => barrier_call::<SYNC, $domain, 2>,
            }
        };
    }
    match domain {
        BarrierDomain::NonShareable => select_access!(0),
        BarrierDomain::InnerShareable => select_access!(1),
        BarrierDomain::OuterShareable => select_access!(2),
        BarrierDomain::FullSystem => select_access!(3),
    }
}

unsafe extern "C" fn barrier_call<const SYNC: u8, const DOMAIN: u8, const ACCESS: u8>(
    context: *mut NativeContext,
) {
    contain(context, |context| {
        let operation = if SYNC == 0 {
            BarrierOperation::DataMemory {
                domain: barrier_domain::<DOMAIN>(),
                access: barrier_access::<ACCESS>(),
            }
        } else {
            BarrierOperation::DataSynchronization {
                domain: barrier_domain::<DOMAIN>(),
                access: barrier_access::<ACCESS>(),
            }
        };
        unsafe { &*context.memory }.memory_barrier(operation);
        Ok(())
    });
}

unsafe extern "C" fn barrier_isb(context: *mut NativeContext) {
    contain(context, |context| {
        unsafe { &*context.memory }.memory_barrier(BarrierOperation::InstructionSynchronization);
        Ok(())
    });
}

macro_rules! cache_call {
    ($name:ident, $kind:expr, $address:expr) => {
        unsafe extern "C" fn $name(context: *mut NativeContext, address: u64) {
            contain(context, |context| {
                unsafe { &*context.memory }.maintain_cache(
                    AddressSpaceId::new(context.address_space),
                    $kind,
                    $address(address),
                )?;
                Ok(())
            });
        }
    };
}

unsafe extern "C" fn cache_instruction_all(context: *mut NativeContext) {
    contain(context, |context| {
        unsafe { &*context.memory }.maintain_cache(
            AddressSpaceId::new(context.address_space),
            CacheMaintenanceKind::InstructionInvalidate,
            None,
        )?;
        Ok(())
    });
}
cache_call!(
    cache_instruction_address,
    CacheMaintenanceKind::InstructionInvalidate,
    |address| Some(GuestVirtualAddress::new(address))
);
cache_call!(
    cache_data_invalidate,
    CacheMaintenanceKind::DataInvalidate,
    |address| Some(GuestVirtualAddress::new(address))
);
cache_call!(
    cache_data_clean,
    CacheMaintenanceKind::DataClean,
    |address| Some(GuestVirtualAddress::new(address))
);
cache_call!(
    cache_data_clean_invalidate,
    CacheMaintenanceKind::DataCleanAndInvalidate,
    |address| Some(GuestVirtualAddress::new(address))
);

#[repr(u64)]
pub(super) enum FpUnaryKind {
    SquareRoot,
    ConvertSingleDouble,
    ConvertDoubleSingle,
    RoundNearestEven,
    RoundPositive,
    RoundNegative,
    RoundZero,
    RoundNearestAway,
    RoundExact,
    RoundCurrent,
    VectorSignedIntToFloat,
    VectorUnsignedIntToFloat,
    ScalarVectorSignedIntToFloat,
    ScalarVectorUnsignedIntToFloat,
}

#[repr(u64)]
pub(super) enum FpBinaryKind {
    Add,
    Subtract,
    Multiply,
    NegatedMultiply,
    Divide,
    VectorDivide,
    VectorMultiplyElement,
}

#[repr(u64)]
pub(super) enum FpFusedKind {
    MultiplyAdd,
    MultiplySubtract,
    NegatedMultiplyAdd,
    NegatedMultiplySubtract,
}

#[repr(u64)]
pub(super) enum FpCompareKind {
    Register,
    Zero,
    Conditional,
}

fn finish_fp(
    context: *mut NativeContext,
    outcome: ExactFpOutcome,
    fpcr: u32,
    fpsr: u32,
    nzcv: u32,
) {
    let Some(context) = (unsafe { context.as_mut() }) else {
        return;
    };
    context.slow_status = STATUS_OK;
    context.data_fault = None;
    if a64_fp_simd::fp_status_traps(outcome.status, fpcr) {
        context.slow_status = STATUS_FP_TRAP;
        return;
    }
    context.slow_result_low = outcome.bits as u64;
    context.slow_result_high = (outcome.bits >> 64) as u64;
    context.slow_result_flags =
        u64::from(fpsr | a64_fp_simd::fp_status_bits(outcome.status)) | (u64::from(nzcv) << 32);
}

fn finish_integer(context: *mut NativeContext, outcome: ExactIntegerOutcome, fpcr: u32, fpsr: u32) {
    let value = if outcome.width == 32 {
        u64::from(outcome.value as u32)
    } else {
        outcome.value
    };
    finish_fp(
        context,
        ExactFpOutcome {
            bits: u128::from(value),
            status: outcome.status,
        },
        fpcr,
        fpsr,
        0,
    );
}

pub(super) unsafe extern "C" fn fp_unary(
    context: *mut NativeContext,
    input_low: u64,
    input_high: u64,
    fpcr: u64,
    fpsr: u64,
    shape: u64,
) {
    let input = u128::from(input_low) | (u128::from(input_high) << 64);
    let kind = shape & 0xff;
    let precision = if (shape >> 8) & 0xff == 0 { 32 } else { 64 };
    let vector_bits = if shape & (1 << 16) != 0 { 128 } else { 64 };
    let outcome = if kind == FpUnaryKind::SquareRoot as u64 {
        a64_fp_simd::exact_scalar_float_square_root(input_low, precision, fpcr as u32)
    } else if kind == FpUnaryKind::ConvertSingleDouble as u64 {
        a64_fp_simd::exact_float_convert(input_low, FloatConversion::SingleToDouble, fpcr as u32)
    } else if kind == FpUnaryKind::ConvertDoubleSingle as u64 {
        a64_fp_simd::exact_float_convert(input_low, FloatConversion::DoubleToSingle, fpcr as u32)
    } else if kind == FpUnaryKind::VectorSignedIntToFloat as u64
        || kind == FpUnaryKind::VectorUnsignedIntToFloat as u64
    {
        let (bits, inexact) = a64_fp_simd::exact_vector_integer_to_float(
            input,
            precision,
            vector_bits,
            kind == FpUnaryKind::VectorSignedIntToFloat as u64,
            fpcr as u32,
        );
        ExactFpOutcome {
            bits,
            status: FpStatus {
                inexact,
                ..FpStatus::default()
            },
        }
    } else if kind == FpUnaryKind::ScalarVectorSignedIntToFloat as u64
        || kind == FpUnaryKind::ScalarVectorUnsignedIntToFloat as u64
    {
        let (bits, inexact) = a64_fp_simd::exact_scalar_vector_integer_to_float(
            input_low,
            precision,
            kind == FpUnaryKind::ScalarVectorSignedIntToFloat as u64,
            fpcr as u32,
        );
        ExactFpOutcome {
            bits: u128::from(bits),
            status: FpStatus {
                inexact,
                ..FpStatus::default()
            },
        }
    } else {
        let operation = match kind {
            value if value == FpUnaryKind::RoundNearestEven as u64 => {
                FloatRoundOperation::NearestEven
            }
            value if value == FpUnaryKind::RoundPositive as u64 => {
                FloatRoundOperation::TowardPositive
            }
            value if value == FpUnaryKind::RoundNegative as u64 => {
                FloatRoundOperation::TowardNegative
            }
            value if value == FpUnaryKind::RoundZero as u64 => FloatRoundOperation::TowardZero,
            value if value == FpUnaryKind::RoundNearestAway as u64 => {
                FloatRoundOperation::NearestAway
            }
            value if value == FpUnaryKind::RoundExact as u64 => FloatRoundOperation::Exact,
            value if value == FpUnaryKind::RoundCurrent as u64 => FloatRoundOperation::CurrentMode,
            _ => return typed_fp_internal(context),
        };
        a64_fp_simd::exact_scalar_float_round(input_low, precision, operation, fpcr as u32)
    };
    finish_fp(context, outcome, fpcr as u32, fpsr as u32, 0);
}

pub(super) unsafe extern "C" fn fp_binary(
    context: *mut NativeContext,
    first_low: u64,
    first_high: u64,
    second_low: u64,
    second_high: u64,
    fpcr: u64,
    fpsr: u64,
    shape: u64,
) {
    let first = u128::from(first_low) | (u128::from(first_high) << 64);
    let second = u128::from(second_low) | (u128::from(second_high) << 64);
    let kind = shape & 0xff;
    let precision = if (shape >> 8) & 0xff == 0 { 32 } else { 64 };
    let vector_bits = if shape & (1 << 16) != 0 { 128 } else { 64 };
    let outcome = match kind {
        value if value == FpBinaryKind::Add as u64 => a64_fp_simd::exact_scalar_float_add(
            first_low,
            second_low,
            precision,
            FloatAddOperation::Add,
            fpcr as u32,
        ),
        value if value == FpBinaryKind::Subtract as u64 => a64_fp_simd::exact_scalar_float_add(
            first_low,
            second_low,
            precision,
            FloatAddOperation::Subtract,
            fpcr as u32,
        ),
        value if value == FpBinaryKind::Multiply as u64 => {
            a64_fp_simd::exact_scalar_float_multiply(
                first_low,
                second_low,
                precision,
                FloatMultiplyOperation::Multiply,
                fpcr as u32,
            )
        }
        value if value == FpBinaryKind::NegatedMultiply as u64 => {
            a64_fp_simd::exact_scalar_float_multiply(
                first_low,
                second_low,
                precision,
                FloatMultiplyOperation::NegatedMultiply,
                fpcr as u32,
            )
        }
        value if value == FpBinaryKind::Divide as u64 => {
            a64_fp_simd::exact_scalar_float_divide(first_low, second_low, precision, fpcr as u32)
        }
        value if value == FpBinaryKind::VectorDivide as u64 => {
            a64_fp_simd::exact_vector_float_divide(
                first,
                second,
                precision,
                vector_bits,
                fpcr as u32,
            )
        }
        value if value == FpBinaryKind::VectorMultiplyElement as u64 => {
            a64_fp_simd::exact_vector_float_multiply_element(
                first,
                second,
                precision,
                vector_bits,
                ((shape >> 24) & 0xff) as u8,
                fpcr as u32,
            )
        }
        _ => return typed_fp_internal(context),
    };
    finish_fp(context, outcome, fpcr as u32, fpsr as u32, 0);
}

pub(super) unsafe extern "C" fn fp_fused(
    context: *mut NativeContext,
    first_low: u64,
    second_low: u64,
    third_low: u64,
    fpcr: u64,
    fpsr: u64,
    shape: u64,
) {
    let operation = match shape & 0xff {
        value if value == FpFusedKind::MultiplyAdd as u64 => {
            FloatFusedMultiplyOperation::MultiplyAdd
        }
        value if value == FpFusedKind::MultiplySubtract as u64 => {
            FloatFusedMultiplyOperation::MultiplySubtract
        }
        value if value == FpFusedKind::NegatedMultiplyAdd as u64 => {
            FloatFusedMultiplyOperation::NegatedMultiplyAdd
        }
        value if value == FpFusedKind::NegatedMultiplySubtract as u64 => {
            FloatFusedMultiplyOperation::NegatedMultiplySubtract
        }
        _ => return typed_fp_internal(context),
    };
    let precision = if (shape >> 8) & 0xff == 0 { 32 } else { 64 };
    let outcome = a64_fp_simd::exact_scalar_float_fused_multiply_add(
        first_low,
        second_low,
        third_low,
        precision,
        operation,
        fpcr as u32,
    );
    finish_fp(context, outcome, fpcr as u32, fpsr as u32, 0);
}

pub(super) unsafe extern "C" fn fp_integer_to_float(
    context: *mut NativeContext,
    input: u64,
    fpcr: u64,
    fpsr: u64,
    shape: u64,
) {
    let source_bits = if (shape >> 16) & 2 != 0 { 64 } else { 32 };
    let destination_bits = if (shape >> 8) & 0xff == 0 { 32 } else { 64 };
    let (bits, inexact) = a64_fp_simd::exact_scalar_integer_to_float(
        input,
        source_bits,
        destination_bits,
        shape & 1 == 0,
        fpcr as u32,
    );
    finish_fp(
        context,
        ExactFpOutcome {
            bits: u128::from(bits),
            status: FpStatus {
                inexact,
                ..FpStatus::default()
            },
        },
        fpcr as u32,
        fpsr as u32,
        0,
    );
}

pub(super) unsafe extern "C" fn fp_float_to_integer(
    context: *mut NativeContext,
    input: u64,
    fpcr: u64,
    fpsr: u64,
    shape: u64,
) {
    let rounding = match (shape >> 24) & 0xff {
        0 => FloatToIntegerRounding::NearestEven,
        1 => FloatToIntegerRounding::NearestAway,
        2 => FloatToIntegerRounding::TowardPositive,
        3 => FloatToIntegerRounding::TowardNegative,
        _ => FloatToIntegerRounding::TowardZero,
    };
    let source_bits = if (shape >> 8) & 0xff == 0 { 32 } else { 64 };
    let width = if (shape >> 16) & 2 != 0 { 64 } else { 32 };
    let fraction = ((shape >> 32) & 0xff) as u8;
    let outcome = a64_fp_simd::exact_float_to_integer(
        input,
        source_bits,
        width,
        shape & 1 == 0,
        rounding,
        if fraction == u8::MAX { 0 } else { fraction },
        fpcr as u32,
    );
    finish_integer(context, outcome, fpcr as u32, fpsr as u32);
}

pub(super) unsafe extern "C" fn fp_compare(
    context: *mut NativeContext,
    first: u64,
    second: u64,
    fpcr: u64,
    fpsr: u64,
    nzcv: u64,
    shape: u64,
) {
    let kind = shape & 0xff;
    if kind == FpCompareKind::Conditional as u64
        && !evaluate_a64(
            Condition::from_encoding(((shape >> 24) & 0xff) as u8),
            nzcv as u32,
        )
    {
        finish_fp(
            context,
            ExactFpOutcome {
                bits: 0,
                status: FpStatus::default(),
            },
            fpcr as u32,
            fpsr as u32,
            (((shape >> 32) & 0xf) as u32) << 28,
        );
        return;
    }
    if kind != FpCompareKind::Register as u64
        && kind != FpCompareKind::Zero as u64
        && kind != FpCompareKind::Conditional as u64
    {
        return typed_fp_internal(context);
    }
    let outcome: ExactCompareOutcome = a64_fp_simd::exact_scalar_float_compare(
        first,
        if kind == FpCompareKind::Zero as u64 {
            0
        } else {
            second
        },
        if (shape >> 8) & 0xff == 0 { 32 } else { 64 },
        shape & (1 << 16) != 0,
        fpcr as u32,
    );
    finish_fp(
        context,
        ExactFpOutcome {
            bits: 0,
            status: outcome.status,
        },
        fpcr as u32,
        fpsr as u32,
        outcome.nzcv,
    );
}

fn typed_fp_internal(context: *mut NativeContext) {
    if let Some(context) = unsafe { context.as_mut() } {
        context.slow_status = STATUS_INTERNAL;
    }
}

fn contain(
    context: *mut NativeContext,
    operation: impl FnOnce(&mut NativeContext) -> Result<(), DataAccessFault>,
) {
    let Some(context) = (unsafe { context.as_mut() }) else {
        return;
    };
    context.slow_status = STATUS_OK;
    context.data_fault = None;
    match catch_unwind(AssertUnwindSafe(|| operation(context))) {
        Ok(Ok(())) => {}
        Ok(Err(fault)) => {
            context.data_fault = Some(fault);
            context.slow_status = STATUS_DATA_FAULT;
        }
        Err(_) => context.slow_status = STATUS_INTERNAL,
    }
}

const fn size<const SIZE: u8>() -> MemoryAccessSize {
    match SIZE {
        1 => MemoryAccessSize::Byte,
        2 => MemoryAccessSize::Halfword,
        4 => MemoryAccessSize::Word,
        8 => MemoryAccessSize::Doubleword,
        16 => MemoryAccessSize::Quadword,
        _ => panic!("invalid direct memory access size"),
    }
}

const fn ordering<const ORDER: u8>() -> MemoryOrdering {
    match ORDER {
        0 => MemoryOrdering::Relaxed,
        1 => MemoryOrdering::Acquire,
        2 => MemoryOrdering::Release,
        3 => MemoryOrdering::AcquireRelease,
        _ => panic!("invalid direct memory ordering"),
    }
}

const fn access<const SIZE: u8, const ORDER: u8>(
    class: MemoryAccessClass,
    aligned: bool,
) -> MemoryAccess {
    MemoryAccess::new(
        size::<SIZE>(),
        if aligned {
            MemoryAlignment::Natural
        } else {
            MemoryAlignment::Unaligned
        },
        ordering::<ORDER>(),
        class,
    )
}

const fn atomic_kind<const KIND: u8>() -> AtomicRmwKind {
    match KIND {
        0 => AtomicRmwKind::Add,
        1 => AtomicRmwKind::Clear,
        2 => AtomicRmwKind::Xor,
        3 => AtomicRmwKind::Set,
        4 => AtomicRmwKind::SignedMaximum,
        5 => AtomicRmwKind::SignedMinimum,
        6 => AtomicRmwKind::UnsignedMaximum,
        7 => AtomicRmwKind::UnsignedMinimum,
        8 => AtomicRmwKind::Swap,
        _ => panic!("invalid direct atomic operation"),
    }
}

const fn barrier_domain<const DOMAIN: u8>() -> BarrierDomain {
    match DOMAIN {
        0 => BarrierDomain::NonShareable,
        1 => BarrierDomain::InnerShareable,
        2 => BarrierDomain::OuterShareable,
        3 => BarrierDomain::FullSystem,
        _ => panic!("invalid direct barrier domain"),
    }
}

const fn barrier_access<const ACCESS: u8>() -> BarrierAccess {
    match ACCESS {
        0 => BarrierAccess::Reads,
        1 => BarrierAccess::Writes,
        2 => BarrierAccess::ReadsAndWrites,
        _ => panic!("invalid direct barrier access"),
    }
}
