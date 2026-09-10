use super::*;
use crate::lcq::Fragment;
use nixe_cpu::memory::{
    AtomicRmwKind, CacheMaintenanceKind, CpuMemory, ExecutableMemory, MemoryAccess,
    MemoryAccessClass, MemoryAccessSize, MemoryAlignment, MemoryOrdering, MemoryValue,
};

#[test]
fn checked_cpu_writes_dirty_captures_but_invalidate_published_code_only_at_ic() {
    for size in [
        MemoryAccessSize::Byte,
        MemoryAccessSize::Halfword,
        MemoryAccessSize::Word,
        MemoryAccessSize::Doubleword,
        MemoryAccessSize::Quadword,
    ] {
        for operation in 0..4 {
            let (process, memory) = fixture();
            writes::writable_alias(&memory);
            let code = publish(&process, &memory, 0x1000);
            let captured = Fragment::capture(&memory, key(0x1000)).unwrap();
            let cursor = memory.invalidation_cursor();
            let mut compiling = process.register().unwrap();
            let compile::Request::Owner(claim) = compiling.claim(key(0x2000)).unwrap() else {
                panic!()
            };
            let mut reader = process.register().unwrap();
            let mut state = A64State::default();
            let mut frame = NativeFrame::new(&mut state, PollBudget::new(4096, 1000).unwrap());
            let lease = memory.acquire_execution_lease();
            let invocation = unsafe { reader.admit(&mut frame, key(0x1000)) }
                .unwrap()
                .unwrap();
            let address = GuestVirtualAddress::new(0x3000);
            let previous = memory
                .read(SPACE, address, MemoryAccess::normal(size))
                .unwrap()
                .value;
            let next = MemoryValue::from_bits(size, previous.bits() ^ 1);
            let access = MemoryAccess::new(
                size,
                MemoryAlignment::Natural,
                MemoryOrdering::AcquireRelease,
                if operation == 2 {
                    MemoryAccessClass::Exclusive
                } else {
                    MemoryAccessClass::Atomic
                },
            );
            match operation {
                0 => assert!(
                    memory
                        .atomic_compare_exchange(SPACE, address, access, previous, next)
                        .unwrap()
                        .stored
                ),
                1 => assert!(
                    memory
                        .atomic_read_modify_write(SPACE, address, access, AtomicRmwKind::Swap, next)
                        .unwrap()
                        .stored
                ),
                2 => {
                    let (_, reservation) = memory.load_exclusive(SPACE, address, access).unwrap();
                    assert!(
                        memory
                            .store_exclusive(SPACE, address, access, next, reservation)
                            .unwrap()
                            .1
                    );
                }
                3 => {
                    memory
                        .write(SPACE, address, MemoryAccess::normal(size), next)
                        .unwrap();
                }
                _ => unreachable!(),
            }
            assert!(!memory.image_is_current(&captured.image));
            assert_eq!(
                memory
                    .read(SPACE, address, MemoryAccess::normal(size))
                    .unwrap()
                    .value,
                next
            );
            assert_eq!(memory.invalidation_cursor(), cursor);
            claim.validate().unwrap();
            assert!(process.snapshot(code).is_ok());
            drop(invocation);
            drop(lease);
            memory
                .maintain_cache(
                    SPACE,
                    CacheMaintenanceKind::InstructionInvalidate,
                    Some(address),
                )
                .unwrap();
            assert!(matches!(process.snapshot(code), Err(Error::StaleUnit)));
            assert_eq!(claim.validate(), Err(Error::StalePublication));
        }
    }
}
