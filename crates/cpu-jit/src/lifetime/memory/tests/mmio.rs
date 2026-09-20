use super::*;
use crate::lcq::invocation::{self, Exit};
use nixe_cpu::{
    exclusive::ExclusiveMonitorState,
    memory::{MemoryAccess, MemoryValue, SyntheticMmio},
};
use nixe_cpu_direct_memory::WorkerFaultContext;
use std::sync::{OnceLock, Weak, atomic::AtomicUsize};

struct Device {
    memory: Arc<OnceLock<Weak<ExecutionMemory>>>,
    calls: Arc<AtomicUsize>,
    drops: Arc<AtomicUsize>,
}

impl Device {
    fn access(&self) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let memory = self.memory.get().unwrap().upgrade().unwrap();
        // Requires the real JIT stop: the source invocation/lease must already
        // be gone, and MMIO must not hold the mapping lock through this callback.
        memory
            .set_permissions(
                SPACE,
                GuestVirtualAddress::new(0x1000),
                4096,
                MemoryPermissions::READ,
            )
            .unwrap();
        memory
            .resize_zeroed_mapping(
                SPACE,
                GuestVirtualAddress::new(0x3000),
                4096,
                0,
                MemoryPermissions::READ_WRITE,
                MemoryMappingPurpose::Normal,
            )
            .unwrap();
        assert_eq!(self.drops.load(Ordering::Relaxed), 0);
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::Relaxed);
    }
}

impl SyntheticMmio for Device {
    fn read(&mut self, _: u64, _: MemoryAccess) -> Result<MemoryValue, Box<str>> {
        self.access();
        Ok(MemoryValue::U64(23))
    }
    fn write(&mut self, _: u64, _: MemoryAccess, value: MemoryValue) -> Result<(), Box<str>> {
        assert_eq!(value, MemoryValue::U64(29));
        self.access();
        Ok(())
    }
}

#[test]
fn native_mmio_completion_can_retire_its_source_and_remove_its_own_device_mapping_once() {
    for write in [false, true] {
        let (process, mut memory) = fixture();
        if write {
            memory
                .initialize_ram(
                    GuestPhysicalPageId::new(1),
                    0,
                    &0xf9000020_u32.to_le_bytes(),
                )
                .unwrap(); // STR X0, [X1]
        }
        let owner = Arc::new(OnceLock::new());
        let calls = Arc::new(AtomicUsize::new(0));
        let drops = Arc::new(AtomicUsize::new(0));
        assert!(memory.add_mmio_page(
            GuestPhysicalPageId::new(3),
            Device {
                memory: owner.clone(),
                calls: calls.clone(),
                drops: drops.clone()
            }
        ));
        assert!(memory.map_page(
            SPACE,
            GuestVirtualAddress::new(0x3000),
            GuestPhysicalPageId::new(3),
            MemoryPermissions::READ_WRITE
        ));
        let memory = Arc::new(memory);
        owner.set(Arc::downgrade(&memory)).unwrap();
        let code = publish(&process, &memory, 0x1000);
        let mut reader = process.register().unwrap();
        let mut state = A64State::default();
        state.set_pc(0x1000);
        state.general_register_storage_mut()[0] = 29;
        state.general_register_storage_mut()[1] = 0x3000;
        let mut monitor = ExclusiveMonitorState::default();
        let mut worker = WorkerFaultContext::register().unwrap();
        let mut frame = NativeFrame::new(&mut state, PollBudget::new(4096, 1000).unwrap());
        let Exit::Memory {
            instruction,
            outcome,
            ..
        } = unsafe {
            invocation::run(
                &mut crate::sampling::Samples::new(),
                &mut reader,
                &mut frame,
                &memory,
                &mut worker,
                &mut monitor,
                key(0x1000),
            )
        }
        .unwrap()
        .unwrap()
        else {
            panic!("MMIO must escape to owned completion")
        };
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        assert!(
            outcome
                .complete(instruction, &mut state, &*memory, &mut monitor, 0)
                .unwrap()
                .is_none()
        );
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(drops.load(Ordering::Relaxed), 1);
        assert_eq!(state.pc(), 0x1004);
        assert_eq!(
            state.general_register_storage_mut()[0],
            if write { 29 } else { 23 }
        );
        assert_eq!(process.lock().phase, Phase::Open);
        assert!(matches!(process.snapshot(code), Err(Error::StaleUnit)));
    }
}
