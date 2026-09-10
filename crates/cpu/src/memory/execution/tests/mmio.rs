use super::*;
use crate::memory::MemoryAccessSize;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{OnceLock, Weak};

const SPACE: AddressSpaceId = AddressSpaceId::new(1);
const DEVICE: GuestVirtualAddress = GuestVirtualAddress::new(0x1000);
const ALIAS: GuestVirtualAddress = GuestVirtualAddress::new(0x3000);
const RAM: GuestVirtualAddress = GuestVirtualAddress::new(0x5000);

struct Device {
    memory: Arc<OnceLock<Weak<ExecutionMemory>>>,
    calls: Arc<AtomicUsize>,
    fail: bool,
    wrong_size: bool,
}

impl Device {
    fn access(&self) -> Result<(), Box<str>> {
        let count = self.calls.fetch_add(1, Ordering::Relaxed);
        let memory = self.memory.get().unwrap().upgrade().unwrap();
        assert!(
            memory.inner.try_lock().is_ok(),
            "MMIO retained the mapping lock"
        );
        memory
            .write(
                SPACE,
                RAM,
                MemoryAccess::normal(MemoryAccessSize::Word),
                MemoryValue::U32(17),
            )
            .unwrap();
        if count == 0 {
            memory
                .resize_zeroed_mapping(
                    SPACE,
                    DEVICE,
                    4096,
                    0,
                    MemoryPermissions::READ_WRITE,
                    MemoryMappingPurpose::Normal,
                )
                .unwrap();
            memory
                .resize_zeroed_mapping(
                    SPACE,
                    DEVICE,
                    0,
                    4096,
                    MemoryPermissions::READ_WRITE,
                    MemoryMappingPurpose::Normal,
                )
                .unwrap();
        }
        if self.fail {
            Err("MMIO callback rejected after its side effect".into())
        } else {
            Ok(())
        }
    }
}

impl SyntheticMmio for Device {
    fn read(&mut self, _: u64, _: MemoryAccess) -> Result<MemoryValue, Box<str>> {
        self.access()?;
        Ok(if self.wrong_size {
            MemoryValue::U8(23)
        } else {
            MemoryValue::U32(23)
        })
    }
    fn write(&mut self, _: u64, _: MemoryAccess, value: MemoryValue) -> Result<(), Box<str>> {
        assert_eq!(value, MemoryValue::U32(29));
        self.access()
    }
}

fn fixture(device: impl SyntheticMmio + 'static) -> Arc<ExecutionMemory> {
    let mut memory = ExecutionMemory::new();
    assert!(memory.add_mmio_page(GuestPhysicalPageId::new(1), device));
    assert!(memory.add_ram_page(GuestPhysicalPageId::new(2)));
    for (address, page) in [(DEVICE, 1), (ALIAS, 1), (RAM, 2)] {
        assert!(memory.map_page(
            SPACE,
            address,
            GuestPhysicalPageId::new(page),
            MemoryPermissions::READ_WRITE
        ));
    }
    Arc::new(memory)
}

#[test]
fn mmio_callbacks_remap_their_address_without_replaying_side_effects_or_losing_errors() {
    for mode in 0..5 {
        let owner = Arc::new(OnceLock::new());
        let calls = Arc::new(AtomicUsize::new(0));
        let memory = fixture(Device {
            memory: owner.clone(),
            calls: calls.clone(),
            fail: matches!(mode, 2 | 3),
            wrong_size: mode == 4,
        });
        owner.set(Arc::downgrade(&memory)).unwrap();
        let access = MemoryAccess::normal(MemoryAccessSize::Word);
        let result = if matches!(mode, 1 | 3) {
            memory
                .write(SPACE, DEVICE, access, MemoryValue::U32(29))
                .map(|result| result.region)
        } else {
            memory.read(SPACE, DEVICE, access).map(|result| {
                assert_eq!(result.value, MemoryValue::U32(23));
                result.region
            })
        };
        match mode {
            0 | 1 => assert_eq!(result.unwrap(), MemoryRegionKind::Device),
            2 | 3 => assert!(
                matches!(result.unwrap_err().reason, DataAccessFaultReason::Device(ref text) if &**text == "MMIO callback rejected after its side effect")
            ),
            4 => assert_eq!(
                result.unwrap_err().reason,
                DataAccessFaultReason::ValueSizeMismatch
            ),
            _ => unreachable!(),
        }
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(
            memory.read(SPACE, DEVICE, access).unwrap().value,
            MemoryValue::U32(0)
        );
        assert_eq!(
            memory.read(SPACE, RAM, access).unwrap().value,
            MemoryValue::U32(17)
        );
        // The physical alias retains the very same mutable handler.
        let _ = memory.read(SPACE, ALIAS, access);
        assert_eq!(calls.load(Ordering::Relaxed), 2);
    }
}

#[test]
fn panicking_mmio_poison_is_reported_without_poisoning_mapping_or_ram_access() {
    struct PanicDevice;
    impl SyntheticMmio for PanicDevice {
        fn read(&mut self, _: u64, _: MemoryAccess) -> Result<MemoryValue, Box<str>> {
            panic!("injected device panic")
        }
        fn write(&mut self, _: u64, _: MemoryAccess, _: MemoryValue) -> Result<(), Box<str>> {
            panic!("poisoned handler must not be called again")
        }
    }
    let memory = fixture(PanicDevice);
    let access = MemoryAccess::normal(MemoryAccessSize::Word);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(
            || memory.read(SPACE, DEVICE, access)
        ))
        .is_err()
    );
    for error in [
        memory.read(SPACE, ALIAS, access).unwrap_err(),
        memory
            .write(SPACE, DEVICE, access, MemoryValue::U32(0))
            .unwrap_err(),
    ] {
        assert!(
            matches!(error.reason, DataAccessFaultReason::HostBacking(ref text) if text.contains("MMIO handler poisoned"))
        );
    }
    assert!(!memory.inner.is_poisoned());
    memory
        .write(SPACE, RAM, access, MemoryValue::U32(31))
        .unwrap();
    assert_eq!(
        memory.read(SPACE, RAM, access).unwrap().value,
        MemoryValue::U32(31)
    );
}
