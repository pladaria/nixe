//! CPU-relative GPU heap offsets and independent 64 KiB reservations share
//! the canonical device memory manager across console frontends.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/service/nvdrv/core/heap_mapper.cpp
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/service/nvdrv/core/nvmap.cpp
use crate::{
    AddressSpaceId, AllocationRange, GuestVirtualAddress, Machine, MemoryError, PAGE_SIZE,
};
use crate::{
    PhysicalMapping,
    device_memory::{DeviceMapping, DeviceMemory, DeviceReservation},
};
use std::sync::{Arc, RwLock};
const RESERVATION_ALIGNMENT: u64 = 64 * 1024;
#[derive(Debug)]
struct Heap {
    process: AddressSpaceId,
    cpu: u64,
    reservation: DeviceReservation,
    physical: Arc<PhysicalMapping>,
}
#[derive(Debug)]
struct Inner {
    device: DeviceMemory,
    heap: RwLock<Option<Heap>>,
}
#[derive(Clone, Debug)]
pub struct DeviceHeapMemory(Arc<Inner>);
#[derive(Debug)]
pub struct DeviceHeapMapping {
    // Drop pins before releasing the address reservation or session heap.
    pin: DeviceMapping,
    memory: DeviceHeapMemory,
    reservation_bytes: u64,
    _reservation: Option<Arc<DeviceReservation>>,
}
impl DeviceHeapMemory {
    pub fn new(device: DeviceMemory) -> Self {
        Self(Arc::new(Inner {
            device,
            heap: RwLock::new(None),
        }))
    }
    pub fn device(&self) -> &DeviceMemory {
        &self.0.device
    }
    pub fn machine(&self) -> &Machine {
        self.device().machine()
    }
    pub fn register_heap(
        &self,
        process: AddressSpaceId,
        cpu: GuestVirtualAddress,
        physical: Arc<PhysicalMapping>,
    ) -> Result<(), MemoryError> {
        let bytes = physical.bytes();
        if physical.machine() != self.machine().identity()
            || !(cpu.get() | bytes).is_multiple_of(PAGE_SIZE)
        {
            return Err(MemoryError::Invalid(
                "SMMU heap belongs to another machine or is unaligned",
            ));
        }
        let mut heap = self.0.heap.write().unwrap();
        if let Some(heap) = &*heap {
            return if (heap.process, heap.cpu, heap.physical.bytes()) == (process, cpu.get(), bytes)
            {
                Ok(())
            } else {
                Err(MemoryError::Invalid(
                    "SMMU session heap changed after registration",
                ))
            };
        }
        let reservation = self.device().reserve(
            bytes.div_ceil(RESERVATION_ALIGNMENT) * RESERVATION_ALIGNMENT,
            RESERVATION_ALIGNMENT,
        )?;
        self.device()
            .track_backing(reservation.address(), &physical)?;
        *heap = Some(Heap {
            process,
            cpu: cpu.get(),
            reservation,
            physical,
        });
        Ok(())
    }
    pub fn map(
        &self,
        ranges: Arc<[AllocationRange]>,
        reservation_bytes: u64,
        cpu: Option<(AddressSpaceId, GuestVirtualAddress)>,
    ) -> Result<Arc<DeviceHeapMapping>, MemoryError> {
        let bytes = ranges.iter().map(|r| r.bytes()).sum::<u64>();
        if reservation_bytes < bytes || !reservation_bytes.is_multiple_of(PAGE_SIZE) {
            return Err(MemoryError::Invalid(
                "SMMU reservation does not contain its CPU backing",
            ));
        }
        let heap = self.0.heap.read().unwrap();
        let address = cpu.and_then(|(process, address)| {
            heap.as_ref()
                .filter(|heap| {
                    heap.process == process
                        && address.get() >= heap.cpu
                        && address
                            .get()
                            .checked_add(reservation_bytes)
                            .is_some_and(|end| end <= heap.cpu + heap.physical.bytes())
                })
                .map(|heap| heap.reservation.address() + address.get() - heap.cpu)
        });
        let reservation = if address.is_none() {
            Some(Arc::new(self.device().reserve(
                reservation_bytes.next_multiple_of(RESERVATION_ALIGNMENT),
                RESERVATION_ALIGNMENT,
            )?))
        } else {
            None
        };
        let address = address.unwrap_or_else(|| reservation.as_ref().unwrap().address());
        let pin = self.device().map(address, ranges, reservation.is_some())?;
        Ok(Arc::new(DeviceHeapMapping {
            pin,
            memory: self.clone(),
            reservation_bytes,
            _reservation: reservation,
        }))
    }
}
impl DeviceHeapMapping {
    pub fn address(&self) -> u64 {
        self.pin.address()
    }
    pub fn reservation_bytes(&self) -> u64 {
        self.reservation_bytes
    }
    pub fn memory_machine(&self) -> crate::MachineId {
        self.memory.machine().identity()
    }
    pub fn read_mapping(
        &self,
        address: u64,
        bytes: u64,
    ) -> Result<Arc<PhysicalMapping>, MemoryError> {
        self.memory.device().read_mapping(address, bytes)
    }
    pub fn physical(&self) -> &Arc<PhysicalMapping> {
        self.pin.physical()
    }
    /// Pin a surviving slice before retiring the old mapping. The address
    /// reservation remains owned until every slice and GPU capture retires.
    pub fn slice(&self, offset: u64, bytes: u64) -> Result<Arc<Self>, MemoryError> {
        let ranges = crate::capture_subrange(self.pin.ranges(), offset, bytes)?;
        let pin = self.memory.device().map(
            self.address() + offset,
            ranges.into(),
            self._reservation.is_some(),
        )?;
        Ok(Arc::new(Self {
            pin,
            memory: self.memory.clone(),
            reservation_bytes: bytes,
            _reservation: self._reservation.clone(),
        }))
    }
    pub fn belongs_to(&self, memory: &DeviceHeapMemory) -> bool {
        self.memory.device().same_device(memory.device())
    }
}
