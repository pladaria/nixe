//! Maxwell NvMap pin policy: CPU-relative heap offsets and independent
//! 64 KiB reservations share the GPU's actual device memory manager.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/service/nvdrv/core/heap_mapper.cpp
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/service/nvdrv/core/nvmap.cpp
use nixe_memory::{
    AddressSpaceId, AllocationRange, GuestVirtualAddress, Machine, MemoryError, PAGE_SIZE,
};
use nixe_memory::{
    PhysicalMapping,
    device_memory::{DeviceMapping, DeviceMemory, DeviceReservation},
};
use std::sync::{Arc, RwLock};
const DEVICE_BITS: u32 = 34;
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
pub struct MaxwellDeviceMemory(Arc<Inner>);
#[derive(Debug)]
pub struct MaxwellDeviceMapping {
    // Drop pins before releasing the address reservation or session heap.
    pin: DeviceMapping,
    memory: MaxwellDeviceMemory,
    _reservation: Option<DeviceReservation>,
}
impl MaxwellDeviceMemory {
    pub fn new(machine: Machine) -> Self {
        Self::from_device(DeviceMemory::new(machine, DEVICE_BITS))
    }
    /// NvCore session heap policy uses the GPU's existing SMMU, as in Eden.
    pub fn from_device(device: DeviceMemory) -> Self {
        assert_eq!(device.address_bits(), DEVICE_BITS);
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
        cpu: Option<(AddressSpaceId, GuestVirtualAddress)>,
    ) -> Result<Arc<MaxwellDeviceMapping>, MemoryError> {
        let bytes = ranges.iter().map(|r| r.bytes()).sum::<u64>();
        let heap = self.0.heap.read().unwrap();
        let address = cpu.and_then(|(process, address)| {
            heap.as_ref()
                .filter(|heap| {
                    heap.process == process
                        && address.get() >= heap.cpu
                        && address
                            .get()
                            .checked_add(bytes)
                            .is_some_and(|end| end <= heap.cpu + heap.physical.bytes())
                })
                .map(|heap| heap.reservation.address() + address.get() - heap.cpu)
        });
        let reservation = if address.is_none() {
            Some(self.device().reserve(
                bytes.div_ceil(RESERVATION_ALIGNMENT) * RESERVATION_ALIGNMENT,
                RESERVATION_ALIGNMENT,
            )?)
        } else {
            None
        };
        let address = address.unwrap_or_else(|| reservation.as_ref().unwrap().address());
        let pin = self.device().map(address, ranges, reservation.is_some())?;
        Ok(Arc::new(MaxwellDeviceMapping {
            pin,
            memory: self.clone(),
            _reservation: reservation,
        }))
    }
}
impl MaxwellDeviceMapping {
    pub fn address(&self) -> u64 {
        self.pin.address()
    }
    pub fn physical(&self) -> &Arc<PhysicalMapping> {
        self.pin.physical()
    }
    pub fn ranges(&self) -> &[AllocationRange] {
        self.pin.ranges()
    }
    pub fn belongs_to(&self, memory: &MaxwellDeviceMemory) -> bool {
        self.memory.device().same_device(memory.device())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn input(
        machine: &Machine,
        bytes: u64,
        value: u8,
    ) -> (nixe_memory::Allocation, Arc<[AllocationRange]>) {
        let allocation = machine.account(bytes).unwrap().allocate(bytes).unwrap();
        let ranges: Arc<[AllocationRange]> =
            vec![AllocationRange::new(allocation.pin(), 0, bytes).unwrap()].into();
        machine
            .write_resolved(&ranges, &vec![value; bytes as usize])
            .unwrap();
        (allocation, ranges)
    }
    fn stream(view: &PhysicalMapping) -> Vec<u8> {
        let mut bytes = vec![0xff; view.bytes() as usize];
        view.capture_stream(0, &mut bytes).unwrap();
        bytes
    }
    #[test]
    fn shared_smmu_pins_belong_to_the_device_across_session_heap_policies() {
        let machine = Machine::new(4 * PAGE_SIZE).unwrap();
        let first = MaxwellDeviceMemory::new(machine.clone());
        let second = MaxwellDeviceMemory::from_device(first.device().clone());
        let other = MaxwellDeviceMemory::new(machine.clone());
        let (_allocation, input) = input(&machine, PAGE_SIZE, 0x29);
        let mapped = first.map(input, None).unwrap();
        assert!(mapped.belongs_to(&second));
        assert!(!mapped.belongs_to(&other));
        assert_eq!(
            stream(&second.device().read_mapping(mapped.address(), 32).unwrap()),
            [0x29; 32]
        );
    }
    #[test]
    fn heap_offsets_cross_discontiguous_objects_and_preserve_unmapped_device_pages() {
        let machine = Machine::new(16 * PAGE_SIZE).unwrap();
        let device = MaxwellDeviceMemory::new(machine.clone());
        let process = AddressSpaceId::new(7);
        let cpu = GuestVirtualAddress::new(0x8000_0000);
        let (_a, first) = input(&machine, PAGE_SIZE, 0x17);
        let (_gap, gap) = input(&machine, PAGE_SIZE, 0xdd);
        let (_b, second) = input(&machine, PAGE_SIZE, 0x29);
        let (_hole, hole) = input(&machine, PAGE_SIZE, 0xee);
        let cpu_pages = [
            first[0].clone(),
            hole[0].clone(),
            second[0].clone(),
            gap[0].clone(),
        ];
        device
            .register_heap(process, cpu, PhysicalMapping::new(&cpu_pages).unwrap())
            .unwrap();
        let first = device.map(first, Some((process, cpu))).unwrap();
        let second = device
            .map(
                second,
                Some((process, cpu.checked_add(2 * PAGE_SIZE).unwrap())),
            )
            .unwrap();
        assert_eq!(second.address(), first.address() + 2 * PAGE_SIZE);
        let view = device
            .device()
            .read_mapping(first.address(), 4 * PAGE_SIZE)
            .unwrap();
        let bytes = stream(&view);
        assert_eq!(&bytes[..PAGE_SIZE as usize], &[0x17; PAGE_SIZE as usize]);
        assert_eq!(
            &bytes[PAGE_SIZE as usize..2 * PAGE_SIZE as usize],
            &[0; PAGE_SIZE as usize]
        );
        assert_eq!(
            &bytes[2 * PAGE_SIZE as usize..3 * PAGE_SIZE as usize],
            &[0x29; PAGE_SIZE as usize]
        );
        assert_eq!(&bytes[3 * PAGE_SIZE as usize..], &[0; PAGE_SIZE as usize]);
        assert_eq!(view.spans().len(), 2);
        assert_eq!(
            view.missing(0..4 * PAGE_SIZE).collect::<Vec<_>>(),
            [PAGE_SIZE..2 * PAGE_SIZE, 3 * PAGE_SIZE..4 * PAGE_SIZE]
        );
        let mapped = view.slice(0, PAGE_SIZE).unwrap();
        assert_eq!(view.containing_offset(&mapped, 0..PAGE_SIZE), Some(0));
        let dense = PhysicalMapping::new(first.ranges()).unwrap();
        assert!(dense.containing_offset(&view, 0..4 * PAGE_SIZE).is_none());
    }
    #[test]
    fn overlapping_heap_pins_release_only_uncovered_pages_and_pin_partial_allocations() {
        let machine = Machine::new(4 * PAGE_SIZE).unwrap();
        let device = MaxwellDeviceMemory::new(machine.clone());
        let process = AddressSpaceId::new(1);
        let cpu = GuestVirtualAddress::new(0x4000_0000);
        let (allocation, ranges) = input(&machine, 2 * PAGE_SIZE, 0x53);
        device
            .register_heap(process, cpu, PhysicalMapping::new(&ranges).unwrap())
            .unwrap();
        let page: Arc<[AllocationRange]> =
            vec![ranges[0].slice(PAGE_SIZE, PAGE_SIZE).unwrap()].into();
        let whole = device
            .map(Arc::clone(&ranges), Some((process, cpu)))
            .unwrap();
        let generation = device.device().generation();
        let alias = device.map(ranges, Some((process, cpu))).unwrap();
        assert_eq!(device.device().generation(), generation);
        drop(alias);
        assert_eq!(
            device.device().generation(),
            generation,
            "overlapping pins do not change mapped coverage"
        );
        let second = device
            .map(page, Some((process, cpu.checked_add(PAGE_SIZE).unwrap())))
            .unwrap();
        assert_eq!(device.device().generation(), generation);
        let base = whole.address();
        let id = allocation.identity();
        drop(allocation);
        drop(whole);
        assert!(
            device.device().generation() > generation,
            "the uncovered first page changes coverage"
        );
        assert_eq!(machine.statistics().allocated_bytes, PAGE_SIZE);
        assert!(machine.pin(id).is_none());
        let view = device.device().read_mapping(base, 2 * PAGE_SIZE).unwrap();
        let bytes = stream(&view);
        assert_eq!(&bytes[..PAGE_SIZE as usize], &[0; PAGE_SIZE as usize]);
        // WalkBlock advances by the pretracked CPU continuity even when its
        // starting SMMU page is unmapped. A direct read of the live second page
        // still consumes the retained partial allocation.
        assert_eq!(&bytes[PAGE_SIZE as usize..], &[0; PAGE_SIZE as usize]);
        assert_eq!(
            stream(
                &device
                    .device()
                    .read_mapping(base + PAGE_SIZE, PAGE_SIZE)
                    .unwrap()
            ),
            vec![0x53; PAGE_SIZE as usize]
        );
        drop(second);
        assert_eq!(machine.statistics().allocated_bytes, 0);
        assert_eq!(
            stream(&device.device().read_mapping(base, 2 * PAGE_SIZE).unwrap()),
            vec![0; 2 * PAGE_SIZE as usize]
        );
    }
    #[test]
    fn contiguous_cpu_heap_block_reads_extend_beyond_the_first_nvmap_mapping() {
        let machine = Machine::new(4 * PAGE_SIZE).unwrap();
        let device = MaxwellDeviceMemory::new(machine.clone());
        let process = AddressSpaceId::new(4);
        let cpu = GuestVirtualAddress::new(0x9000_0000);
        let (_allocation, ranges) = input(&machine, 4 * PAGE_SIZE, 0x61);
        machine
            .write_resolved(
                &[ranges[0].slice(PAGE_SIZE, PAGE_SIZE).unwrap()],
                &vec![0x72; PAGE_SIZE as usize],
            )
            .unwrap();
        device
            .register_heap(process, cpu, PhysicalMapping::new(&ranges).unwrap())
            .unwrap();
        let first = device
            .map(
                vec![ranges[0].slice(0, PAGE_SIZE).unwrap()].into(),
                Some((process, cpu)),
            )
            .unwrap();
        let view = device
            .device()
            .read_mapping(first.address(), 2 * PAGE_SIZE)
            .unwrap();
        let bytes = stream(&view);
        assert_eq!(&bytes[..PAGE_SIZE as usize], &[0x61; PAGE_SIZE as usize]);
        assert_eq!(&bytes[PAGE_SIZE as usize..], &[0x72; PAGE_SIZE as usize]);
        assert_eq!(view.spans().len(), 1);
        assert!(view.missing(0..2 * PAGE_SIZE).next().is_none());
        let second = device
            .map(
                vec![ranges[0].slice(PAGE_SIZE, PAGE_SIZE).unwrap()].into(),
                Some((process, cpu.checked_add(PAGE_SIZE).unwrap())),
            )
            .unwrap();
        drop(second);
        assert_eq!(
            stream(
                &device
                    .device()
                    .read_mapping(first.address(), 2 * PAGE_SIZE)
                    .unwrap()
            ),
            bytes
        );
    }
    #[test]
    fn independent_device_allocations_keep_alignment_holes_and_reuse_after_last_pin() {
        let machine = Machine::new(4 * PAGE_SIZE).unwrap();
        let device = MaxwellDeviceMemory::new(machine.clone());
        let (_a, first) = input(&machine, PAGE_SIZE, 0x6a);
        let (_b, second) = input(&machine, PAGE_SIZE, 0x7b);
        let first = device.map(first, None).unwrap();
        let base = first.address();
        let second = device.map(second.clone(), None).unwrap();
        assert_eq!(second.address(), base + RESERVATION_ALIGNMENT);
        let view = device
            .device()
            .read_mapping(base + PAGE_SIZE - 3, RESERVATION_ALIGNMENT - PAGE_SIZE + 6)
            .unwrap();
        let bytes = stream(&view);
        assert_eq!(&bytes[..3], &[0x6a; 3]);
        assert!(bytes[3..bytes.len() - 3].iter().all(|&byte| byte == 0));
        assert_eq!(&bytes[bytes.len() - 3..], &[0x7b; 3]);
        drop(first);
        let replacement = device.map(second.ranges().to_vec().into(), None).unwrap();
        assert_eq!(replacement.address(), base);
        assert_eq!(
            stream(&device.device().read_mapping(base, 3).unwrap()),
            [0x7b; 3]
        );
    }
}
