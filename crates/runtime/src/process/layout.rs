use super::*;

/// Runtime interpretation of the address-space selector validated by NPDM.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProcessAddressSpace {
    Bit32,
    Bit32NoReserved,
    Bit64Old,
    Bit64,
}

/// Horizon kernel generation governing process virtual-region availability.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ProcessMemoryLayoutProfile {
    Horizon1,
    #[default]
    Horizon2Plus,
}

impl ProcessAddressSpace {
    pub(super) const fn from_npdm(value: AddressSpaceType) -> Self {
        match value {
            AddressSpaceType::AddressSpace32Bit => Self::Bit32,
            AddressSpaceType::AddressSpace32BitNoReserved => Self::Bit32NoReserved,
            AddressSpaceType::AddressSpace64BitOld => Self::Bit64Old,
            AddressSpaceType::AddressSpace64Bit => Self::Bit64,
        }
    }

    pub const fn exclusive_limit(self) -> u64 {
        match self {
            Self::Bit32 | Self::Bit32NoReserved => 1_u64 << 32,
            Self::Bit64Old => 1_u64 << 36,
            Self::Bit64 => 1_u64 << 39,
        }
    }
}

/// One reserved guest-virtual region reported through platform process APIs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProcessVirtualRegion {
    base: GuestVirtualAddress,
    size: u64,
}

impl ProcessVirtualRegion {
    #[must_use]
    pub const fn new(base: GuestVirtualAddress, size: u64) -> Self {
        Self { base, size }
    }

    #[must_use]
    pub const fn base(self) -> GuestVirtualAddress {
        self.base
    }

    #[must_use]
    pub const fn size(self) -> u64 {
        self.size
    }

    pub(super) const fn end(self) -> u64 {
        self.base.get() + self.size
    }
}

/// Runtime-owned Horizon process virtual-memory layout.
///
/// Region dimensions and the 39-bit placement policy follow the public
/// Atmosphere `svc_memory_map.hpp` and `KPageTableBase::InitializeForProcess`
/// definitions. The ASLR window may contain the concrete heap, alias, and
/// stack reservations; it is not a disjoint allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProcessMemoryLayout {
    aslr: ProcessVirtualRegion,
    heap: ProcessVirtualRegion,
    alias: ProcessVirtualRegion,
    stack: ProcessVirtualRegion,
    memory_capacity: u64,
}

impl ProcessMemoryLayout {
    pub(super) fn for_address_space(
        profile: ProcessMemoryLayoutProfile,
        address_space: ProcessAddressSpace,
        process_code_start: u64,
        process_code_end: u64,
        memory_capacity: u64,
    ) -> Result<Self, ProcessBuildError> {
        if profile == ProcessMemoryLayoutProfile::Horizon1
            && address_space == ProcessAddressSpace::Bit64
        {
            return Err(error(
                ProcessBuildStage::Metadata,
                "39-bit process address spaces require Horizon 2.0.0 or newer",
            ));
        }
        let (aslr, alias, heap, stack) = match address_space {
            ProcessAddressSpace::Bit32 => (
                region(0x0020_0000, 0xffe0_0000),
                region(0x4000_0000, 0x4000_0000),
                region(0x8000_0000, 0x4000_0000),
                region(0x0020_0000, 0x3fe0_0000),
            ),
            ProcessAddressSpace::Bit32NoReserved => (
                region(0x0020_0000, 0xffe0_0000),
                region(0x4000_0000, 0),
                region(0x4000_0000, 0x8000_0000),
                region(0x0020_0000, 0x3fe0_0000),
            ),
            ProcessAddressSpace::Bit64Old => (
                region(0x0800_0000, 0xf_f800_0000),
                region(0x8000_0000, 0x1_8000_0000),
                region(0x2_0000_0000, 0x2_0000_0000),
                region(0x0800_0000, 0x7800_0000),
            ),
            ProcessAddressSpace::Bit64 => {
                let code_start = process_code_start & !(HORIZON_REGION_ALIGNMENT - 1);
                let code_end = align_up(process_code_end, HORIZON_REGION_ALIGNMENT)?;
                let aslr = region(0x0800_0000, (1_u64 << 39) - 0x0800_0000);
                if code_start < aslr.base().get() || code_end > aslr.end() {
                    return Err(error(
                        ProcessBuildStage::Placement,
                        "process code is outside the 39-bit Horizon ASLR window",
                    ));
                }

                let [_, stack, alias, heap] = layout_39_bit_regions(aslr, code_start, code_end)?;
                (aslr, alias, heap, stack)
            }
        };
        Ok(Self {
            aslr,
            heap,
            alias,
            stack,
            memory_capacity,
        })
    }

    #[must_use]
    pub const fn aslr(self) -> ProcessVirtualRegion {
        self.aslr
    }

    #[must_use]
    pub const fn heap(self) -> ProcessVirtualRegion {
        self.heap
    }

    #[must_use]
    pub const fn alias(self) -> ProcessVirtualRegion {
        self.alias
    }

    #[must_use]
    pub const fn stack(self) -> ProcessVirtualRegion {
        self.stack
    }

    /// Returns the process commit limit used for memory accounting.
    #[must_use]
    pub const fn memory_capacity(self) -> u64 {
        self.memory_capacity
    }
}

/// Caller-controlled process identities and relocatable image placement.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProcessBuildConfig {
    pub process_id: u64,
    pub address_space_id: AddressSpaceId,
    pub cpu_profile: GuestCpuProfile,
    pub memory_layout_profile: ProcessMemoryLayoutProfile,
    pub image_base: GuestVirtualAddress,
    /// Physical-memory resource limit assigned to the emulated process.
    pub physical_memory_limit: u64,
    /// Frequency exposed by architectural counter registers.
    pub architectural_timer_frequency: u64,
}

impl Default for ProcessBuildConfig {
    fn default() -> Self {
        Self {
            process_id: 1,
            address_space_id: AddressSpaceId::new(1),
            cpu_profile: GuestCpuProfile::switch_1(),
            memory_layout_profile: ProcessMemoryLayoutProfile::Horizon2Plus,
            image_base: GuestVirtualAddress::new(DEFAULT_IMAGE_BASE),
            physical_memory_limit: DEFAULT_PHYSICAL_MEMORY_LIMIT,
            // Horizon exposes the Switch 1 system counter at 19.2 MHz:
            // https://switchbrew.org/w/index.php?title=SVC&oldid=14679#svcGetSystemTick
            architectural_timer_frequency: 19_200_000,
        }
    }
}

const fn region(base: u64, size: u64) -> ProcessVirtualRegion {
    ProcessVirtualRegion::new(GuestVirtualAddress::new(base), size)
}

fn layout_39_bit_regions(
    aslr: ProcessVirtualRegion,
    code_start: u64,
    code_end: u64,
) -> Result<[ProcessVirtualRegion; 4], ProcessBuildError> {
    // Region kinds use Horizon's deterministic no-ASLR ordering:
    // kernel-map, stack, alias, heap.
    let sizes = [0x10_0000_0000, 0x8000_0000, 0x10_0000_0000, 0x2_0000_0000];
    let mut by_descending_size = [(0_usize, sizes[0]); 4];
    for (kind, entry) in by_descending_size.iter_mut().enumerate() {
        *entry = (kind, sizes[kind]);
    }
    by_descending_size.sort_by_key(|entry| std::cmp::Reverse(entry.1));

    let allocation_starts = [aslr.base().get(), code_end];
    let mut allocation_sizes = [code_start - aslr.base().get(), aslr.end() - code_end];
    let mut assignment = [usize::MAX; 4];
    for (kind, size) in by_descending_size {
        let allocation = usize::from(allocation_sizes[1] >= allocation_sizes[0]);
        if allocation_sizes[allocation] < size {
            return Err(error(
                ProcessBuildStage::Placement,
                "39-bit Horizon regions do not fit around process code",
            ));
        }
        allocation_sizes[allocation] -= size;
        assignment[kind] = allocation;
    }

    let mut result = [region(0, 0); 4];
    for (allocation, start) in allocation_starts.into_iter().enumerate() {
        let mut cursor = start;
        for kind in 0..sizes.len() {
            if assignment[kind] == allocation {
                result[kind] = region(cursor, sizes[kind]);
                cursor = cursor.checked_add(sizes[kind]).ok_or_else(|| {
                    error(ProcessBuildStage::Placement, "Horizon region overflows")
                })?;
            }
        }
    }
    Ok(result)
}
