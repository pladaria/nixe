//! Immutable Switch 1 Maxwell capability-profile schema.
//!
//! The discovery fields represented here follow the pinned libnx public ABI:
//! https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/nvidia/ioctl.h#L47-L106
//!
//! The built-in GM20B profile is populated from pinned public sources and is
//! validated before an ABI adapter exposes it to guest software.

use std::fmt::{Display, Formatter};

use nixe_gpu::GpuClassId;

macro_rules! raw_u32_type {
    ($documentation:literal, $name:ident) => {
        #[doc = $documentation]
        #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        #[repr(transparent)]
        pub struct $name(u32);

        impl $name {
            #[must_use]
            pub const fn from_raw(raw: u32) -> Self {
                Self(raw)
            }

            #[must_use]
            pub const fn raw(self) -> u32 {
                self.0
            }
        }
    };
}

/// Stable emulator identifier for one immutable GPU profile.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct GpuProfileId(&'static str);

impl GpuProfileId {
    #[must_use]
    pub const fn new(identifier: &'static str) -> Self {
        Self(identifier)
    }

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        self.0
    }
}

impl Display for GpuProfileId {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.0)
    }
}

raw_u32_type!(
    "Raw NVIDIA architecture identifier exposed by GPU discovery.",
    GpuArchitecture
);
raw_u32_type!(
    "Raw NVIDIA implementation identifier exposed by GPU discovery.",
    GpuImplementation
);
raw_u32_type!(
    "Raw NVIDIA silicon revision exposed by GPU discovery.",
    GpuRevision
);
raw_u32_type!("Guest-visible GPU interconnect type.", GpuBusType);
raw_u32_type!("One guest GPU page size in bytes.", GpuPageSize);
raw_u32_type!(
    "Guest ABI mask describing the available big page sizes.",
    GpuPageSizeMask
);
raw_u32_type!("Guest-visible shader architecture version.", ShaderVersion);

/// One shader-program-header format version.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct MaxwellShaderProgramHeaderVersion(u16);

impl MaxwellShaderProgramHeaderVersion {
    #[must_use]
    pub const fn new(raw: u16) -> Self {
        Self(raw)
    }

    #[must_use]
    pub const fn raw(self) -> u16 {
        self.0
    }
}

/// Inclusive shader-program-header versions understood by one producer.
///
/// The two-field command encoding is pinned to NVIDIA's public class header:
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L3153-L3159>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellShaderProgramHeaderVersionRange {
    current: MaxwellShaderProgramHeaderVersion,
    oldest_supported: MaxwellShaderProgramHeaderVersion,
}

impl MaxwellShaderProgramHeaderVersionRange {
    #[must_use]
    pub const fn new(
        current: MaxwellShaderProgramHeaderVersion,
        oldest_supported: MaxwellShaderProgramHeaderVersion,
    ) -> Self {
        Self {
            current,
            oldest_supported,
        }
    }

    #[must_use]
    pub const fn current(self) -> MaxwellShaderProgramHeaderVersion {
        self.current
    }

    #[must_use]
    pub const fn oldest_supported(self) -> MaxwellShaderProgramHeaderVersion {
        self.oldest_supported
    }

    #[must_use]
    pub const fn is_well_ordered(self) -> bool {
        self.oldest_supported.raw() <= self.current.raw()
    }

    #[must_use]
    pub const fn overlaps(self, other: Self) -> bool {
        self.is_well_ordered()
            && other.is_well_ordered()
            && self.oldest_supported.raw() <= other.current.raw()
            && other.oldest_supported.raw() <= self.current.raw()
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self.current.raw() as u32 | ((self.oldest_supported.raw() as u32) << 16)
    }
}

/// One AAM version.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct MaxwellAamVersion(u16);

impl MaxwellAamVersion {
    #[must_use]
    pub const fn new(raw: u16) -> Self {
        Self(raw)
    }

    #[must_use]
    pub const fn raw(self) -> u16 {
        self.0
    }
}

/// Inclusive AAM versions accepted by one profile.
///
/// The two-field command encoding is pinned to NVIDIA's public class header:
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L1110-L1116>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellAamVersionRange {
    current: MaxwellAamVersion,
    oldest_supported: MaxwellAamVersion,
}

impl MaxwellAamVersionRange {
    #[must_use]
    pub const fn new(current: MaxwellAamVersion, oldest_supported: MaxwellAamVersion) -> Self {
        Self {
            current,
            oldest_supported,
        }
    }

    #[must_use]
    pub const fn current(self) -> MaxwellAamVersion {
        self.current
    }

    #[must_use]
    pub const fn oldest_supported(self) -> MaxwellAamVersion {
        self.oldest_supported
    }

    #[must_use]
    pub const fn is_well_ordered(self) -> bool {
        self.oldest_supported.raw() <= self.current.raw()
    }

    #[must_use]
    pub const fn overlaps(self, other: Self) -> bool {
        self.is_well_ordered()
            && other.is_well_ordered()
            && self.oldest_supported.raw() <= other.current.raw()
            && other.oldest_supported.raw() <= self.current.raw()
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self.current.raw() as u32 | ((self.oldest_supported.raw() as u32) << 16)
    }
}

/// Number of implemented bits in one guest GPU address.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct AddressBitCount(u8);

impl AddressBitCount {
    #[must_use]
    pub const fn from_raw(bits: u8) -> Self {
        Self(bits)
    }

    #[must_use]
    pub const fn bits(self) -> u8 {
        self.0
    }
}

/// Raw feature flags returned by the verified GPU-characteristics ABI.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct GpuFeatureFlags(u64);

impl GpuFeatureFlags {
    #[must_use]
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

/// Fixed-width chip name carried by the GPU-characteristics wire structure.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct ChipName([u8; 8]);

impl ChipName {
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 8]) -> Self {
        Self(bytes)
    }

    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 8] {
        &self.0
    }
}

/// Numeric NVIDIA identity returned by GPU discovery.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellChipsetIdentity {
    architecture: GpuArchitecture,
    implementation: GpuImplementation,
    revision: GpuRevision,
    chip_name: ChipName,
}

impl MaxwellChipsetIdentity {
    #[must_use]
    pub const fn architecture(self) -> GpuArchitecture {
        self.architecture
    }

    #[must_use]
    pub const fn implementation(self) -> GpuImplementation {
        self.implementation
    }

    #[must_use]
    pub const fn revision(self) -> GpuRevision {
        self.revision
    }

    #[must_use]
    pub const fn chip_name(self) -> ChipName {
        self.chip_name
    }
}

/// Active and maximum graphics-processing topology.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellTopology {
    gpc_count: u32,
    tpc_per_gpc: u32,
    gpc_enable_mask: u32,
    tpc_enable_masks: &'static [u32],
    maximum_gpc_count: u32,
    maximum_fbp_count: u32,
    fbp_enable_mask: u32,
    maximum_ltc_per_fbp: u32,
    maximum_lts_per_ltc: u32,
    maximum_texture_units_per_tpc: u32,
}

impl MaxwellTopology {
    #[must_use]
    pub const fn gpc_count(self) -> u32 {
        self.gpc_count
    }

    #[must_use]
    pub const fn tpc_per_gpc(self) -> u32 {
        self.tpc_per_gpc
    }

    #[must_use]
    pub const fn gpc_enable_mask(self) -> u32 {
        self.gpc_enable_mask
    }

    #[must_use]
    pub const fn tpc_enable_masks(self) -> &'static [u32] {
        self.tpc_enable_masks
    }

    #[must_use]
    pub const fn maximum_gpc_count(self) -> u32 {
        self.maximum_gpc_count
    }

    #[must_use]
    pub const fn maximum_fbp_count(self) -> u32 {
        self.maximum_fbp_count
    }

    #[must_use]
    pub const fn fbp_enable_mask(self) -> u32 {
        self.fbp_enable_mask
    }

    #[must_use]
    pub const fn maximum_ltc_per_fbp(self) -> u32 {
        self.maximum_ltc_per_fbp
    }

    #[must_use]
    pub const fn maximum_lts_per_ltc(self) -> u32 {
        self.maximum_lts_per_ltc
    }

    #[must_use]
    pub const fn maximum_texture_units_per_tpc(self) -> u32 {
        self.maximum_texture_units_per_tpc
    }
}

/// GPU engine class identifiers advertised to the guest driver.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellClassCapabilities {
    two_d: GpuClassId,
    three_d: GpuClassId,
    compute: GpuClassId,
    gpfifo: GpuClassId,
    inline_to_memory: GpuClassId,
    dma_copy: GpuClassId,
}

impl MaxwellClassCapabilities {
    #[must_use]
    pub const fn two_d(self) -> GpuClassId {
        self.two_d
    }

    #[must_use]
    pub const fn three_d(self) -> GpuClassId {
        self.three_d
    }

    #[must_use]
    pub const fn compute(self) -> GpuClassId {
        self.compute
    }

    #[must_use]
    pub const fn gpfifo(self) -> GpuClassId {
        self.gpfifo
    }

    #[must_use]
    pub const fn inline_to_memory(self) -> GpuClassId {
        self.inline_to_memory
    }

    #[must_use]
    pub const fn dma_copy(self) -> GpuClassId {
        self.dma_copy
    }
}

/// Guest GPU virtual-address geometry reported by discovery.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellVirtualAddressCapabilities {
    address_bits: AddressBitCount,
    pde_coverage_bits: AddressBitCount,
}

impl MaxwellVirtualAddressCapabilities {
    #[must_use]
    pub const fn address_bits(self) -> AddressBitCount {
        self.address_bits
    }

    #[must_use]
    pub const fn pde_coverage_bits(self) -> AddressBitCount {
        self.pde_coverage_bits
    }
}

/// Page-size and video-memory discovery fields.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellMemoryCapabilities {
    onboard_video_memory_bytes: u64,
    small_page_size: GpuPageSize,
    big_page_size: GpuPageSize,
    compression_page_size: GpuPageSize,
    available_big_page_sizes: GpuPageSizeMask,
}

impl MaxwellMemoryCapabilities {
    #[must_use]
    pub const fn onboard_video_memory_bytes(self) -> u64 {
        self.onboard_video_memory_bytes
    }

    #[must_use]
    pub const fn small_page_size(self) -> GpuPageSize {
        self.small_page_size
    }

    #[must_use]
    pub const fn big_page_size(self) -> GpuPageSize {
        self.big_page_size
    }

    #[must_use]
    pub const fn compression_page_size(self) -> GpuPageSize {
        self.compression_page_size
    }

    #[must_use]
    pub const fn available_big_page_sizes(self) -> GpuPageSizeMask {
        self.available_big_page_sizes
    }
}

/// L2 and compression metadata reported by GPU discovery.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellCacheCapabilities {
    l2_cache_bytes: u64,
    rop_l2_enable_masks: [u32; 2],
    compression_bit_store_base: Option<u64>,
}

impl MaxwellCacheCapabilities {
    #[must_use]
    pub const fn l2_cache_bytes(self) -> u64 {
        self.l2_cache_bytes
    }

    #[must_use]
    pub const fn rop_l2_enable_masks(self) -> [u32; 2] {
        self.rop_l2_enable_masks
    }

    #[must_use]
    pub const fn compression_bit_store_base(self) -> Option<u64> {
        self.compression_bit_store_base
    }
}

/// Bus type exposed by the GPU-characteristics operation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellInterconnectCapabilities {
    bus_type: GpuBusType,
}

impl MaxwellInterconnectCapabilities {
    #[must_use]
    pub const fn bus_type(self) -> GpuBusType {
        self.bus_type
    }
}

/// Z-cull context and region geometry.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellZCullCapabilities {
    context_size: u32,
    width_align_pixels: u32,
    height_align_pixels: u32,
    pixel_squares_by_aliquots: u32,
    aliquot_total: u32,
    region_byte_multiplier: u32,
    region_header_size: u32,
    subregion_header_size: u32,
    subregion_width_align_pixels: u32,
    subregion_height_align_pixels: u32,
    subregion_count: u32,
}

impl MaxwellZCullCapabilities {
    #[must_use]
    pub const fn context_size(self) -> u32 {
        self.context_size
    }

    #[must_use]
    pub const fn width_align_pixels(self) -> u32 {
        self.width_align_pixels
    }

    #[must_use]
    pub const fn height_align_pixels(self) -> u32 {
        self.height_align_pixels
    }

    #[must_use]
    pub const fn pixel_squares_by_aliquots(self) -> u32 {
        self.pixel_squares_by_aliquots
    }

    #[must_use]
    pub const fn aliquot_total(self) -> u32 {
        self.aliquot_total
    }

    #[must_use]
    pub const fn region_byte_multiplier(self) -> u32 {
        self.region_byte_multiplier
    }

    #[must_use]
    pub const fn region_header_size(self) -> u32 {
        self.region_header_size
    }

    #[must_use]
    pub const fn subregion_header_size(self) -> u32 {
        self.subregion_header_size
    }

    #[must_use]
    pub const fn subregion_width_align_pixels(self) -> u32 {
        self.subregion_width_align_pixels
    }

    #[must_use]
    pub const fn subregion_height_align_pixels(self) -> u32 {
        self.subregion_height_align_pixels
    }

    #[must_use]
    pub const fn subregion_count(self) -> u32 {
        self.subregion_count
    }
}

/// Shader-multiprocessor architecture fields returned by discovery.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellShaderCapabilities {
    sm_version: ShaderVersion,
    spa_version: ShaderVersion,
    sph_versions: MaxwellShaderProgramHeaderVersionRange,
    warp_count: u32,
}

impl MaxwellShaderCapabilities {
    #[must_use]
    pub const fn sm_version(self) -> ShaderVersion {
        self.sm_version
    }

    #[must_use]
    pub const fn spa_version(self) -> ShaderVersion {
        self.spa_version
    }

    #[must_use]
    pub const fn sph_versions(self) -> MaxwellShaderProgramHeaderVersionRange {
        self.sph_versions
    }

    #[must_use]
    pub const fn warp_count(self) -> u32 {
        self.warp_count
    }
}

/// Complete immutable capability source for one Switch 1 Maxwell GPU.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellGpuProfile {
    id: GpuProfileId,
    chipset: MaxwellChipsetIdentity,
    topology: MaxwellTopology,
    classes: MaxwellClassCapabilities,
    virtual_address: MaxwellVirtualAddressCapabilities,
    memory: MaxwellMemoryCapabilities,
    cache: MaxwellCacheCapabilities,
    interconnect: MaxwellInterconnectCapabilities,
    z_cull: MaxwellZCullCapabilities,
    shader: MaxwellShaderCapabilities,
    aam_versions: MaxwellAamVersionRange,
    feature_flags: GpuFeatureFlags,
}

impl MaxwellGpuProfile {
    #[must_use]
    pub const fn id(self) -> GpuProfileId {
        self.id
    }

    #[must_use]
    pub const fn chipset(self) -> MaxwellChipsetIdentity {
        self.chipset
    }

    #[must_use]
    pub const fn topology(self) -> MaxwellTopology {
        self.topology
    }

    #[must_use]
    pub const fn classes(self) -> MaxwellClassCapabilities {
        self.classes
    }

    #[must_use]
    pub const fn virtual_address(self) -> MaxwellVirtualAddressCapabilities {
        self.virtual_address
    }

    #[must_use]
    pub const fn memory(self) -> MaxwellMemoryCapabilities {
        self.memory
    }

    #[must_use]
    pub const fn cache(self) -> MaxwellCacheCapabilities {
        self.cache
    }

    #[must_use]
    pub const fn interconnect(self) -> MaxwellInterconnectCapabilities {
        self.interconnect
    }

    #[must_use]
    pub const fn z_cull(self) -> MaxwellZCullCapabilities {
        self.z_cull
    }

    #[must_use]
    pub const fn shader(self) -> MaxwellShaderCapabilities {
        self.shader
    }

    #[must_use]
    pub const fn aam_versions(self) -> MaxwellAamVersionRange {
        self.aam_versions
    }

    #[must_use]
    pub const fn feature_flags(self) -> GpuFeatureFlags {
        self.feature_flags
    }

    /// Checks relationships which cannot be represented by individual field
    /// types alone.
    pub const fn validate(self) -> Result<(), MaxwellProfileValidationError> {
        let topology = self.topology;
        let maximum_gpc_count = topology.maximum_gpc_count;
        if maximum_gpc_count == 0 || maximum_gpc_count > u32::BITS {
            return Err(MaxwellProfileValidationError::InvalidMaximumGpcCount {
                count: maximum_gpc_count,
            });
        }
        if topology.gpc_count == 0 || topology.gpc_count > maximum_gpc_count {
            return Err(MaxwellProfileValidationError::InvalidGpcCount {
                count: topology.gpc_count,
                maximum: maximum_gpc_count,
            });
        }

        let valid_gpc_mask = low_bits_mask(maximum_gpc_count);
        if topology.gpc_enable_mask & !valid_gpc_mask != 0
            || topology.gpc_enable_mask.count_ones() != topology.gpc_count
        {
            return Err(MaxwellProfileValidationError::InconsistentGpcMask {
                mask: topology.gpc_enable_mask,
                count: topology.gpc_count,
                maximum: maximum_gpc_count,
            });
        }

        if topology.tpc_enable_masks.len() != maximum_gpc_count as usize {
            return Err(MaxwellProfileValidationError::TpcMaskCount {
                actual: topology.tpc_enable_masks.len(),
                expected: maximum_gpc_count,
            });
        }
        if topology.tpc_per_gpc == 0 || topology.tpc_per_gpc > u32::BITS {
            return Err(MaxwellProfileValidationError::InvalidTpcCount {
                count: topology.tpc_per_gpc,
            });
        }
        let valid_tpc_mask = low_bits_mask(topology.tpc_per_gpc);
        let mut gpc = 0;
        while gpc < topology.tpc_enable_masks.len() {
            let mask = topology.tpc_enable_masks[gpc];
            let gpc_bit = 1_u32 << gpc;
            if mask & !valid_tpc_mask != 0
                || (topology.gpc_enable_mask & gpc_bit == 0 && mask != 0)
                || (topology.gpc_enable_mask & gpc_bit != 0 && mask == 0)
            {
                return Err(MaxwellProfileValidationError::InconsistentTpcMask {
                    gpc: gpc as u32,
                    mask,
                    maximum_tpc_count: topology.tpc_per_gpc,
                    gpc_enabled: topology.gpc_enable_mask & gpc_bit != 0,
                });
            }
            gpc += 1;
        }

        let virtual_address = self.virtual_address;
        let address_bits = virtual_address.address_bits.bits();
        let pde_coverage_bits = virtual_address.pde_coverage_bits.bits();
        if address_bits == 0 || address_bits > u64::BITS as u8 {
            return Err(
                MaxwellProfileValidationError::InvalidVirtualAddressBitCount { bits: address_bits },
            );
        }
        if pde_coverage_bits == 0 || pde_coverage_bits >= address_bits {
            return Err(MaxwellProfileValidationError::InvalidPdeCoverage {
                bits: pde_coverage_bits,
                virtual_address_bits: address_bits,
            });
        }

        let memory = self.memory;
        let small_page_size = memory.small_page_size.raw();
        if !is_nonzero_power_of_two(small_page_size) {
            return Err(MaxwellProfileValidationError::InvalidSmallPageSize {
                bytes: small_page_size,
            });
        }
        let big_page_size = memory.big_page_size.raw();
        if !is_nonzero_power_of_two(big_page_size) {
            return Err(MaxwellProfileValidationError::InvalidBigPageSize {
                bytes: big_page_size,
            });
        }
        if memory.available_big_page_sizes.raw() & big_page_size == 0 {
            return Err(MaxwellProfileValidationError::BigPageSizeNotAdvertised {
                bytes: big_page_size,
                available_mask: memory.available_big_page_sizes.raw(),
            });
        }
        let compression_page_size = memory.compression_page_size.raw();
        if !is_nonzero_power_of_two(compression_page_size) {
            return Err(MaxwellProfileValidationError::InvalidCompressionPageSize {
                bytes: compression_page_size,
            });
        }

        let classes = [
            self.classes.two_d,
            self.classes.three_d,
            self.classes.compute,
            self.classes.gpfifo,
            self.classes.inline_to_memory,
            self.classes.dma_copy,
        ];
        let mut index = 0;
        while index < classes.len() {
            let class = classes[index];
            if class.0 == 0 {
                return Err(MaxwellProfileValidationError::MissingGpuClass);
            }
            let mut previous = 0;
            while previous < index {
                if classes[previous].0 == class.0 {
                    return Err(MaxwellProfileValidationError::DuplicateGpuClass { class });
                }
                previous += 1;
            }
            index += 1;
        }

        let shader = self.shader;
        if shader.sm_version.raw() == 0
            || shader.sm_version.raw() != shader.spa_version.raw()
            || shader.warp_count == 0
        {
            return Err(
                MaxwellProfileValidationError::InconsistentShaderArchitecture {
                    sm_version: shader.sm_version,
                    spa_version: shader.spa_version,
                    warp_count: shader.warp_count,
                },
            );
        }
        if !shader.sph_versions.is_well_ordered() {
            return Err(
                MaxwellProfileValidationError::InvalidShaderProgramHeaderVersionRange {
                    current: shader.sph_versions.current(),
                    oldest_supported: shader.sph_versions.oldest_supported(),
                },
            );
        }
        if !self.aam_versions.is_well_ordered() {
            return Err(MaxwellProfileValidationError::InvalidAamVersionRange {
                current: self.aam_versions.current(),
                oldest_supported: self.aam_versions.oldest_supported(),
            });
        }

        Ok(())
    }
}

const fn low_bits_mask(bit_count: u32) -> u32 {
    if bit_count == u32::BITS {
        u32::MAX
    } else {
        (1_u32 << bit_count) - 1
    }
}

const fn is_nonzero_power_of_two(value: u32) -> bool {
    value != 0 && value.is_power_of_two()
}

/// A contradictory or incomplete immutable Maxwell capability profile.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellProfileValidationError {
    InvalidMaximumGpcCount {
        count: u32,
    },
    InvalidGpcCount {
        count: u32,
        maximum: u32,
    },
    InconsistentGpcMask {
        mask: u32,
        count: u32,
        maximum: u32,
    },
    TpcMaskCount {
        actual: usize,
        expected: u32,
    },
    InvalidTpcCount {
        count: u32,
    },
    InconsistentTpcMask {
        gpc: u32,
        mask: u32,
        maximum_tpc_count: u32,
        gpc_enabled: bool,
    },
    InvalidVirtualAddressBitCount {
        bits: u8,
    },
    InvalidPdeCoverage {
        bits: u8,
        virtual_address_bits: u8,
    },
    InvalidSmallPageSize {
        bytes: u32,
    },
    InvalidBigPageSize {
        bytes: u32,
    },
    BigPageSizeNotAdvertised {
        bytes: u32,
        available_mask: u32,
    },
    InvalidCompressionPageSize {
        bytes: u32,
    },
    MissingGpuClass,
    DuplicateGpuClass {
        class: GpuClassId,
    },
    InconsistentShaderArchitecture {
        sm_version: ShaderVersion,
        spa_version: ShaderVersion,
        warp_count: u32,
    },
    InvalidShaderProgramHeaderVersionRange {
        current: MaxwellShaderProgramHeaderVersion,
        oldest_supported: MaxwellShaderProgramHeaderVersion,
    },
    InvalidAamVersionRange {
        current: MaxwellAamVersion,
        oldest_supported: MaxwellAamVersion,
    },
}

impl Display for MaxwellProfileValidationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for MaxwellProfileValidationError {}

// The discovery values and their exact guest ABI representation are documented
// by pinned libnx commit dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb:
// https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/nvidia/ioctl.h#L47-L106
//
// NVIDIA's version 1.0 Tegra X1 architecture whitepaper independently records
// one GPC containing two active SMM blocks (page 13), which yields mask 0b11
// for the sole enabled GPC:
// https://images.nvidia.com/content/pdf/tegra/Tegra-X1-whitepaper-v1.0.pdf
const SWITCH_1_GM20B_TPC_MASKS: [u32; 1] = [0b11];

/// Immutable Switch 1 Tegra X1 GM20B capability profile.
pub const SWITCH_1_GM20B_PROFILE: MaxwellGpuProfile = MaxwellGpuProfile {
    id: GpuProfileId::new("switch1-gm20b"),
    chipset: MaxwellChipsetIdentity {
        architecture: GpuArchitecture::from_raw(0x120),
        implementation: GpuImplementation::from_raw(0x0b),
        revision: GpuRevision::from_raw(0xa1),
        chip_name: ChipName::from_bytes(*b"gm20b\0\0\0"),
    },
    topology: MaxwellTopology {
        gpc_count: 1,
        tpc_per_gpc: 2,
        gpc_enable_mask: 0b1,
        tpc_enable_masks: &SWITCH_1_GM20B_TPC_MASKS,
        maximum_gpc_count: 1,
        maximum_fbp_count: 1,
        fbp_enable_mask: 0,
        maximum_ltc_per_fbp: 2,
        maximum_lts_per_ltc: 1,
        maximum_texture_units_per_tpc: 0,
    },
    classes: MaxwellClassCapabilities {
        two_d: GpuClassId(0x902d),
        three_d: GpuClassId(0xb197),
        compute: GpuClassId(0xb1c0),
        gpfifo: GpuClassId(0xb06f),
        inline_to_memory: GpuClassId(0xa140),
        dma_copy: GpuClassId(0xb0b5),
    },
    virtual_address: MaxwellVirtualAddressCapabilities {
        address_bits: AddressBitCount::from_raw(0x28),
        pde_coverage_bits: AddressBitCount::from_raw(0x1b),
    },
    memory: MaxwellMemoryCapabilities {
        onboard_video_memory_bytes: 0,
        small_page_size: GpuPageSize::from_raw(0x1_000),
        big_page_size: GpuPageSize::from_raw(0x2_0000),
        compression_page_size: GpuPageSize::from_raw(0x2_0000),
        available_big_page_sizes: GpuPageSizeMask::from_raw(0x3_0000),
    },
    cache: MaxwellCacheCapabilities {
        l2_cache_bytes: 0x4_0000,
        rop_l2_enable_masks: [0x2_1d70, 0],
        compression_bit_store_base: None,
    },
    interconnect: MaxwellInterconnectCapabilities {
        bus_type: GpuBusType::from_raw(0x20),
    },
    z_cull: MaxwellZCullCapabilities {
        // This value is also recorded by the pinned public Switch frontend
        // implementation at commit b15cbf9bcf4e02f182abfd8ff84921f436b8c464:
        // https://ni.4a.si/anonymous/yuzu/commit/src?id=b15cbf9bcf4e02f182abfd8ff84921f436b8c464
        context_size: 1,
        width_align_pixels: 0x20,
        height_align_pixels: 0x20,
        pixel_squares_by_aliquots: 0x400,
        aliquot_total: 0x800,
        region_byte_multiplier: 0x20,
        region_header_size: 0x20,
        subregion_header_size: 0xc0,
        subregion_width_align_pixels: 0x20,
        subregion_height_align_pixels: 0x40,
        subregion_count: 0x10,
    },
    shader: MaxwellShaderCapabilities {
        sm_version: ShaderVersion::from_raw(0x503),
        spa_version: ShaderVersion::from_raw(0x503),
        // Switch 1 command streams select and check SPH format version 3.
        sph_versions: MaxwellShaderProgramHeaderVersionRange::new(
            MaxwellShaderProgramHeaderVersion::new(3),
            MaxwellShaderProgramHeaderVersion::new(3),
        ),
        warp_count: 0x80,
    },
    // Switch 1 command streams select and check AAM version 2.
    aam_versions: MaxwellAamVersionRange::new(MaxwellAamVersion::new(2), MaxwellAamVersion::new(2)),
    feature_flags: GpuFeatureFlags::from_raw(0x55),
};

const _: () = match SWITCH_1_GM20B_PROFILE.validate() {
    Ok(()) => (),
    Err(_) => panic!("the built-in Switch 1 GM20B profile is inconsistent"),
};

#[cfg(test)]
mod tests {
    use super::*;

    const TPC_MASKS: [u32; 2] = [0b11, 0b01];
    const PROFILE: MaxwellGpuProfile = MaxwellGpuProfile {
        id: GpuProfileId::new("synthetic-maxwell"),
        chipset: MaxwellChipsetIdentity {
            architecture: GpuArchitecture::from_raw(1),
            implementation: GpuImplementation::from_raw(2),
            revision: GpuRevision::from_raw(3),
            chip_name: ChipName::from_bytes(*b"testgpu\0"),
        },
        topology: MaxwellTopology {
            gpc_count: 2,
            tpc_per_gpc: 2,
            gpc_enable_mask: 0b11,
            tpc_enable_masks: &TPC_MASKS,
            maximum_gpc_count: 2,
            maximum_fbp_count: 1,
            fbp_enable_mask: 1,
            maximum_ltc_per_fbp: 2,
            maximum_lts_per_ltc: 1,
            maximum_texture_units_per_tpc: 4,
        },
        classes: MaxwellClassCapabilities {
            two_d: GpuClassId(0x10),
            three_d: GpuClassId(0x20),
            compute: GpuClassId(0x30),
            gpfifo: GpuClassId(0x40),
            inline_to_memory: GpuClassId(0x50),
            dma_copy: GpuClassId(0x60),
        },
        virtual_address: MaxwellVirtualAddressCapabilities {
            address_bits: AddressBitCount::from_raw(40),
            pde_coverage_bits: AddressBitCount::from_raw(27),
        },
        memory: MaxwellMemoryCapabilities {
            onboard_video_memory_bytes: 0,
            small_page_size: GpuPageSize::from_raw(0x1_000),
            big_page_size: GpuPageSize::from_raw(0x1_000),
            compression_page_size: GpuPageSize::from_raw(0x2_000),
            available_big_page_sizes: GpuPageSizeMask::from_raw(0x3_000),
        },
        cache: MaxwellCacheCapabilities {
            l2_cache_bytes: 0x8_000,
            rop_l2_enable_masks: [1, 0],
            compression_bit_store_base: None,
        },
        interconnect: MaxwellInterconnectCapabilities {
            bus_type: GpuBusType::from_raw(7),
        },
        z_cull: MaxwellZCullCapabilities {
            context_size: 4,
            width_align_pixels: 8,
            height_align_pixels: 16,
            pixel_squares_by_aliquots: 32,
            aliquot_total: 64,
            region_byte_multiplier: 128,
            region_header_size: 256,
            subregion_header_size: 512,
            subregion_width_align_pixels: 1,
            subregion_height_align_pixels: 2,
            subregion_count: 3,
        },
        shader: MaxwellShaderCapabilities {
            sm_version: ShaderVersion::from_raw(0x101),
            spa_version: ShaderVersion::from_raw(0x101),
            sph_versions: MaxwellShaderProgramHeaderVersionRange::new(
                MaxwellShaderProgramHeaderVersion::new(4),
                MaxwellShaderProgramHeaderVersion::new(2),
            ),
            warp_count: 8,
        },
        aam_versions: MaxwellAamVersionRange::new(
            MaxwellAamVersion::new(5),
            MaxwellAamVersion::new(2),
        ),
        feature_flags: GpuFeatureFlags::from_raw(0xa5),
    };

    #[test]
    fn profile_exposes_every_discovery_group_without_mutation() {
        const ID: &str = PROFILE.id().as_str();
        const GPU_VA_BITS: u8 = PROFILE.virtual_address().address_bits().bits();

        assert_eq!(ID, "synthetic-maxwell");
        assert_eq!(GPU_VA_BITS, 40);
        assert_eq!(PROFILE.chipset().architecture().raw(), 1);
        assert_eq!(PROFILE.chipset().implementation().raw(), 2);
        assert_eq!(PROFILE.chipset().revision().raw(), 3);
        assert_eq!(PROFILE.chipset().chip_name().as_bytes(), b"testgpu\0");
        assert_eq!(PROFILE.topology().tpc_enable_masks(), &[0b11, 0b01]);
        assert_eq!(PROFILE.classes().three_d(), GpuClassId(0x20));
        assert_eq!(PROFILE.memory().big_page_size().raw(), 0x1_000);
        assert_eq!(PROFILE.cache().l2_cache_bytes(), 0x8_000);
        assert_eq!(PROFILE.interconnect().bus_type().raw(), 7);
        assert_eq!(PROFILE.z_cull().subregion_count(), 3);
        assert_eq!(PROFILE.shader().sm_version().raw(), 0x101);
        assert_eq!(PROFILE.shader().sph_versions().raw(), 0x0002_0004);
        assert_eq!(PROFILE.aam_versions().raw(), 0x0002_0005);
        assert_eq!(PROFILE.feature_flags().raw(), 0xa5);
    }

    #[test]
    fn profile_value_is_copyable_and_contains_no_runtime_owned_storage() {
        let first = PROFILE;
        let second = first;

        assert_eq!(first, second);
        assert!(!std::mem::needs_drop::<MaxwellGpuProfile>());
    }

    #[test]
    fn switch_1_gm20b_profile_is_internally_consistent() {
        assert_eq!(SWITCH_1_GM20B_PROFILE.validate(), Ok(()));
        assert_eq!(
            SWITCH_1_GM20B_PROFILE.shader().sph_versions().raw(),
            0x0003_0003
        );
        assert_eq!(SWITCH_1_GM20B_PROFILE.aam_versions().raw(), 0x0002_0002);
        assert_eq!(
            SWITCH_1_GM20B_PROFILE.topology().tpc_enable_masks(),
            &[0b11]
        );
    }

    #[test]
    fn validation_rejects_cross_field_contradictions() {
        const INVALID_TPC_MASKS: [u32; 2] = [0b100, 0b01];
        let invalid = MaxwellGpuProfile {
            topology: MaxwellTopology {
                tpc_enable_masks: &INVALID_TPC_MASKS,
                ..PROFILE.topology
            },
            ..PROFILE
        };

        assert_eq!(
            invalid.validate(),
            Err(MaxwellProfileValidationError::InconsistentTpcMask {
                gpc: 0,
                mask: 0b100,
                maximum_tpc_count: 2,
                gpc_enabled: true,
            })
        );

        let invalid = MaxwellGpuProfile {
            virtual_address: MaxwellVirtualAddressCapabilities {
                address_bits: AddressBitCount::from_raw(27),
                pde_coverage_bits: AddressBitCount::from_raw(27),
            },
            ..PROFILE
        };
        assert_eq!(
            invalid.validate(),
            Err(MaxwellProfileValidationError::InvalidPdeCoverage {
                bits: 27,
                virtual_address_bits: 27,
            })
        );

        let invalid = MaxwellGpuProfile {
            memory: MaxwellMemoryCapabilities {
                big_page_size: GpuPageSize::from_raw(0x4_000),
                ..PROFILE.memory
            },
            ..PROFILE
        };
        assert_eq!(
            invalid.validate(),
            Err(MaxwellProfileValidationError::BigPageSizeNotAdvertised {
                bytes: 0x4_000,
                available_mask: 0x3_000,
            })
        );

        let invalid = MaxwellGpuProfile {
            classes: MaxwellClassCapabilities {
                compute: PROFILE.classes.three_d,
                ..PROFILE.classes
            },
            ..PROFILE
        };
        assert_eq!(
            invalid.validate(),
            Err(MaxwellProfileValidationError::DuplicateGpuClass {
                class: PROFILE.classes.three_d,
            })
        );

        let invalid = MaxwellGpuProfile {
            shader: MaxwellShaderCapabilities {
                spa_version: ShaderVersion::from_raw(0x102),
                ..PROFILE.shader
            },
            ..PROFILE
        };
        assert_eq!(
            invalid.validate(),
            Err(
                MaxwellProfileValidationError::InconsistentShaderArchitecture {
                    sm_version: ShaderVersion::from_raw(0x101),
                    spa_version: ShaderVersion::from_raw(0x102),
                    warp_count: 8,
                }
            )
        );

        let invalid = MaxwellGpuProfile {
            shader: MaxwellShaderCapabilities {
                sph_versions: MaxwellShaderProgramHeaderVersionRange::new(
                    MaxwellShaderProgramHeaderVersion::new(2),
                    MaxwellShaderProgramHeaderVersion::new(3),
                ),
                ..PROFILE.shader
            },
            ..PROFILE
        };
        assert_eq!(
            invalid.validate(),
            Err(
                MaxwellProfileValidationError::InvalidShaderProgramHeaderVersionRange {
                    current: MaxwellShaderProgramHeaderVersion::new(2),
                    oldest_supported: MaxwellShaderProgramHeaderVersion::new(3),
                }
            )
        );

        let invalid = MaxwellGpuProfile {
            aam_versions: MaxwellAamVersionRange::new(
                MaxwellAamVersion::new(2),
                MaxwellAamVersion::new(3),
            ),
            ..PROFILE
        };
        assert_eq!(
            invalid.validate(),
            Err(MaxwellProfileValidationError::InvalidAamVersionRange {
                current: MaxwellAamVersion::new(2),
                oldest_supported: MaxwellAamVersion::new(3),
            })
        );
    }
}
