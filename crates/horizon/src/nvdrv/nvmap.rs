use std::collections::BTreeMap;
use std::fmt::{Display, Formatter};

use nixe_memory::{
    AddressSpaceId, CanonicalBackingRange, CanonicalRangeAccessError, GuestVirtualAddress,
    MemoryPermissions,
};

// Switch nvmap ABI values and parameter behavior:
// https://switchbrew.org/w/index.php?title=NV_services&oldid=14790#/dev/nvmap
// The libnx wrappers pin flags bit 0 to read/write access and zero-initialize
// the remaining ABI fields:
// https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/nvidia/ioctl/nvmap.c
const NVMAP_MINIMUM_ALIGNMENT: u32 = 0x1000;
const NVMAP_SYSTEM_HEAP_MASK: u32 = 0x4000_0000;
const NVMAP_READ_WRITE_FLAG: u32 = 1;
const NVMAP_UNCACHED_FLAG: u32 = 1 << 1;
const NVMAP_VALID_FLAGS: u32 = NVMAP_READ_WRITE_FLAG | NVMAP_UNCACHED_FLAG;
const NVMAP_INVALID_KIND: u8 = 0xff;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) struct NvMapOwner {
    process_id: u64,
}

impl NvMapOwner {
    pub(super) const fn new(process_id: u64) -> Self {
        Self { process_id }
    }
}

/// Guest-visible handle to one `nvmap` object.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct NvMapHandle(u32);

impl NvMapHandle {
    pub const fn new(raw: u32) -> Self {
        Self(raw)
    }

    pub const fn raw(self) -> u32 {
        self.0
    }
}

impl Display for NvMapHandle {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "nvmap-handle:{:#010x}", self.0)
    }
}

/// Stable internal identity of one lifetime-managed `nvmap` object.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct NvMapObjectId(u64);

impl NvMapObjectId {
    const FIRST: Self = Self(1);

    pub const fn raw(self) -> u64 {
        self.0
    }
}

impl Display for NvMapObjectId {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "nvmap-object:{:#018x}", self.0)
    }
}

/// Guest-visible ID used to import another handle to an existing object.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct NvMapExportedId(u32);

impl NvMapExportedId {
    pub const fn new(raw: u32) -> Self {
        Self(raw)
    }

    pub const fn raw(self) -> u32 {
        self.0
    }
}

impl Display for NvMapExportedId {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "nvmap-id:{:#010x}", self.0)
    }
}

/// CPU mapping retained only for ABI output and pointer-free diagnostics.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct NvMapCpuMapping {
    address_space: AddressSpaceId,
    address: GuestVirtualAddress,
}

impl NvMapCpuMapping {
    pub const fn new(address_space: AddressSpaceId, address: GuestVirtualAddress) -> Self {
        Self {
            address_space,
            address,
        }
    }

    pub const fn address_space(self) -> AddressSpaceId {
        self.address_space
    }

    pub const fn address(self) -> GuestVirtualAddress {
        self.address
    }
}

/// Allocation attributes attached to an object after `NVMAP_IOC_ALLOC`.
///
/// These describe allocation policy only. Image dimensions, pitch, planes,
/// and layout belong to a view and never become properties of the storage.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct NvMapAllocationMetadata {
    heap_mask: u32,
    flags: u32,
    alignment: u32,
    kind: u8,
}

impl NvMapAllocationMetadata {
    pub const fn new(heap_mask: u32, flags: u32, alignment: u32, kind: u8) -> Self {
        Self {
            // A zero request selects the regular system heap. The observable
            // NVMAP_HANDLE_PARAM_HEAP value is still the system-heap bit.
            heap_mask: if heap_mask == 0 {
                NVMAP_SYSTEM_HEAP_MASK
            } else {
                heap_mask
            },
            flags,
            alignment,
            kind,
        }
    }

    pub const fn heap_mask(self) -> u32 {
        self.heap_mask
    }

    pub const fn flags(self) -> u32 {
        self.flags
    }

    pub const fn alignment(self) -> u32 {
        self.alignment
    }

    pub const fn kind(self) -> u8 {
        self.kind
    }

    pub(super) const fn required_permissions(self) -> MemoryPermissions {
        if self.flags & NVMAP_READ_WRITE_FLAG != 0 {
            MemoryPermissions::READ_WRITE
        } else {
            MemoryPermissions::READ
        }
    }

    pub(super) const fn validate(self) -> Result<(), NvMapStateError> {
        if self.heap_mask != NVMAP_SYSTEM_HEAP_MASK
            || self.flags & !NVMAP_VALID_FLAGS != 0
            || self.alignment < NVMAP_MINIMUM_ALIGNMENT
            || !self.alignment.is_power_of_two()
            || self.kind == NVMAP_INVALID_KIND
        {
            return Err(NvMapStateError::BadParameter);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct NvMapCanonicalStorage {
    allocation: NvMapAllocationMetadata,
    cpu_mapping: NvMapCpuMapping,
    backing: CanonicalBackingRange,
}

/// Retained semantic `nvmap` object independent from handles and exported IDs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NvMapObject {
    id: NvMapObjectId,
    size: u32,
    storage: Option<NvMapCanonicalStorage>,
}

impl NvMapObject {
    pub const fn id(&self) -> NvMapObjectId {
        self.id
    }

    pub const fn size(&self) -> u32 {
        self.size
    }

    pub fn allocation_metadata(&self) -> Option<NvMapAllocationMetadata> {
        self.storage.as_ref().map(|storage| storage.allocation)
    }

    pub fn cpu_mapping(&self) -> Option<NvMapCpuMapping> {
        self.storage.as_ref().map(|storage| storage.cpu_mapping)
    }

    /// Returns the canonical byte identity retained by this object.
    pub fn backing(&self) -> Option<&CanonicalBackingRange> {
        self.storage.as_ref().map(|storage| &storage.backing)
    }

    /// Constructs an image interpretation without transferring memory
    /// ownership from this object to the view.
    pub fn image_view(
        &self,
        metadata: NvMapImageViewMetadata,
    ) -> Result<NvMapImageView, NvMapViewError> {
        let Some(storage) = &self.storage else {
            return Err(NvMapViewError::UnallocatedObject);
        };
        if metadata.planes.is_empty() {
            return Err(NvMapViewError::MissingPlanes);
        }
        for plane in metadata.planes.iter() {
            let end = plane
                .offset
                .checked_add(plane.size)
                .ok_or(NvMapViewError::RangeOverflow)?;
            if plane.size == 0 || end > u64::from(self.size) || end > storage.backing.size() {
                return Err(NvMapViewError::PlaneOutsideObject);
            }
        }
        Ok(NvMapImageView {
            object_id: self.id,
            backing: storage.backing.clone(),
            metadata,
        })
    }
}

/// Byte range and row pitch of one image plane.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct NvMapPlaneMetadata {
    offset: u64,
    size: u64,
    pitch: u32,
}

impl NvMapPlaneMetadata {
    pub const fn new(offset: u64, size: u64, pitch: u32) -> Self {
        Self {
            offset,
            size,
            pitch,
        }
    }

    pub const fn offset(self) -> u64 {
        self.offset
    }

    pub const fn size(self) -> u64 {
        self.size
    }

    pub const fn pitch(self) -> u32 {
        self.pitch
    }
}

/// Image interpretation carried separately from an `nvmap` allocation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NvMapImageViewMetadata {
    width: u32,
    height: u32,
    format: u32,
    kind: u32,
    layout: u32,
    block_height_log2: u32,
    planes: Box<[NvMapPlaneMetadata]>,
}

impl NvMapImageViewMetadata {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        width: u32,
        height: u32,
        format: u32,
        kind: u32,
        layout: u32,
        block_height_log2: u32,
        planes: Vec<NvMapPlaneMetadata>,
    ) -> Self {
        Self {
            width,
            height,
            format,
            kind,
            layout,
            block_height_log2,
            planes: planes.into_boxed_slice(),
        }
    }

    pub const fn width(&self) -> u32 {
        self.width
    }

    pub const fn height(&self) -> u32 {
        self.height
    }

    pub const fn format(&self) -> u32 {
        self.format
    }

    pub const fn kind(&self) -> u32 {
        self.kind
    }

    pub const fn layout(&self) -> u32 {
        self.layout
    }

    pub const fn block_height_log2(&self) -> u32 {
        self.block_height_log2
    }

    pub fn planes(&self) -> &[NvMapPlaneMetadata] {
        &self.planes
    }
}

/// One retained image view over canonical `nvmap` bytes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NvMapImageView {
    object_id: NvMapObjectId,
    backing: CanonicalBackingRange,
    metadata: NvMapImageViewMetadata,
}

impl NvMapImageView {
    pub const fn object_id(&self) -> NvMapObjectId {
        self.object_id
    }

    pub const fn metadata(&self) -> &NvMapImageViewMetadata {
        &self.metadata
    }

    pub fn read_plane(&self, index: usize) -> Result<Vec<u8>, NvMapViewError> {
        let plane = self
            .metadata
            .planes
            .get(index)
            .ok_or(NvMapViewError::UnknownPlane)?;
        let size = usize::try_from(plane.size).map_err(|_| NvMapViewError::RangeOverflow)?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(size)
            .map_err(|_| NvMapViewError::ResourceExhausted)?;
        output.resize(size, 0);
        self.backing
            .read(plane.offset, &mut output)
            .map_err(NvMapViewError::Backing)?;
        Ok(output)
    }
}

/// Invalid construction or access of an `nvmap` resource view.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NvMapViewError {
    UnallocatedObject,
    MissingPlanes,
    UnknownPlane,
    PlaneOutsideObject,
    RangeOverflow,
    ResourceExhausted,
    Backing(CanonicalRangeAccessError),
}

impl Display for NvMapViewError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnallocatedObject => formatter.write_str("nvmap object has no canonical backing"),
            Self::MissingPlanes => formatter.write_str("nvmap image view has no planes"),
            Self::UnknownPlane => formatter.write_str("nvmap image view plane does not exist"),
            Self::PlaneOutsideObject => {
                formatter.write_str("nvmap image view plane exceeds its object")
            }
            Self::RangeOverflow => formatter.write_str("nvmap image view range overflows"),
            Self::ResourceExhausted => {
                formatter.write_str("nvmap image view resources are exhausted")
            }
            Self::Backing(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for NvMapViewError {}

#[derive(Debug)]
struct NvMapObjectRecord {
    object: NvMapObject,
    owner: NvMapOwner,
    handle_references: u32,
    exported_id: Option<NvMapExportedId>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct NvMapHandleRecord {
    object_id: NvMapObjectId,
    owner: NvMapOwner,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct NvMapExportedIdRecord {
    object_id: NvMapObjectId,
    owner: NvMapOwner,
}

/// All `nvmap` object, handle, and exported-ID state for one shared client.
#[derive(Debug)]
pub(super) struct NvMapObjects {
    next_object_id: u64,
    next_handle: u32,
    next_exported_id: u32,
    objects: BTreeMap<NvMapObjectId, NvMapObjectRecord>,
    handles: BTreeMap<NvMapHandle, NvMapHandleRecord>,
    exported_ids: BTreeMap<NvMapExportedId, NvMapExportedIdRecord>,
}

impl Default for NvMapObjects {
    fn default() -> Self {
        Self {
            next_object_id: NvMapObjectId::FIRST.raw(),
            next_handle: 1,
            next_exported_id: 1,
            objects: BTreeMap::new(),
            handles: BTreeMap::new(),
            exported_ids: BTreeMap::new(),
        }
    }
}

impl NvMapObjects {
    pub(super) fn create(
        &mut self,
        owner: NvMapOwner,
        size: u32,
    ) -> Result<NvMapHandle, NvMapStateError> {
        if size == 0 {
            return Err(NvMapStateError::BadParameter);
        }
        let object_id = NvMapObjectId(self.next_object_id);
        let next_object_id = self
            .next_object_id
            .checked_add(1)
            .ok_or(NvMapStateError::InvalidState)?;
        let handle = NvMapHandle::new(self.next_handle);
        let next_handle = self
            .next_handle
            .checked_add(1)
            .ok_or(NvMapStateError::InvalidState)?;

        self.objects.insert(
            object_id,
            NvMapObjectRecord {
                object: NvMapObject {
                    id: object_id,
                    size,
                    storage: None,
                },
                owner,
                handle_references: 1,
                exported_id: None,
            },
        );
        self.handles
            .insert(handle, NvMapHandleRecord { object_id, owner });
        self.next_object_id = next_object_id;
        self.next_handle = next_handle;
        Ok(handle)
    }

    pub(super) fn import(
        &mut self,
        owner: NvMapOwner,
        exported_id: NvMapExportedId,
    ) -> Result<NvMapHandle, NvMapStateError> {
        let exported = self
            .exported_ids
            .get(&exported_id)
            .copied()
            .ok_or(NvMapStateError::BadParameter)?;
        if exported.owner != owner {
            return Err(NvMapStateError::InvalidOwner);
        }
        let object_id = exported.object_id;
        let handle = NvMapHandle::new(self.next_handle);
        let next_handle = self
            .next_handle
            .checked_add(1)
            .ok_or(NvMapStateError::InvalidState)?;
        let record = self
            .objects
            .get_mut(&object_id)
            .ok_or(NvMapStateError::InvalidState)?;
        if record.owner != owner {
            return Err(NvMapStateError::InvalidOwner);
        }
        let references = record
            .handle_references
            .checked_add(1)
            .ok_or(NvMapStateError::InvalidState)?;

        record.handle_references = references;
        self.handles
            .insert(handle, NvMapHandleRecord { object_id, owner });
        self.next_handle = next_handle;
        Ok(handle)
    }

    pub(super) fn allocation_size(
        &self,
        owner: NvMapOwner,
        handle: NvMapHandle,
    ) -> Result<u32, NvMapStateError> {
        let object = self.object_by_owned_handle(owner, handle)?;
        if object.storage.is_some() {
            return Err(NvMapStateError::AlreadyAllocated);
        }
        Ok(object.size)
    }

    pub(super) fn allocate(
        &mut self,
        owner: NvMapOwner,
        handle: NvMapHandle,
        allocation: NvMapAllocationMetadata,
        cpu_mapping: NvMapCpuMapping,
        backing: CanonicalBackingRange,
    ) -> Result<(), NvMapStateError> {
        allocation.validate()?;
        if backing.size() == 0
            || backing.size() != u64::from(self.object_by_owned_handle(owner, handle)?.size())
            || backing.segments().iter().any(|segment| {
                !segment
                    .permissions()
                    .contains(allocation.required_permissions())
            })
        {
            return Err(NvMapStateError::InvalidBacking);
        }
        let handle_record = self
            .handles
            .get(&handle)
            .copied()
            .ok_or(NvMapStateError::BadParameter)?;
        if handle_record.owner != owner {
            return Err(NvMapStateError::InvalidOwner);
        }
        let object_id = handle_record.object_id;
        let object = &mut self
            .objects
            .get_mut(&object_id)
            .ok_or(NvMapStateError::InvalidState)?
            .object;
        if object.storage.is_some() {
            return Err(NvMapStateError::AlreadyAllocated);
        }
        object.storage = Some(NvMapCanonicalStorage {
            allocation,
            cpu_mapping,
            backing,
        });
        Ok(())
    }

    pub(super) fn free(
        &mut self,
        owner: NvMapOwner,
        handle: NvMapHandle,
    ) -> Result<NvMapFreeResult, NvMapStateError> {
        let handle_record = self
            .handles
            .get(&handle)
            .copied()
            .ok_or(NvMapStateError::BadParameter)?;
        if handle_record.owner != owner {
            return Err(NvMapStateError::InvalidOwner);
        }
        let object_id = handle_record.object_id;
        let record = self
            .objects
            .get_mut(&object_id)
            .ok_or(NvMapStateError::InvalidState)?;
        if record.owner != owner {
            return Err(NvMapStateError::InvalidOwner);
        }
        let size = record.object.size;
        if record.handle_references > 1 {
            record.handle_references -= 1;
            self.handles.remove(&handle);
            return Ok(NvMapFreeResult {
                address: 0,
                size,
                flags: 0,
            });
        }
        if record.handle_references != 1 {
            return Err(NvMapStateError::InvalidState);
        }

        let address = record
            .object
            .cpu_mapping()
            .map_or(0, |mapping| mapping.address().get());
        let flags = record.object.allocation_metadata().map_or(0, |allocation| {
            u32::from(allocation.flags() & NVMAP_UNCACHED_FLAG != 0)
        });
        let exported_id = record.exported_id;
        self.handles.remove(&handle);
        self.objects.remove(&object_id);
        if let Some(exported_id) = exported_id {
            self.exported_ids.remove(&exported_id);
        }
        Ok(NvMapFreeResult {
            address,
            size,
            flags,
        })
    }

    pub(super) fn parameter(
        &self,
        owner: NvMapOwner,
        handle: NvMapHandle,
        parameter: u32,
    ) -> Result<u32, NvMapStateError> {
        let object = self.object_by_owned_handle(owner, handle)?;
        let allocation = object.allocation_metadata();
        match parameter {
            1 => Ok(object.size),
            2 => Ok(allocation.map_or(0, NvMapAllocationMetadata::alignment)),
            4 => Ok(NVMAP_SYSTEM_HEAP_MASK),
            5 => Ok(allocation.map_or(0, |metadata| u32::from(metadata.kind()))),
            6 => Ok(0),
            _ => Err(NvMapStateError::BadParameter),
        }
    }

    pub(super) fn exported_id(
        &mut self,
        owner: NvMapOwner,
        handle: NvMapHandle,
    ) -> Result<NvMapExportedId, NvMapStateError> {
        let handle_record = self
            .handles
            .get(&handle)
            .copied()
            .ok_or(NvMapStateError::BadParameter)?;
        if handle_record.owner != owner {
            return Err(NvMapStateError::InvalidOwner);
        }
        let object_id = handle_record.object_id;
        if let Some(exported_id) = self
            .objects
            .get(&object_id)
            .ok_or(NvMapStateError::InvalidState)?
            .exported_id
        {
            return Ok(exported_id);
        }
        let exported_id = NvMapExportedId::new(self.next_exported_id);
        let next_exported_id = self
            .next_exported_id
            .checked_add(1)
            .ok_or(NvMapStateError::InvalidState)?;
        self.exported_ids
            .insert(exported_id, NvMapExportedIdRecord { object_id, owner });
        self.objects
            .get_mut(&object_id)
            .ok_or(NvMapStateError::InvalidState)?
            .exported_id = Some(exported_id);
        self.next_exported_id = next_exported_id;
        Ok(exported_id)
    }

    pub(super) fn object_by_exported_id(
        &self,
        exported_id: NvMapExportedId,
    ) -> Option<NvMapObject> {
        self.exported_ids
            .get(&exported_id)
            .and_then(|record| self.objects.get(&record.object_id))
            .map(|record| record.object.clone())
    }

    pub(super) fn object_snapshot_by_handle(&self, handle: NvMapHandle) -> Option<NvMapObject> {
        self.object_by_handle(handle).ok().cloned()
    }

    pub(super) fn object_by_handle(
        &self,
        handle: NvMapHandle,
    ) -> Result<&NvMapObject, NvMapStateError> {
        self.handles
            .get(&handle)
            .and_then(|record| self.objects.get(&record.object_id))
            .map(|record| &record.object)
            .ok_or(NvMapStateError::BadParameter)
    }

    fn object_by_owned_handle(
        &self,
        owner: NvMapOwner,
        handle: NvMapHandle,
    ) -> Result<&NvMapObject, NvMapStateError> {
        let handle_record = self
            .handles
            .get(&handle)
            .ok_or(NvMapStateError::BadParameter)?;
        if handle_record.owner != owner {
            return Err(NvMapStateError::InvalidOwner);
        }
        let record = self
            .objects
            .get(&handle_record.object_id)
            .ok_or(NvMapStateError::InvalidState)?;
        if record.owner != owner {
            return Err(NvMapStateError::InvalidOwner);
        }
        Ok(&record.object)
    }

    #[cfg(test)]
    pub(super) fn handle_references(&self, handle: NvMapHandle) -> Result<u32, NvMapStateError> {
        let object_id = self
            .handles
            .get(&handle)
            .map(|record| record.object_id)
            .ok_or(NvMapStateError::BadParameter)?;
        self.objects
            .get(&object_id)
            .map(|record| record.handle_references)
            .ok_or(NvMapStateError::InvalidState)
    }

    pub(super) fn clear(&mut self) -> usize {
        let released = self.objects.len();
        self.handles.clear();
        self.exported_ids.clear();
        self.objects.clear();
        released
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct NvMapFreeResult {
    pub address: u64,
    pub size: u32,
    pub flags: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum NvMapStateError {
    BadParameter,
    InvalidState,
    AlreadyAllocated,
    InvalidBacking,
    InvalidOwner,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_object_reference_operation_rejects_a_foreign_owner() {
        let owner = NvMapOwner::new(7);
        let foreign = NvMapOwner::new(8);
        let mut objects = NvMapObjects::default();
        let handle = objects.create(owner, 0x1000).unwrap();

        assert_eq!(
            objects.allocation_size(foreign, handle),
            Err(NvMapStateError::InvalidOwner)
        );
        assert_eq!(
            objects.parameter(foreign, handle, 1),
            Err(NvMapStateError::InvalidOwner)
        );
        assert_eq!(
            objects.exported_id(foreign, handle),
            Err(NvMapStateError::InvalidOwner)
        );
        assert_eq!(
            objects.free(foreign, handle),
            Err(NvMapStateError::InvalidOwner)
        );

        let exported_id = objects.exported_id(owner, handle).unwrap();
        assert_eq!(
            objects.import(foreign, exported_id),
            Err(NvMapStateError::InvalidOwner)
        );
        assert!(objects.object_by_handle(handle).is_ok());
    }
}
