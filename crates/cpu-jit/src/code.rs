//! Instruction inputs follow Eden's current-page cache and explicit IC operations.
//! Device-owned code is resolved outside the native compilation frame.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/arm/dynarmic/arm_dynarmic_64.cpp
use nixe_memory::{
    Access, AccessResult, AddressSpace, AddressSpaceId, AllocationRange, CodeFetch,
    GuestVirtualAddress, MappingObservation, MemoryError, NativeAdmission, NativeCodePage,
    PAGE_SIZE,
};
use std::sync::Arc;
const CODE_PAGE_BYTES: usize = PAGE_SIZE as usize;
pub(crate) struct CodeCache {
    page: Option<NativeCodePage>,
    bytes: Box<[u8; CODE_PAGE_BYTES]>,
    captured: Option<ResolvedCode>,
}
impl Default for CodeCache {
    fn default() -> Self {
        Self {
            page: None,
            bytes: Box::new([0; CODE_PAGE_BYTES]),
            captured: None,
        }
    }
}
pub(crate) enum Lookup {
    Ready(u32),
    Capture(CodeCapture),
}
pub struct CodeCapture {
    space: AddressSpaceId,
    address: GuestVirtualAddress,
    ranges: Vec<AllocationRange>,
}
pub struct CodeRead {
    space: AddressSpaceId,
    address: GuestVirtualAddress,
    access: Option<Access<AccessResult>>,
    mappings: Vec<MappingObservation>,
}
pub struct ResolvedCode {
    space: AddressSpaceId,
    address: GuestVirtualAddress,
    bytes: Box<[u8; CODE_PAGE_BYTES]>,
    mappings: Vec<MappingObservation>,
}
impl ResolvedCode {
    pub(crate) fn page_start(&self) -> u64 {
        self.address.get() & !(PAGE_SIZE - 1)
    }
    pub(crate) fn space(&self) -> AddressSpaceId {
        self.space
    }
}
impl CodeCache {
    pub fn clear(&mut self) {
        self.page = None;
        self.captured = None;
    }
    pub fn fetch(
        &mut self,
        space: &AddressSpace,
        native: &NativeAdmission<'_>,
        address: GuestVirtualAddress,
    ) -> Result<Lookup, MemoryError> {
        if let Some(word) = self
            .page
            .as_ref()
            .and_then(|page| page.read(native, address, &self.bytes))
        {
            return Ok(Lookup::Ready(word));
        }
        if let Some(page) = &self.captured
            && page.space == space.identity()
            && page.page_start() == address.get() & !(PAGE_SIZE - 1)
            && page.mappings.iter().all(|mapping| !mapping.is_invalid())
        {
            space.check_native_access(native, address, nixe_memory::MemoryPermissions::EXECUTE)?;
            let offset = (address.get() & (PAGE_SIZE - 1)) as usize;
            if !address.get().is_multiple_of(4) {
                return Err(MemoryError::Invalid("unaligned A64 instruction fetch"));
            }
            return Ok(Lookup::Ready(u32::from_le_bytes(
                page.bytes[offset..offset + 4].try_into().unwrap(),
            )));
        }
        self.clear();
        match space.fetch_code_native(native, address, &mut self.bytes)? {
            CodeFetch::Ready { word, page } => {
                self.page = Some(page);
                Ok(Lookup::Ready(word))
            }
            CodeFetch::Capture { ranges } => Ok(Lookup::Capture(CodeCapture {
                space: space.identity(),
                address,
                ranges,
            })),
        }
    }
    pub fn install(
        &mut self,
        space: AddressSpaceId,
        code: ResolvedCode,
    ) -> Result<(), MemoryError> {
        if code.space != space {
            return Err(MemoryError::Invalid(
                "resolved code belongs to another address space",
            ));
        }
        self.page = None;
        self.captured = code
            .mappings
            .iter()
            .all(|mapping| !mapping.is_invalid())
            .then_some(code);
        Ok(())
    }
}
impl CodeCapture {
    pub fn address(&self) -> GuestVirtualAddress {
        self.address
    }
    pub(crate) fn covers(&self, address: GuestVirtualAddress) -> bool {
        self.address.get() & !(PAGE_SIZE - 1) == address.get() & !(PAGE_SIZE - 1)
    }
    pub fn start(self, space: &AddressSpace) -> Result<CodeRead, MemoryError> {
        if self.space != space.identity() {
            return Err(MemoryError::Invalid(
                "code input belongs to another address space",
            ));
        }
        let mappings = self
            .ranges
            .iter()
            .map(|range| space.observe_mapping(range))
            .collect::<Result<Vec<_>, _>>()?;
        let access = space.machine().read_cached_captured(&self.ranges)?;
        Ok(CodeRead {
            space: self.space,
            address: self.address,
            access: Some(access),
            mappings,
        })
    }
}
impl CodeRead {
    pub fn cancel(mut self) -> Option<Arc<MemoryError>> {
        match self.access.as_mut() {
            Some(Access::Pending(pending)) => pending.poll().and_then(Result::err),
            _ => None,
        }
    }
    pub fn set_waker(&mut self, wake: Arc<dyn Fn() + Send + Sync>) {
        match self.access.as_mut().expect("code result already consumed") {
            Access::Pending(pending) => pending.set_waker(wake),
            Access::Ready(_) => wake(),
        }
    }
    pub fn poll(&mut self) -> Option<Result<ResolvedCode, Arc<MemoryError>>> {
        let result = match self.access.as_mut().expect("code result already consumed") {
            Access::Ready(_) => None,
            Access::Pending(pending) => Some(pending.poll()?),
        };
        let result = match (self.access.take().unwrap(), result) {
            (_, Some(result)) => result,
            (Access::Ready(result), None) => Ok(result),
            _ => unreachable!(),
        };
        Some(result.and_then(|result| {
            let AccessResult::Read(bytes) = result else {
                return Err(Arc::new(MemoryError::Invalid("invalid code read result")));
            };
            let bytes = bytes.into_boxed_slice().try_into().map_err(|_| {
                Arc::new(MemoryError::Invalid(
                    "code read did not return one instruction page",
                ))
            })?;
            Ok(ResolvedCode {
                space: self.space,
                address: self.address,
                bytes,
                mappings: std::mem::take(&mut self.mappings),
            })
        }))
    }
}
