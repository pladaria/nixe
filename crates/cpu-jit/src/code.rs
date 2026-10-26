//! Code inputs captured before leaving native translation admission.
use nixe_memory::{
    Access, AccessResult, AddressSpace, AddressSpaceId, AllocationRange, CodeFetch,
    GuestVirtualAddress, MappingObservation, MemoryError, NativeAdmission, Observation,
};
use std::{collections::HashMap, sync::Arc};

#[derive(Default)]
pub(crate) struct CodeCache(HashMap<u64, CodeWord>);
pub(crate) enum Lookup {
    Ready(u32),
    Capture(CodeCapture),
}
/// Immutable code input. It may outlive a remap but cannot be reused by a new
/// translation after its physical or mapping observation becomes invalid.
pub struct CodeCapture {
    space: AddressSpaceId,
    address: GuestVirtualAddress,
    ranges: Vec<AllocationRange>,
    device_owned: bool,
}
struct CodeWord {
    address: GuestVirtualAddress,
    word: u32,
    observation: Observation,
    _mappings: Vec<MappingObservation>,
}
/// The scheduler retains this owned continuation while the graphics owner
/// resolves code bytes. Neither a C++ frame nor native admission survives here.
pub struct CodeRead {
    space: AddressSpaceId,
    address: GuestVirtualAddress,
    device_owned: bool,
    access: Option<Access<AccessResult>>,
    observation: Observation,
    mappings: Vec<MappingObservation>,
}
pub struct ResolvedCode {
    space: AddressSpaceId,
    device_owned: bool,
    word: CodeWord,
}
impl ResolvedCode {
    pub fn address(&self) -> GuestVirtualAddress {
        self.word.address
    }
    pub(crate) fn space(&self) -> AddressSpaceId {
        self.space
    }
}
impl CodeCache {
    pub fn fetch(
        &mut self,
        space: &AddressSpace,
        native: &NativeAdmission<'_>,
        address: GuestVirtualAddress,
    ) -> Result<Lookup, MemoryError> {
        match space.fetch_code_native(native, address)? {
            CodeFetch::Ready { word, .. } => Ok(Lookup::Ready(word)),
            CodeFetch::Capture {
                ranges,
                device_owned,
            } => {
                if let Some(word) = self.0.get(&address.get())
                    && !word.observation.is_invalid()
                {
                    return Ok(Lookup::Ready(word.word));
                }
                self.0.remove(&address.get());
                Ok(Lookup::Capture(CodeCapture {
                    space: space.identity(),
                    address,
                    ranges,
                    device_owned,
                }))
            }
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
        // Ordinary RAM now has a physical code watch and remains a direct
        // indexed read. Only device-owned code needs an owned-word cache.
        if code.device_owned && !code.word.observation.is_invalid() {
            self.0.insert(code.word.address.get(), code.word);
        }
        Ok(())
    }
}
impl CodeCapture {
    pub fn address(&self) -> GuestVirtualAddress {
        self.address
    }
    pub fn start(self, space: &AddressSpace) -> Result<CodeRead, MemoryError> {
        if self.space != space.identity() {
            return Err(MemoryError::Invalid(
                "code input belongs to another address space",
            ));
        }
        let (access, observation) = space.machine().access_code_captured(self.ranges.clone())?;
        let mappings = self
            .ranges
            .iter()
            .map(|range| space.observe_mapping(range, &observation))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(CodeRead {
            space: self.space,
            address: self.address,
            device_owned: self.device_owned,
            access: Some(access),
            observation,
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
            let bytes: [u8; 4] = bytes.as_ref().try_into().map_err(|_| {
                Arc::new(MemoryError::Invalid(
                    "code read did not return one A64 word",
                ))
            })?;
            Ok(ResolvedCode {
                space: self.space,
                device_owned: self.device_owned,
                word: CodeWord {
                    address: self.address,
                    word: u32::from_le_bytes(bytes),
                    observation: self.observation.clone(),
                    _mappings: std::mem::take(&mut self.mappings),
                },
            })
        }))
    }
}
