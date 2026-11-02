//! Code inputs captured before leaving native translation admission.
use nixe_memory::{
    Access, AccessResult, AddressSpace, AddressSpaceId, AllocationRange, CodeFetch,
    GuestVirtualAddress, MappingObservation, MemoryError, NativeAdmission, NativeCodePage,
    Observation,
};
use std::{collections::HashMap, sync::Arc};

#[derive(Default)]
pub(crate) struct CodeCache {
    page: Option<NativeCodePage>,
    device_words: HashMap<u64, CodeWord>,
}
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
        if let Some(word) = self
            .page
            .as_ref()
            .and_then(|page| page.read(native, address))
        {
            return Ok(Lookup::Ready(word));
        }
        match space.fetch_code_native(native, address)? {
            CodeFetch::Ready { word, page } => {
                self.page = Some(page);
                Ok(Lookup::Ready(word))
            }
            CodeFetch::Capture {
                ranges,
                device_owned,
            } => {
                if let Some(word) = self.device_words.get(&address.get())
                    && !word.observation.is_invalid()
                {
                    return Ok(Lookup::Ready(word.word));
                }
                self.device_words.remove(&address.get());
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
            self.device_words.insert(code.word.address.get(), code.word);
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
            let bytes: [u8; 4] = bytes.as_slice().try_into().map_err(|_| {
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

#[cfg(test)]
mod tests {
    use super::*;
    use nixe_memory::{AddressWidth, Machine, MemoryPermissions, PAGE_SIZE};

    #[test]
    #[cfg(target_os = "linux")]
    #[ignore = "controlled native code-reader measurement"]
    fn native_code_reader_throughput() {
        let pages = 16;
        let bytes = pages * PAGE_SIZE;
        let machine = Machine::new(bytes).unwrap();
        let allocation = machine.account(bytes).unwrap().allocate(bytes).unwrap();
        let words: Vec<_> = (0..bytes / 4).map(|index| index as u32).collect();
        machine
            .write_resolved(
                &[AllocationRange::new(allocation.pin(), 0, bytes).unwrap()],
                &words
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let space = AddressSpace::new(machine.clone(), AddressWidth::Bits32).unwrap();
        let base = GuestVirtualAddress::new(PAGE_SIZE);
        space
            .map(base, allocation, 0, bytes, MemoryPermissions::EXECUTE)
            .unwrap();
        let captured = space
            .capture(base, bytes, MemoryPermissions::EXECUTE)
            .unwrap();
        let (access, _observation) = machine.access_code_captured(captured).unwrap();
        access.into_ready().unwrap();
        let context = space.register_core(Arc::new(|| {
            panic!("unexpected transition during benchmark")
        }));
        let native = context.enter_native().unwrap();
        let mut cache = CodeCache::default();
        let iterations = std::env::var("NIXE_CODE_READER_ITERS")
            .ok()
            .map(|value| value.parse::<u64>().unwrap())
            .unwrap_or(1000);
        let cpu_ns = || {
            std::fs::read_to_string("/proc/thread-self/schedstat")
                .unwrap()
                .split_whitespace()
                .next()
                .unwrap()
                .parse::<u64>()
                .unwrap()
        };
        let begin_cpu = cpu_ns();
        let begin = std::time::Instant::now();
        let mut checksum = 0u64;
        for _ in 0..iterations {
            for (index, expected) in words.iter().enumerate() {
                let address = GuestVirtualAddress::new(PAGE_SIZE + index as u64 * 4);
                let Lookup::Ready(word) = cache
                    .fetch(&space, &native, std::hint::black_box(address))
                    .unwrap()
                else {
                    panic!("tracked code unexpectedly requires capture");
                };
                assert_eq!(word, *expected);
                checksum += u64::from(word);
            }
        }
        let wall_ns = begin.elapsed().as_nanos();
        let elapsed_cpu = cpu_ns() - begin_cpu;
        let count = iterations * words.len() as u64;
        assert_eq!(
            checksum,
            iterations * (words.len() as u64 - 1) * words.len() as u64 / 2
        );
        println!(
            "CODE_READER_BENCH {{\"words\":{count},\"cpu_ns\":{elapsed_cpu},\"wall_ns\":{wall_ns},\"checksum\":{checksum}}}"
        );
    }
}
