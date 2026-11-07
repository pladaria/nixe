//! Shader bytecode lookup and invalidation indexed by device address.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/shader_cache.cpp
use crate::{MaxwellGpuAccessError, MaxwellGpuMemory, MaxwellShaderStage};
use nixe_memory::device_memory::DeviceMemory;
use nixe_shader_recompiler::maxwell::header::ProgramHeader;
use rustc_hash::FxHashMap as HashMap;
use std::sync::{Arc, Mutex};

const PAGE_BITS: u32 = 14;
const PAGE_SIZE: u64 = 1 << PAGE_BITS;

/// Immutable shader facts. Current channel selections retain these facts,
/// independently of cache page marks and owned compiler environments.
#[derive(Debug)]
pub struct ShaderInfo {
    pub unique_hash: u64,
    pub size_bytes: usize,
    pub header: ProgramHeader,
}
struct Entry {
    address: u64,
    end: u64,
    info: Arc<ShaderInfo>,
}
#[derive(Default)]
struct Invalidations {
    pages: HashMap<u64, Vec<Arc<Entry>>>,
}
struct Cache {
    memory: DeviceMemory,
    lookup: Mutex<HashMap<u64, Arc<Entry>>>,
    invalidations: Mutex<Invalidations>,
}
impl Drop for Cache {
    fn drop(&mut self) {
        for entry in self.lookup.get_mut().unwrap().values() {
            self.memory
                .update_pages_cached_count(entry.address, entry.end - entry.address, -1)
                .expect("registered shader cache pages");
        }
    }
}
#[derive(Clone)]
pub struct ShaderCache(Arc<Cache>);
impl ShaderCache {
    pub fn new(memory: DeviceMemory) -> Self {
        Self(Arc::new(Cache {
            memory,
            lookup: Mutex::new(HashMap::default()),
            invalidations: Mutex::new(Invalidations::default()),
        }))
    }
    pub fn try_get(&self, address: u64) -> Option<Arc<ShaderInfo>> {
        self.0
            .lookup
            .lock()
            .unwrap()
            .get(&address)
            .map(|entry| Arc::clone(&entry.info))
    }
    pub fn make_shader_info(
        &self,
        space: &MaxwellGpuMemory,
        address: u64,
        request: &super::recompiler::CaptureRequest,
        access: &mut nixe_gpu::MemoryAccess<'_>,
    ) -> nixe_shader_recompiler::Result<Arc<ShaderInfo>> {
        let info = Arc::new(super::recompiler::analyze_code(space, request, access)?);
        let bytes = info.size_bytes as u64;
        let end = address
            .checked_add(bytes)
            .ok_or_else(|| super::recompiler::invalid("Shader cache range overflows"))?;
        let entry = Arc::new(Entry {
            address,
            end,
            info: Arc::clone(&info),
        });
        // Register publishes one address entry and its page index together;
        // only these cache-owned entries mark device pages cached.
        let mut invalidations = self.0.invalidations.lock().unwrap();
        let mut lookup = self.0.lookup.lock().unwrap();
        if let Some(old) = lookup.get(&address) {
            return Ok(Arc::clone(&old.info));
        }
        self.0
            .memory
            .update_pages_cached_count(address, bytes, 1)
            .map_err(|e| super::recompiler::invalid(e.to_string()))?;
        lookup.insert(address, Arc::clone(&entry));
        for page in address >> PAGE_BITS..end.div_ceil(PAGE_SIZE) {
            invalidations
                .pages
                .entry(page)
                .or_default()
                .push(Arc::clone(&entry));
        }
        Ok(info)
    }
    pub fn invalidate_region(&self, address: u64, bytes: u64) {
        if bytes == 0 {
            return;
        }
        let end = address + bytes;
        let mut invalidations = self.0.invalidations.lock().unwrap();
        let mut removed = Vec::<Arc<Entry>>::new();
        for page in address >> PAGE_BITS..end.div_ceil(PAGE_SIZE) {
            if let Some(entries) = invalidations.pages.get(&page) {
                for entry in entries {
                    if entry.address < end
                        && address < entry.end
                        && !removed.iter().any(|old| Arc::ptr_eq(old, entry))
                    {
                        removed.push(Arc::clone(entry));
                    }
                }
            }
        }
        if removed.is_empty() {
            return;
        }
        let mut lookup = self.0.lookup.lock().unwrap();
        for entry in &removed {
            lookup.remove(&entry.address);
            for page in entry.address >> PAGE_BITS..entry.end.div_ceil(PAGE_SIZE) {
                let empty = if let Some(entries) = invalidations.pages.get_mut(&page) {
                    entries.retain(|old| !Arc::ptr_eq(old, entry));
                    entries.is_empty()
                } else {
                    false
                };
                if empty {
                    invalidations.pages.remove(&page);
                }
            }
            self.0
                .memory
                .update_pages_cached_count(entry.address, entry.end - entry.address, -1)
                .expect("invalidated shader cache pages");
        }
    }
    pub fn invalidate_device_write(&self, memory: &DeviceMemory, address: u64, bytes: u64) {
        self.invalidate_region(address, bytes);
        memory.apply_device_aliases(address, bytes, |alias, size| {
            self.invalidate_region(alias, size)
        });
    }
}

/// Only the graphics owner reads mutable guest memory. Compilation workers
/// consume recording environments with their own instruction/data bytes.
pub struct ShaderMemory<'a, 'm> {
    address_space: &'a MaxwellGpuMemory,
    access: &'a mut nixe_gpu::MemoryAccess<'m>,
}
impl<'a, 'm> ShaderMemory<'a, 'm> {
    pub fn new(
        address_space: &'a MaxwellGpuMemory,
        access: &'a mut nixe_gpu::MemoryAccess<'m>,
    ) -> Self {
        Self {
            address_space,
            access,
        }
    }
    pub fn read_code(
        &mut self,
        address: u64,
        size: usize,
    ) -> nixe_shader_recompiler::Result<Vec<u8>> {
        let mut output = vec![0; size];
        self.read_code_into(address, &mut output)?;
        Ok(output)
    }
    pub fn read_code_into(
        &mut self,
        address: u64,
        output: &mut [u8],
    ) -> nixe_shader_recompiler::Result<()> {
        let address = self
            .address_space
            .address(address)
            .map_err(super::recompiler::invalid)?;
        self.address_space
            .read_into(address, output, self.access)
            .map_err(super::recompiler::invalid)
    }
    pub fn instruction(&self, address: u64) -> nixe_shader_recompiler::Result<u64> {
        let address = self
            .address_space
            .address(address)
            .map_err(super::recompiler::invalid)?;
        self.address_space
            .read_u64(address)
            .map_err(|e| super::recompiler::invalid(e.to_string()))
    }
    pub fn read_data(
        &self,
        stage: MaxwellShaderStage,
        address: u64,
        size: usize,
    ) -> Result<Arc<[u8]>, ShaderCaptureError> {
        if size > 64 * 1024 {
            return Err(ShaderCaptureError::ReadTooLarge {
                requested: size,
                limit: 64 * 1024,
            });
        }
        let fail = |error| ShaderCaptureError::Memory {
            stage,
            address,
            error,
        };
        let address = self
            .address_space
            .address(address)
            .map_err(MaxwellGpuAccessError::Address)
            .map_err(fail)?;
        let mut bytes = vec![0; size];
        self.address_space
            .read_into_unsafe(address, &mut bytes)
            .map_err(fail)?;
        Ok(bytes.into())
    }
}
#[derive(Debug)]
pub enum ShaderCaptureError {
    ReadTooLarge {
        requested: usize,
        limit: usize,
    },
    Memory {
        stage: MaxwellShaderStage,
        address: u64,
        error: MaxwellGpuAccessError,
    },
}
impl std::fmt::Display for ShaderCaptureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ReadTooLarge { requested, limit } => write!(
                f,
                "shader capture requests {requested} bytes, limit {limit}"
            ),
            Self::Memory {
                stage,
                address,
                error,
            } => write!(f, "{stage:?} shader capture at {address:#x}: {error}"),
        }
    }
}
