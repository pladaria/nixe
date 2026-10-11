//! Configured TTF exposed by `pl:u` in read-only shared memory.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use nixe_cpu::memory::{ExecutionMemory, MemoryPermissions};
use nixe_runtime::SharedMemoryObject;

// IDs and the shared-memory ABI are documented by libnx. The configured
// replacement font is shared across all supported system-font type requests.
// https://github.com/switchbrew/libnx/blob/master/nx/source/services/pl.c
// https://switchbrew.org/wiki/Glue_services#SharedFontType
const MEMORY_SIZE: usize = 0x110_0000;
const FONT_TYPE_COUNT: usize = 7;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct FontRegion {
    pub offset: u32,
    pub size: u32,
}

#[derive(Debug, Default)]
struct FontState {
    memory: Option<SharedMemoryObject>,
    region: Option<FontRegion>,
}

/// Connections and system-font types share the configured font's backing.
#[derive(Clone, Debug, Default)]
pub struct SharedFontSession {
    font: Option<PathBuf>,
    state: Arc<Mutex<FontState>>,
}

impl SharedFontSession {
    #[must_use]
    pub fn new(font: Option<PathBuf>) -> Self {
        Self {
            font,
            ..Self::default()
        }
    }

    pub(crate) fn valid_type(font: u32) -> bool {
        (font as usize) < FONT_TYPE_COUNT
    }

    #[cfg(test)]
    fn region(&self) -> Option<FontRegion> {
        self.state.lock().unwrap().region
    }

    pub(crate) fn shared_memory(
        &self,
        memory: &ExecutionMemory,
    ) -> Result<SharedMemoryObject, String> {
        let mut state = self.state.lock().unwrap();
        if state.memory.is_none() {
            state.memory = Some(
                SharedMemoryObject::for_process(memory, MEMORY_SIZE, MemoryPermissions::READ)
                    .map_err(|error| error.to_string())?,
            );
        }
        Ok(state.memory.as_ref().unwrap().clone())
    }

    pub(crate) fn request_load(
        &self,
        memory: &ExecutionMemory,
        font: u32,
    ) -> Result<FontRegion, String> {
        let shared = self.shared_memory(memory)?;
        let mut state = self.state.lock().unwrap();
        if let Some(region) = state.region {
            return Ok(region);
        }
        let path = self.font.as_ref().ok_or_else(|| {
            format!("font type {font} requires a TTF file; configure system.font with its path")
        })?;
        let bytes = std::fs::read(path)
            .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
        validate_sfnt(&bytes)
            .map_err(|error| format!("invalid font {}: {error}", path.display()))?;
        bytes
            .len()
            .checked_add(8)
            .filter(|end| *end <= MEMORY_SIZE)
            .ok_or_else(|| "fonts exceed the 17 MiB shared-memory region".to_owned())?;
        // Preserve the decrypted BFTTF header preceding each TTF, including
        // the encoded size used by Nintendo's font runtime.
        // https://ni.4a.si/anonymous/yuzu/tree/src/core/hle/service/ns/pl_u.cpp?id=99eaa2e6f2e04a761e2bc3f92b071352ec88a71f
        let mut header = [0; 8];
        header[..4].copy_from_slice(&0x7f9a_0218_u32.to_be_bytes());
        header[4..].copy_from_slice(&((bytes.len() as u32) ^ 0x4962_1806).to_le_bytes());
        shared
            .write(0, &header)
            .map_err(|error| error.to_string())?;
        shared.write(8, &bytes).map_err(|error| error.to_string())?;
        let region = FontRegion {
            offset: 8,
            size: bytes.len() as u32,
        };
        state.region = Some(region);
        log::debug!(
            "loaded shared font: type={font} file={} offset={:#x} size={:#x}",
            path.display(),
            region.offset,
            region.size
        );
        Ok(region)
    }
}

fn validate_sfnt(bytes: &[u8]) -> Result<(), &'static str> {
    if bytes.len() < 12 || !matches!(&bytes[..4], b"\0\x01\0\0" | b"OTTO" | b"true") {
        return Err("missing SFNT header");
    }
    let count = u16::from_be_bytes(bytes[4..6].try_into().unwrap()) as usize;
    let directory_end = 12 + count * 16;
    if count == 0 || directory_end > bytes.len() {
        return Err("truncated or empty SFNT table directory");
    }
    for entry in bytes[12..directory_end].chunks_exact(16) {
        let offset = u32::from_be_bytes(entry[8..12].try_into().unwrap()) as u64;
        let size = u32::from_be_bytes(entry[12..16].try_into().unwrap()) as u64;
        if offset + size > bytes.len() as u64 {
            return Err("SFNT table exceeds file bounds");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_font_does_not_publish_a_loaded_region() {
        let session = SharedFontSession::default();
        let memory = ExecutionMemory::new();
        let shared = session.shared_memory(&memory).unwrap();
        assert_eq!(shared.size(), MEMORY_SIZE);
        assert_eq!(shared.remote_permissions(), MemoryPermissions::READ);
        assert!(
            session
                .request_load(&memory, 0)
                .unwrap_err()
                .contains("system.font")
        );
        assert_eq!(session.region(), None);
    }

    #[test]
    fn validates_sfnt_table_bounds() {
        let mut font = vec![0; 32];
        font[..4].copy_from_slice(&[0, 1, 0, 0]);
        font[4..6].copy_from_slice(&1_u16.to_be_bytes());
        font[20..24].copy_from_slice(&28_u32.to_be_bytes());
        font[24..28].copy_from_slice(&4_u32.to_be_bytes());
        assert_eq!(validate_sfnt(&font), Ok(()));
        font[24..28].copy_from_slice(&5_u32.to_be_bytes());
        assert_eq!(validate_sfnt(&font), Err("SFNT table exceeds file bounds"));
        assert!(validate_sfnt(&font[..20]).is_err());
    }

    #[test]
    fn publishes_font_bytes_once_and_keeps_offsets_stable_across_sessions() {
        let directory = tempfile::tempdir().unwrap();
        let mut font = vec![0; 32];
        font[..4].copy_from_slice(&[0, 1, 0, 0]);
        font[4..6].copy_from_slice(&1_u16.to_be_bytes());
        font[20..24].copy_from_slice(&28_u32.to_be_bytes());
        font[24..28].copy_from_slice(&4_u32.to_be_bytes());
        font[28..].copy_from_slice(b"test");
        let path = directory.path().join("font.ttf");
        std::fs::write(&path, &font).unwrap();
        let session = SharedFontSession::new(Some(path.clone()));
        let memory = ExecutionMemory::new();
        let shared = session.shared_memory(&memory).unwrap();
        let first = session.request_load(&memory, 0).unwrap();
        let second = session.clone().request_load(&memory, 5).unwrap();
        assert_eq!(
            first,
            FontRegion {
                offset: 8,
                size: 32
            }
        );
        assert_eq!(
            second,
            FontRegion {
                offset: 8,
                size: 32
            }
        );
        let mut payload = [0; 32];
        shared.read(second.offset as usize, &mut payload).unwrap();
        assert_eq!(payload.as_slice(), font);
        std::fs::remove_file(path).unwrap();
        assert_eq!(session.clone().request_load(&memory, 0).unwrap(), first);
        assert_eq!(session.request_load(&memory, 1).unwrap(), first);
        assert_eq!(session.region(), Some(first));
    }

    #[test]
    fn bundled_dejavu_sans_is_published_unchanged_for_every_supported_font_type() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets/DejaVuSans.ttf");
        let bytes = std::fs::read(&path).unwrap();
        let session = SharedFontSession::new(Some(path));
        let memory = ExecutionMemory::new();
        let region = session.request_load(&memory, 0).unwrap();
        assert_eq!(region.size as usize, bytes.len());
        for font in 1..FONT_TYPE_COUNT as u32 {
            assert_eq!(session.request_load(&memory, font).unwrap(), region);
        }
        let shared = session.shared_memory(&memory).unwrap();
        let mut payload = vec![0; bytes.len()];
        shared.read(region.offset as usize, &mut payload).unwrap();
        assert_eq!(payload, bytes);
    }
}
