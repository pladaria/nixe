//! Configured TTF exposed by `pl:u` in read-only shared memory.

use std::path::PathBuf;
use std::sync::Arc;

use nixe_memory::{AllocationRange, MemoryAccount, MemoryPermissions};
use nixe_runtime::SharedMemoryObject;

// https://github.com/switchbrew/libnx/blob/master/nx/source/services/pl.c
// https://switchbrew.org/wiki/Glue_services#SharedFontType
const MEMORY_SIZE: usize = 0x110_0000;
const FONT_TYPE_COUNT: usize = 7;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct FontRegion {
    pub offset: u32,
    pub size: u32,
}

#[derive(Debug)]
struct FontState {
    memory: SharedMemoryObject,
    region: Result<FontRegion, String>,
}

/// One system allocation, initialized before any client can map it.
#[derive(Clone, Debug)]
pub struct SharedFontSession(Arc<FontState>);

impl SharedFontSession {
    /// Prepares the configured replacement font. Invalid or absent font data is
    /// reported when a client requests it; physical allocation failures stop
    /// construction. Font types share the same immutable configured payload.
    pub fn new(account: &MemoryAccount, font: Option<PathBuf>) -> Result<Self, String> {
        let allocation = account
            .allocate(MEMORY_SIZE as u64)
            .map_err(|e| e.to_string())?;
        // Prepare once, as the platform service constructor does in Eden:
        // https://git.eden-emu.dev/eden-emu/eden/src/branch/master/src/core/hle/service/ns/platform_service_manager.cpp
        let region = match font {
            None => Err(
                "system fonts require a TTF file; configure system.font with its path".to_owned(),
            ),
            Some(path) => match prepare_font(&path) {
                Err(error) => Err(error),
                Ok(payload) => {
                    let bytes = payload.len() as u64;
                    let range = AllocationRange::new(
                        allocation.pin_range(0, bytes).map_err(|e| e.to_string())?,
                        0,
                        bytes,
                    )
                    .map_err(|e| e.to_string())?;
                    // No mapping or device can own this private allocation yet.
                    account
                        .machine()
                        .write_resolved(&[range], &payload)
                        .map_err(|e| e.to_string())?;
                    Ok(FontRegion {
                        offset: 8,
                        size: (payload.len() - 8) as u32,
                    })
                }
            },
        };
        let memory = SharedMemoryObject::new(allocation, MEMORY_SIZE, MemoryPermissions::READ)
            .map_err(|e| e.to_string())?;
        Ok(Self(Arc::new(FontState { memory, region })))
    }

    pub(crate) fn valid_type(font: u32) -> bool {
        (font as usize) < FONT_TYPE_COUNT
    }

    pub(crate) fn shared_memory(&self) -> SharedMemoryObject {
        self.0.memory.clone()
    }

    pub(crate) fn request_load(&self, _font: u32) -> Result<FontRegion, String> {
        self.0.region.clone()
    }
}

fn prepare_font(path: &std::path::Path) -> Result<Vec<u8>, String> {
    let bytes =
        std::fs::read(path).map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    validate_sfnt(&bytes).map_err(|error| format!("invalid font {}: {error}", path.display()))?;
    bytes
        .len()
        .checked_add(8)
        .filter(|end| *end <= MEMORY_SIZE)
        .ok_or_else(|| "fonts exceed the 17 MiB shared-memory region".to_owned())?;
    let mut payload = Vec::with_capacity(bytes.len() + 8);
    // Decrypted BFTTF header followed by the unchanged TTF payload.
    // https://git.eden-emu.dev/eden-emu/eden/src/branch/master/src/core/hle/service/ns/platform_service_manager.cpp
    payload.extend_from_slice(&0x7f9a_0218_u32.to_be_bytes());
    payload.extend_from_slice(&((bytes.len() as u32) ^ 0x4962_1806).to_le_bytes());
    payload.extend_from_slice(&bytes);
    Ok(payload)
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

    fn account() -> MemoryAccount {
        nixe_memory::Machine::new(32 * 1024 * 1024)
            .unwrap()
            .account(32 * 1024 * 1024)
            .unwrap()
    }

    #[test]
    fn missing_font_does_not_publish_a_loaded_region() {
        let account = account();
        let session = SharedFontSession::new(&account, None).unwrap();
        let shared = session.shared_memory();
        assert_eq!(shared.size(), MEMORY_SIZE);
        assert_eq!(shared.remote_permissions(), MemoryPermissions::READ);
        assert!(session.request_load(0).unwrap_err().contains("system.font"));
        assert_eq!(account.used_bytes(), MEMORY_SIZE as u64);
    }

    #[test]
    fn validates_sfnt_table_bounds() {
        let mut font = vec![0; 32];
        font[..4].copy_from_slice(&[0, 1, 0, 0]);
        font[4..6].copy_from_slice(&1_u16.to_be_bytes());
        font[20..24].copy_from_slice(&28_u32.to_be_bytes());
        font[24..28].copy_from_slice(&4_u32.to_be_bytes());
        assert_eq!((validate_sfnt(&font)).unwrap(), ());
        font[24..28].copy_from_slice(&5_u32.to_be_bytes());
        assert!(matches!(
            validate_sfnt(&font),
            Err("SFNT table exceeds file bounds")
        ));
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
        let account = account();
        let session = SharedFontSession::new(&account, Some(path.clone())).unwrap();
        let shared = session.shared_memory();
        let first = session.request_load(0).unwrap();
        let second = session.clone().request_load(5).unwrap();
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
        let nixe_memory::Access::Ready(payload) = shared.read(second.offset as usize, 32).unwrap()
        else {
            panic!("private font unexpectedly deferred")
        };
        assert_eq!(payload.as_ref(), font);
        std::fs::remove_file(path).unwrap();
        assert_eq!(session.clone().request_load(0).unwrap(), first);
        assert_eq!(session.request_load(1).unwrap(), first);
        assert_eq!(account.used_bytes(), MEMORY_SIZE as u64);
    }

    #[test]
    fn bundled_dejavu_sans_is_published_unchanged_for_every_supported_font_type() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets/DejaVuSans.ttf");
        let bytes = std::fs::read(&path).unwrap();
        let account = account();
        let session = SharedFontSession::new(&account, Some(path)).unwrap();
        let region = session.request_load(0).unwrap();
        assert_eq!(region.size as usize, bytes.len());
        for font in 1..FONT_TYPE_COUNT as u32 {
            assert_eq!(session.request_load(font).unwrap(), region);
        }
        let shared = session.shared_memory();
        let nixe_memory::Access::Ready(payload) =
            shared.read(region.offset as usize, bytes.len()).unwrap()
        else {
            panic!("private font unexpectedly deferred")
        };
        assert_eq!(payload.as_ref(), bytes);
    }
}
