//! Text and Nintendo symbol fonts exposed by `pl:u` in read-only shared memory.

use std::path::PathBuf;
use std::sync::Arc;

use nixe_memory::{AllocationRange, MemoryAccount, MemoryPermissions};
use nixe_runtime::SharedMemoryObject;

// https://github.com/switchbrew/libnx/blob/master/nx/source/services/pl.c
// https://switchbrew.org/wiki/Glue_services#SharedFontType
const MEMORY_SIZE: usize = 0x110_0000;
const FONT_TYPE_COUNT: usize = 7;
const NINTENDO_EXTENDED: usize = 5;
// Eden uses this same open-source replacement for both Nintendo extension
// fonts. Its original copyright, license and provenance accompany the asset.
// https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/core/file_sys/system_archive/shared_font.cpp
const SYMBOL_FONT: &[u8] = include_bytes!("../../../assets/YuzuOSSExtension.ttf");

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct FontRegion {
    pub offset: u32,
    pub size: u32,
}

#[derive(Debug)]
struct FontState {
    memory: SharedMemoryObject,
    regions: [Result<FontRegion, String>; FONT_TYPE_COUNT],
}

/// One system allocation, initialized before any client can map it.
#[derive(Clone, Debug)]
pub struct SharedFontSession(Arc<FontState>);

impl SharedFontSession {
    /// Prepares the configured text replacement and bundled symbol fonts once.
    /// Invalid or absent font data is reported when its type is requested;
    /// physical allocation failures stop construction.
    pub fn new(account: &MemoryAccount, font: Option<PathBuf>) -> Result<Self, String> {
        let allocation = account
            .allocate(MEMORY_SIZE as u64)
            .map_err(|e| e.to_string())?;
        // Prepare once, as the platform service constructor does in Eden:
        // https://git.eden-emu.dev/eden-emu/eden/src/branch/master/src/core/hle/service/ns/platform_service_manager.cpp
        let mut payload = Vec::new();
        let text_region = match font {
            None => Err(
                "system fonts require a TTF file; configure system.font with its path".to_owned(),
            ),
            Some(path) => std::fs::read(&path)
                .map_err(|error| format!("cannot read {}: {error}", path.display()))
                .and_then(|bytes| {
                    append_font(&mut payload, &bytes)
                        .map_err(|error| format!("cannot load {}: {error}", path.display()))
                }),
        };
        let mut regions = std::array::from_fn(|_| text_region.clone());
        for region in &mut regions[NINTENDO_EXTENDED..] {
            *region = append_font(&mut payload, SYMBOL_FONT);
        }
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
        let memory = SharedMemoryObject::new(allocation, MEMORY_SIZE, MemoryPermissions::READ)
            .map_err(|e| e.to_string())?;
        Ok(Self(Arc::new(FontState { memory, regions })))
    }

    pub(crate) fn valid_type(font: u32) -> bool {
        (font as usize) < FONT_TYPE_COUNT
    }

    pub(crate) fn shared_memory(&self) -> SharedMemoryObject {
        self.0.memory.clone()
    }

    pub(crate) fn request_load(&self, font: u32) -> Result<FontRegion, String> {
        self.0.regions[font as usize].clone()
    }
}

fn append_font(payload: &mut Vec<u8>, bytes: &[u8]) -> Result<FontRegion, String> {
    validate_sfnt(bytes).map_err(|error| format!("invalid shared font: {error}"))?;
    let offset = payload.len();
    offset
        .checked_add(bytes.len())
        .and_then(|size| size.checked_add(8))
        .filter(|end| *end <= MEMORY_SIZE)
        .ok_or_else(|| "fonts exceed the 17 MiB shared-memory region".to_owned())?;
    // Decrypted BFTTF header followed by the unchanged TTF payload.
    // https://git.eden-emu.dev/eden-emu/eden/src/branch/master/src/core/hle/service/ns/platform_service_manager.cpp
    payload.extend_from_slice(&0x7f9a_0218_u32.to_be_bytes());
    payload.extend_from_slice(&((bytes.len() as u32) ^ 0x4962_1806).to_le_bytes());
    payload.extend_from_slice(bytes);
    Ok(FontRegion {
        offset: (offset + 8) as u32,
        size: bytes.len() as u32,
    })
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
}
