//! Machine-managed window geometry stored beside the selected `nixe.toml`.

use std::fs;
use std::io;
use std::path::Path;

pub(super) const FILE_NAME: &str = "nixe.cfg";

// Bump this when the binary layout changes. Unknown versions are discarded.
const VERSION: u32 = 1;
const FILE_BYTES: usize = 21;
const MAX_DIMENSION: u32 = 32_768;

/// Physical-pixel client dimensions and, where available, desktop position.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowState {
    pub width: u32,
    pub height: u32,
    pub position: Option<(i32, i32)>,
}

impl WindowState {
    /// Reads the current format. A missing file uses defaults; invalid data is an error.
    pub fn load(path: &Path) -> io::Result<Option<Self>> {
        let bytes = match fs::read(path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error),
        };
        if bytes.len() != FILE_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unexpected window configuration length",
            ));
        }
        let version = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        if version != VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported window configuration version {version}; expected {VERSION}"),
            ));
        }
        let width = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let height = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        if !valid_dimensions(width, height) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid window dimensions",
            ));
        }
        let position = match bytes[20] {
            0 => None,
            1 => Some((
                i32::from_le_bytes(bytes[12..16].try_into().unwrap()),
                i32::from_le_bytes(bytes[16..20].try_into().unwrap()),
            )),
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "invalid window position marker",
                ));
            }
        };
        Ok(Some(Self {
            width,
            height,
            position,
        }))
    }

    /// Saves a fixed-width, little-endian record with a version header.
    pub fn save(self, path: &Path) -> io::Result<()> {
        if !valid_dimensions(self.width, self.height) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "invalid window dimensions",
            ));
        }
        let mut bytes = [0; FILE_BYTES];
        bytes[0..4].copy_from_slice(&VERSION.to_le_bytes());
        bytes[4..8].copy_from_slice(&self.width.to_le_bytes());
        bytes[8..12].copy_from_slice(&self.height.to_le_bytes());
        if let Some((x, y)) = self.position {
            bytes[12..16].copy_from_slice(&x.to_le_bytes());
            bytes[16..20].copy_from_slice(&y.to_le_bytes());
            bytes[20] = 1;
        }
        fs::write(path, bytes)
    }
}

const fn valid_dimensions(width: u32, height: u32) -> bool {
    width > 0 && width <= MAX_DIMENSION && height > 0 && height <= MAX_DIMENSION
}
