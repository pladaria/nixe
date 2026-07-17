use std::collections::HashMap;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs;
use std::path::{Path, PathBuf};

use crate::crypto::decrypt_ecb_blocks;

/// Selects one of the three NCA key-area encryption-key families.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum KeyAreaKeyIndex {
    Application = 0,
    Ocean = 1,
    System = 2,
}

impl KeyAreaKeyIndex {
    pub(crate) fn from_raw(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Application),
            1 => Some(Self::Ocean),
            2 => Some(Self::System),
            _ => None,
        }
    }

    pub(crate) const fn key_name(self) -> &'static str {
        match self {
            Self::Application => "key_area_key_application",
            Self::Ocean => "key_area_key_ocean",
            Self::System => "key_area_key_system",
        }
    }
}

/// Supplies only the key material required to open an NCA.
///
/// Implementations return copies so an archive never retains a reference to a
/// mutable global key store. Key material must be provided by the caller and is
/// never bundled or downloaded by this crate.
pub trait NcaKeyProvider {
    fn header_key(&self) -> Option<[u8; 32]>;

    fn key_area_key(&self, generation: u8, index: KeyAreaKeyIndex) -> Option<[u8; 16]>;

    /// Returns a decrypted title key for the requested rights ID.
    fn title_key(&self, rights_id: &[u8; 16], generation: u8) -> Option<[u8; 16]>;
}

/// User-supplied keys parsed from the conventional prod.keys and title.keys
/// text formats.
pub struct NcaKeySet {
    header_key: Option<[u8; 32]>,
    key_area_keys: HashMap<(u8, KeyAreaKeyIndex), [u8; 16]>,
    title_keks: HashMap<u8, [u8; 16]>,
    encrypted_title_keys: HashMap<[u8; 16], [u8; 16]>,
}

impl NcaKeySet {
    /// Reads key files from paths selected by the caller.
    pub fn from_files(
        prod_keys: impl AsRef<Path>,
        title_keys: Option<&Path>,
    ) -> Result<Self, KeySetError> {
        let prod_path = prod_keys.as_ref();
        let prod_text = fs::read_to_string(prod_path).map_err(|source| KeySetError::Io {
            path: prod_path.to_path_buf(),
            source,
        })?;

        let title_text = title_keys
            .map(|path| {
                fs::read_to_string(path).map_err(|source| KeySetError::Io {
                    path: path.to_path_buf(),
                    source,
                })
            })
            .transpose()?;

        Self::from_text(&prod_text, title_text.as_deref())
    }

    /// Parses caller-owned key text without retaining the source strings.
    pub fn from_text(prod_keys: &str, title_keys: Option<&str>) -> Result<Self, KeySetError> {
        let mut key_set = Self {
            header_key: None,
            key_area_keys: HashMap::new(),
            title_keks: HashMap::new(),
            encrypted_title_keys: HashMap::new(),
        };

        for (line_index, line) in prod_keys.lines().enumerate() {
            let Some((name, value)) = assignment(line) else {
                continue;
            };
            let name = name.to_ascii_lowercase();

            if name == "header_key" {
                key_set.header_key =
                    Some(parse_hex::<32>(value, "prod.keys", line_index + 1, &name)?);
                continue;
            }

            let key_area_families = [
                ("key_area_key_application_", KeyAreaKeyIndex::Application),
                ("key_area_key_ocean_", KeyAreaKeyIndex::Ocean),
                ("key_area_key_system_", KeyAreaKeyIndex::System),
            ];
            if let Some((generation, family)) = key_area_families
                .iter()
                .find_map(|(prefix, family)| parse_generation(&name, prefix).map(|g| (g, *family)))
            {
                let key = parse_hex::<16>(value, "prod.keys", line_index + 1, &name)?;
                key_set.key_area_keys.insert((generation, family), key);
                continue;
            }

            if let Some(generation) = parse_generation(&name, "titlekek_") {
                let key = parse_hex::<16>(value, "prod.keys", line_index + 1, &name)?;
                key_set.title_keks.insert(generation, key);
            }
        }

        if let Some(title_keys) = title_keys {
            for (line_index, line) in title_keys.lines().enumerate() {
                let Some((rights_id, encrypted_key)) = assignment(line) else {
                    continue;
                };
                let rights_id =
                    parse_hex::<16>(rights_id, "title.keys", line_index + 1, "rights ID")?;
                let encrypted_key = parse_hex::<16>(
                    encrypted_key,
                    "title.keys",
                    line_index + 1,
                    "encrypted title key",
                )?;
                key_set
                    .encrypted_title_keys
                    .insert(rights_id, encrypted_key);
            }
        }

        Ok(key_set)
    }

    pub fn has_header_key(&self) -> bool {
        self.header_key.is_some()
    }

    pub fn key_area_key_count(&self) -> usize {
        self.key_area_keys.len()
    }

    pub fn title_key_count(&self) -> usize {
        self.encrypted_title_keys.len()
    }

    /// Adds an encrypted title key obtained from caller-owned metadata such as
    /// an NSP ticket. The matching titlekek is still resolved from prod.keys.
    pub fn insert_encrypted_title_key(
        &mut self,
        rights_id: [u8; 16],
        encrypted_title_key: [u8; 16],
    ) {
        self.encrypted_title_keys
            .insert(rights_id, encrypted_title_key);
    }
}

impl NcaKeyProvider for NcaKeySet {
    fn header_key(&self) -> Option<[u8; 32]> {
        self.header_key
    }

    fn key_area_key(&self, generation: u8, index: KeyAreaKeyIndex) -> Option<[u8; 16]> {
        self.key_area_keys.get(&(generation, index)).copied()
    }

    fn title_key(&self, rights_id: &[u8; 16], generation: u8) -> Option<[u8; 16]> {
        let mut title_key = self.encrypted_title_keys.get(rights_id).copied()?;
        let title_kek = self.title_keks.get(&generation)?;
        decrypt_ecb_blocks(title_kek, &mut title_key);
        Some(title_key)
    }
}

impl std::fmt::Debug for NcaKeySet {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NcaKeySet")
            .field("has_header_key", &self.header_key.is_some())
            .field("key_area_key_count", &self.key_area_keys.len())
            .field("title_kek_count", &self.title_keks.len())
            .field("title_key_count", &self.encrypted_title_keys.len())
            .finish()
    }
}

impl Drop for NcaKeySet {
    fn drop(&mut self) {
        if let Some(key) = &mut self.header_key {
            key.fill(0);
        }
        for key in self.key_area_keys.values_mut() {
            key.fill(0);
        }
        for key in self.title_keks.values_mut() {
            key.fill(0);
        }
        for key in self.encrypted_title_keys.values_mut() {
            key.fill(0);
        }
    }
}

fn assignment(line: &str) -> Option<(&str, &str)> {
    let line = line
        .split_once('#')
        .map_or(line, |(before_comment, _)| before_comment)
        .trim();
    if line.is_empty() {
        return None;
    }
    let (name, value) = line.split_once('=')?;
    let name = name.trim();
    let value = value.trim();
    (!name.is_empty() && !value.is_empty()).then_some((name, value))
}

fn parse_generation(name: &str, prefix: &str) -> Option<u8> {
    let suffix = name.strip_prefix(prefix)?;
    if suffix.len() != 2 {
        return None;
    }
    u8::from_str_radix(suffix, 16).ok()
}

fn parse_hex<const N: usize>(
    value: &str,
    source: &'static str,
    line: usize,
    field: &str,
) -> Result<[u8; N], KeySetError> {
    if value.len() != N * 2 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(KeySetError::InvalidValue {
            source,
            line,
            field: field.to_owned(),
            expected_bytes: N,
        });
    }

    let mut result = [0_u8; N];
    for (index, byte) in result.iter_mut().enumerate() {
        let start = index * 2;
        *byte = u8::from_str_radix(&value[start..start + 2], 16).map_err(|_| {
            KeySetError::InvalidValue {
                source,
                line,
                field: field.to_owned(),
                expected_bytes: N,
            }
        })?;
    }
    Ok(result)
}

#[derive(Debug)]
pub enum KeySetError {
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    InvalidValue {
        source: &'static str,
        line: usize,
        field: String,
        expected_bytes: usize,
    },
}

impl Display for KeySetError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { path, source } => {
                write!(
                    formatter,
                    "failed to read key file {}: {source}",
                    path.display()
                )
            }
            Self::InvalidValue {
                source,
                line,
                field,
                expected_bytes,
            } => write!(
                formatter,
                "invalid {field} in {source} at line {line}: expected {expected_bytes} bytes of hexadecimal data"
            ),
        }
    }
}

impl Error for KeySetError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::InvalidValue { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_only_nca_keys_and_redacts_debug_output() {
        let keys = NcaKeySet::from_text(
            "header_key = 000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f\n\
             key_area_key_application_00 = 00112233445566778899aabbccddeeff\n\
             irrelevant_key = 0123\n",
            None,
        )
        .unwrap();

        assert!(keys.has_header_key());
        assert_eq!(keys.key_area_key_count(), 1);
        let debug = format!("{keys:?}");
        assert!(!debug.contains("00010203"));
        assert!(!debug.contains("00112233"));
    }

    #[test]
    fn rejects_malformed_recognized_key() {
        let error = NcaKeySet::from_text("header_key = 00", None).unwrap_err();
        assert!(matches!(error, KeySetError::InvalidValue { .. }));
    }

    #[test]
    fn ignores_comments_and_blank_lines() {
        let keys = NcaKeySet::from_text(
            "# local keys\nheader_key = aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa # comment\n",
            None,
        )
        .unwrap();
        assert!(keys.has_header_key());
    }
}
