//! Container-independent access to canonical package metadata and Control content.

use swiitx_loader_content::{
    CnmtContentMeta, CnmtContentType, CnmtExtendedHeader, CnmtLoader, Hfs0Archive, NacpLanguage,
    NacpLoader, NcaContentType, NcaKeyProvider, NcaKeySet, NcaLoader, NcaSectionType, NspArchive,
    NszArchive, Pfs0Loader, RomFsLoader, XczPartition,
};
use swiitx_loader_storage::{FormatLoader, LoadError, StorageRef};

use crate::{ControlIcon, ControlMetadata};

pub(crate) trait PackageContent {
    fn entry_count(&self) -> usize;
    fn entry_name(&self, index: usize) -> &str;
    fn entry_size(&self, index: usize) -> u64;
    fn open_entry_at(&self, index: usize) -> Result<StorageRef, LoadError>;
}

impl PackageContent for NspArchive {
    fn entry_count(&self) -> usize {
        self.entries().len()
    }

    fn entry_name(&self, index: usize) -> &str {
        self.entries()[index].name()
    }

    fn entry_size(&self, index: usize) -> u64 {
        self.entries()[index].size()
    }

    fn open_entry_at(&self, index: usize) -> Result<StorageRef, LoadError> {
        self.open_entry(&self.entries()[index])
    }
}

impl PackageContent for NszArchive {
    fn entry_count(&self) -> usize {
        self.entries().len()
    }

    fn entry_name(&self, index: usize) -> &str {
        self.entries()[index].logical_name()
    }

    fn entry_size(&self, index: usize) -> u64 {
        self.entries()[index].logical_size()
    }

    fn open_entry_at(&self, index: usize) -> Result<StorageRef, LoadError> {
        self.open_entry(&self.entries()[index])
    }
}

impl PackageContent for XczPartition {
    fn entry_count(&self) -> usize {
        self.entries().len()
    }

    fn entry_name(&self, index: usize) -> &str {
        self.entries()[index].logical_name()
    }

    fn entry_size(&self, index: usize) -> u64 {
        self.entries()[index].logical_size()
    }

    fn open_entry_at(&self, index: usize) -> Result<StorageRef, LoadError> {
        self.open_entry(&self.entries()[index])
    }
}

impl PackageContent for Hfs0Archive {
    fn entry_count(&self) -> usize {
        self.entries().len()
    }

    fn entry_name(&self, index: usize) -> &str {
        self.entries()[index].name()
    }

    fn entry_size(&self, index: usize) -> u64 {
        self.entries()[index].size()
    }

    fn open_entry_at(&self, index: usize) -> Result<StorageRef, LoadError> {
        let entry = &self.entries()[index];
        let integrity = self.validate_entry(entry)?;
        if !integrity.is_valid() {
            return Err(LoadError::invalid(
                "HFS0 package content",
                format!(
                    "entry {:?} failed its advertised hash validation",
                    entry.name()
                ),
            ));
        }
        self.open_entry(entry)
    }
}

pub(crate) fn load_canonical_content_meta<C: PackageContent + ?Sized>(
    archive: &C,
    keys: Option<&dyn NcaKeyProvider>,
) -> Result<CnmtContentMeta, LoadError> {
    let metadata = load_canonical_content_metas(archive, keys)?;
    if metadata.len() != 1 {
        return Err(LoadError::invalid(
            "CNMT",
            format!(
                "package contains {} .cnmt.nca entries; expected exactly one",
                metadata.len()
            ),
        ));
    }
    Ok(metadata
        .into_iter()
        .next()
        .expect("validated metadata count"))
}

pub(crate) fn load_canonical_content_metas<C: PackageContent + ?Sized>(
    archive: &C,
    keys: Option<&dyn NcaKeyProvider>,
) -> Result<Vec<CnmtContentMeta>, LoadError> {
    let meta_entries: Vec<_> = (0..archive.entry_count())
        .filter(|index| {
            archive
                .entry_name(*index)
                .to_ascii_lowercase()
                .ends_with(".cnmt.nca")
        })
        .collect();
    if meta_entries.is_empty() {
        return Err(LoadError::invalid(
            "CNMT",
            "package contains no .cnmt.nca entries",
        ));
    }
    let mut metadata = Vec::with_capacity(meta_entries.len());
    for index in meta_entries {
        let content_meta = load_content_meta_entry(archive, index, keys).map_err(|error| {
            LoadError::invalid(
                "CNMT",
                format!("entry {:?}: {error}", archive.entry_name(index)),
            )
        })?;
        if metadata.iter().any(|existing| existing == &content_meta) {
            continue;
        }
        if metadata.iter().any(|existing: &CnmtContentMeta| {
            existing.title_id == content_meta.title_id
                && existing.version == content_meta.version
                && existing.content_meta_type == content_meta.content_meta_type
        }) {
            return Err(LoadError::invalid(
                "CNMT",
                format!(
                    "conflicting metadata records claim title {:016X}, type {}, version {}",
                    content_meta.title_id,
                    content_meta.content_meta_type,
                    content_meta.version.raw()
                ),
            ));
        }
        metadata.push(content_meta);
    }
    Ok(metadata)
}

fn load_content_meta_entry<C: PackageContent + ?Sized>(
    archive: &C,
    entry_index: usize,
    keys: Option<&dyn NcaKeyProvider>,
) -> Result<CnmtContentMeta, LoadError> {
    let storage = archive.open_entry_at(entry_index)?;
    let nca = match keys {
        Some(keys) => NcaLoader::load_with_key_provider(storage, keys)?,
        None => NcaLoader::load(storage)?,
    };
    if nca.header().content_type() != NcaContentType::Meta {
        return Err(LoadError::invalid(
            "CNMT",
            "the .cnmt.nca entry is not a meta-content NCA",
        ));
    }

    let pfs0_sections: Vec<_> = nca
        .sections()
        .iter()
        .filter(|section| section.section_type() == NcaSectionType::Pfs0)
        .collect();
    if pfs0_sections.len() != 1 {
        return Err(LoadError::invalid(
            "CNMT",
            format!(
                "meta NCA contains {} PFS0 sections; expected exactly one",
                pfs0_sections.len()
            ),
        ));
    }

    let section = pfs0_sections[0];
    let integrity = section.validate_integrity()?;
    if !integrity.is_valid() {
        return Err(LoadError::invalid(
            "CNMT",
            format!(
                "meta NCA PFS0 section {} failed integrity validation: {:?}",
                section.index(),
                integrity.checks()
            ),
        ));
    }

    let pfs0 = Pfs0Loader::load(section.payload_storage()?)?;
    let cnmt_entries: Vec<_> = pfs0
        .entries()
        .iter()
        .filter(|entry| entry.name().to_ascii_lowercase().ends_with(".cnmt"))
        .collect();
    if cnmt_entries.len() != 1 {
        return Err(LoadError::invalid(
            "CNMT",
            format!(
                "meta NCA PFS0 contains {} .cnmt entries; expected exactly one",
                cnmt_entries.len()
            ),
        ));
    }

    CnmtLoader::load(pfs0.open_entry(cnmt_entries[0])?)
}

pub(crate) fn load_control_metadata<C: PackageContent + ?Sized>(
    archive: &C,
    content_meta: &CnmtContentMeta,
    keys: Option<&dyn NcaKeyProvider>,
) -> Result<Option<ControlMetadata>, LoadError> {
    let controls: Vec<_> = content_meta
        .contents
        .iter()
        .filter(|content| content.content_type == CnmtContentType::Control)
        .collect();
    let content = match controls.as_slice() {
        [] => return Ok(None),
        [content] => *content,
        _ => {
            return Err(LoadError::invalid(
                "Control NCA",
                format!(
                    "canonical CNMT contains {} Control records; expected at most one",
                    controls.len()
                ),
            ));
        }
    };

    let expected_name = format!("{}.nca", hex(&content.content_id));
    let entries: Vec<_> = (0..archive.entry_count())
        .filter(|index| {
            archive
                .entry_name(*index)
                .eq_ignore_ascii_case(&expected_name)
        })
        .collect();
    let entry_index = match entries.as_slice() {
        [index] => *index,
        [] => {
            return Err(LoadError::invalid(
                "Control NCA",
                format!("canonical content {expected_name} is missing from the package"),
            ));
        }
        _ => {
            return Err(LoadError::invalid(
                "Control NCA",
                format!("multiple package entries match {expected_name}"),
            ));
        }
    };
    if archive.entry_size(entry_index) != content.size {
        return Err(LoadError::invalid(
            "Control NCA",
            format!(
                "CNMT declares {} bytes for {expected_name}, but the entry has {}",
                content.size,
                archive.entry_size(entry_index)
            ),
        ));
    }

    let storage = archive.open_entry_at(entry_index)?;
    let nca = match keys {
        Some(keys) => NcaLoader::load_with_key_provider(storage, keys)?,
        None => NcaLoader::load(storage)?,
    };
    if nca.header().content_type() != NcaContentType::Control {
        return Err(LoadError::invalid(
            "Control NCA",
            "canonical Control content is not a Control NCA",
        ));
    }
    let expected_title_id = match &content_meta.extended_header {
        CnmtExtendedHeader::Patch { application_id, .. } => *application_id,
        _ => content_meta.title_id,
    };
    if nca.header().title_id() != expected_title_id {
        return Err(LoadError::invalid(
            "Control NCA",
            format!(
                "title ID {:016X} does not match expected title ID {:016X}",
                nca.header().title_id(),
                expected_title_id
            ),
        ));
    }

    let sections: Vec<_> = nca
        .sections()
        .iter()
        .filter(|section| section.section_type() == NcaSectionType::RomFs)
        .collect();
    let section = match sections.as_slice() {
        [section] => *section,
        _ => {
            return Err(LoadError::invalid(
                "Control NCA",
                format!(
                    "contains {} usable RomFS sections; expected exactly one",
                    sections.len()
                ),
            ));
        }
    };
    let integrity = section.validate_integrity()?;
    if !integrity.is_valid() {
        return Err(LoadError::invalid(
            "Control NCA",
            format!(
                "RomFS section failed integrity validation: {:?}",
                integrity.checks()
            ),
        ));
    }

    let romfs = RomFsLoader::load(section.payload_storage()?)?;
    let nacp_storage = romfs
        .open("/control.nacp")?
        .ok_or_else(|| LoadError::invalid("Control NCA", "RomFS does not contain control.nacp"))?;
    let nacp = NacpLoader::load(nacp_storage)?;

    let mut icons = Vec::new();
    for language in NacpLanguage::ALL {
        let filename = format!("icon_{}.dat", language.icon_suffix());
        let path = format!("/{filename}");
        if let Some(storage) = romfs.open(&path)? {
            icons.push(ControlIcon::load(language, filename, storage)?);
        }
    }

    Ok(Some(ControlMetadata::new(nacp, content, icons)))
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut result = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        result.push(char::from(DIGITS[usize::from(byte >> 4)]));
        result.push(char::from(DIGITS[usize::from(byte & 0x0F)]));
    }
    result
}

pub(crate) fn import_ticket_keys<C: PackageContent + ?Sized>(
    archive: &C,
    keys: &mut NcaKeySet,
) -> Vec<String> {
    const ENCRYPTED_TITLE_KEY_OFFSET: u64 = 0x180;
    const RIGHTS_ID_OFFSET: u64 = 0x2A0;
    const REQUIRED_TICKET_SIZE: u64 = RIGHTS_ID_OFFSET + 16;

    let mut warnings = Vec::new();
    for index in (0..archive.entry_count()).filter(|index| {
        archive
            .entry_name(*index)
            .to_ascii_lowercase()
            .ends_with(".tik")
    }) {
        let result = (|| {
            if archive.entry_size(index) < REQUIRED_TICKET_SIZE {
                return Err(LoadError::invalid("ticket", "ticket is truncated"));
            }
            let storage = archive.open_entry_at(index)?;
            let mut encrypted_title_key = [0_u8; 16];
            let mut rights_id = [0_u8; 16];
            storage.read_at(ENCRYPTED_TITLE_KEY_OFFSET, &mut encrypted_title_key)?;
            storage.read_at(RIGHTS_ID_OFFSET, &mut rights_id)?;
            keys.insert_encrypted_title_key(rights_id, encrypted_title_key);
            Ok::<_, LoadError>(())
        })();
        if let Err(error) = result {
            warnings.push(format!("{}: {error}", archive.entry_name(index)));
        }
    }
    warnings
}
