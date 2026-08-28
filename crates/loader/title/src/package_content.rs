//! Container-independent access to canonical package metadata and Control content.

use nixe_loader_content::{
    CnmtContentMeta, CnmtContentType, CnmtExtendedHeader, CnmtLoader, Hfs0Archive, NacpLanguage,
    NacpLoader, NcaContentType, NcaKeyProvider, NcaKeySet, NcaLoader, NcaSectionType, NspArchive,
    NszArchive, Pfs0Loader, RomFsLoader, XczPartition,
};
use nixe_loader_storage::{FormatLoader, LoadError, StorageRef};

use crate::{ContentType, PackageFormat, PackageMetadata};
use crate::{ControlIcon, ControlMetadata};

pub(crate) trait PackageContent {
    const FORMAT: PackageFormat;

    fn entry_count(&self) -> usize;
    fn entry_name(&self, index: usize) -> &str;
    fn entry_size(&self, index: usize) -> u64;
    fn open_entry_at(&self, index: usize) -> Result<StorageRef, LoadError>;
}

impl PackageContent for NspArchive {
    const FORMAT: PackageFormat = PackageFormat::Nsp;

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
    const FORMAT: PackageFormat = PackageFormat::Nsz;

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
    const FORMAT: PackageFormat = PackageFormat::Xcz;

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
    const FORMAT: PackageFormat = PackageFormat::Xci;

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
        // Opening canonical launch content must remain lazy. Whole-entry HFS0
        // hashing belongs to the explicit archive-verification path; the NCA
        // loader validates the metadata and sections actually consumed.
        self.open_entry(&self.entries()[index])
    }
}

pub(crate) fn load_canonical_content_metas<C: PackageContent + ?Sized>(
    archive: &C,
    keys: Option<&dyn NcaKeyProvider>,
) -> Result<Vec<CnmtContentMeta>, LoadError> {
    let mut metadata = Vec::new();
    let mut found_entry = false;
    for index in (0..archive.entry_count()).filter(|index| {
        archive
            .entry_name(*index)
            .to_ascii_lowercase()
            .ends_with(".cnmt.nca")
    }) {
        found_entry = true;
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
    if !found_entry {
        return Err(LoadError::invalid(
            "CNMT",
            "package contains no .cnmt.nca entries",
        ));
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
    let mut controls = content_meta
        .contents
        .iter()
        .filter(|content| content.content_type == CnmtContentType::Control);
    let Some(content) = controls.next() else {
        return Ok(None);
    };
    if controls.next().is_some() {
        return Err(LoadError::invalid(
            "Control NCA",
            format!(
                "canonical CNMT contains {} Control records; expected at most one",
                2 + controls.count()
            ),
        ));
    }

    let expected_name = format!("{}.nca", hex(&content.content_id));
    let entry_index = match unique_entry_index(archive, &expected_name) {
        UniqueEntry::One(index) => index,
        UniqueEntry::Missing => {
            return Err(LoadError::invalid(
                "Control NCA",
                format!("canonical content {expected_name} is missing from the package"),
            ));
        }
        UniqueEntry::Ambiguous => {
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

pub(crate) fn open_canonical_content(
    package: &PackageMetadata,
    content: &nixe_loader_content::CnmtContentInfo,
    format: PackageFormat,
    keys: Option<&dyn NcaKeyProvider>,
) -> Result<nixe_loader_content::NcaArchive, LoadError> {
    let expected_name = format!("{}.nca", hex(&content.content_id));
    let storage = match format {
        PackageFormat::Nsp => {
            let archive = nixe_loader_content::NspLoader::load(package.source.clone())?;
            open_canonical_entry(&archive, &expected_name, content.size)?
        }
        PackageFormat::Nsz => {
            let archive = nixe_loader_content::NszLoader::load(package.source.clone())?;
            open_canonical_entry(&archive, &expected_name, content.size)?
        }
        PackageFormat::Xci => {
            let archive = nixe_loader_content::XciLoader::load(package.source.clone())?;
            open_canonical_entry(
                archive.secure_partition()?.archive(),
                &expected_name,
                content.size,
            )?
        }
        PackageFormat::Xcz => {
            let archive = nixe_loader_content::XczLoader::load(package.source.clone())?;
            open_canonical_entry(archive.secure_partition()?, &expected_name, content.size)?
        }
    };
    let nca = match keys {
        Some(keys) => NcaLoader::load_with_key_provider(storage, keys)?,
        None => NcaLoader::load(storage)?,
    };
    let expected_type = match canonical_nca_content_type(package.content_type, content.content_type)
    {
        Some(content_type) => content_type,
        None => {
            return Err(LoadError::invalid(
                "canonical package content",
                format!(
                    "unsupported canonical content type {}",
                    content.content_type
                ),
            ));
        }
    };
    if nca.header().content_type() != expected_type {
        return Err(LoadError::invalid(
            "canonical package content",
            format!(
                "{expected_name} is {:?}, expected {expected_type:?}",
                nca.header().content_type()
            ),
        ));
    }
    let expected_title_id = match package.content_type {
        ContentType::Patch => package.application_id.get(),
        ContentType::Application | ContentType::AddOnContent | ContentType::Delta => {
            package.title_id.get()
        }
    };
    if nca.header().title_id() != expected_title_id {
        return Err(LoadError::invalid(
            "canonical package content",
            format!(
                "{expected_name} has title ID {:016X}, expected {expected_title_id:016X}",
                nca.header().title_id()
            ),
        ));
    }
    Ok(nca)
}

fn canonical_nca_content_type(
    package_type: ContentType,
    content_type: CnmtContentType,
) -> Option<NcaContentType> {
    // Add-on CNMT records call their payload Data even though the corresponding
    // NCA header calls it PublicData. hacPack enforces the same title/NCA pairing:
    // https://github.com/DarkMatterCore/hacPack/blob/e506cb58b7843d86df7518156debd28f3b575638/main.c#L572-L575
    match (package_type, content_type) {
        (ContentType::AddOnContent, CnmtContentType::Data) => Some(NcaContentType::PublicData),
        (_, CnmtContentType::Program) => Some(NcaContentType::Program),
        (_, CnmtContentType::Data) => Some(NcaContentType::Data),
        (_, CnmtContentType::Control) => Some(NcaContentType::Control),
        (_, CnmtContentType::Meta) => Some(NcaContentType::Meta),
        (_, CnmtContentType::HtmlDocument | CnmtContentType::LegalInformation) => {
            Some(NcaContentType::Manual)
        }
        (_, CnmtContentType::DeltaFragment | CnmtContentType::Unknown(_)) => None,
    }
}

fn open_canonical_entry<C: PackageContent + ?Sized>(
    archive: &C,
    expected_name: &str,
    expected_size: u64,
) -> Result<StorageRef, LoadError> {
    let index = match unique_entry_index(archive, expected_name) {
        UniqueEntry::One(index) => index,
        UniqueEntry::Missing => {
            return Err(LoadError::invalid(
                "canonical package content",
                format!("{expected_name} is missing"),
            ));
        }
        UniqueEntry::Ambiguous => {
            return Err(LoadError::invalid(
                "canonical package content",
                format!("{expected_name} is duplicated or case-ambiguous"),
            ));
        }
    };
    let actual_size = archive.entry_size(index);
    if actual_size != expected_size {
        return Err(LoadError::invalid(
            "canonical package content",
            format!("{expected_name} has size {actual_size}, expected {expected_size}"),
        ));
    }
    archive.open_entry_at(index)
}

fn unique_entry_index<C: PackageContent + ?Sized>(archive: &C, name: &str) -> UniqueEntry {
    let mut matches = (0..archive.entry_count())
        .filter(|index| archive.entry_name(*index).eq_ignore_ascii_case(name));
    let first = matches.next();
    if matches.next().is_some() {
        UniqueEntry::Ambiguous
    } else if let Some(index) = first {
        UniqueEntry::One(index)
    } else {
        UniqueEntry::Missing
    }
}

enum UniqueEntry {
    Missing,
    One(usize),
    Ambiguous,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_on_data_maps_strictly_to_public_data_nca() {
        assert_eq!(
            canonical_nca_content_type(ContentType::AddOnContent, CnmtContentType::Data),
            Some(NcaContentType::PublicData)
        );
        for package_type in [
            ContentType::Application,
            ContentType::Patch,
            ContentType::Delta,
        ] {
            assert_eq!(
                canonical_nca_content_type(package_type, CnmtContentType::Data),
                Some(NcaContentType::Data)
            );
        }
    }
}
