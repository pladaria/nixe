use swiitx_loader_content::{
    CnmtContentMeta, CnmtContentType, CnmtLoader, NacpLanguage, NacpLoader, NcaContentType,
    NcaKeyProvider, NcaKeySet, NcaLoader, NcaSectionType, NspArchive, Pfs0Loader, RomFsLoader,
};
use swiitx_loader_storage::{FormatLoader, LoadError};

use crate::{ControlIcon, ControlMetadata};

pub(crate) fn load_canonical_content_meta(
    archive: &NspArchive,
    keys: Option<&dyn NcaKeyProvider>,
) -> Result<CnmtContentMeta, LoadError> {
    let meta_entries: Vec<_> = archive
        .entries()
        .iter()
        .filter(|entry| entry.name().to_ascii_lowercase().ends_with(".cnmt.nca"))
        .collect();
    if meta_entries.len() != 1 {
        return Err(LoadError::invalid(
            "CNMT",
            format!(
                "package contains {} .cnmt.nca entries; expected exactly one",
                meta_entries.len()
            ),
        ));
    }

    let storage = archive.open_entry(meta_entries[0])?;
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

pub(crate) fn load_control_metadata(
    archive: &NspArchive,
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
    let entries: Vec<_> = archive
        .entries()
        .iter()
        .filter(|entry| entry.name().eq_ignore_ascii_case(&expected_name))
        .collect();
    let entry = match entries.as_slice() {
        [entry] => *entry,
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
    if entry.size() != content.size {
        return Err(LoadError::invalid(
            "Control NCA",
            format!(
                "CNMT declares {} bytes for {expected_name}, but the entry has {}",
                content.size,
                entry.size()
            ),
        ));
    }

    let storage = archive.open_entry(entry)?;
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
    if nca.header().title_id() != content_meta.title_id {
        return Err(LoadError::invalid(
            "Control NCA",
            format!(
                "title ID {:016X} does not match CNMT title ID {:016X}",
                nca.header().title_id(),
                content_meta.title_id
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

pub(crate) fn import_ticket_keys(archive: &NspArchive, keys: &mut NcaKeySet) -> Vec<String> {
    const ENCRYPTED_TITLE_KEY_OFFSET: u64 = 0x180;
    const RIGHTS_ID_OFFSET: u64 = 0x2A0;
    const REQUIRED_TICKET_SIZE: u64 = RIGHTS_ID_OFFSET + 16;

    let mut warnings = Vec::new();
    for entry in archive
        .entries()
        .iter()
        .filter(|entry| entry.name().to_ascii_lowercase().ends_with(".tik"))
    {
        let result = (|| {
            if entry.size() < REQUIRED_TICKET_SIZE {
                return Err(LoadError::invalid("ticket", "ticket is truncated"));
            }
            let storage = archive.open_entry(entry)?;
            let mut encrypted_title_key = [0_u8; 16];
            let mut rights_id = [0_u8; 16];
            storage.read_at(ENCRYPTED_TITLE_KEY_OFFSET, &mut encrypted_title_key)?;
            storage.read_at(RIGHTS_ID_OFFSET, &mut rights_id)?;
            keys.insert_encrypted_title_key(rights_id, encrypted_title_key);
            Ok::<_, LoadError>(())
        })();
        if let Err(error) = result {
            warnings.push(format!("{}: {error}", entry.name()));
        }
    }
    warnings
}
