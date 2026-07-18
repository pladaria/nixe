use swiitx_loader_content::{
    CnmtContentMeta, CnmtLoader, NcaContentType, NcaKeyProvider, NcaKeySet, NcaLoader,
    NcaSectionType, NspArchive, Pfs0Loader,
};
use swiitx_loader_storage::{FormatLoader, LoadError};

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
