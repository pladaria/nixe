use std::env;
use std::path::PathBuf;
use std::sync::Arc;

use nixe_loader_content::{
    CnmtLoader, NcaContentType, NcaKeySet, NcaLoader, NcaSectionType, NspLoader, Pfs0Loader,
};
use nixe_loader_storage::{FileStorage, FormatLoader, StorageRef};

#[test]
#[ignore = "requires caller-owned NSP and key files"]
fn opens_and_validates_caller_owned_nsp() {
    let nsp_path = required_path("NIXE_TEST_NSP");
    let prod_keys = required_path("NIXE_PROD_KEYS");
    let title_keys = env::var_os("NIXE_TITLE_KEYS").map(PathBuf::from);
    let mut keys = NcaKeySet::from_files(&prod_keys, title_keys.as_deref()).unwrap();

    let nsp_storage: StorageRef = Arc::new(FileStorage::open(&nsp_path).unwrap());
    let nsp = NspLoader::load(nsp_storage).unwrap();

    for ticket_entry in nsp
        .entries()
        .iter()
        .filter(|entry| entry.name().ends_with(".tik"))
    {
        let ticket = nsp.open_entry(ticket_entry).unwrap();
        assert!(ticket.len().unwrap() >= 0x2B0, "ticket is truncated");
        let mut encrypted_title_key = [0_u8; 16];
        let mut rights_id = [0_u8; 16];
        ticket.read_at(0x180, &mut encrypted_title_key).unwrap();
        ticket.read_at(0x2A0, &mut rights_id).unwrap();
        keys.insert_encrypted_title_key(rights_id, encrypted_title_key);
    }

    let mut nca_count = 0;
    let mut cnmt_count = 0;

    for entry in nsp
        .entries()
        .iter()
        .filter(|entry| entry.name().ends_with(".nca"))
    {
        let storage = nsp.open_entry(entry).unwrap();
        let nca = NcaLoader::load_with_key_provider(storage, &keys).unwrap();
        assert!(
            !nca.sections().is_empty(),
            "{} has no sections",
            entry.name()
        );

        for section in nca.sections() {
            let report = section.validate_integrity().unwrap();
            if !matches!(
                section.section_type(),
                NcaSectionType::Bktr | NcaSectionType::Unknown { .. }
            ) {
                assert!(
                    report.is_valid(),
                    "{} section {} failed integrity: {:?}",
                    entry.name(),
                    section.index(),
                    report.checks()
                );
            }
        }
        if nca.header().content_type() == NcaContentType::Meta {
            let section = nca
                .sections()
                .iter()
                .find(|section| section.section_type() == NcaSectionType::Pfs0)
                .expect("meta NCA has no PFS0 section");
            assert!(section.validate_integrity().unwrap().is_valid());
            let pfs0 = Pfs0Loader::load(section.payload_storage().unwrap()).unwrap();
            let cnmt_entry = pfs0
                .entries()
                .iter()
                .find(|entry| entry.name().ends_with(".cnmt"))
                .expect("meta NCA PFS0 has no binary CNMT");
            let cnmt = CnmtLoader::load(pfs0.open_entry(cnmt_entry).unwrap()).unwrap();
            assert_eq!(cnmt.title_id, nca.header().title_id());
            assert!(!cnmt.contents.is_empty());
            cnmt_count += 1;
        }
        nca_count += 1;
    }

    assert!(nca_count > 0, "the NSP contains no NCA entries");
    assert!(cnmt_count > 0, "the NSP contains no canonical CNMT");
}

fn required_path(name: &str) -> PathBuf {
    env::var_os(name)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("{name} must point to a caller-owned file"))
}
