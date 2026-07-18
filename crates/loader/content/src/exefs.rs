use swiitx_loader_storage::{FormatLoader, LoadError, StorageRef};

use crate::Pfs0Entry;
use crate::nca::{NcaSection, NcaSectionType};
use crate::pfs0::Pfs0Archive;

/// Loads Executable File System (ExeFS) images.
///
/// Switch ExeFS images use the PFS0 byte-level format, but contain executable
/// modules and process metadata rather than package contents. This loader
/// delegates all structural validation to the shared PFS0 parser while
/// retaining ExeFS-specific error context.
#[derive(Debug)]
pub struct ExeFsLoader;

impl ExeFsLoader {
    /// Loads ExeFS from an already selected NCA section.
    ///
    /// The caller remains responsible for selecting the executable section of
    /// the appropriate Program NCA. This method deliberately does not guess
    /// among multiple PFS0 sections.
    pub fn load_nca_section(section: &NcaSection) -> Result<ExeFsArchive, LoadError> {
        if section.section_type() != NcaSectionType::Pfs0 {
            return Err(LoadError::invalid(
                Self::FORMAT_NAME,
                "NCA section is not a PFS0 section",
            ));
        }

        Self::load(section.payload_storage()?)
    }
}

impl FormatLoader for ExeFsLoader {
    type Output = ExeFsArchive;

    const FORMAT_NAME: &'static str = "ExeFS";

    fn load(storage: StorageRef) -> Result<Self::Output, LoadError> {
        Ok(ExeFsArchive {
            pfs0: Pfs0Archive::parse(storage, Self::FORMAT_NAME)?,
        })
    }
}

/// Parsed, read-only view of an Executable File System image.
///
/// ExeFS is structurally PFS0. This type adds executable-content semantics
/// without duplicating the PFS0 parser or copying entry contents into memory.
#[derive(Debug)]
pub struct ExeFsArchive {
    pfs0: Pfs0Archive,
}

impl ExeFsArchive {
    /// Returns entries in their original PFS0 order.
    pub fn entries(&self) -> &[Pfs0Entry] {
        self.pfs0.entries()
    }

    /// Finds an entry by its exact, case-sensitive name.
    pub fn entry(&self, name: &str) -> Option<&Pfs0Entry> {
        self.pfs0.entry(name)
    }

    /// Opens an entry as an independent bounded storage view.
    pub fn open_entry(&self, entry: &Pfs0Entry) -> Result<StorageRef, LoadError> {
        self.pfs0.open_entry(entry)
    }

    /// Finds and opens an entry by its exact, case-sensitive name.
    pub fn open(&self, name: &str) -> Result<Option<StorageRef>, LoadError> {
        self.pfs0.open(name)
    }

    /// Returns the `main` executable entry when present.
    pub fn main(&self) -> Option<&Pfs0Entry> {
        self.entry("main")
    }

    /// Returns the `main.npdm` process metadata entry when present.
    pub fn main_npdm(&self) -> Option<&Pfs0Entry> {
        self.entry("main.npdm")
    }

    /// Returns the byte offset at which file data begins.
    pub const fn data_offset(&self) -> u64 {
        self.pfs0.data_offset()
    }

    /// Returns the generic PFS0 view backing this ExeFS archive.
    pub const fn as_pfs0(&self) -> &Pfs0Archive {
        &self.pfs0
    }

    /// Converts this semantic ExeFS view back into its generic PFS0 view.
    pub fn into_pfs0(self) -> Pfs0Archive {
        self.pfs0
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use sha2::{Digest, Sha256};
    use swiitx_loader_storage::{Storage, StorageError};

    use crate::{NcaContentType, NcaLoader};

    use super::*;

    #[derive(Debug)]
    struct VecStorage(Vec<u8>);

    impl Storage for VecStorage {
        fn len(&self) -> Result<u64, StorageError> {
            u64::try_from(self.0.len()).map_err(|_| StorageError::OutOfBounds)
        }

        fn read_at(&self, offset: u64, buffer: &mut [u8]) -> Result<(), StorageError> {
            let start = usize::try_from(offset).map_err(|_| StorageError::OutOfBounds)?;
            let end = start
                .checked_add(buffer.len())
                .ok_or(StorageError::OutOfBounds)?;
            let source = self.0.get(start..end).ok_or(StorageError::OutOfBounds)?;
            buffer.copy_from_slice(source);
            Ok(())
        }
    }

    fn storage(bytes: Vec<u8>) -> StorageRef {
        Arc::new(VecStorage(bytes))
    }

    fn build_exefs(files: &[(&str, &[u8])]) -> Vec<u8> {
        let mut strings = Vec::new();
        let mut name_offsets = Vec::new();
        for (name, _) in files {
            name_offsets.push(u32::try_from(strings.len()).unwrap());
            strings.extend_from_slice(name.as_bytes());
            strings.push(0);
        }

        let mut result = Vec::new();
        result.extend_from_slice(b"PFS0");
        result.extend_from_slice(&u32::try_from(files.len()).unwrap().to_le_bytes());
        result.extend_from_slice(&u32::try_from(strings.len()).unwrap().to_le_bytes());
        result.extend_from_slice(&0_u32.to_le_bytes());

        let mut file_offset = 0_u64;
        for ((_, data), name_offset) in files.iter().zip(name_offsets) {
            result.extend_from_slice(&file_offset.to_le_bytes());
            result.extend_from_slice(&u64::try_from(data.len()).unwrap().to_le_bytes());
            result.extend_from_slice(&name_offset.to_le_bytes());
            result.extend_from_slice(&0_u32.to_le_bytes());
            file_offset += u64::try_from(data.len()).unwrap();
        }

        result.extend_from_slice(&strings);
        for (_, data) in files {
            result.extend_from_slice(data);
        }
        result
    }

    fn load_bytes(bytes: Vec<u8>) -> Result<ExeFsArchive, LoadError> {
        ExeFsLoader::load(storage(bytes))
    }

    #[test]
    fn loads_empty_exefs() {
        let archive = load_bytes(build_exefs(&[])).unwrap();

        assert!(archive.entries().is_empty());
        assert_eq!(archive.data_offset(), 0x10);
        assert!(archive.main().is_none());
        assert!(archive.main_npdm().is_none());
    }

    #[test]
    fn preserves_order_and_exposes_exefs_entries() {
        let archive = load_bytes(build_exefs(&[
            ("rtld", b"loader"),
            ("main", b"program"),
            ("main.npdm", b"metadata"),
            ("future.module", b"unknown"),
        ]))
        .unwrap();

        let names: Vec<_> = archive.entries().iter().map(Pfs0Entry::name).collect();
        assert_eq!(names, ["rtld", "main", "main.npdm", "future.module"]);
        assert_eq!(archive.main().unwrap().name(), "main");
        assert_eq!(archive.main_npdm().unwrap().name(), "main.npdm");
        assert!(archive.entry("MAIN").is_none());
        assert!(archive.open("MAIN").unwrap().is_none());
        assert_eq!(archive.as_pfs0().entries(), archive.entries());
    }

    #[test]
    fn opens_bounded_entries_including_empty_files() {
        let archive = load_bytes(build_exefs(&[("main", b"abc"), ("empty", b"")])).unwrap();

        let main = archive.open_entry(archive.main().unwrap()).unwrap();
        assert_eq!(main.len().unwrap(), 3);
        let mut bytes = [0_u8; 3];
        main.read_at(0, &mut bytes).unwrap();
        assert_eq!(&bytes, b"abc");
        assert!(matches!(
            main.read_at(2, &mut [0_u8; 2]),
            Err(StorageError::OutOfBounds)
        ));

        let empty = archive.open("empty").unwrap().unwrap();
        assert_eq!(empty.len().unwrap(), 0);
        empty.read_at(0, &mut []).unwrap();
        assert!(matches!(
            empty.read_at(0, &mut [0_u8; 1]),
            Err(StorageError::OutOfBounds)
        ));
    }

    #[test]
    fn reports_malformed_pfs0_as_exefs() {
        for bytes in [b"NOPE".to_vec(), b"PFS0".to_vec()] {
            assert!(matches!(
                load_bytes(bytes),
                Err(LoadError::InvalidFormat {
                    format: "ExeFS",
                    ..
                })
            ));
        }

        let mut truncated_metadata = build_exefs(&[("main", b"")]);
        truncated_metadata.truncate(0x10);
        assert!(matches!(
            load_bytes(truncated_metadata),
            Err(LoadError::InvalidFormat {
                format: "ExeFS",
                ..
            })
        ));
    }

    #[test]
    fn preserves_pfs0_validation_with_exefs_context() {
        let mut duplicate = build_exefs(&[("main", b"a"), ("main", b"b")]);
        assert!(matches!(
            load_bytes(duplicate.clone()),
            Err(LoadError::InvalidFormat {
                format: "ExeFS",
                ..
            })
        ));

        duplicate[0x10..0x18].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(matches!(
            load_bytes(duplicate),
            Err(LoadError::InvalidFormat {
                format: "ExeFS",
                ..
            })
        ));

        let mut invalid_name = build_exefs(&[("main", b"")]);
        invalid_name[0x20..0x24].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(matches!(
            load_bytes(invalid_name),
            Err(LoadError::InvalidFormat {
                format: "ExeFS",
                ..
            })
        ));
    }

    #[test]
    fn loads_selected_program_nca_pfs0_section() {
        let exefs = build_exefs(&[("main", b"synthetic-nso"), ("main.npdm", b"npdm")]);
        let nca = NcaLoader::load(storage(build_program_nca(&exefs))).unwrap();

        assert_eq!(nca.header().content_type(), NcaContentType::Program);
        let section = &nca.sections()[0];
        let archive = ExeFsLoader::load_nca_section(section).unwrap();

        let main = archive.open("main").unwrap().unwrap();
        let mut bytes = [0_u8; 13];
        main.read_at(0, &mut bytes).unwrap();
        assert_eq!(&bytes, b"synthetic-nso");
    }

    #[test]
    fn rejects_incompatible_nca_section() {
        let mut bytes = build_program_nca(&build_exefs(&[]));
        bytes[0x402] = 0xFF;
        bytes[0x403] = 0xFF;
        let fs_hash: [u8; 32] = Sha256::digest(&bytes[0x400..0x600]).into();
        bytes[0x280..0x2A0].copy_from_slice(&fs_hash);
        let nca = NcaLoader::load(storage(bytes)).unwrap();

        assert!(matches!(
            ExeFsLoader::load_nca_section(&nca.sections()[0]),
            Err(LoadError::InvalidFormat {
                format: "ExeFS",
                ..
            })
        ));
    }

    fn build_program_nca(exefs: &[u8]) -> Vec<u8> {
        const MEDIA_UNIT_SIZE: u64 = 0x200;
        const SECTION_OFFSET: usize = 0xC00;
        const DATA_OFFSET: usize = 0x200;
        const BLOCK_SIZE: usize = 0x100;

        let data_size = exefs.len().div_ceil(BLOCK_SIZE) * BLOCK_SIZE;
        let unaligned_section_size = DATA_OFFSET + data_size;
        let section_size =
            unaligned_section_size.div_ceil(MEDIA_UNIT_SIZE as usize) * MEDIA_UNIT_SIZE as usize;
        let mut nca = vec![0_u8; SECTION_OFFSET + section_size];
        nca[0x200..0x204].copy_from_slice(b"NCA3");
        nca[0x204] = 0;
        nca[0x205] = 0;
        nca[0x206] = 1;
        put_u64(&mut nca, 0x208, (SECTION_OFFSET + section_size) as u64);
        put_u64(&mut nca, 0x210, 0x0100_0000_0000_1000);
        put_u32(
            &mut nca,
            0x240,
            u32::try_from(SECTION_OFFSET as u64 / MEDIA_UNIT_SIZE).unwrap(),
        );
        put_u32(
            &mut nca,
            0x244,
            u32::try_from((SECTION_OFFSET + section_size) as u64 / MEDIA_UNIT_SIZE).unwrap(),
        );

        let data_start = SECTION_OFFSET + DATA_OFFSET;
        nca[data_start..data_start + exefs.len()].copy_from_slice(exefs);
        let data_hash: [u8; 32] = Sha256::digest(&nca[data_start..data_start + data_size]).into();
        nca[SECTION_OFFSET..SECTION_OFFSET + 0x20].copy_from_slice(&data_hash);
        let master_hash: [u8; 32] =
            Sha256::digest(&nca[SECTION_OFFSET..SECTION_OFFSET + 0x20]).into();

        let fs = &mut nca[0x400..0x600];
        fs[2] = 1;
        fs[3] = 2;
        fs[4] = 1;
        fs[0x08..0x28].copy_from_slice(&master_hash);
        put_u32(fs, 0x28, BLOCK_SIZE as u32);
        put_u32(fs, 0x2C, 2);
        put_u64(fs, 0x30, 0);
        put_u64(fs, 0x38, 0x20);
        put_u64(fs, 0x40, DATA_OFFSET as u64);
        put_u64(fs, 0x48, data_size as u64);

        let fs_hash: [u8; 32] = Sha256::digest(&nca[0x400..0x600]).into();
        nca[0x280..0x2A0].copy_from_slice(&fs_hash);
        nca
    }

    fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
        bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }
}
