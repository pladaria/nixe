use std::collections::BTreeSet;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use swiitx_loader_content::{CnmtContentMeta, NcaKeyProvider, NcaKeySet, NspLoader, XciLoader};
use swiitx_loader_storage::{FileStorage, FormatLoader, LoadError, StorageRef};

use crate::discovery::{directory_files, package_format};
use crate::package_content::{
    PackageContent, import_ticket_keys, load_canonical_content_metas, load_control_metadata,
};
use crate::{
    ApplicationId, DirectoryScanOptions, PackageFormat, PackageMetadata, PackageMetadataError,
    TitleError,
};

/// Collection of package metadata discovered in one or more locations.
#[derive(Debug, Default)]
pub struct TitleCatalog {
    packages: Vec<PackageMetadata>,
}

impl TitleCatalog {
    /// Creates an empty title catalog.
    pub const fn new() -> Self {
        Self {
            packages: Vec::new(),
        }
    }

    /// Creates a catalog from metadata produced by content loaders.
    pub fn from_packages(packages: Vec<PackageMetadata>) -> Self {
        Self { packages }
    }

    /// Recursively scans supported packages without decryption keys.
    pub fn scan_directory(path: impl AsRef<Path>) -> Result<Self, TitleError> {
        Self::scan_directory_with_options(path, DirectoryScanOptions::default())
    }

    /// Scans supported packages using the supplied directory options.
    pub fn scan_directory_with_options(
        path: impl AsRef<Path>,
        options: DirectoryScanOptions,
    ) -> Result<Self, TitleError> {
        Self::scan_directory_impl(path.as_ref(), None, options)
    }

    /// Recursively scans supported packages using caller-owned NCA keys.
    ///
    /// Encrypted title keys present in package tickets are imported into the
    /// supplied key set before canonical content metadata is loaded.
    pub fn scan_directory_with_key_set(
        path: impl AsRef<Path>,
        keys: &mut NcaKeySet,
    ) -> Result<Self, TitleError> {
        Self::scan_directory_with_key_set_and_options(path, keys, DirectoryScanOptions::default())
    }

    /// Scans supported packages with caller-owned keys and directory options.
    pub fn scan_directory_with_key_set_and_options(
        path: impl AsRef<Path>,
        keys: &mut NcaKeySet,
        options: DirectoryScanOptions,
    ) -> Result<Self, TitleError> {
        Self::scan_directory_impl(path.as_ref(), Some(keys), options)
    }

    fn scan_directory_impl(
        path: &Path,
        mut keys: Option<&mut NcaKeySet>,
        options: DirectoryScanOptions,
    ) -> Result<Self, TitleError> {
        let metadata = fs::metadata(path).map_err(|source| TitleError::Io {
            path: path.to_owned(),
            source,
        })?;
        if !metadata.is_dir() {
            return Err(TitleError::NotDirectory {
                path: path.to_owned(),
            });
        }

        let candidates = directory_files(path, options)
            .map_err(|error| TitleError::Io {
                path: error.path,
                source: error.source,
            })?
            .into_iter()
            .filter(|candidate| package_format(candidate).is_some())
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            return Err(TitleError::NoSupportedPackages {
                path: path.to_owned(),
            });
        }

        let mut catalog = Self::new();
        for candidate in candidates {
            let format = package_format(&candidate)
                .expect("catalog candidates must have a supported package format");
            let storage = FileStorage::open(&candidate).map_err(|source| TitleError::Package {
                path: candidate.clone(),
                source: LoadError::Storage(source),
            })?;
            let storage: StorageRef = Arc::new(storage);
            match format {
                PackageFormat::Nsp => {
                    let archive =
                        NspLoader::load(storage.clone()).map_err(|source| TitleError::Package {
                            path: candidate.clone(),
                            source,
                        })?;
                    add_package_contents(
                        &mut catalog,
                        &candidate,
                        storage,
                        &archive,
                        keys.as_deref_mut(),
                        false,
                    )?;
                }
                PackageFormat::Xci => {
                    let archive =
                        XciLoader::load(storage.clone()).map_err(|source| TitleError::Package {
                            path: candidate.clone(),
                            source,
                        })?;
                    let secure =
                        archive
                            .secure_partition()
                            .map_err(|source| TitleError::Package {
                                path: candidate.clone(),
                                source,
                            })?;
                    add_package_contents(
                        &mut catalog,
                        &candidate,
                        storage,
                        secure.archive(),
                        keys.as_deref_mut(),
                        true,
                    )?;
                }
            }
        }

        Ok(catalog)
    }

    /// Adds package metadata to the catalog.
    pub fn add(&mut self, package: PackageMetadata) {
        self.packages.push(package);
    }

    /// Converts canonical binary CNMT into package metadata and adds it.
    pub fn add_content_meta(
        &mut self,
        content_meta: &CnmtContentMeta,
        source: StorageRef,
    ) -> Result<(), crate::PackageMetadataError> {
        self.add(PackageMetadata::from_content_meta(content_meta, source)?);
        Ok(())
    }

    /// Returns every package in discovery order.
    pub fn packages(&self) -> &[PackageMetadata] {
        &self.packages
    }

    /// Returns packages associated with one application.
    pub fn packages_for(
        &self,
        application_id: ApplicationId,
    ) -> impl Iterator<Item = &PackageMetadata> {
        self.packages
            .iter()
            .filter(move |package| package.application_id == application_id)
    }

    /// Returns each discovered application relationship once, in discovery order.
    pub fn application_ids(&self) -> impl Iterator<Item = ApplicationId> + '_ {
        let mut seen = BTreeSet::new();
        self.packages.iter().filter_map(move |package| {
            seen.insert(package.application_id)
                .then_some(package.application_id)
        })
    }
}

fn add_package_contents<C: PackageContent + ?Sized>(
    catalog: &mut TitleCatalog,
    path: &Path,
    source_storage: StorageRef,
    contents: &C,
    mut keys: Option<&mut NcaKeySet>,
    ignore_unsupported_metadata: bool,
) -> Result<(), TitleError> {
    if let Some(keys) = keys.as_deref_mut() {
        let _warnings = import_ticket_keys(contents, keys);
    }
    let key_provider = keys.as_deref().map(|keys| keys as &dyn NcaKeyProvider);
    let content_metas = load_canonical_content_metas(contents, key_provider).map_err(|source| {
        TitleError::Package {
            path: path.to_owned(),
            source,
        }
    })?;
    for content_meta in content_metas {
        let package = PackageMetadata::from_content_meta(&content_meta, source_storage.clone());
        let mut package = match package {
            Ok(package) => package,
            Err(PackageMetadataError::UnsupportedContentMetaType { .. })
                if ignore_unsupported_metadata =>
            {
                continue;
            }
            Err(source) => {
                return Err(TitleError::PackageMetadata {
                    path: path.to_owned(),
                    source,
                });
            }
        };
        let control_metadata = load_control_metadata(contents, &content_meta, key_provider)
            .map_err(|source| TitleError::Package {
                path: path.to_owned(),
                source,
            })?;
        package.set_control_metadata(control_metadata);
        catalog.add(package);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    use sha2::{Digest, Sha256};
    use swiitx_loader_content::{CnmtExtendedHeader, CnmtInstallType, CnmtMetaType, CnmtPlatform};
    use swiitx_loader_storage::{Storage, StorageError};

    use super::*;
    use crate::{ContentType, TitleResolver};

    const FIRST_APPLICATION_ID: u64 = 0x0100_1234_0000_0000;
    const SECOND_APPLICATION_ID: u64 = 0x0100_5678_0000_0000;

    static NEXT_TEMPORARY_DIRECTORY: AtomicU64 = AtomicU64::new(0);

    struct TemporaryDirectory(PathBuf);

    impl TemporaryDirectory {
        fn new() -> Self {
            let sequence = NEXT_TEMPORARY_DIRECTORY.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "swiitx-title-catalog-{}-{sequence}",
                std::process::id()
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }

        fn write(&self, name: &str, bytes: &[u8]) -> PathBuf {
            let path = self.0.join(name);
            fs::write(&path, bytes).unwrap();
            path
        }
    }

    impl Drop for TemporaryDirectory {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).unwrap();
        }
    }

    #[derive(Debug)]
    struct EmptyStorage;

    impl Storage for EmptyStorage {
        fn len(&self) -> Result<u64, StorageError> {
            Ok(0)
        }

        fn read_at(&self, offset: u64, buffer: &mut [u8]) -> Result<(), StorageError> {
            if offset == 0 && buffer.is_empty() {
                Ok(())
            } else {
                Err(StorageError::OutOfBounds)
            }
        }
    }

    fn application_content_meta(title_id: u64, version: u32) -> CnmtContentMeta {
        CnmtContentMeta {
            title_id,
            version: version.into(),
            content_meta_type: CnmtMetaType::Application,
            platform: CnmtPlatform::Nx,
            extended_header_size: 0x10,
            attributes: 0,
            storage_id: 0,
            install_type: CnmtInstallType::Full,
            committed: true,
            required_download_system_version: 0.into(),
            reserved: [0; 4],
            extended_header: CnmtExtendedHeader::Application {
                patch_id: title_id + 0x800,
                required_system_version: 0.into(),
                required_application_version: 0.into(),
            },
            contents: Vec::new(),
            content_meta: Vec::new(),
            extended_data_size: 0,
            digest: [0; 32],
        }
    }

    fn related_content_meta(
        title_id: u64,
        version: u32,
        content_meta_type: CnmtMetaType,
        extended_header: CnmtExtendedHeader,
    ) -> CnmtContentMeta {
        let mut metadata = application_content_meta(title_id, version);
        metadata.content_meta_type = content_meta_type;
        metadata.extended_header = extended_header;
        metadata
    }

    #[test]
    fn adds_canonical_content_meta_in_discovery_order() {
        let mut catalog = TitleCatalog::new();
        let first_source: StorageRef = Arc::new(EmptyStorage);
        let second_source: StorageRef = Arc::new(EmptyStorage);

        catalog
            .add_content_meta(
                &application_content_meta(0x0100_1234_0000_0000, 1),
                first_source.clone(),
            )
            .unwrap();
        catalog
            .add_content_meta(
                &application_content_meta(0x0100_5678_0000_0000, 2),
                second_source.clone(),
            )
            .unwrap();

        assert_eq!(catalog.packages().len(), 2);
        assert_eq!(
            catalog.packages()[0].title_id,
            crate::TitleId::new(0x0100_1234_0000_0000)
        );
        assert_eq!(
            catalog.packages()[1].title_id,
            crate::TitleId::new(0x0100_5678_0000_0000)
        );
        assert!(Arc::ptr_eq(&catalog.packages()[0].source, &first_source));
        assert!(Arc::ptr_eq(&catalog.packages()[1].source, &second_source));
    }

    #[test]
    fn does_not_add_content_meta_when_conversion_fails() {
        let mut catalog = TitleCatalog::new();
        let source: StorageRef = Arc::new(EmptyStorage);
        let mut metadata = application_content_meta(0x0100_1234_0000_0000, 1);
        metadata.content_meta_type = CnmtMetaType::SystemData;
        metadata.extended_header = CnmtExtendedHeader::None;

        assert!(catalog.add_content_meta(&metadata, source).is_err());
        assert!(catalog.packages().is_empty());
    }

    #[test]
    fn lists_unique_application_ids_in_discovery_order() {
        let source: StorageRef = Arc::new(EmptyStorage);
        let catalog = TitleCatalog::from_packages(vec![
            PackageMetadata::from_content_meta(
                &related_content_meta(
                    FIRST_APPLICATION_ID + 0x800,
                    1,
                    CnmtMetaType::Patch,
                    CnmtExtendedHeader::Patch {
                        application_id: FIRST_APPLICATION_ID,
                        required_system_version: 0.into(),
                        extended_data_size: 0,
                        reserved: [0; 8],
                    },
                ),
                source.clone(),
            )
            .unwrap(),
            PackageMetadata::from_content_meta(
                &application_content_meta(SECOND_APPLICATION_ID, 0),
                source.clone(),
            )
            .unwrap(),
            PackageMetadata::from_content_meta(
                &related_content_meta(
                    FIRST_APPLICATION_ID + 0x1001,
                    0,
                    CnmtMetaType::AddOnContent,
                    CnmtExtendedHeader::AddOnContent {
                        application_id: FIRST_APPLICATION_ID,
                        required_application_version: 0.into(),
                        content_accessibilities: 0,
                        padding: [0; 3],
                        data_patch_id: 0,
                    },
                ),
                source,
            )
            .unwrap(),
        ]);

        assert_eq!(
            catalog.application_ids().collect::<Vec<_>>(),
            vec![
                ApplicationId::new(FIRST_APPLICATION_ID),
                ApplicationId::new(SECOND_APPLICATION_ID)
            ]
        );
    }

    #[test]
    fn recursively_scans_nsps_in_sorted_order_and_resolves_titles() {
        let directory = TemporaryDirectory::new();
        let base = synthetic_nsp(
            FIRST_APPLICATION_ID,
            0,
            SyntheticMetaType::Application,
            false,
        );
        let patch = synthetic_nsp(
            FIRST_APPLICATION_ID + 0x800,
            5,
            SyntheticMetaType::Patch {
                application_id: FIRST_APPLICATION_ID,
            },
            false,
        );
        let add_on = synthetic_nsp(
            FIRST_APPLICATION_ID + 0x1001,
            1,
            SyntheticMetaType::AddOnContent {
                application_id: FIRST_APPLICATION_ID,
            },
            false,
        );
        directory.write("c-dlc.NSP", &add_on);
        directory.write("b-update.nsp", &patch);
        directory.write("a-base.nsp", &base);
        directory.write("ignored.bin", b"unsupported format");
        let nested = directory.path().join("nested");
        fs::create_dir(&nested).unwrap();
        fs::write(
            nested.join("nested.nsp"),
            synthetic_nsp(
                SECOND_APPLICATION_ID,
                0,
                SyntheticMetaType::Application,
                false,
            ),
        )
        .unwrap();

        let catalog = TitleCatalog::scan_directory(directory.path()).unwrap();
        let titles = TitleResolver::resolve_all(&catalog).unwrap();

        assert_eq!(catalog.packages().len(), 4);
        assert_eq!(catalog.packages()[0].content_type, ContentType::Application);
        assert_eq!(catalog.packages()[1].content_type, ContentType::Patch);
        assert_eq!(
            catalog.packages()[2].content_type,
            ContentType::AddOnContent
        );
        assert_eq!(catalog.packages()[3].content_type, ContentType::Application);
        assert_eq!(
            catalog.packages()[0].source.len().unwrap(),
            base.len() as u64
        );
        assert_eq!(titles.len(), 2);
        assert_eq!(
            titles[0].application_id,
            ApplicationId::new(FIRST_APPLICATION_ID)
        );
        assert_eq!(titles[0].patch.as_ref().unwrap().version.raw(), 5);
        assert_eq!(titles[0].add_ons.len(), 1);
        assert_eq!(
            titles[1].application_id,
            ApplicationId::new(SECOND_APPLICATION_ID)
        );
    }

    #[test]
    fn can_disable_recursive_directory_scanning() {
        let directory = TemporaryDirectory::new();
        directory.write(
            "root.nsp",
            &synthetic_nsp(
                FIRST_APPLICATION_ID,
                0,
                SyntheticMetaType::Application,
                false,
            ),
        );
        let nested = directory.path().join("nested");
        fs::create_dir(&nested).unwrap();
        fs::write(
            nested.join("nested.nsp"),
            synthetic_nsp(
                SECOND_APPLICATION_ID,
                0,
                SyntheticMetaType::Application,
                false,
            ),
        )
        .unwrap();

        let options = DirectoryScanOptions::default().with_recursive(false);
        let catalog = TitleCatalog::scan_directory_with_options(directory.path(), options).unwrap();

        assert_eq!(catalog.packages().len(), 1);
        assert_eq!(
            catalog.packages()[0].application_id,
            ApplicationId::new(FIRST_APPLICATION_ID)
        );
    }

    #[test]
    fn keyed_scan_imports_package_ticket_keys() {
        let directory = TemporaryDirectory::new();
        directory.write(
            "base.nsp",
            &synthetic_nsp(
                FIRST_APPLICATION_ID,
                0,
                SyntheticMetaType::Application,
                true,
            ),
        );
        let mut keys = NcaKeySet::from_text("", None).unwrap();

        let catalog =
            TitleCatalog::scan_directory_with_key_set(directory.path(), &mut keys).unwrap();

        assert_eq!(catalog.packages().len(), 1);
        assert_eq!(keys.title_key_count(), 1);
    }

    #[test]
    fn rejects_non_directory_and_empty_directory() {
        let directory = TemporaryDirectory::new();
        let file = directory.write("package.txt", b"not a package");

        assert!(matches!(
            TitleCatalog::scan_directory(&file),
            Err(TitleError::NotDirectory { path }) if path == file
        ));
        assert!(matches!(
            TitleCatalog::scan_directory(directory.path()),
            Err(TitleError::NoSupportedPackages { path }) if path == directory.path()
        ));
    }

    #[test]
    fn reports_the_path_of_a_malformed_nsp() {
        let directory = TemporaryDirectory::new();
        let package = directory.write("broken.nsp", b"not an NSP");

        assert!(matches!(
            TitleCatalog::scan_directory(directory.path()),
            Err(TitleError::Package { path, .. }) if path == package
        ));
    }

    #[test]
    fn scans_mixed_nsp_and_xci_in_order_and_reads_multiple_xci_titles() {
        let directory = TemporaryDirectory::new();
        directory.write(
            "a-digital.nsp",
            &synthetic_nsp(
                SECOND_APPLICATION_ID,
                0,
                SyntheticMetaType::Application,
                false,
            ),
        );
        let image = synthetic_xci(&[
            (FIRST_APPLICATION_ID, 0, SyntheticMetaType::Application),
            (
                FIRST_APPLICATION_ID + 0x800,
                3,
                SyntheticMetaType::Patch {
                    application_id: FIRST_APPLICATION_ID,
                },
            ),
        ]);
        directory.write("b-game.XcI", &image);

        let catalog = TitleCatalog::scan_directory(directory.path()).unwrap();
        let titles = TitleResolver::resolve_all(&catalog).unwrap();

        assert_eq!(catalog.packages().len(), 3);
        assert_eq!(
            catalog.packages()[0].application_id,
            ApplicationId::new(SECOND_APPLICATION_ID)
        );
        assert_eq!(
            catalog.packages()[1].application_id,
            ApplicationId::new(FIRST_APPLICATION_ID)
        );
        assert_eq!(titles.len(), 2);
        assert_eq!(titles[1].patch.as_ref().unwrap().version.raw(), 3);
        assert!(
            catalog
                .packages()
                .iter()
                .skip(1)
                .all(|package| package.source.len().unwrap() == image.len() as u64)
        );
    }

    enum SyntheticMetaType {
        Application,
        Patch { application_id: u64 },
        AddOnContent { application_id: u64 },
    }

    fn synthetic_nsp(
        title_id: u64,
        version: u32,
        meta_type: SyntheticMetaType,
        include_ticket: bool,
    ) -> Vec<u8> {
        let cnmt = synthetic_cnmt(title_id, version, meta_type);
        let inner_pfs0 = build_pfs0(&[("ContentMeta.cnmt", cnmt.as_slice())]);
        let meta_nca = build_meta_nca(title_id, &inner_pfs0);
        if include_ticket {
            let ticket = vec![0_u8; 0x2B0];
            build_pfs0(&[
                ("meta.cnmt.nca", meta_nca.as_slice()),
                ("title.tik", ticket.as_slice()),
            ])
        } else {
            build_pfs0(&[("meta.cnmt.nca", meta_nca.as_slice())])
        }
    }

    fn synthetic_xci(records: &[(u64, u32, SyntheticMetaType)]) -> Vec<u8> {
        let meta_ncas = records
            .iter()
            .enumerate()
            .map(|(index, (title_id, version, meta_type))| {
                let meta_type = match meta_type {
                    SyntheticMetaType::Application => SyntheticMetaType::Application,
                    SyntheticMetaType::Patch { application_id } => SyntheticMetaType::Patch {
                        application_id: *application_id,
                    },
                    SyntheticMetaType::AddOnContent { application_id } => {
                        SyntheticMetaType::AddOnContent {
                            application_id: *application_id,
                        }
                    }
                };
                let cnmt = synthetic_cnmt(*title_id, *version, meta_type);
                let pfs0 = build_pfs0(&[("ContentMeta.cnmt", cnmt.as_slice())]);
                (
                    format!("meta-{index}.cnmt.nca"),
                    build_meta_nca(*title_id, &pfs0),
                )
            })
            .collect::<Vec<_>>();
        let secure_files = meta_ncas
            .iter()
            .map(|(name, bytes)| (name.as_str(), bytes.as_slice()))
            .collect::<Vec<_>>();
        let secure = build_hfs0(&secure_files);
        let root = build_hfs0(&[("secure", secure.as_slice())]);
        let root_header_size = 0x10 + 0x40 + "secure".len() + 1;
        let root_offset = 0x200_usize;
        let image_size = root_offset + root.len();
        let page_count = image_size.div_ceil(0x200);
        let mut image = vec![0_u8; page_count * 0x200];
        image[0x100..0x104].copy_from_slice(b"HEAD");
        image[0x118..0x11c].copy_from_slice(&((page_count - 1) as u32).to_le_bytes());
        image[0x130..0x138].copy_from_slice(&(root_offset as u64).to_le_bytes());
        image[0x138..0x140].copy_from_slice(&(root_header_size as u64).to_le_bytes());
        image[0x140..0x160].copy_from_slice(&Sha256::digest(&root[..root_header_size]));
        image[root_offset..root_offset + root.len()].copy_from_slice(&root);
        image
    }

    fn build_hfs0(files: &[(&str, &[u8])]) -> Vec<u8> {
        let mut strings = Vec::new();
        let mut name_offsets = Vec::new();
        for (name, _) in files {
            name_offsets.push(strings.len() as u32);
            strings.extend_from_slice(name.as_bytes());
            strings.push(0);
        }
        let mut hfs0 = Vec::new();
        hfs0.extend_from_slice(b"HFS0");
        hfs0.extend_from_slice(&(files.len() as u32).to_le_bytes());
        hfs0.extend_from_slice(&(strings.len() as u32).to_le_bytes());
        hfs0.extend_from_slice(&0_u32.to_le_bytes());
        let mut offset = 0_u64;
        for ((_, data), name_offset) in files.iter().zip(name_offsets) {
            hfs0.extend_from_slice(&offset.to_le_bytes());
            hfs0.extend_from_slice(&(data.len() as u64).to_le_bytes());
            hfs0.extend_from_slice(&name_offset.to_le_bytes());
            hfs0.extend_from_slice(&(data.len() as u32).to_le_bytes());
            hfs0.extend_from_slice(&[0; 8]);
            hfs0.extend_from_slice(&Sha256::digest(data));
            offset += data.len() as u64;
        }
        hfs0.extend_from_slice(&strings);
        for (_, data) in files {
            hfs0.extend_from_slice(data);
        }
        hfs0
    }

    fn synthetic_cnmt(title_id: u64, version: u32, meta_type: SyntheticMetaType) -> Vec<u8> {
        let (raw_type, extended_header) = match meta_type {
            SyntheticMetaType::Application => {
                let mut header = vec![0_u8; 0x10];
                put_u64(&mut header, 0, title_id + 0x800);
                (0x80, header)
            }
            SyntheticMetaType::Patch { application_id } => {
                let mut header = vec![0_u8; 0x18];
                put_u64(&mut header, 0, application_id);
                (0x81, header)
            }
            SyntheticMetaType::AddOnContent { application_id } => {
                let mut header = vec![0_u8; 0x10];
                put_u64(&mut header, 0, application_id);
                (0x82, header)
            }
        };

        let mut cnmt = vec![0_u8; 0x20];
        put_u64(&mut cnmt, 0, title_id);
        put_u32(&mut cnmt, 8, version);
        cnmt[0x0C] = raw_type;
        put_u16(&mut cnmt, 0x0E, extended_header.len() as u16);
        cnmt[0x17] = 1;
        cnmt.extend_from_slice(&extended_header);
        cnmt.extend_from_slice(&[0xA5; 0x20]);
        cnmt
    }

    fn build_pfs0(files: &[(&str, &[u8])]) -> Vec<u8> {
        let mut strings = Vec::new();
        let mut name_offsets = Vec::new();
        for (name, _) in files {
            name_offsets.push(strings.len() as u32);
            strings.extend_from_slice(name.as_bytes());
            strings.push(0);
        }

        let mut pfs0 = Vec::new();
        pfs0.extend_from_slice(b"PFS0");
        pfs0.extend_from_slice(&(files.len() as u32).to_le_bytes());
        pfs0.extend_from_slice(&(strings.len() as u32).to_le_bytes());
        pfs0.extend_from_slice(&0_u32.to_le_bytes());
        let mut relative_offset = 0_u64;
        for ((_, data), name_offset) in files.iter().zip(name_offsets) {
            pfs0.extend_from_slice(&relative_offset.to_le_bytes());
            pfs0.extend_from_slice(&(data.len() as u64).to_le_bytes());
            pfs0.extend_from_slice(&name_offset.to_le_bytes());
            pfs0.extend_from_slice(&0_u32.to_le_bytes());
            relative_offset += data.len() as u64;
        }
        pfs0.extend_from_slice(&strings);
        for (_, data) in files {
            pfs0.extend_from_slice(data);
        }
        pfs0
    }

    fn build_meta_nca(title_id: u64, payload: &[u8]) -> Vec<u8> {
        const SECTION_OFFSET: usize = 0xC00;
        const SECTION_SIZE: usize = 0x400;
        const DATA_OFFSET: usize = 0x200;
        const BLOCK_SIZE: usize = 0x200;
        assert!(payload.len() <= BLOCK_SIZE);

        let mut nca = vec![0_u8; SECTION_OFFSET + SECTION_SIZE];
        nca[0x200..0x204].copy_from_slice(b"NCA3");
        nca[0x205] = 1;
        nca[0x206] = 1;
        put_u64(&mut nca, 0x208, (SECTION_OFFSET + SECTION_SIZE) as u64);
        put_u64(&mut nca, 0x210, title_id);
        put_u32(&mut nca, 0x240, (SECTION_OFFSET / 0x200) as u32);
        put_u32(
            &mut nca,
            0x244,
            ((SECTION_OFFSET + SECTION_SIZE) / 0x200) as u32,
        );

        let data_start = SECTION_OFFSET + DATA_OFFSET;
        nca[data_start..data_start + payload.len()].copy_from_slice(payload);
        let data_hash: [u8; 32] = Sha256::digest(payload).into();
        nca[SECTION_OFFSET..SECTION_OFFSET + 0x20].copy_from_slice(&data_hash);
        let master_hash: [u8; 32] =
            Sha256::digest(&nca[SECTION_OFFSET..SECTION_OFFSET + 0x20]).into();

        let fs_header = &mut nca[0x400..0x600];
        fs_header[2] = 1;
        fs_header[3] = 2;
        fs_header[4] = 1;
        fs_header[0x08..0x28].copy_from_slice(&master_hash);
        put_u32(fs_header, 0x28, BLOCK_SIZE as u32);
        put_u64(fs_header, 0x30, 0);
        put_u64(fs_header, 0x38, 0x20);
        put_u64(fs_header, 0x40, DATA_OFFSET as u64);
        put_u64(fs_header, 0x48, payload.len() as u64);
        let fs_hash: [u8; 32] = Sha256::digest(&nca[0x400..0x600]).into();
        nca[0x280..0x2A0].copy_from_slice(&fs_hash);
        nca
    }

    fn put_u16(bytes: &mut [u8], offset: usize, value: u16) {
        bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
        bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }
}
