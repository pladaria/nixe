use std::collections::BTreeSet;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use nixe_loader_content::{
    BktrPatch, CnmtContentInfo, CnmtContentType, CnmtExtendedHeader, NcaArchive, NcaKeyProvider,
    NcaKeySet, NcaSectionType, RomFsLoader,
};
use nixe_loader_executable::{NpdmLoader, NroLoader, NsoLoader};
use nixe_loader_storage::{FileStorage, FormatLoader, LoadError, StorageRef};
use nixe_loader_title::{
    ApplicationId, DirectoryScanOptions, PackageMetadata, ResolvedTitle, TitleCatalog, TitleError,
    TitleId, TitleResolver,
};

use crate::launch_plan::{
    AddOnContent, LaunchKind, LaunchModule, LaunchModuleImage, LaunchPlan, MAX_ADD_ON_MOUNTS,
    MAX_LAUNCH_ADD_ONS, MAX_LAUNCH_MODULES, ModuleRole, MountProvenance, PackagedIdentity,
    ReadOnlyMount,
};

/// Configuration consumed atomically by [`Launcher::build`].
pub struct LauncherInput {
    path: PathBuf,
    keys: Option<NcaKeySet>,
    directory_options: DirectoryScanOptions,
    application_id: Option<ApplicationId>,
}

impl LauncherInput {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            keys: None,
            directory_options: DirectoryScanOptions::default(),
            application_id: None,
        }
    }
    pub fn with_keys(mut self, keys: NcaKeySet) -> Self {
        self.keys = Some(keys);
        self
    }
    pub const fn with_directory_options(mut self, options: DirectoryScanOptions) -> Self {
        self.directory_options = options;
        self
    }
    pub const fn with_application_id(mut self, application_id: ApplicationId) -> Self {
        self.application_id = Some(application_id);
        self
    }
}

impl std::fmt::Debug for LauncherInput {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LauncherInput")
            .field("path", &self.path)
            .field("keys", &self.keys.as_ref().map(|_| "<caller keys>"))
            .field("directory_options", &self.directory_options)
            .field("application_id", &self.application_id)
            .finish()
    }
}

/// Stage at which launch-plan construction failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaunchStage {
    PathDetection,
    TitleDiscovery,
    TitleResolution,
    ProgramContent,
    ProcessMetadata,
    ExecutableModules,
    PrimaryFileSystem,
    AddOnContent,
}

/// Structured, context-preserving launch-plan construction failure.
#[derive(Debug)]
pub struct LaunchError {
    stage: LaunchStage,
    path: PathBuf,
    application_id: Option<ApplicationId>,
    title_id: Option<TitleId>,
    content_id: Option<[u8; 16]>,
    module: Option<String>,
    source: Box<LaunchErrorSource>,
}

#[derive(Debug)]
enum LaunchErrorSource {
    Io(std::io::Error),
    Title(TitleError),
    Load(LoadError),
    Invalid(String),
}

impl LaunchError {
    fn invalid(stage: LaunchStage, path: &Path, reason: impl Into<String>) -> Self {
        Self {
            stage,
            path: path.to_owned(),
            application_id: None,
            title_id: None,
            content_id: None,
            module: None,
            source: Box::new(LaunchErrorSource::Invalid(reason.into())),
        }
    }
    fn title(stage: LaunchStage, path: &Path, source: TitleError) -> Self {
        Self {
            stage,
            path: path.to_owned(),
            application_id: None,
            title_id: None,
            content_id: None,
            module: None,
            source: Box::new(LaunchErrorSource::Title(source)),
        }
    }
    fn load(stage: LaunchStage, path: &Path, source: LoadError) -> Self {
        Self {
            stage,
            path: path.to_owned(),
            application_id: None,
            title_id: None,
            content_id: None,
            module: None,
            source: Box::new(LaunchErrorSource::Load(source)),
        }
    }
    fn application(mut self, id: ApplicationId) -> Self {
        self.application_id = Some(id);
        self
    }
    fn title_id(mut self, id: TitleId) -> Self {
        self.title_id = Some(id);
        self
    }
    fn content(mut self, id: [u8; 16]) -> Self {
        self.content_id = Some(id);
        self
    }
    fn module(mut self, name: impl Into<String>) -> Self {
        self.module = Some(name.into());
        self
    }
    pub const fn stage(&self) -> LaunchStage {
        self.stage
    }
    pub fn path(&self) -> &Path {
        &self.path
    }
    pub const fn application_id(&self) -> Option<ApplicationId> {
        self.application_id
    }
    pub const fn title_id_value(&self) -> Option<TitleId> {
        self.title_id
    }
    pub const fn content_id(&self) -> Option<&[u8; 16]> {
        self.content_id.as_ref()
    }
    pub fn module_name(&self) -> Option<&str> {
        self.module.as_deref()
    }
}

impl Display for LaunchError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "cannot construct launch plan at {:?} for {}",
            self.stage,
            self.path.display()
        )?;
        if let Some(id) = self.application_id {
            write!(formatter, ", application {id}")?;
        }
        if let Some(id) = self.title_id {
            write!(formatter, ", title {id}")?;
        }
        if let Some(name) = &self.module {
            write!(formatter, ", module {name:?}")?;
        }
        write!(formatter, ": ")?;
        match self.source.as_ref() {
            LaunchErrorSource::Io(error) => Display::fmt(error, formatter),
            LaunchErrorSource::Title(error) => Display::fmt(error, formatter),
            LaunchErrorSource::Load(error) => Display::fmt(error, formatter),
            LaunchErrorSource::Invalid(reason) => formatter.write_str(reason),
        }
    }
}

impl Error for LaunchError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self.source.as_ref() {
            LaunchErrorSource::Io(error) => Some(error),
            LaunchErrorSource::Title(error) => Some(error),
            LaunchErrorSource::Load(error) => Some(error),
            LaunchErrorSource::Invalid(_) => None,
        }
    }
}

/// Coordinates path, title, content, and executable loaders.
#[derive(Debug)]
pub struct Launcher;

impl Launcher {
    /// Builds one complete immutable plan or returns no partial result.
    pub fn build(mut input: LauncherInput) -> Result<LaunchPlan, LaunchError> {
        let metadata = std::fs::metadata(&input.path).map_err(|source| LaunchError {
            stage: LaunchStage::PathDetection,
            path: input.path.clone(),
            application_id: None,
            title_id: None,
            content_id: None,
            module: None,
            source: Box::new(LaunchErrorSource::Io(source)),
        })?;
        if metadata.is_dir() {
            let path = input.path.clone();
            return Self::build_directory(&path, &mut input);
        }
        if !metadata.is_file() {
            return Err(LaunchError::invalid(
                LaunchStage::PathDetection,
                &input.path,
                "path is not a regular file or directory",
            ));
        }
        let extension = input
            .path
            .extension()
            .and_then(|value| value.to_str())
            .unwrap_or_default();
        if extension.eq_ignore_ascii_case("nro") {
            Self::build_homebrew(&input.path)
        } else if ["nsp", "nsz", "xci", "xcz"]
            .iter()
            .any(|candidate| extension.eq_ignore_ascii_case(candidate))
        {
            let catalog = match input.keys.as_mut() {
                Some(keys) => TitleCatalog::load_package_with_key_set(&input.path, keys),
                None => TitleCatalog::load_package(&input.path),
            }
            .map_err(|error| LaunchError::title(LaunchStage::TitleDiscovery, &input.path, error))?;
            Self::build_from_catalog(
                &input.path,
                catalog,
                input.application_id,
                input.keys.as_ref(),
            )
        } else {
            Err(LaunchError::invalid(
                LaunchStage::PathDetection,
                &input.path,
                "unsupported file type",
            ))
        }
    }

    /// Builds a launch plan from a title already resolved by a shared library scan.
    ///
    /// This avoids discovering and resolving the same packages again after an
    /// application has selected a title by ID.
    pub fn build_resolved_title(
        resolved: ResolvedTitle,
        keys: &NcaKeySet,
    ) -> Result<LaunchPlan, LaunchError> {
        let context = PathBuf::from(format!("resolved-title-{}", resolved.application_id));
        build_packaged(&context, resolved, Some(keys))
    }

    fn build_directory(path: &Path, input: &mut LauncherInput) -> Result<LaunchPlan, LaunchError> {
        let catalog = match input.keys.as_mut() {
            Some(keys) => TitleCatalog::scan_directory_with_key_set_and_options(
                path,
                keys,
                input.directory_options,
            ),
            None => TitleCatalog::scan_directory_with_options(path, input.directory_options),
        }
        .map_err(|error| LaunchError::title(LaunchStage::TitleDiscovery, path, error))?;
        Self::build_from_catalog(path, catalog, input.application_id, input.keys.as_ref())
    }

    fn build_from_catalog(
        path: &Path,
        catalog: TitleCatalog,
        selection: Option<ApplicationId>,
        keys: Option<&NcaKeySet>,
    ) -> Result<LaunchPlan, LaunchError> {
        let application_id = match selection {
            Some(id) => id,
            None => {
                let ids = catalog.application_ids().collect::<Vec<_>>();
                match ids.as_slice() {
                    [id] => *id,
                    [] => {
                        return Err(LaunchError::invalid(
                            LaunchStage::TitleResolution,
                            path,
                            "input contains no application relationship",
                        ));
                    }
                    _ => {
                        return Err(LaunchError::invalid(
                            LaunchStage::TitleResolution,
                            path,
                            "input resolves to multiple applications; select one explicitly",
                        ));
                    }
                }
            }
        };
        let resolved = TitleResolver::resolve(&catalog, application_id).map_err(|error| {
            LaunchError::title(LaunchStage::TitleResolution, path, error)
                .application(application_id)
        })?;
        build_packaged(
            path,
            resolved,
            keys.map(|value| value as &dyn NcaKeyProvider),
        )
    }

    fn build_homebrew(path: &Path) -> Result<LaunchPlan, LaunchError> {
        let storage: StorageRef = Arc::new(
            FileStorage::open(path)
                .map_err(LoadError::Storage)
                .map_err(|error| LaunchError::load(LaunchStage::ExecutableModules, path, error))?,
        );
        let image = NroLoader::load(storage)
            .map_err(|error| LaunchError::load(LaunchStage::ExecutableModules, path, error))?;
        let primary = image
            .assets()
            .and_then(|assets| assets.romfs())
            .map(|storage| RomFsLoader::load(storage.clone()))
            .transpose()
            .map_err(|error| LaunchError::load(LaunchStage::PrimaryFileSystem, path, error))?
            .map(|romfs| ReadOnlyMount::new(MountProvenance::HomebrewAsset, None, romfs));
        let module = LaunchModule::new(
            "main".into(),
            ModuleRole::Homebrew,
            LaunchModuleImage::Nro(image),
        );
        Ok(LaunchPlan::new(
            LaunchKind::Homebrew,
            vec![module],
            0,
            primary,
            Vec::new(),
            None,
        ))
    }
}

fn build_packaged(
    path: &Path,
    resolved: ResolvedTitle,
    keys: Option<&dyn NcaKeyProvider>,
) -> Result<LaunchPlan, LaunchError> {
    let build_started = Instant::now();
    if resolved.add_ons.len() > MAX_LAUNCH_ADD_ONS {
        return Err(LaunchError::invalid(
            LaunchStage::AddOnContent,
            path,
            "resolved add-on count exceeds the launch limit",
        )
        .application(resolved.application_id));
    }
    let effective = resolved
        .patch
        .as_ref()
        .filter(|package| !records(package, CnmtContentType::Program).is_empty())
        .unwrap_or(&resolved.base);
    let program_record = exactly_one_record(effective, CnmtContentType::Program, "Program")
        .map_err(|reason| {
            LaunchError::invalid(LaunchStage::ProgramContent, path, reason)
                .application(resolved.application_id)
                .title_id(effective.title_id)
        })?;
    let program_started = Instant::now();
    let program = effective
        .open_content(program_record, keys)
        .map_err(|error| {
            LaunchError::load(LaunchStage::ProgramContent, path, error)
                .application(resolved.application_id)
                .title_id(effective.title_id)
                .content(program_record.content_id)
        })?;
    log::debug!("program NCA opened in {:?}", program_started.elapsed());
    let exefs_started = Instant::now();
    let mut exefs_candidates = Vec::new();
    for section in program
        .sections()
        .iter()
        .filter(|section| section.section_type() == NcaSectionType::Pfs0)
    {
        let archive = nixe_loader_content::ExeFsLoader::load_nca_section(section)
            .map_err(|error| LaunchError::load(LaunchStage::ProgramContent, path, error))?;
        if archive.main().is_some() && archive.main_npdm().is_some() {
            exefs_candidates.push(archive);
        }
    }
    let exefs = match exefs_candidates.len() {
        1 => exefs_candidates.pop().expect("validated ExeFS count"),
        _ => {
            return Err(LaunchError::invalid(
                LaunchStage::ProgramContent,
                path,
                format!(
                    "effective Program NCA contains {} PFS0 sections with main and main.npdm; expected one",
                    exefs_candidates.len()
                ),
            )
            .application(resolved.application_id)
            .title_id(effective.title_id)
            .content(program_record.content_id));
        }
    };
    log::debug!("ExeFS located and parsed in {:?}", exefs_started.elapsed());
    let npdm_started = Instant::now();
    let npdm_entry = exefs.main_npdm().ok_or_else(|| {
        LaunchError::invalid(
            LaunchStage::ProcessMetadata,
            path,
            "effective ExeFS has no main.npdm",
        )
    })?;
    let npdm = NpdmLoader::load(
        exefs
            .open_entry(npdm_entry)
            .map_err(|error| LaunchError::load(LaunchStage::ProcessMetadata, path, error))?,
    )
    .map_err(|error| LaunchError::load(LaunchStage::ProcessMetadata, path, error))?;
    if npdm.program_id() != resolved.application_id.get() {
        return Err(LaunchError::invalid(
            LaunchStage::ProcessMetadata,
            path,
            format!(
                "NPDM program ID {:016X} does not match application {}",
                npdm.program_id(),
                resolved.application_id
            ),
        )
        .application(resolved.application_id));
    }
    log::debug!("main.npdm loaded in {:?}", npdm_started.elapsed());
    let modules_started = Instant::now();
    let (modules, entry_module) = load_modules(path, &exefs)?;
    log::debug!(
        "{} executable module(s) loaded in {:?}",
        modules.len(),
        modules_started.elapsed()
    );
    let primary_started = Instant::now();
    let primary = load_primary_mount(path, &resolved, effective, &program, keys)?;
    log::debug!(
        "primary filesystem resolved in {:?}",
        primary_started.elapsed()
    );
    let add_ons_started = Instant::now();
    let add_ons = load_add_ons(path, &resolved, keys)?;
    log::debug!(
        "{} add-on(s) resolved in {:?}",
        add_ons.len(),
        add_ons_started.elapsed()
    );
    let identity = PackagedIdentity::new(
        resolved.application_id,
        effective.title_id,
        effective.version,
        program_record.content_id,
        npdm,
    );
    let plan = LaunchPlan::new(
        LaunchKind::Packaged(Box::new(identity)),
        modules,
        entry_module,
        primary,
        add_ons,
        resolved.control_metadata().cloned(),
    );
    log::debug!(
        "packaged launch plan built in {:?}",
        build_started.elapsed()
    );
    Ok(plan)
}

fn load_modules(
    path: &Path,
    exefs: &nixe_loader_content::ExeFsArchive,
) -> Result<(Vec<LaunchModule>, usize), LaunchError> {
    let mut candidates = Vec::new();
    let mut roles = BTreeSet::new();
    for entry in exefs.entries() {
        let name = entry.name();
        if name == "main.npdm" {
            continue;
        }
        let role = module_role(name).ok_or_else(|| {
            LaunchError::invalid(
                LaunchStage::ExecutableModules,
                path,
                "ExeFS entry has no verified executable module role",
            )
            .module(name)
        })?;
        if !roles.insert(role) {
            return Err(LaunchError::invalid(
                LaunchStage::ExecutableModules,
                path,
                "duplicate executable module role",
            )
            .module(name));
        }
        candidates.push((role, name.to_owned(), entry));
    }
    if candidates.len() > MAX_LAUNCH_MODULES {
        return Err(LaunchError::invalid(
            LaunchStage::ExecutableModules,
            path,
            "module count exceeds the launch limit",
        ));
    }
    candidates.sort_by_key(|(role, _, _)| *role);
    let main_count = candidates
        .iter()
        .filter(|(role, _, _)| *role == ModuleRole::Main)
        .count();
    if main_count != 1 {
        return Err(LaunchError::invalid(
            LaunchStage::ExecutableModules,
            path,
            format!("ExeFS contains {main_count} main modules; expected one"),
        ));
    }
    let mut modules = Vec::with_capacity(candidates.len());
    let mut module_ids = BTreeSet::new();
    for (role, name, entry) in candidates {
        let module_started = Instant::now();
        let storage = exefs.open_entry(entry).map_err(|error| {
            LaunchError::load(LaunchStage::ExecutableModules, path, error).module(&name)
        })?;
        let image = NsoLoader::load(storage).map_err(|error| {
            LaunchError::load(LaunchStage::ExecutableModules, path, error).module(&name)
        })?;
        if !module_ids.insert(*image.executable().module_id()) {
            return Err(LaunchError::invalid(
                LaunchStage::ExecutableModules,
                path,
                "duplicate executable module identity",
            )
            .module(&name));
        }
        modules.push(LaunchModule::new(
            name.into_boxed_str(),
            role,
            LaunchModuleImage::Nso(image),
        ));
        let module = modules.last().expect("module was just inserted");
        log::debug!(
            "module {} ({:?}) loaded in {:?}",
            module.name(),
            module.role(),
            module_started.elapsed()
        );
    }
    let entry = modules
        .iter()
        .position(|module| module.role() == ModuleRole::RuntimeLoader)
        .or_else(|| {
            modules
                .iter()
                .position(|module| module.role() == ModuleRole::Main)
        })
        .expect("validated main module");
    Ok((modules, entry))
}

fn module_role(name: &str) -> Option<ModuleRole> {
    match name {
        "rtld" => Some(ModuleRole::RuntimeLoader),
        "main" => Some(ModuleRole::Main),
        "sdk" => Some(ModuleRole::Sdk),
        _ => name
            .strip_prefix("subsdk")
            .filter(|suffix| !suffix.is_empty() && suffix.bytes().all(|byte| byte.is_ascii_digit()))
            .and_then(|suffix| suffix.parse::<u16>().ok())
            .map(ModuleRole::SubSdk),
    }
}

fn load_primary_mount(
    path: &Path,
    resolved: &ResolvedTitle,
    effective: &PackageMetadata,
    program: &NcaArchive,
    keys: Option<&dyn NcaKeyProvider>,
) -> Result<Option<ReadOnlyMount>, LaunchError> {
    let data = records(effective, CnmtContentType::Data);
    if data.len() > 1 {
        return Err(LaunchError::invalid(
            LaunchStage::PrimaryFileSystem,
            path,
            "effective application declares multiple primary Data contents",
        ));
    }
    if let Some(record) = data.first() {
        let nca = effective.open_content(record, keys).map_err(|error| {
            LaunchError::load(LaunchStage::PrimaryFileSystem, path, error)
                .title_id(effective.title_id)
                .content(record.content_id)
        })?;
        let is_bktr = nca
            .sections()
            .iter()
            .any(|section| section.section_type() == NcaSectionType::Bktr);
        let base_nca = if is_bktr && effective.title_id != resolved.base.title_id {
            let matching = records(&resolved.base, CnmtContentType::Data)
                .into_iter()
                .filter(|base| base.id_offset == record.id_offset)
                .collect::<Vec<_>>();
            let base_record = match matching.as_slice() {
                [record] => *record,
                _ => {
                    return Err(LaunchError::invalid(
                        LaunchStage::PrimaryFileSystem,
                        path,
                        "patch Data content has no unique base record with the same ID offset",
                    )
                    .content(record.content_id));
                }
            };
            Some(
                resolved
                    .base
                    .open_content(base_record, keys)
                    .map_err(|error| {
                        LaunchError::load(LaunchStage::PrimaryFileSystem, path, error)
                            .title_id(resolved.base.title_id)
                            .content(base_record.content_id)
                    })?,
            )
        } else {
            None
        };
        let provenance = if base_nca.is_some() {
            MountProvenance::BaseAndPatch
        } else if effective.title_id == resolved.base.title_id {
            MountProvenance::Base
        } else {
            MountProvenance::Patch
        };
        return mount_nca(path, &nca, base_nca.as_ref(), provenance, record.content_id).map(Some);
    }
    let base_program = if effective.title_id != resolved.base.title_id
        && program
            .sections()
            .iter()
            .any(|section| section.section_type() == NcaSectionType::Bktr)
    {
        let record = exactly_one_record(&resolved.base, CnmtContentType::Program, "base Program")
            .map_err(|reason| {
            LaunchError::invalid(LaunchStage::PrimaryFileSystem, path, reason)
        })?;
        Some(resolved.base.open_content(record, keys).map_err(|error| {
            LaunchError::load(LaunchStage::PrimaryFileSystem, path, error)
                .title_id(resolved.base.title_id)
                .content(record.content_id)
        })?)
    } else {
        None
    };
    let provenance = if base_program.is_some() {
        MountProvenance::BaseAndPatch
    } else if effective.title_id == resolved.base.title_id {
        MountProvenance::Base
    } else {
        MountProvenance::Patch
    };
    mount_nca(
        path,
        program,
        base_program.as_ref(),
        provenance,
        exactly_one_record(effective, CnmtContentType::Program, "Program")
            .expect("effective Program was validated")
            .content_id,
    )
    .map(Some).or_else(|error| {
        if matches!(error.source.as_ref(), LaunchErrorSource::Invalid(reason) if reason == "NCA has no RomFS section") {
            Ok(None)
        } else {
            Err(error)
        }
    })
}

fn mount_nca(
    path: &Path,
    nca: &NcaArchive,
    base: Option<&NcaArchive>,
    provenance: MountProvenance,
    content_id: [u8; 16],
) -> Result<ReadOnlyMount, LaunchError> {
    let mount_started = Instant::now();
    let sections = nca
        .sections()
        .iter()
        .filter(|section| {
            matches!(
                section.section_type(),
                NcaSectionType::RomFs | NcaSectionType::Bktr
            )
        })
        .collect::<Vec<_>>();
    let section = match sections.as_slice() {
        [section] => *section,
        [] => {
            return Err(LaunchError::invalid(
                LaunchStage::PrimaryFileSystem,
                path,
                "NCA has no RomFS section",
            ));
        }
        _ => {
            return Err(LaunchError::invalid(
                LaunchStage::PrimaryFileSystem,
                path,
                "NCA has multiple RomFS sections",
            ));
        }
    };
    let romfs = if section.section_type() == NcaSectionType::Bktr {
        let base = base.ok_or_else(|| {
            LaunchError::invalid(
                LaunchStage::PrimaryFileSystem,
                path,
                "BKTR content has no matching base NCA",
            )
        })?;
        let base_sections = base
            .sections()
            .iter()
            .filter(|candidate| candidate.section_type() == NcaSectionType::RomFs)
            .collect::<Vec<_>>();
        let base_section = match base_sections.as_slice() {
            [section] => *section,
            _ => {
                return Err(LaunchError::invalid(
                    LaunchStage::PrimaryFileSystem,
                    path,
                    "base NCA does not contain exactly one RomFS section",
                ));
            }
        };
        BktrPatch::open(base_section, section)
            .and_then(|patch| patch.load_romfs())
            .map_err(|error| LaunchError::load(LaunchStage::PrimaryFileSystem, path, error))?
    } else {
        RomFsLoader::load(
            section
                .payload_storage()
                .map_err(|error| LaunchError::load(LaunchStage::PrimaryFileSystem, path, error))?,
        )
        .map_err(|error| LaunchError::load(LaunchStage::PrimaryFileSystem, path, error))?
    };
    let file_count = romfs.files().len();
    let mount = ReadOnlyMount::new(provenance, Some(content_id), romfs);
    log::debug!(
        "RomFS ({provenance:?}, {file_count} files) indexed in {:?}",
        mount_started.elapsed()
    );
    Ok(mount)
}

fn load_add_ons(
    path: &Path,
    resolved: &ResolvedTitle,
    keys: Option<&dyn NcaKeyProvider>,
) -> Result<Vec<AddOnContent>, LaunchError> {
    let base_id = resolved
        .control_metadata()
        .map(|control| control.nacp.add_on_content_base_id)
        .filter(|value| *value != 0);
    let mut indices = BTreeSet::new();
    let mut content_ids = BTreeSet::new();
    let mut result = Vec::with_capacity(resolved.add_ons.len());
    for package in &resolved.add_ons {
        if let CnmtExtendedHeader::AddOnContent { data_patch_id, .. } =
            &package.canonical_content_meta().extended_header
            && *data_patch_id != 0
        {
            return Err(LaunchError::invalid(
                LaunchStage::AddOnContent,
                path,
                format!("add-on data patch {data_patch_id:016X} is not present in ResolvedTitle"),
            )
            .title_id(package.title_id));
        }
        let mut data = records(package, CnmtContentType::Data);
        data.sort_by_key(|record| (record.id_offset, record.content_id));
        if data.is_empty() || data.len() > MAX_ADD_ON_MOUNTS {
            return Err(LaunchError::invalid(
                LaunchStage::AddOnContent,
                path,
                format!("add-on declares {} Data contents", data.len()),
            )
            .title_id(package.title_id));
        }
        let mut mounts = Vec::with_capacity(data.len());
        for record in data {
            if !content_ids.insert(record.content_id) {
                return Err(LaunchError::invalid(
                    LaunchStage::AddOnContent,
                    path,
                    "duplicate add-on Data content identity",
                )
                .title_id(package.title_id)
                .content(record.content_id));
            }
            let nca = package.open_content(record, keys).map_err(|error| {
                LaunchError::load(LaunchStage::AddOnContent, path, error)
                    .title_id(package.title_id)
                    .content(record.content_id)
            })?;
            mounts.push(
                mount_nca(path, &nca, None, MountProvenance::AddOn, record.content_id).map_err(
                    |mut error| {
                        error.stage = LaunchStage::AddOnContent;
                        error.title_id = Some(package.title_id);
                        error.content_id = Some(record.content_id);
                        error
                    },
                )?,
            );
        }
        let horizon_index = base_id
            .map(|base| {
                package
                    .title_id
                    .get()
                    .checked_sub(base)
                    .and_then(|value| u32::try_from(value).ok())
                    .ok_or_else(|| {
                        LaunchError::invalid(
                            LaunchStage::AddOnContent,
                            path,
                            "add-on title ID is incompatible with NACP add-on base ID",
                        )
                        .title_id(package.title_id)
                    })
            })
            .transpose()?;
        if let Some(index) = horizon_index
            && !indices.insert(index)
        {
            return Err(LaunchError::invalid(
                LaunchStage::AddOnContent,
                path,
                "duplicate Horizon add-on index",
            )
            .title_id(package.title_id));
        }
        result.push(AddOnContent::new(
            package.title_id,
            package.version,
            horizon_index,
            mounts,
        ));
    }
    Ok(result)
}

fn records(package: &PackageMetadata, content_type: CnmtContentType) -> Vec<&CnmtContentInfo> {
    package
        .canonical_content_meta()
        .contents
        .iter()
        .filter(|record| record.content_type == content_type)
        .collect()
}

fn exactly_one_record<'a>(
    package: &'a PackageMetadata,
    content_type: CnmtContentType,
    label: &str,
) -> Result<&'a CnmtContentInfo, String> {
    let records = records(package, content_type);
    match records.as_slice() {
        [record] => Ok(*record),
        _ => Err(format!(
            "canonical metadata contains {} {label} records; expected one",
            records.len()
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    use crate::{LaunchKind, LaunchModuleImage};

    fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn minimal_nro() -> Vec<u8> {
        let mut bytes = vec![0; 0x2800];
        bytes[0x10..0x14].copy_from_slice(b"NRO0");
        put_u32(&mut bytes, 0x18, 0x2800);
        put_u32(&mut bytes, 0x20, 0);
        put_u32(&mut bytes, 0x24, 0x1000);
        put_u32(&mut bytes, 0x28, 0x1000);
        put_u32(&mut bytes, 0x2c, 0x1000);
        put_u32(&mut bytes, 0x30, 0x2000);
        put_u32(&mut bytes, 0x34, 0x800);
        put_u32(&mut bytes, 0x38, 0x800);
        bytes[0x40..0x60].fill(0x5a);
        bytes
    }

    #[test]
    fn module_roles_have_stable_dependency_order() {
        let mut roles = [
            module_role("sdk").unwrap(),
            module_role("subsdk12").unwrap(),
            module_role("main").unwrap(),
            module_role("rtld").unwrap(),
            module_role("subsdk2").unwrap(),
        ];
        roles.sort();
        assert_eq!(
            roles,
            [
                ModuleRole::RuntimeLoader,
                ModuleRole::Main,
                ModuleRole::SubSdk(2),
                ModuleRole::SubSdk(12),
                ModuleRole::Sdk,
            ]
        );
        assert_eq!(module_role("subsdk"), None);
        assert_eq!(module_role("subsdk-1"), None);
        assert_eq!(module_role("other"), None);
    }

    #[test]
    fn standalone_nro_builds_an_explicit_homebrew_plan() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("demo.NRO");
        fs::write(&path, minimal_nro()).unwrap();
        let plan = Launcher::build(LauncherInput::new(&path)).unwrap();
        assert!(matches!(plan.kind(), LaunchKind::Homebrew));
        assert!(plan.packaged_identity().is_none());
        assert!(plan.effective_policy().is_none());
        assert!(plan.primary_file_system().is_none());
        assert!(plan.add_ons().is_empty());
        assert_eq!(plan.modules().len(), 1);
        assert_eq!(plan.entry_module().role(), ModuleRole::Homebrew);
        assert!(matches!(
            plan.entry_module().image(),
            LaunchModuleImage::Nro(_)
        ));
    }

    #[test]
    fn path_detection_rejects_missing_and_unsupported_inputs() {
        let directory = tempfile::tempdir().unwrap();
        let missing = directory.path().join("missing.nro");
        let error = Launcher::build(LauncherInput::new(&missing)).unwrap_err();
        assert_eq!(error.stage(), LaunchStage::PathDetection);
        let unsupported = directory.path().join("readme.txt");
        fs::write(&unsupported, b"not executable").unwrap();
        let error = Launcher::build(LauncherInput::new(&unsupported)).unwrap_err();
        assert_eq!(error.stage(), LaunchStage::PathDetection);
    }
}
