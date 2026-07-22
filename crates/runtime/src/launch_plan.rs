use std::fmt::{Debug, Formatter};
use std::sync::Arc;

use nixe_loader_content::{ApplicationVersion, RomFsArchive};
use nixe_loader_executable::{EffectiveNpdmPolicy, Npdm, NroImage, NsoImage};
use nixe_loader_title::{ApplicationId, ControlMetadata, TitleId};

/// Maximum executable modules retained by one launch plan.
pub const MAX_LAUNCH_MODULES: usize = 64;
/// Maximum resolved add-on titles retained by one launch plan.
pub const MAX_LAUNCH_ADD_ONS: usize = 2_000;
/// Maximum Data contents retained for one add-on title.
pub const MAX_ADD_ON_MOUNTS: usize = 64;

/// Security and content identity for a packaged application.
#[derive(Clone)]
pub struct PackagedIdentity {
    application_id: ApplicationId,
    effective_title_id: TitleId,
    effective_version: ApplicationVersion,
    program_content_id: [u8; 16],
    npdm: Npdm,
}

impl Debug for PackagedIdentity {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PackagedIdentity")
            .field("application_id", &self.application_id)
            .field("effective_title_id", &self.effective_title_id)
            .field("effective_version", &self.effective_version)
            .field("program_content_id", &self.program_content_id)
            .field(
                "npdm_program_id",
                &format_args!("{:016X}", self.npdm.program_id()),
            )
            .field("process_name", &self.npdm.name_str())
            .field("product_code", &self.npdm.product_code_str())
            .finish_non_exhaustive()
    }
}

impl PackagedIdentity {
    pub(crate) fn new(
        application_id: ApplicationId,
        effective_title_id: TitleId,
        effective_version: ApplicationVersion,
        program_content_id: [u8; 16],
        npdm: Npdm,
    ) -> Self {
        Self {
            application_id,
            effective_title_id,
            effective_version,
            program_content_id,
            npdm,
        }
    }

    pub const fn application_id(&self) -> ApplicationId {
        self.application_id
    }
    pub const fn effective_title_id(&self) -> TitleId {
        self.effective_title_id
    }
    pub const fn effective_version(&self) -> ApplicationVersion {
        self.effective_version
    }
    pub const fn program_content_id(&self) -> &[u8; 16] {
        &self.program_content_id
    }
    pub const fn npdm(&self) -> &Npdm {
        &self.npdm
    }
    pub const fn effective_policy(&self) -> &EffectiveNpdmPolicy {
        self.npdm.effective_policy()
    }
}

/// Distinguishes official packaged software from standalone homebrew.
#[derive(Debug)]
pub enum LaunchKind {
    Packaged(Box<PackagedIdentity>),
    Homebrew,
}

/// Stable semantic role of one executable module.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ModuleRole {
    RuntimeLoader,
    Main,
    SubSdk(u16),
    Sdk,
    Homebrew,
}

/// A validated relocatable executable retained without shadow segment metadata.
pub enum LaunchModuleImage {
    Nso(NsoImage),
    Nro(NroImage),
}

impl Debug for LaunchModuleImage {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Nso(image) => formatter.debug_tuple("Nso").field(image).finish(),
            Self::Nro(image) => formatter.debug_tuple("Nro").field(image).finish(),
        }
    }
}

/// One module in deterministic dependency/load order.
#[derive(Debug)]
pub struct LaunchModule {
    name: Box<str>,
    role: ModuleRole,
    image: LaunchModuleImage,
}

impl LaunchModule {
    pub(crate) fn new(name: Box<str>, role: ModuleRole, image: LaunchModuleImage) -> Self {
        Self { name, role, image }
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub const fn role(&self) -> ModuleRole {
        self.role
    }
    pub const fn image(&self) -> &LaunchModuleImage {
        &self.image
    }
}

/// Diagnostic provenance of an effective read-only filesystem.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MountProvenance {
    Base,
    Patch,
    BaseAndPatch,
    AddOn,
    AddOnPatch,
    HomebrewAsset,
}

/// Lazy read-only RomFS view used by later process mount creation.
#[derive(Clone, Debug)]
pub struct ReadOnlyMount {
    provenance: MountProvenance,
    content_id: Option<[u8; 16]>,
    romfs: Arc<RomFsArchive>,
}

impl ReadOnlyMount {
    pub(crate) fn new(
        provenance: MountProvenance,
        content_id: Option<[u8; 16]>,
        romfs: RomFsArchive,
    ) -> Self {
        Self {
            provenance,
            content_id,
            romfs: Arc::new(romfs),
        }
    }
    pub const fn provenance(&self) -> MountProvenance {
        self.provenance
    }
    /// Returns canonical NCA content identity, or `None` for an NRO asset.
    pub const fn content_id(&self) -> Option<&[u8; 16]> {
        self.content_id.as_ref()
    }
    pub fn romfs(&self) -> &RomFsArchive {
        &self.romfs
    }
}

/// One resolved add-on title and its canonically ordered Data mounts.
#[derive(Clone, Debug)]
pub struct AddOnContent {
    title_id: TitleId,
    version: ApplicationVersion,
    horizon_index: Option<u32>,
    mounts: Box<[ReadOnlyMount]>,
}

impl AddOnContent {
    pub(crate) fn new(
        title_id: TitleId,
        version: ApplicationVersion,
        horizon_index: Option<u32>,
        mounts: Vec<ReadOnlyMount>,
    ) -> Self {
        Self {
            title_id,
            version,
            horizon_index,
            mounts: mounts.into_boxed_slice(),
        }
    }
    pub const fn title_id(&self) -> TitleId {
        self.title_id
    }
    pub const fn version(&self) -> ApplicationVersion {
        self.version
    }
    pub const fn horizon_index(&self) -> Option<u32> {
        self.horizon_index
    }
    pub fn mounts(&self) -> &[ReadOnlyMount] {
        &self.mounts
    }
}

/// Immutable, format-independent boundary consumed by future process creation.
pub struct LaunchPlan {
    kind: LaunchKind,
    modules: Box<[LaunchModule]>,
    entry_module: usize,
    primary_file_system: Option<ReadOnlyMount>,
    add_ons: Box<[AddOnContent]>,
    control: Option<ControlMetadata>,
}

impl LaunchPlan {
    pub(crate) fn new(
        kind: LaunchKind,
        modules: Vec<LaunchModule>,
        entry_module: usize,
        primary_file_system: Option<ReadOnlyMount>,
        add_ons: Vec<AddOnContent>,
        control: Option<ControlMetadata>,
    ) -> Self {
        Self {
            kind,
            modules: modules.into_boxed_slice(),
            entry_module,
            primary_file_system,
            add_ons: add_ons.into_boxed_slice(),
            control,
        }
    }

    pub const fn kind(&self) -> &LaunchKind {
        &self.kind
    }
    pub fn packaged_identity(&self) -> Option<&PackagedIdentity> {
        match &self.kind {
            LaunchKind::Packaged(identity) => Some(identity.as_ref()),
            LaunchKind::Homebrew => None,
        }
    }
    pub fn modules(&self) -> &[LaunchModule] {
        &self.modules
    }
    pub fn entry_module(&self) -> &LaunchModule {
        &self.modules[self.entry_module]
    }
    /// Returns the entry module's stable index in [`Self::modules`].
    pub const fn entry_module_index(&self) -> usize {
        self.entry_module
    }
    pub const fn primary_file_system(&self) -> Option<&ReadOnlyMount> {
        self.primary_file_system.as_ref()
    }
    pub fn add_ons(&self) -> &[AddOnContent] {
        &self.add_ons
    }
    pub const fn control_metadata(&self) -> Option<&ControlMetadata> {
        self.control.as_ref()
    }
    pub fn effective_policy(&self) -> Option<&EffectiveNpdmPolicy> {
        self.packaged_identity()
            .map(PackagedIdentity::effective_policy)
    }
}

impl Debug for LaunchPlan {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LaunchPlan")
            .field("kind", &self.kind)
            .field("modules", &self.modules)
            .field("entry_module", &self.entry_module)
            .field("primary_file_system", &self.primary_file_system)
            .field("add_ons", &self.add_ons)
            .field("control", &self.control)
            .finish()
    }
}
