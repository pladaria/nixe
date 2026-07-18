use swiitx_loader_content::{
    ApplicationControlProperty, CnmtContentInfo, NacpLanguage, SupportedLanguages,
};
use swiitx_loader_storage::{LoadError, StorageRef};

const MAX_ICON_SIZE: u64 = 16 * 1024 * 1024;

/// Parsed application control metadata and its localized icon files.
#[derive(Clone, Debug)]
pub struct ControlMetadata {
    pub nacp: ApplicationControlProperty,
    pub content_id: [u8; 16],
    pub declared_content_size: u64,
    pub declared_content_hash: [u8; 32],
    icons: Vec<ControlIcon>,
}

impl ControlMetadata {
    pub(crate) fn new(
        nacp: ApplicationControlProperty,
        content: &CnmtContentInfo,
        icons: Vec<ControlIcon>,
    ) -> Self {
        Self {
            nacp,
            content_id: content.content_id,
            declared_content_size: content.size,
            declared_content_hash: content.hash,
            icons,
        }
    }

    pub const fn supported_languages(&self) -> SupportedLanguages {
        self.nacp.supported_languages
    }

    pub fn icons(&self) -> &[ControlIcon] {
        &self.icons
    }

    pub fn icon(&self, language: NacpLanguage) -> Option<&ControlIcon> {
        self.icons.iter().find(|icon| icon.language == language)
    }

    /// Chooses the first requested icon, then American English, then the first
    /// icon in canonical language order.
    pub fn preferred_icon(&self, preferences: &[NacpLanguage]) -> Option<&ControlIcon> {
        preferences
            .iter()
            .copied()
            .chain([NacpLanguage::AmericanEnglish])
            .chain(NacpLanguage::ALL)
            .find_map(|language| self.icon(language))
    }
}

impl PartialEq for ControlMetadata {
    fn eq(&self, other: &Self) -> bool {
        self.nacp == other.nacp
            && self.content_id == other.content_id
            && self.declared_content_size == other.declared_content_size
            && self.declared_content_hash == other.declared_content_hash
            && self.icons == other.icons
    }
}

impl Eq for ControlMetadata {}

/// One localized JPEG icon retained as a bounded RomFS storage view.
#[derive(Clone)]
pub struct ControlIcon {
    pub language: NacpLanguage,
    pub filename: String,
    pub size: u64,
    storage: StorageRef,
}

impl ControlIcon {
    pub(crate) fn load(
        language: NacpLanguage,
        filename: String,
        storage: StorageRef,
    ) -> Result<Self, LoadError> {
        let size = storage.len()?;
        if !(3..=MAX_ICON_SIZE).contains(&size) {
            return Err(LoadError::invalid(
                "Control NCA",
                format!("icon {filename} has invalid size {size}"),
            ));
        }
        let mut signature = [0_u8; 3];
        storage.read_at(0, &mut signature)?;
        if signature != [0xFF, 0xD8, 0xFF] {
            return Err(LoadError::invalid(
                "Control NCA",
                format!("icon {filename} is not a JPEG image"),
            ));
        }
        Ok(Self {
            language,
            filename,
            size,
            storage,
        })
    }

    pub fn storage(&self) -> StorageRef {
        self.storage.clone()
    }

    pub fn bytes(&self) -> Result<Vec<u8>, LoadError> {
        let size = usize::try_from(self.size)
            .map_err(|_| LoadError::invalid("Control NCA", "icon size is invalid"))?;
        let mut bytes = vec![0_u8; size];
        self.storage.read_at(0, &mut bytes)?;
        Ok(bytes)
    }
}

impl PartialEq for ControlIcon {
    fn eq(&self, other: &Self) -> bool {
        self.language == other.language
            && self.filename == other.filename
            && self.size == other.size
    }
}

impl Eq for ControlIcon {}

impl std::fmt::Debug for ControlIcon {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ControlIcon")
            .field("language", &self.language)
            .field("filename", &self.filename)
            .field("size", &self.size)
            .field("storage", &"<storage>")
            .finish()
    }
}
