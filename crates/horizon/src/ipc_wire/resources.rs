//! Service subsystem owners shared by sessions independently of CPU execution.
use std::sync::Arc;

#[derive(Debug)]
pub(crate) struct GraphicsServices {
    pub video: crate::VideoSystem,
}

#[derive(Debug)]
pub(crate) struct AudioServices {
    pub backend: Option<Arc<dyn nixe_audio::AudioBackend>>,
    pub renderers: crate::AudioRendererManagerSession,
    pub opus: crate::OpusManagerSession,
}

#[derive(Debug)]
pub(crate) struct FileServices {
    pub save_data: Option<crate::SaveDataSystem>,
    pub shared_fonts: crate::SharedFontSession,
    pub archives: Arc<crate::SystemArchives>,
}

#[derive(Debug)]
pub(crate) struct SystemServices {
    pub operation_mode: crate::OperationMode,
    pub time: crate::TimeEnvironment,
    pub hid: crate::HidSystem,
    pub keyboard: Option<nixe_applets::keyboard::KeyboardHost>,
    pub settings: crate::SettingsEnvironment,
    pub application_language: Option<crate::SystemLanguage>,
    pub application_display_version: Option<[u8; 16]>,
    pub user_account_switch_locked: Option<bool>,
    pub diagnostics: crate::HorizonDiagnostics,
}

#[derive(Debug)]
pub(crate) struct ServiceResources {
    pub payloads: super::payload::PayloadBudgets,
    pub graphics: GraphicsServices,
    pub audio: AudioServices,
    pub files: FileServices,
    pub system: SystemServices,
}
