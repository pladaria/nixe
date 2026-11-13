//! Service subsystem owners shared by sessions independently of CPU execution.
use std::sync::Arc;

#[derive(Debug)]
pub(crate) struct GraphicsServices {
    pub video: Option<crate::VideoSystem>,
}

impl GraphicsServices {
    pub fn require_video(
        &self,
        service: &'static str,
        command_id: u32,
    ) -> Result<&crate::VideoSystem, super::IpcWireError> {
        self.video
            .as_ref()
            .ok_or(super::IpcWireError::UnsupportedService(
                super::UnsupportedServiceOperation::CommandVariant {
                    service,
                    command_id,
                    detail: "the selected platform's graphics services are not implemented",
                },
            ))
    }
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
    pub profiles: Arc<crate::user::ProfileManager>,
    pub operation_mode: crate::OperationMode,
    pub time: crate::TimeEnvironment,
    pub hid: crate::HidSystem,
    pub keyboard: Option<nixe_applets::keyboard::KeyboardHost>,
    pub settings: crate::SettingsEnvironment,
    pub application_language: Option<crate::SystemLanguage>,
    pub application_display_version: Option<[u8; 16]>,
    pub application_pseudo_device_seed: Option<u64>,
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
