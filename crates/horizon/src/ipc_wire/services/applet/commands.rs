//! Semantic command identities for the applet service family.

use crate::object::{AppletObject, AppletProxyKind};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum RootCommand {
    OpenApplicationProxy,
    OpenApplicationProxy2,
    OpenSystemAppletProxy,
}

impl RootCommand {
    pub(super) const fn decode(command_id: u32) -> Option<Self> {
        match command_id {
            0 => Some(Self::OpenApplicationProxy),
            10 => Some(Self::OpenApplicationProxy2),
            100 => Some(Self::OpenSystemAppletProxy),
            _ => None,
        }
    }

    pub(super) const fn proxy_kind(self) -> AppletProxyKind {
        match self {
            Self::OpenApplicationProxy | Self::OpenApplicationProxy2 => {
                AppletProxyKind::Application
            }
            Self::OpenSystemAppletProxy => AppletProxyKind::SystemApplet,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ProxyCommand {
    CommonStateGetter,
    SelfController,
    WindowController,
    AudioController,
    DisplayController,
    LibraryAppletCreator,
    DebugFunctions,
    RoleFunctions,
    GlobalStateController,
    ApplicationCreator,
    AppletCommonFunctions,
    GraphicsScalingParametersFunctions,
    PerformanceFunctions,
}

impl ProxyCommand {
    pub(super) const fn decode(kind: AppletProxyKind, command_id: u32) -> Option<Self> {
        match command_id {
            0 => Some(Self::CommonStateGetter),
            1 => Some(Self::SelfController),
            2 => Some(Self::WindowController),
            3 => Some(Self::AudioController),
            4 => Some(Self::DisplayController),
            11 => Some(Self::LibraryAppletCreator),
            20 => Some(Self::RoleFunctions),
            40 if matches!(kind, AppletProxyKind::Application) => Some(Self::PerformanceFunctions),
            60 if matches!(kind, AppletProxyKind::Application) => {
                Some(Self::GraphicsScalingParametersFunctions)
            }
            21 if matches!(kind, AppletProxyKind::SystemApplet) => {
                Some(Self::GlobalStateController)
            }
            22 if matches!(kind, AppletProxyKind::SystemApplet) => Some(Self::ApplicationCreator),
            23 if matches!(kind, AppletProxyKind::SystemApplet) => {
                Some(Self::AppletCommonFunctions)
            }
            1000 => Some(Self::DebugFunctions),
            _ => None,
        }
    }

    pub(super) fn child(self, kind: AppletProxyKind) -> AppletObject {
        match self {
            Self::CommonStateGetter => AppletObject::CommonStateGetter,
            Self::SelfController => AppletObject::SelfController,
            Self::WindowController => AppletObject::WindowController,
            Self::AudioController => AppletObject::AudioController(std::sync::Arc::new(
                std::sync::Mutex::new(crate::object::AppletAudioController::default()),
            )),
            Self::PerformanceFunctions => AppletObject::PerformanceFunctions,
            Self::GraphicsScalingParametersFunctions => {
                AppletObject::GraphicsScalingParametersFunctions
            }
            Self::DisplayController => AppletObject::DisplayController,
            Self::LibraryAppletCreator => AppletObject::LibraryAppletCreator,
            Self::DebugFunctions => AppletObject::DebugFunctions,
            Self::RoleFunctions => match kind {
                AppletProxyKind::Application => AppletObject::ApplicationFunctions,
                AppletProxyKind::SystemApplet => AppletObject::HomeMenuFunctions,
            },
            Self::GlobalStateController => AppletObject::GlobalStateController,
            Self::ApplicationCreator => AppletObject::ApplicationCreator,
            Self::AppletCommonFunctions => AppletObject::AppletCommonFunctions,
        }
    }
}

macro_rules! command_enum {
    ($name:ident { $($id:literal => $variant:ident),+ $(,)? }) => {
        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        pub(super) enum $name {
            $($variant),+
        }

        impl $name {
            pub(super) const fn decode(command_id: u32) -> Option<Self> {
                match command_id {
                    $($id => Some(Self::$variant),)+
                    _ => None,
                }
            }
        }
    };
}

command_enum!(CommonStateGetterCommand {
    0 => GetEventHandle,
    1 => ReceiveMessage,
    5 => GetOperationMode,
    6 => GetPerformanceMode,
    9 => GetCurrentFocusState,
    50 => IsVrModeEnabled,
    51 => SetVrModeEnabled,
    53 => BeginVrModeEx,
    54 => EndVrModeEx,
    60 => GetDefaultDisplayResolution,
    66 => SetCpuBoostMode,
    67 => CancelCpuBoostMode,
    900 => SetRequestExitToLibraryAppletAtExecuteNextProgramEnabled,
});

command_enum!(SelfControllerCommand {
    0 => Exit,
    1 => LockExit,
    2 => UnlockExit,
    9 => GetLibraryAppletLaunchableEvent,
    11 => SetOperationModeChangedNotification,
    12 => SetPerformanceModeChangedNotification,
    13 => SetFocusHandlingMode,
    14 => SetRestartMessageEnabled,
    16 => SetOutOfFocusSuspendingEnabled,
    19 => SetAlbumImageOrientation,
    40 => CreateManagedDisplayLayer,
    44 => CreateManagedDisplaySeparableLayer,
    62 => SetIdleTimeDetectionExtension,
    63 => GetIdleTimeDetectionExtension,
    90 => GetAccumulatedSuspendedTickValue,
    91 => GetAccumulatedSuspendedTickChangedEvent,
    2000 => GetGraphicsSupervisor,
    2400 => SetRtcScreenSharingAudioEnabled,
});

command_enum!(WindowControllerCommand {
    1 => GetAppletResourceUserId,
    10 => AcquireForegroundRights,
});

command_enum!(AudioControllerCommand {
    0 => SetExpectedMasterVolume,
    1 => GetMainAppletExpectedMasterVolume,
    2 => GetLibraryAppletExpectedMasterVolume,
    3 => ChangeMainAppletMasterVolume,
    4 => SetTransparentVolumeRate,
});

command_enum!(ApplicationFunctionsCommand {
    20 => EnsureSaveData,
    1 => PopLaunchParameter,
    21 => GetDesiredLanguage,
    22 => SetTerminateResult,
    23 => GetDisplayVersion,
    40 => NotifyRunning,
    50 => GetPseudoDeviceId,
    67 => SetGamePlayRecordingState,
    110 => QueryApplicationPlayStatistics,
    111 => QueryApplicationPlayStatisticsByUid,
    130 => GetGpuErrorDetectedSystemEvent,
    320 => Query320,
    321 => Query321,
});

command_enum!(LibraryAppletCreatorCommand {
    0 => LibraryApplet,
    10 => Storage,
    11 => TransferMemoryStorage,
});

// https://switchbrew.org/wiki/Applet_Manager_services#ILibraryAppletAccessor
command_enum!(LibraryAppletAccessorCommand {
    0 => GetAppletStateChangedEvent,
    1 => CheckFinished,
    10 => Start,
    30 => GetResult,
    100 => PushInData,
    101 => PopOutData,
    103 => PushInteractiveInData,
    104 => PopInteractiveOutData,
    105 => GetPopOutDataEvent,
    106 => GetPopInteractiveOutDataEvent,
});

command_enum!(StorageCommand { 0 => Open });

command_enum!(StorageAccessorCommand {
    0 => GetSize,
    10 => Write,
    11 => Read,
});
