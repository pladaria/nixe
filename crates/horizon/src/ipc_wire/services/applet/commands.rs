//! Semantic command identities for the applet service family.

use crate::object::{AppletObject, AppletProxyKind};

pub(super) const PERFORMANCE_MODE_NORMAL: u32 = 0;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum RootCommand {
    OpenApplicationProxy,
    OpenSystemAppletProxy,
}

impl RootCommand {
    pub(super) const fn decode(command_id: u32) -> Option<Self> {
        match command_id {
            0 => Some(Self::OpenApplicationProxy),
            100 => Some(Self::OpenSystemAppletProxy),
            _ => None,
        }
    }

    pub(super) const fn proxy_kind(self) -> AppletProxyKind {
        match self {
            Self::OpenApplicationProxy => AppletProxyKind::Application,
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

    pub(super) const fn child(self, kind: AppletProxyKind) -> AppletObject {
        match self {
            Self::CommonStateGetter => AppletObject::CommonStateGetter,
            Self::SelfController => AppletObject::SelfController,
            Self::WindowController => AppletObject::WindowController,
            Self::AudioController => AppletObject::AudioController,
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
});

command_enum!(SelfControllerCommand {
    0 => Exit,
    1 => LockExit,
    2 => UnlockExit,
    9 => GetLibraryAppletLaunchableEvent,
    11 => SetOperationModeChangedNotification,
    12 => SetPerformanceModeChangedNotification,
    13 => SetFocusHandlingMode,
    16 => SetOutOfFocusSuspendingEnabled,
    40 => CreateManagedDisplayLayer,
});

command_enum!(WindowControllerCommand {
    1 => GetAppletResourceUserId,
    10 => AcquireForegroundRights,
});

command_enum!(ApplicationFunctionsCommand {
    22 => SetTerminateResult,
    40 => NotifyRunning,
});

command_enum!(LibraryAppletCreatorCommand {
    0 => CreateLibraryApplet,
    10 => CreateStorage,
});

command_enum!(LibraryAppletAccessorCommand {
    0 => GetAppletStateChangedEvent,
    10 => Start,
    100 => PushInData,
});

command_enum!(StorageCommand { 0 => Open });

command_enum!(StorageAccessorCommand {
    0 => GetSize,
    10 => Write,
    11 => Read,
});
