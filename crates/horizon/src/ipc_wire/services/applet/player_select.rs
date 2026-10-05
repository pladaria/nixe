//! HLE user selection uses the local profile selected for application launch.
//! Selection still obeys the applet's exclusion list and account requirements;
//! profile creation, editing and online qualification are separate operations.
//! https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/applets/psel.h
//! https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/applets/psel.c
use super::super::prelude::*;
use crate::{object::LibraryAppletLaunchRequest, user::UserIdentity};

fn unsupported(detail: &'static str) -> IpcWireError {
    IpcWireError::UnsupportedService(UnsupportedServiceOperation::CommandVariant {
        service: "PlayerSelect applet",
        command_id: 10,
        detail,
    })
}

pub(super) fn run(
    launch: &LibraryAppletLaunchRequest,
    user: UserIdentity,
) -> Result<Vec<u8>, IpcWireError> {
    if launch.mode != LibraryAppletMode::AllForeground {
        return Err(unsupported("PlayerSelect requires AllForeground mode"));
    }
    let [common, arg] = launch.input_storages.as_slice() else {
        return Err(IpcWireError::Malformed(
            "PlayerSelect requires common and settings storages",
        ));
    };
    if common.len() != 32 || request_u32(common, 0) != Some(1) || request_u32(common, 4) != Some(32)
    {
        return Err(IpcWireError::Malformed(
            "invalid PlayerSelect common arguments",
        ));
    }
    let version = request_u32(common, 8).unwrap();
    let size = match version {
        1 => 0x98,
        0x10000 | 0x20000 => 0xa0,
        _ => return Err(unsupported("unsupported PlayerSelect API version")),
    };
    if arg.len() != size {
        return Err(IpcWireError::Malformed(
            "invalid PlayerSelect settings size",
        ));
    }
    if request_u32(arg, 0) != Some(0) {
        return Err(unsupported(
            "profile creation, editing and online account dialogs are not implemented",
        ));
    }
    if arg[0x90] != 0 {
        return Err(unsupported("an online-linked account is required"));
    }
    let uid = user.id().encode();
    if arg[8..0x88]
        .chunks_exact(16)
        .any(|excluded| excluded == uid)
    {
        return Err(unsupported(
            "the selected local profile is excluded by the application",
        ));
    }
    // PselUiReturnArg aligns the 16-byte AccountUid to 8 bytes after Result.
    // This is the configured user's selection, not an invented online account
    // or a blanket successful result for every playerSelect mode.
    let mut output = vec![0; 24];
    output[8..24].copy_from_slice(&uid);
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn configured_user_selection_obeys_exclusions_and_online_requirements() {
        let mut common = vec![0; 32];
        common[..4].copy_from_slice(&1u32.to_le_bytes());
        common[4..8].copy_from_slice(&32u32.to_le_bytes());
        common[8..12].copy_from_slice(&0x20000u32.to_le_bytes());
        let mut launch = LibraryAppletLaunchRequest {
            applet_id: LibraryAppletId::PlayerSelect,
            mode: LibraryAppletMode::AllForeground,
            input_storages: vec![common, vec![0; 0xa0]],
        };
        let output = run(&launch, crate::user::DEFAULT_USER).unwrap();
        assert_eq!(output.len(), 24);
        assert_eq!(&output[8..24], &crate::user::DEFAULT_USER.id().encode());
        launch.input_storages[1][8..24].copy_from_slice(&crate::user::DEFAULT_USER.id().encode());
        assert!(run(&launch, crate::user::DEFAULT_USER).is_err());
        launch.input_storages[1][8..24].fill(0);
        launch.input_storages[1][0x90] = 1;
        assert!(run(&launch, crate::user::DEFAULT_USER).is_err());
    }
}
