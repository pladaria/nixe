//! Controller-support configuration through the virtual HID controller owner,
//! following Eden's default frontend, independently of host-device attachment.
//! https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/core/frontend/applets/controller.cpp
//! https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/core/hle/service/am/frontend/applet_controller.cpp
//! Wire ABI:
//! https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/applets/hid_la.h
//! https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/applets/hid_la.c
use super::super::prelude::*;
use crate::object::LibraryAppletLaunchRequest;

fn unsupported(detail: &'static str) -> IpcWireError {
    IpcWireError::UnsupportedService(UnsupportedServiceOperation::CommandVariant {
        service: "ControllerSupport applet",
        command_id: 10,
        detail,
    })
}

pub(super) fn run(
    launch: &LibraryAppletLaunchRequest,
    hid: &HidSystem,
) -> Result<Vec<u8>, IpcWireError> {
    if launch.mode != LibraryAppletMode::AllForeground {
        return Err(unsupported("ControllerSupport requires AllForeground mode"));
    }
    let keep_connected = decode(&launch.input_storages)?;
    let selected =
        hid.configure_full_key_controller(keep_connected)
            .map_err(|error| match error {
                crate::HidInputError::Memory(error) => IpcWireError::SystemMemory(error),
                crate::HidInputError::Vibration(error) => {
                    IpcWireError::InputBackend(error.to_string().into_boxed_str())
                }
            })?;
    let mut output = vec![0; 12];
    output[0] = 1;
    output[4..8].copy_from_slice(&selected.to_le_bytes());
    log::debug!(
        "controller applet configured virtual FullKey player: id={selected:#x} keep-connected={keep_connected}"
    );
    Ok(output)
}

fn decode(inputs: &[Vec<u8>]) -> Result<bool, IpcWireError> {
    let [common, private, arg] = inputs else {
        return Err(IpcWireError::Malformed(
            "ControllerSupport requires common, private and argument storages",
        ));
    };
    if common.len() != 0x20
        || request_u32(common, 0) != Some(1)
        || request_u32(common, 4) != Some(0x20)
        || private.len() != 0x14
        || request_u32(private, 0) != Some(0x14)
        || request_u32(private, 4) != u32::try_from(arg.len()).ok()
    {
        return Err(IpcWireError::Malformed(
            "invalid ControllerSupport storage header or size",
        ));
    }
    let version = request_u32(common, 8).unwrap();
    let (arg_size, players) = match version {
        3..=5 => (0x21c, 4),
        7..=9 => (0x430, 8),
        _ => return Err(unsupported("unsupported ControllerSupport API version")),
    };
    if arg.len() != arg_size {
        return Err(IpcWireError::Malformed(
            "invalid ControllerSupport argument size",
        ));
    }
    if private[8..12] != [0; 4] {
        return Err(unsupported(
            "system UI, strap guide, firmware update or key remapping is required",
        ));
    }
    if request_u32(private, 12).unwrap() & 1 == 0 {
        return Err(unsupported(
            "requested controller styles exclude the supported FullKey controller",
        ));
    }
    let single = arg[5] != 0;
    let min = arg[0] as i8;
    let max = arg[1] as i8;
    if !single && (min < 0 || max < 1 || min > max || max > players) {
        return Err(IpcWireError::Malformed(
            "invalid ControllerSupport player count range",
        ));
    }
    if !single && min > 1 {
        return Err(unsupported("additional player connections are required"));
    }
    Ok(arg[2] != 0)
}
