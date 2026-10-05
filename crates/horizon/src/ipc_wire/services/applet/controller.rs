//! Controller-support completion when the existing HID connection satisfies
//! the request. Cases requiring pairing, firmware changes or UI remain explicit.
//! The ABI and automatic application-mode completion are documented by libnx:
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
    let selected = hid.connected_full_key_id();
    complete(&launch.input_storages, selected)
}

fn complete(inputs: &[Vec<u8>], selected: Option<u32>) -> Result<Vec<u8>, IpcWireError> {
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
        7 | 8 => (0x430, 8),
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
            "requested controller styles exclude the connected FullKey controller",
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
    if arg[2] == 0 {
        return Err(unsupported(
            "controller disconnection and reconnection UI is required",
        ));
    }
    let selected = selected
        .ok_or_else(|| unsupported("no controller is connected; connection UI is required"))?;
    if !single && min > 1 {
        return Err(unsupported("additional player connections are required"));
    }
    // One published FullKey controller, selected by its real NPad ID. Do not
    // invent a connection or report a count that does not exist in shared HID.
    let mut output = vec![0; 12];
    output[0] = 1;
    output[4..8].copy_from_slice(&selected.to_le_bytes());
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn inputs(version: u32, size: usize) -> Vec<Vec<u8>> {
        let mut common = vec![0; 32];
        common[0..4].copy_from_slice(&1u32.to_le_bytes());
        common[4..8].copy_from_slice(&32u32.to_le_bytes());
        common[8..12].copy_from_slice(&version.to_le_bytes());
        let mut private = vec![0; 20];
        private[..4].copy_from_slice(&20u32.to_le_bytes());
        private[4..8].copy_from_slice(&(size as u32).to_le_bytes());
        private[12..16].copy_from_slice(&1u32.to_le_bytes());
        let mut arg = vec![0; size];
        arg[..7].copy_from_slice(&[1, 4, 1, 1, 1, 0, 0]);
        vec![common, private, arg]
    }
    #[test]
    fn completion_reports_the_connected_id_and_requires_actual_connections() {
        for (version, size) in [(3, 0x21c), (5, 0x21c), (7, 0x430), (8, 0x430)] {
            let mut input = inputs(version, size);
            assert_eq!(
                complete(&input, Some(0)).unwrap(),
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            );
            assert!(complete(&input, None).is_err());
            input[2][0] = 2;
            assert!(complete(&input, Some(0)).is_err());
        }
    }
}
