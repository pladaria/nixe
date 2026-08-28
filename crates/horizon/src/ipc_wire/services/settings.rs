use super::super::*;

const FIRMWARE_VERSION_SIZE: usize = 0x100;

fn emulated_firmware_version() -> [u8; FIRMWARE_VERSION_SIZE] {
    // SetSysFirmwareVersion's verified field layout is defined by libnx:
    // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/services/set.h
    // The field values reproduce the documented retail NX 1.0.0 system
    // version title rather than identifying the emulator:
    // https://switchbrew.org/w/index.php?title=System_Version_Title&oldid=14763
    let mut version = [0; FIRMWARE_VERSION_SIZE];
    version[0] = 1;
    version[4] = 15;
    version[8..10].copy_from_slice(b"NX");
    version[0x28..0x50].copy_from_slice(b"84b8da475a02261c456e6472b403b31416480165");
    version[0x68..0x6d].copy_from_slice(b"1.0.0");
    version[0x80..0xa4].copy_from_slice(b"NintendoSDK Firmware for NX 1.0.0-15");
    version
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SystemSettingsCommand {
    GetFirmwareVersion,
    GetFirmwareVersion2,
}

impl SystemSettingsCommand {
    const fn decode(command_id: u32) -> Option<Self> {
        match command_id {
            3 => Some(Self::GetFirmwareVersion),
            4 => Some(Self::GetFirmwareVersion2),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum UserSettingsCommand {
    GetLanguageCode,
    GetAvailableLanguageCodes,
    MakeLanguageCode,
    GetAvailableLanguageCodeCount,
    GetRegionCode,
    GetAvailableLanguageCodes2,
    GetAvailableLanguageCodeCount2,
}

impl UserSettingsCommand {
    const fn decode(command_id: u32) -> Option<Self> {
        match command_id {
            0 => Some(Self::GetLanguageCode),
            1 => Some(Self::GetAvailableLanguageCodes),
            2 => Some(Self::MakeLanguageCode),
            3 => Some(Self::GetAvailableLanguageCodeCount),
            4 => Some(Self::GetRegionCode),
            5 => Some(Self::GetAvailableLanguageCodes2),
            6 => Some(Self::GetAvailableLanguageCodeCount2),
            _ => None,
        }
    }
}

// libnx uses both firmware commands with a fixed-size 0x100-byte output
// pointer:
// https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/set.c
pub(in crate::ipc_wire) fn dispatch_system_settings(
    process: &ExceptionProcessContext<'_>,
    request: CmifRequest<'_>,
    receive_statics: &ReceiveStatics,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let Some(command) = SystemSettingsCommand::decode(request.command_id) else {
        return unsupported_service_command("set:sys", request.command_id);
    };
    match command {
        SystemSettingsCommand::GetFirmwareVersion | SystemSettingsCommand::GetFirmwareVersion2 => {
            let ReceiveStatics::Entries(descriptors) = receive_statics else {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            };
            let Some(descriptor) = descriptors.first() else {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            };
            if usize::from(descriptor.size) < FIRMWARE_VERSION_SIZE {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            write_bytes(
                process,
                GuestVirtualAddress::new(descriptor.address),
                &emulated_firmware_version(),
            )?;
            log::debug!("set:sys returned emulated firmware version 1.0.0");
            Ok((
                encode_response(request.token, HorizonIpcResult::SUCCESS, &[], None)?,
                None,
            ))
        }
    }
}

// Commands 1/3 are the pre-4.0 pointer-buffer forms; 5/6 are the current
// map-alias forms:
// https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/set.c
pub(in crate::ipc_wire) fn dispatch_user_settings(
    process: &ExceptionProcessContext<'_>,
    session: &UserSettingsSession,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let Some(command) = UserSettingsCommand::decode(request.command_id) else {
        return unsupported_service_command("set", request.command_id);
    };
    match command {
        UserSettingsCommand::GetLanguageCode => {
            if has_ipc_descriptors(hipc) {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            let language = session.environment().language();
            log::debug!("set returned current language {language:?}");
            Ok((
                encode_response(
                    request.token,
                    HorizonIpcResult::SUCCESS,
                    &language.code().to_le_bytes(),
                    None,
                )?,
                None,
            ))
        }
        command @ (UserSettingsCommand::GetAvailableLanguageCodes
        | UserSettingsCommand::GetAvailableLanguageCodes2) => {
            let descriptor = match command {
                UserSettingsCommand::GetAvailableLanguageCodes => match &hipc.receive_statics {
                    ReceiveStatics::Entries(descriptors)
                        if descriptors.len() == 1
                            && hipc.receive_buffers.is_empty()
                            && descriptors[0].size > 0 =>
                    {
                        BufferDescriptor {
                            address: descriptors[0].address,
                            size: u64::from(descriptors[0].size),
                            mode: BufferMode::Normal,
                        }
                    }
                    _ => {
                        return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
                    }
                },
                UserSettingsCommand::GetAvailableLanguageCodes2 => {
                    match hipc.receive_buffers.as_slice() {
                        [descriptor]
                            if descriptor.size > 0
                                && descriptor.mode != BufferMode::Invalid
                                && matches!(hipc.receive_statics, ReceiveStatics::None) =>
                        {
                            *descriptor
                        }
                        _ => {
                            return cmif_error(
                                request.token,
                                HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                            );
                        }
                    }
                }
                _ => unreachable!("matched an available-language command"),
            };
            if !hipc.send_statics.is_empty()
                || !hipc.send_buffers.is_empty()
                || !hipc.exchange_buffers.is_empty()
                || !hipc.copy_handles.is_empty()
                || !hipc.move_handles.is_empty()
            {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            let capacity = usize::try_from(descriptor.size / 8).unwrap_or(usize::MAX);
            let count = capacity.min(SystemLanguage::AVAILABLE.len());
            let mut codes = Vec::with_capacity(count * 8);
            for language in &SystemLanguage::AVAILABLE[..count] {
                codes.extend_from_slice(&language.code().to_le_bytes());
            }
            write_descriptor_bytes(process, descriptor, &codes)?;
            let count = u32::try_from(count).expect("language table fits in a u32");
            log::debug!("set returned {count} available language codes");
            Ok((
                encode_response(
                    request.token,
                    HorizonIpcResult::SUCCESS,
                    &count.to_le_bytes(),
                    None,
                )?,
                None,
            ))
        }
        UserSettingsCommand::MakeLanguageCode => {
            if has_ipc_descriptors(hipc) {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            let Some(language) = request_u32(request.data, 0).and_then(SystemLanguage::from_raw)
            else {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            };
            Ok((
                encode_response(
                    request.token,
                    HorizonIpcResult::SUCCESS,
                    &language.code().to_le_bytes(),
                    None,
                )?,
                None,
            ))
        }
        UserSettingsCommand::GetAvailableLanguageCodeCount
        | UserSettingsCommand::GetAvailableLanguageCodeCount2 => {
            if has_ipc_descriptors(hipc) {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            let count = u32::try_from(SystemLanguage::AVAILABLE.len())
                .expect("language table fits in a u32");
            Ok((
                encode_response(
                    request.token,
                    HorizonIpcResult::SUCCESS,
                    &count.to_le_bytes(),
                    None,
                )?,
                None,
            ))
        }
        UserSettingsCommand::GetRegionCode => {
            if has_ipc_descriptors(hipc) {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            let region = session.environment().region() as u32;
            log::debug!("set returned region {:?}", session.environment().region());
            Ok((
                encode_response(
                    request.token,
                    HorizonIpcResult::SUCCESS,
                    &region.to_le_bytes(),
                    None,
                )?,
                None,
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emulated_firmware_uses_the_verified_setsys_layout() {
        let version = emulated_firmware_version();

        assert_eq!(&version[..3], &[1, 0, 0]);
        assert_eq!(&version[4..6], &[15, 0]);
        assert_eq!(&version[8..10], b"NX");
        assert_eq!(&version[0x68..0x6d], b"1.0.0");
        assert_eq!(version.len(), FIRMWARE_VERSION_SIZE);
    }
}
