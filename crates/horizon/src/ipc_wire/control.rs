//! CMIF session-control commands shared by Horizon services.

use nixe_memory::GuestVirtualAddress;
use nixe_runtime::ExceptionProcessContext;

use super::io::{encode_response, write_response};
use super::message::CmifRequest;
use super::{IpcWireError, unsupported_service_command};
use crate::{HorizonIpcObject, HorizonIpcResult};

const CMIF_COMMAND_CONTROL: u16 = 5;
const CMIF_COMMAND_CONTROL_WITH_CONTEXT: u16 = 7;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CmifControlCommand {
    ConvertCurrentObjectToDomain,
    CloneCurrentObject,
    QueryPointerBufferSize,
    CloneCurrentObjectEx,
}

impl CmifControlCommand {
    const fn decode(command_id: u32) -> Option<Self> {
        match command_id {
            0 => Some(Self::ConvertCurrentObjectToDomain),
            2 => Some(Self::CloneCurrentObject),
            3 => Some(Self::QueryPointerBufferSize),
            4 => Some(Self::CloneCurrentObjectEx),
            _ => None,
        }
    }
}

/// Handles one CMIF control request, returning `false` for ordinary service
/// commands which must continue through service dispatch.
pub(super) fn dispatch_control(
    process: &mut ExceptionProcessContext<'_>,
    address: GuestVirtualAddress,
    size: usize,
    handle: u32,
    target: &HorizonIpcObject,
    request: &CmifRequest<'_>,
) -> Result<bool, IpcWireError> {
    if !matches!(
        request.command_type,
        CMIF_COMMAND_CONTROL | CMIF_COMMAND_CONTROL_WITH_CONTEXT
    ) {
        return Ok(false);
    }
    let Some(control_command) = CmifControlCommand::decode(request.command_id) else {
        return unsupported_service_command("CMIF control", request.command_id);
    };

    match (control_command, target) {
        (CmifControlCommand::ConvertCurrentObjectToDomain, HorizonIpcObject::Applet(applet)) => {
            // libnx converts appletOE to a domain before opening the
            // application proxy. The control command and returned root object
            // ID follow its pinned CMIF implementation:
            // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/sf/service.h#L250-L266
            let object_id = applet.convert_to_domain();
            write_domain_conversion(process, address, size, request.token, object_id)?;
            log::debug!("appletOE converted to domain with root object {object_id:#x}");
        }
        (
            CmifControlCommand::ConvertCurrentObjectToDomain,
            HorizonIpcObject::SemanticService(service),
        ) => {
            let object_id = service.convert_to_domain();
            write_domain_conversion(process, address, size, request.token, object_id)?;
            log::debug!(
                "{:?} converted to domain with root object {object_id:#x}",
                String::from_utf8_lossy(service.service().name())
            );
        }
        (
            CmifControlCommand::ConvertCurrentObjectToDomain,
            HorizonIpcObject::ParentalControl(factory),
        ) => {
            // The public pctl client converts the factory before asking it to
            // create IParentalControlService. Keep that real domain boundary:
            // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/pctl.c#L20-L24
            let object_id = factory.convert_to_domain();
            write_domain_conversion(process, address, size, request.token, object_id)?;
            log::debug!("pctl converted to domain with root object {object_id:#x}");
        }
        (CmifControlCommand::ConvertCurrentObjectToDomain, HorizonIpcObject::Time(service)) => {
            let object_id = service.convert_to_domain();
            write_domain_conversion(process, address, size, request.token, object_id)?;
            log::debug!("time:u converted to domain with root object {object_id:#x}");
        }
        (
            CmifControlCommand::ConvertCurrentObjectToDomain,
            HorizonIpcObject::NetworkInterface(manager),
        ) => {
            let object_id = manager.convert_to_domain();
            write_domain_conversion(process, address, size, request.token, object_id)?;
            log::debug!("nifm:u converted to domain with root object {object_id:#x}");
        }
        (CmifControlCommand::ConvertCurrentObjectToDomain, HorizonIpcObject::Account(account)) => {
            let object_id = account.convert_to_domain();
            write_domain_conversion(process, address, size, request.token, object_id)?;
            log::debug!("acc:u0 converted to domain with root object {object_id:#x}");
        }
        (CmifControlCommand::ConvertCurrentObjectToDomain, HorizonIpcObject::Bsd(session)) => {
            let object_id = session.convert_to_domain();
            write_domain_conversion(process, address, size, request.token, object_id)?;
            log::debug!("bsd:u converted to domain with root object {object_id:#x}");
        }
        (
            CmifControlCommand::CloneCurrentObject | CmifControlCommand::CloneCurrentObjectEx,
            HorizonIpcObject::SemanticService(service),
        ) => {
            // CloneCurrentObject (2) and its Ex form (4) return a moved
            // session handle. A clone retains the source domain table.
            // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/sf/cmif.h#L308-L337
            let cloned_handle = install_clone(
                process,
                address,
                size,
                request.token,
                HorizonIpcObject::SemanticService(service.clone()),
                "cloning a content-service session handle",
            )?;
            log::debug!(
                "{:?} cloned session {handle:#x} as {cloned_handle:#x}",
                String::from_utf8_lossy(service.service().name())
            );
        }
        (
            CmifControlCommand::CloneCurrentObject | CmifControlCommand::CloneCurrentObjectEx,
            HorizonIpcObject::Vi(vi),
        ) => {
            install_clone(
                process,
                address,
                size,
                request.token,
                HorizonIpcObject::Vi(vi.clone()),
                "cloning a VI session handle",
            )?;
        }
        (
            CmifControlCommand::CloneCurrentObject | CmifControlCommand::CloneCurrentObjectEx,
            HorizonIpcObject::NvDrv(nvdrv),
        ) => {
            // CMIF clones share nvdrv client state but retain distinct
            // connection identities for descriptor ownership.
            let cloned_session =
                nvdrv
                    .clone_connection()
                    .ok_or(IpcWireError::HostResourceExhausted(
                        "cloning an nvdrv connection",
                    ))?;
            install_clone(
                process,
                address,
                size,
                request.token,
                HorizonIpcObject::NvDrv(cloned_session),
                "installing a cloned nvdrv session handle",
            )?;
        }
        (
            CmifControlCommand::CloneCurrentObject | CmifControlCommand::CloneCurrentObjectEx,
            HorizonIpcObject::ParentalControl(factory),
        ) => {
            install_clone(
                process,
                address,
                size,
                request.token,
                HorizonIpcObject::ParentalControl(factory.clone()),
                "cloning a pctl session handle",
            )?;
        }
        (
            CmifControlCommand::CloneCurrentObject | CmifControlCommand::CloneCurrentObjectEx,
            HorizonIpcObject::Bsd(session),
        ) => {
            let cloned_handle = install_clone(
                process,
                address,
                size,
                request.token,
                HorizonIpcObject::Bsd(session.clone()),
                "cloning a BSD session handle",
            )?;
            log::debug!("bsd:u cloned session {handle:#x} as {cloned_handle:#x}");
        }
        (CmifControlCommand::QueryPointerBufferSize, _) => {
            // Zero makes libnx use map-alias buffers, which the descriptor
            // bridge validates explicitly.
            let response = encode_response(
                request.token,
                HorizonIpcResult::SUCCESS,
                &0_u16.to_le_bytes(),
                None,
            )?;
            write_response(process, address, size, &response)?;
        }
        _ => return unsupported_service_command("CMIF control", request.command_id),
    }
    Ok(true)
}

fn write_domain_conversion(
    process: &mut ExceptionProcessContext<'_>,
    address: GuestVirtualAddress,
    size: usize,
    token: u32,
    object_id: u32,
) -> Result<(), IpcWireError> {
    let response = encode_response(
        token,
        HorizonIpcResult::SUCCESS,
        &object_id.to_le_bytes(),
        None,
    )?;
    write_response(process, address, size, &response)
}

fn install_clone(
    process: &mut ExceptionProcessContext<'_>,
    address: GuestVirtualAddress,
    size: usize,
    token: u32,
    object: HorizonIpcObject,
    allocation: &'static str,
) -> Result<u32, IpcWireError> {
    let handle = process
        .handles_mut()
        .insert(object)
        .map_err(|_| IpcWireError::HostResourceExhausted(allocation))?;
    let response = match encode_response(token, HorizonIpcResult::SUCCESS, &[], Some(handle)) {
        Ok(response) => response,
        Err(error) => {
            let _ = process.handles_mut().close(handle);
            return Err(error);
        }
    };
    if let Err(error) = write_response(process, address, size, &response) {
        let _ = process.handles_mut().close(handle);
        return Err(error);
    }
    Ok(handle)
}
