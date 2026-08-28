//! Wire decoding for the `aoc:u` add-on-content service.

use crate::ipc_wire::IpcWireError;
use crate::ipc_wire::buffer::one_receive_buffer;
use crate::ipc_wire::io::request_u32;
use crate::ipc_wire::message::{CmifRequest, HipcRequest};
use crate::{IpcRequest, MAX_IPC_LIST_ENTRIES};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AddOnContentCommand {
    CountLegacy,
    ListLegacy,
    Count,
    List,
    PrepareLegacy,
    Prepare,
    GetListChangedEvent,
}

impl AddOnContentCommand {
    const fn decode(command_id: u32) -> Option<Self> {
        match command_id {
            0 => Some(Self::CountLegacy),
            1 => Some(Self::ListLegacy),
            2 => Some(Self::Count),
            3 => Some(Self::List),
            6 => Some(Self::PrepareLegacy),
            7 => Some(Self::Prepare),
            8 => Some(Self::GetListChangedEvent),
            _ => None,
        }
    }
}

pub(in crate::ipc_wire) fn decode_root_request(
    request: &CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<Option<IpcRequest>, IpcWireError> {
    let Some(command) = AddOnContentCommand::decode(request.command_id) else {
        return Ok(None);
    };
    // Versioned command IDs follow the documented aoc:u ABI:
    // https://switchbrew.org/w/index.php?title=NS_services&oldid=14328#aoc:u
    match command {
        AddOnContentCommand::CountLegacy => Ok(Some(IpcRequest::GetIndexedAddOnContentCount)),
        AddOnContentCommand::Count => {
            if hipc.pid.is_none() {
                return Ok(None);
            }
            Ok(Some(IpcRequest::GetIndexedAddOnContentCount))
        }
        command @ (AddOnContentCommand::ListLegacy | AddOnContentCommand::List) => {
            if command == AddOnContentCommand::List && hipc.pid.is_none() {
                return Ok(None);
            }
            let offset = request_u32(request.data, 0)
                .and_then(|value| usize::try_from(value).ok())
                .ok_or(IpcWireError::Malformed(
                    "aoc:u list request omits its start index",
                ))?;
            let requested = request_u32(request.data, 4)
                .and_then(|value| usize::try_from(value).ok())
                .ok_or(IpcWireError::Malformed(
                    "aoc:u list request omits its entry count",
                ))?;
            let descriptor = one_receive_buffer(hipc)?;
            let capacity = usize::try_from(descriptor.size / 4)
                .map_err(|_| IpcWireError::Malformed("aoc:u output buffer is too large"))?;
            Ok(Some(IpcRequest::ListIndexedAddOnContent {
                offset,
                max_entries: requested.min(capacity).min(MAX_IPC_LIST_ENTRIES),
            }))
        }
        command @ (AddOnContentCommand::PrepareLegacy | AddOnContentCommand::Prepare) => {
            if command == AddOnContentCommand::Prepare && hipc.pid.is_none() {
                return Ok(None);
            }
            let horizon_index = request_u32(request.data, 0).ok_or(IpcWireError::Malformed(
                "aoc:u prepare request omits its content index",
            ))?;
            Ok(Some(IpcRequest::PrepareAddOnContent { horizon_index }))
        }
        AddOnContentCommand::GetListChangedEvent => {
            Ok(Some(IpcRequest::GetAddOnContentListChangedEvent))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipc_wire::message::COMMAND_BUFFER_SIZE;

    fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
        bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }

    #[test]
    fn current_process_commands_require_pid_and_bound_list_capacity() {
        let mut count = [0_u8; COMMAND_BUFFER_SIZE];
        put_u32(&mut count, 0, 4);
        put_u32(&mut count, 4, 10 | (1 << 31));
        put_u32(&mut count, 8, 1);
        put_u64(&mut count, 12, 7);
        put_u32(&mut count, 32, 0x4943_4653);
        put_u32(&mut count, 40, 2);
        let hipc = HipcRequest::decode(&count).unwrap();
        let request = CmifRequest::decode(&hipc, false).unwrap();
        assert_eq!(
            decode_root_request(&request, &hipc).unwrap(),
            Some(IpcRequest::GetIndexedAddOnContentCount)
        );

        let mut list = [0_u8; COMMAND_BUFFER_SIZE];
        put_u32(&mut list, 0, 4 | (1 << 24));
        put_u32(&mut list, 4, 10 | (1 << 31));
        put_u32(&mut list, 8, 1);
        put_u64(&mut list, 12, 7);
        put_u32(&mut list, 20, 16);
        put_u32(&mut list, 24, 0x1000);
        put_u32(&mut list, 28, 0);
        put_u32(&mut list, 32, 0x4943_4653);
        put_u32(&mut list, 40, 3);
        put_u32(&mut list, 48, 2);
        put_u32(&mut list, 52, 10);
        let hipc = HipcRequest::decode(&list).unwrap();
        let request = CmifRequest::decode(&hipc, false).unwrap();
        assert_eq!(
            decode_root_request(&request, &hipc).unwrap(),
            Some(IpcRequest::ListIndexedAddOnContent {
                offset: 2,
                max_entries: 4,
            })
        );

        let mut without_pid = [0_u8; COMMAND_BUFFER_SIZE];
        put_u32(&mut without_pid, 0, 4);
        put_u32(&mut without_pid, 4, 8);
        put_u32(&mut without_pid, 16, 0x4943_4653);
        put_u32(&mut without_pid, 24, 2);
        let hipc = HipcRequest::decode(&without_pid).unwrap();
        let request = CmifRequest::decode(&hipc, false).unwrap();
        assert_eq!(decode_root_request(&request, &hipc).unwrap(), None);
    }
}
