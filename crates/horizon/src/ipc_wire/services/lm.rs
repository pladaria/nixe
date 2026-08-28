use super::super::*;

use crate::diagnostics::{GuestLogLevel, GuestLogSeverity};
use crate::object::GuestLogMessage;

const LOG_PACKET_HEADER_SIZE: usize = 0x18;
const MAX_LOG_PACKET_BYTES: usize = 1024 * 1024;
const MAX_ACTIVE_LOG_MESSAGES: usize = 64;
const MAX_ASSEMBLED_LOG_BYTES: usize = 1024 * 1024;
const LOG_PACKET_FLAG_HEAD: u8 = 1 << 0;
const LOG_PACKET_FLAG_TAIL: u8 = 1 << 1;
const LOG_CHUNK_TEXT: u64 = 2;
const LOG_CHUNK_LINE_NUMBER: u64 = 3;
const LOG_CHUNK_FILE_NAME: u64 = 4;
const LOG_CHUNK_FUNCTION_NAME: u64 = 5;
const LOG_CHUNK_MODULE_NAME: u64 = 6;
const LOG_CHUNK_THREAD_NAME: u64 = 7;
const LOG_CHUNK_PROCESS_NAME: u64 = 10;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LogManagerCommand {
    OpenLogger,
}

impl LogManagerCommand {
    const fn decode(command_id: u32) -> Option<Self> {
        match command_id {
            0 => Some(Self::OpenLogger),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LoggerCommand {
    Log,
}

impl LoggerCommand {
    const fn decode(command_id: u32) -> Option<Self> {
        match command_id {
            0 => Some(Self::Log),
            _ => None,
        }
    }
}

pub(in crate::ipc_wire) fn dispatch_log_manager(
    process: &mut ExceptionProcessContext<'_>,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let Some(command) = LogManagerCommand::decode(request.command_id) else {
        return unsupported_service_command("lm", request.command_id);
    };

    match command {
        // ILogService::OpenLogger receives the kernel-substituted client PID
        // descriptor and returns one ILogger object:
        // https://github.com/Atmosphere-NX/Atmosphere/blob/cb4b882e3b176480ac57a1161a85ff175c3f162c/libraries/libstratosphere/source/lm/sf/lm_i_log_service.hpp#L23-L26
        LogManagerCommand::OpenLogger => {
            if hipc.pid.is_none()
                || !hipc.copy_handles.is_empty()
                || !hipc.move_handles.is_empty()
                || !hipc.send_statics.is_empty()
                || !hipc.send_buffers.is_empty()
                || !hipc.receive_buffers.is_empty()
                || !hipc.exchange_buffers.is_empty()
                || !matches!(hipc.receive_statics, ReceiveStatics::None)
            {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            let process_id = process.process_id();
            let handle = process
                .handles_mut()
                .insert(HorizonIpcObject::Logger(LoggerSession::new(process_id)))
                .map_err(|_| {
                    IpcWireError::HostResourceExhausted("installing an lm logger handle")
                })?;
            log::debug!("lm opened ILogger handle {handle:#x} for process {process_id}");
            Ok((
                encode_response(request.token, HorizonIpcResult::SUCCESS, &[], Some(handle))?,
                Some(handle),
            ))
        }
    }
}

pub(in crate::ipc_wire) fn dispatch_logger(
    process: &mut ExceptionProcessContext<'_>,
    logger: &LoggerSession,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
    guest_logs_level: GuestLogLevel,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let Some(command) = LoggerCommand::decode(request.command_id) else {
        return unsupported_service_command("ILogger", request.command_id);
    };

    match command {
        // ILogger::Log takes one sf::InAutoSelectBuffer. The packet header and
        // TLV keys follow Atmosphere's pinned reconstruction of the ABI:
        // https://github.com/Atmosphere-NX/Atmosphere/blob/cb4b882e3b176480ac57a1161a85ff175c3f162c/libraries/libstratosphere/source/lm/sf/lm_i_log_service.hpp#L19-L21
        // https://github.com/Atmosphere-NX/Atmosphere/blob/cb4b882e3b176480ac57a1161a85ff175c3f162c/libraries/libstratosphere/source/lm/impl/lm_log_packet_header.hpp#L20-L48
        LoggerCommand::Log => {
            if hipc.pid.is_some()
                || !hipc.copy_handles.is_empty()
                || !hipc.move_handles.is_empty()
                || !hipc.receive_buffers.is_empty()
                || !hipc.exchange_buffers.is_empty()
                || !matches!(hipc.receive_statics, ReceiveStatics::None)
            {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            let (address, size) = one_auto_select_input(hipc)?;
            if size > MAX_LOG_PACKET_BYTES {
                return Err(IpcWireError::Malformed(
                    "ILogger packet exceeds the bounded diagnostic payload",
                ));
            }
            let mut bytes = vec![0; size];
            if size != 0 {
                read_bytes(process, GuestVirtualAddress::new(address), &mut bytes)?;
            }

            // Nintendo's logger returns success when the packet body is
            // malformed. Preserve that guest-visible behavior while retaining
            // a host warning so broken packet producers remain diagnosable:
            // https://github.com/Atmosphere-NX/Atmosphere/blob/cb4b882e3b176480ac57a1161a85ff175c3f162c/libraries/libstratosphere/source/lm/srv/lm_logger_impl.cpp#L91-L96
            let packet_result = match validate_log_packet_frame(&bytes) {
                Ok(()) => {
                    bytes[..8].copy_from_slice(&logger.process_id().to_le_bytes());
                    write_bytes(process, GuestVirtualAddress::new(address), &bytes[..8])?;
                    ingest_log_packet(logger, &bytes)
                }
                Err(reason) => Err(reason),
            };
            match packet_result {
                Ok(messages) => {
                    for message in messages {
                        emit_guest_log(logger.process_id(), message, guest_logs_level);
                    }
                }
                Err(reason) => log::warn!(
                    "[guest] ILogger discarded a malformed packet from process {}: {reason}",
                    logger.process_id()
                ),
            }
            semantic_success(request.token, false, &[], &[], &[], None)
        }
    }
}

fn validate_log_packet_frame(bytes: &[u8]) -> Result<(), &'static str> {
    let header = bytes
        .get(..LOG_PACKET_HEADER_SIZE)
        .ok_or("packet header is truncated")?;
    let payload_size = usize::try_from(u32::from_le_bytes(header[20..24].try_into().unwrap()))
        .expect("u32 log payload size fits usize");
    if LOG_PACKET_HEADER_SIZE.checked_add(payload_size) != Some(bytes.len()) {
        return Err("packet size does not match its header");
    }
    Ok(())
}

fn ingest_log_packet(
    logger: &LoggerSession,
    bytes: &[u8],
) -> Result<Vec<GuestLogMessage>, &'static str> {
    validate_log_packet_frame(bytes)?;
    let mut completed = Vec::new();
    let header = &bytes[..LOG_PACKET_HEADER_SIZE];
    let thread_id = u64::from_le_bytes(header[8..16].try_into().unwrap());
    let flags = header[16];
    let severity = header[18];
    let verbosity = header[19];
    let payload = &bytes[LOG_PACKET_HEADER_SIZE..];

    logger.with_state(|state| {
        if flags & LOG_PACKET_FLAG_HEAD != 0 {
            if state.messages.len() >= MAX_ACTIVE_LOG_MESSAGES
                && !state.messages.contains_key(&thread_id)
            {
                return Err("too many fragmented log messages are active");
            }
            state.messages.insert(
                thread_id,
                GuestLogMessage {
                    severity,
                    verbosity,
                    ..GuestLogMessage::default()
                },
            );
        }

        let Some(message) = state.messages.get_mut(&thread_id) else {
            // The hardware logger accepts orphan continuation packets. They
            // carry no independently usable context, so there is nothing
            // meaningful for the host diagnostic sink to emit.
            return Ok(());
        };
        if let Err(error) = parse_log_chunks(payload, message) {
            state.messages.remove(&thread_id);
            return Err(error);
        }
        if flags & LOG_PACKET_FLAG_TAIL != 0 {
            completed.push(
                state
                    .messages
                    .remove(&thread_id)
                    .expect("active ILogger message exists"),
            );
        }
        Ok(())
    })?;
    Ok(completed)
}

fn parse_log_chunks(payload: &[u8], message: &mut GuestLogMessage) -> Result<(), &'static str> {
    let mut offset = 0;
    while offset < payload.len() {
        let (key, next) = decode_uleb128(payload, offset)?;
        let (size, next) = decode_uleb128(payload, next)?;
        let size = usize::try_from(size).map_err(|_| "log chunk size exceeds the host range")?;
        let end = next.checked_add(size).ok_or("log chunk size overflows")?;
        let chunk = payload.get(next..end).ok_or("log chunk is truncated")?;
        let current_size = assembled_log_size(message);
        match key {
            LOG_CHUNK_TEXT => append_log_field(&mut message.text, chunk, current_size)?,
            LOG_CHUNK_LINE_NUMBER if chunk.len() == 4 => {
                message.line = Some(u32::from_le_bytes(chunk.try_into().unwrap()));
            }
            LOG_CHUNK_FILE_NAME => append_log_field(&mut message.file, chunk, current_size)?,
            LOG_CHUNK_FUNCTION_NAME => {
                append_log_field(&mut message.function, chunk, current_size)?;
            }
            LOG_CHUNK_MODULE_NAME => append_log_field(&mut message.module, chunk, current_size)?,
            LOG_CHUNK_THREAD_NAME => {
                append_log_field(&mut message.thread_name, chunk, current_size)?;
            }
            LOG_CHUNK_PROCESS_NAME => {
                append_log_field(&mut message.process_name, chunk, current_size)?;
            }
            _ => {}
        }
        offset = end;
    }
    Ok(())
}

fn append_log_field(
    field: &mut Vec<u8>,
    bytes: &[u8],
    current_size: usize,
) -> Result<(), &'static str> {
    if current_size
        .checked_add(bytes.len())
        .is_none_or(|size| size > MAX_ASSEMBLED_LOG_BYTES)
    {
        return Err("fragmented log message exceeds the bounded diagnostic payload");
    }
    field.extend_from_slice(bytes);
    Ok(())
}

fn assembled_log_size(message: &GuestLogMessage) -> usize {
    message.text.len()
        + message.module.len()
        + message.file.len()
        + message.function.len()
        + message.thread_name.len()
        + message.process_name.len()
}

fn decode_uleb128(bytes: &[u8], mut offset: usize) -> Result<(u64, usize), &'static str> {
    let mut value = 0_u64;
    for shift in (0..=63).step_by(7) {
        let byte = *bytes.get(offset).ok_or("log chunk ULEB128 is truncated")?;
        offset += 1;
        if shift == 63 && byte & 0xfe != 0 {
            return Err("log chunk ULEB128 overflows");
        }
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok((value, offset));
        }
    }
    Err("log chunk ULEB128 overflows")
}

fn emit_guest_log(process_id: u64, message: GuestLogMessage, policy: GuestLogLevel) {
    let Some(severity) = GuestLogSeverity::decode(message.severity) else {
        log::warn!(
            "[guest] ILogger discarded a packet from process {process_id} with unknown severity {}",
            message.severity
        );
        return;
    };
    let Some(level) = policy.resolve(severity) else {
        return;
    };
    if !log::log_enabled!(target: "nixe_horizon::guest", level) {
        return;
    }

    let module = sanitize_label(if message.module.is_empty() {
        &message.process_name
    } else {
        &message.module
    });
    let prefix = match (module.is_empty(), message.verbosity) {
        (true, 0) => "[guest]".to_owned(),
        (true, verbosity) => format!("[guest v={verbosity}]"),
        (false, 0) => format!("[guest:{module}]"),
        (false, verbosity) => format!("[guest:{module} v={verbosity}]"),
    };
    let file = sanitize_text(&message.file);
    let function = sanitize_text(&message.function);
    let context = match (file.is_empty(), message.line, function.is_empty()) {
        (false, Some(line), false) => format!(" [{file}:{line} {function}]"),
        (false, Some(line), true) => format!(" [{file}:{line}]"),
        (false, None, false) => format!(" [{file} {function}]"),
        (false, None, true) => format!(" [{file}]"),
        (true, _, false) => format!(" [{function}]"),
        (true, _, true) => String::new(),
    };
    let text = sanitize_text(&message.text);
    let text = text.trim_end_matches(['\r', '\n']);
    if text.is_empty() {
        log::log!(target: "nixe_horizon::guest", level, "{prefix}{context}");
        return;
    }
    for line in text.split('\n') {
        log::log!(
            target: "nixe_horizon::guest",
            level,
            "{prefix} {}{context}",
            line.trim_end_matches('\r')
        );
    }
}

fn sanitize_label(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes)
        .chars()
        .take(32)
        .map(|character| match character {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '_' | '-' | '.' => character,
            _ => '_',
        })
        .collect()
}

fn sanitize_text(bytes: &[u8]) -> String {
    let mut output = String::new();
    for character in String::from_utf8_lossy(bytes).chars() {
        if character.is_control() && !matches!(character, '\n' | '\r' | '\t') {
            output.extend(character.escape_default());
        } else {
            output.push(character);
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn packet(thread_id: u64, flags: u8, severity: u8, chunks: &[(u8, &[u8])]) -> Vec<u8> {
        let mut payload = Vec::new();
        for (key, bytes) in chunks {
            payload.push(*key);
            payload.push(u8::try_from(bytes.len()).unwrap());
            payload.extend_from_slice(bytes);
        }
        let mut packet = vec![0; LOG_PACKET_HEADER_SIZE];
        packet[8..16].copy_from_slice(&thread_id.to_le_bytes());
        packet[16] = flags;
        packet[18] = severity;
        packet[20..24].copy_from_slice(&(payload.len() as u32).to_le_bytes());
        packet.extend_from_slice(&payload);
        packet
    }

    #[test]
    fn parser_reassembles_fragmented_text_and_context() {
        let logger = LoggerSession::new(7);
        let head = packet(
            9,
            LOG_PACKET_FLAG_HEAD,
            2,
            &[
                (LOG_CHUNK_MODULE_NAME as u8, b"sdk"),
                (LOG_CHUNK_TEXT as u8, b"hel"),
            ],
        );
        assert!(ingest_log_packet(&logger, &head).unwrap().is_empty());

        let tail = packet(9, LOG_PACKET_FLAG_TAIL, 2, &[(LOG_CHUNK_TEXT as u8, b"lo")]);
        let messages = ingest_log_packet(&logger, &tail).unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].severity, 2);
        assert_eq!(messages[0].module, b"sdk");
        assert_eq!(messages[0].text, b"hello");
    }

    #[test]
    fn parser_rejects_truncated_packets_without_retaining_the_message() {
        let logger = LoggerSession::new(7);
        let mut malformed = packet(
            9,
            LOG_PACKET_FLAG_HEAD | LOG_PACKET_FLAG_TAIL,
            1,
            &[(LOG_CHUNK_TEXT as u8, b"hello")],
        );
        malformed.pop();
        assert!(matches!(
            ingest_log_packet(&logger, &malformed),
            Err("packet size does not match its header")
        ));
        assert!(logger.with_state(|state| state.messages.is_empty()));
    }

    #[test]
    fn diagnostic_labels_do_not_pass_terminal_control_sequences() {
        assert_eq!(sanitize_label(b"main\x1b[31m"), "main__31m");
        assert_eq!(sanitize_text(b"bad\x1b[31m"), "bad\\u{1b}[31m");
    }
}
