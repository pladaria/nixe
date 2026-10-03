use super::*;
use crate::GuestLogLevel;
use crate::diagnostics::GuestLogSeverity;

/// OutputDebugString takes a pointer and byte count, not a C string. Empty
/// ranges succeed before pointer validation; copy failures are InvalidPointer.
/// https://github.com/Atmosphere-NX/Atmosphere/blob/master/libraries/libmesosphere/source/svc/kern_svc_debug_string.cpp
/// https://github.com/Atmosphere-NX/Atmosphere/blob/master/libraries/libmesosphere/source/kern_debug_log.cpp
pub(super) fn output_debug_string(
    context: &mut ExceptionDispatchContext<'_>,
    policy: GuestLogLevel,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    let mut address = read_register(context.thread().state(), 0);
    let mut remaining = read_register(context.thread().state(), 1);
    if remaining == 0 {
        result(context, HorizonKernelResult::SUCCESS);
        return resume();
    }
    if address
        .checked_add(remaining)
        .is_none_or(|end| end > context.process().address_space_limit())
    {
        result(context, HorizonKernelResult::INVALID_POINTER);
        return resume();
    }

    // Host filtering does not change guest memory validation. No allocation is
    // proportional to a guest-provided length; long output uses several records.
    let level = policy
        .resolve(GuestLogSeverity::Info)
        .filter(|level| log::log_enabled!(target: "nixe_horizon::guest", *level));
    let mut bytes = [0_u8; 0x1000];
    let mut pending = 0;
    while remaining != 0 {
        let count = remaining.min((bytes.len() - pending) as u64) as usize;
        if let Err(fault) = context.process().memory().read_bytes(
            context.process().cpu().address_space_id(),
            GuestVirtualAddress::new(address),
            &mut bytes[pending..pending + count],
        ) {
            let diagnostic = HorizonSvcFault::GuestMemory {
                immediate: 0x27,
                fault,
            };
            return if diagnostic.guest_result().is_some() {
                reject(context, diagnostic)
            } else {
                ExceptionDispatchOutcome::Fault(diagnostic)
            };
        }
        address += count as u64;
        remaining -= count as u64;
        let filled = pending + count;
        let emitted = if remaining == 0 {
            filled
        } else {
            complete_utf8_prefix(&bytes[..filled])
        };
        if let Some(level) = level {
            emit(&bytes[..emitted], level);
        }
        bytes.copy_within(emitted..filled, 0);
        pending = filled - emitted;
    }
    result(context, HorizonKernelResult::SUCCESS);
    resume()
}

// Preserve a trailing partial UTF-8 character across host log chunks, while
// allowing malformed bytes to be displayed through lossy decoding.
fn complete_utf8_prefix(bytes: &[u8]) -> usize {
    let mut offset = 0;
    while let Err(error) = std::str::from_utf8(&bytes[offset..]) {
        offset += error.valid_up_to();
        match error.error_len() {
            Some(length) => offset += length,
            None => return offset,
        }
    }
    bytes.len()
}

fn emit(bytes: &[u8], level: log::Level) {
    if bytes.is_empty() {
        return;
    }
    let mut text = String::new();
    for character in String::from_utf8_lossy(bytes).chars() {
        if character.is_control() && !matches!(character, '\n' | '\r' | '\t') {
            text.extend(character.escape_default());
        } else {
            text.push(character);
        }
    }
    // Prefix every host line, including empty lines, just like ILogger output.
    for line in text.trim_end_matches(['\r', '\n']).split('\n') {
        log::log!(target: "nixe_horizon::guest", level, "[guest] {}", line.trim_end_matches('\r'));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_string_chunk_boundary_preserves_incomplete_utf8_after_invalid_bytes() {
        for (bytes, length) in [
            (&b"hello"[..], 5),
            (&b"a\xe2\x82"[..], 1),
            (&b"a\xff\xe2\x82"[..], 2),
            (&b"a\xe2\x82\xac"[..], 4),
        ] {
            assert_eq!(complete_utf8_prefix(bytes), length);
        }
    }
}
