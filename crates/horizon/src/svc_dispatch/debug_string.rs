use super::*;
use crate::diagnostics::GuestLogSeverity;

/// OutputDebugString takes a pointer and byte count, not a C string. Empty
/// ranges succeed before pointer validation; copy failures are InvalidPointer.
/// https://github.com/Atmosphere-NX/Atmosphere/blob/master/libraries/libmesosphere/source/svc/kern_svc_debug_string.cpp
/// https://github.com/Atmosphere-NX/Atmosphere/blob/master/libraries/libmesosphere/source/kern_debug_log.cpp
impl HorizonSvcDispatcher {
    pub(super) fn output_debug_string(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        let address = GuestVirtualAddress::new(read_register(context.thread().state(), 0));
        let count = read_register(context.thread().state(), 1);
        if count == 0 {
            result(context, HorizonKernelResult::SUCCESS);
            return resume();
        }
        if address
            .get()
            .checked_add(count)
            .is_none_or(|end| end > context.process().address_space_limit())
        {
            result(context, HorizonKernelResult::INVALID_POINTER);
            return resume();
        }
        let ranges = match context.process().memory().space().capture(
            address,
            count,
            nixe_memory::MemoryPermissions::READ,
        ) {
            Ok(ranges) => ranges,
            Err(error) => {
                return reject(
                    context,
                    HorizonSvcFault::MemoryAccess {
                        immediate: 0x27,
                        fault: std::sync::Arc::new(error),
                    },
                );
            }
        };
        let machine = context.process().memory().account().machine().clone();
        let level = self
            .diagnostics
            .guest_logs_level
            .resolve(GuestLogSeverity::Info)
            .filter(|level| log::log_enabled!(target: "nixe_horizon::guest", *level));
        self.start_owned_svc(context, std::sync::Arc::default(), None, async move {
            let fault = |fault| HorizonSvcFault::MemoryAccess {
                immediate: 0x27,
                fault,
            };
            let ranges = nixe_memory::CapturedMemory::new(ranges)
                .map_err(|e| fault(std::sync::Arc::new(e)))?;
            let mut remaining = count;
            let mut offset = 0;
            let mut bytes = [0; 0x1000];
            let mut pending_utf8 = 0;
            while remaining != 0 {
                let count = remaining.min((bytes.len() - pending_utf8) as u64) as usize;
                let output = &mut bytes[pending_utf8..pending_utf8 + count];
                match machine
                    .read_captured_into(&ranges, offset, output)
                    .map_err(|e| fault(std::sync::Arc::new(e)))?
                {
                    nixe_memory::Access::Ready(()) => {}
                    nixe_memory::Access::Pending(pending) => match pending.await.map_err(fault)? {
                        nixe_memory::AccessResult::Read(data) if data.len() == output.len() => {
                            output.copy_from_slice(&data)
                        }
                        _ => return Err(runtime_fault("debug string read completion")),
                    },
                }
                offset += count as u64;
                remaining -= count as u64;
                let filled = pending_utf8 + count;
                let emitted = if remaining == 0 {
                    filled
                } else {
                    complete_utf8_prefix(&bytes[..filled])
                };
                if let Some(level) = level {
                    emit(&bytes[..emitted], level);
                }
                bytes.copy_within(emitted..filled, 0);
                pending_utf8 = filled - emitted;
            }
            Ok(super::memory_wait::SvcCompletion::success(None))
        })
    }
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
