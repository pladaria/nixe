//! Ordered packet spans. Payloads remain borrowed from the decoded packet;
//! only the backend's final byte payload is retained. Exceptional shadow replay
//! uses scalar semantics, while ordinary transfers scale with physical spans.
use super::*;
use crate::pushbuffer::dispatch::class_packet_methods;

pub(super) fn stream_packet<E>(
    channel: &mut MaxwellGpuChannel,
    submission: FrontendSubmissionId,
    packet: &MaxwellDecodedPacket,
    consume: &mut impl for<'a> FnMut(MaxwellEngineEvent<'a>) -> Result<(), E>,
) -> Result<bool, MaxwellEngineStreamError<E>> {
    let MaxwellDecodedPacket::Methods(packet_methods) = packet else {
        return Ok(false);
    };
    let decoded = packet_methods.methods();
    let Some(first) = decoded.first() else {
        return Ok(false);
    };
    let cb = first.method().0 == 0x238c || (0x2390..=0x23fc).contains(&first.method().0);
    let inline = first.method().0 == 0x1b4;
    let indices = matches!(first.method().0, 0x1300 | 0x15e8 | 0x15ec);
    if decoded.len() < 2 || (!cb && !inline && !indices) {
        return Ok(false);
    }
    let data = if cb && first.method().0 == 0x238c {
        decoded.slice(1..)
    } else {
        decoded
    };
    if data.is_empty()
        || if cb {
            // Decoded packet methods are monotonic; endpoint bounds prove the
            // entire CB aperture without reconstructing each word's provenance.
            data.first().unwrap().method().0 < 0x2390 || data.last().unwrap().method().0 > 0x23fc
        } else {
            data.first().unwrap().method() != first.method()
                || data.last().unwrap().method() != first.method()
        }
    {
        return Ok(false);
    }
    let Some((class, subchannel)) = class_packet_methods(channel, submission, packet)
        .map_err(|error| MaxwellEngineStreamError::dispatch(error.into()))?
    else {
        return Ok(false);
    };
    if class == threed::CLASS {
        let (profile, state) = channel.three_d_with_profile_mut();
        if state.pending_notification.is_some()
            || state.mme().shadow_ram_control().value()
                == Some(&MaxwellThreeDMmeShadowRamControl::MethodReplay)
        {
            return Ok(false);
        }
        if indices {
            threed::inline_index_words(state, subchannel, data)
                .map_err(MaxwellEngineStreamError::dispatch)?;
            return Ok(true);
        }
        if cb {
            if first.method().0 == 0x238c {
                threed::preflight(
                    profile,
                    MaxwellMethodDispatch::emitted_by_mme(
                        MaxwellMethodSource::decoded(first, subchannel),
                        class,
                    ),
                    state,
                )
                .map_err(MaxwellEngineStreamError::dispatch)?;
            }
            let (upload, count) = threed::constant_buffer_words(profile, state, subchannel, data)
                .map_err(MaxwellEngineStreamError::dispatch)?;
            consume(MaxwellEngineEvent {
                instances: None,
                operation: PendingEngineOperation::ThreeDInlineConstantBuffer(upload),
                three_d: state.operation_state(),
                compute: None,
                inline_words: Some(data.slice(..count)),
            })
            .map_err(MaxwellEngineStreamError::Consumer)?;
            if count < data.len() {
                threed::preflight(
                    profile,
                    MaxwellMethodDispatch::emitted_by_mme(
                        MaxwellMethodSource::decoded(data.get(count).unwrap(), subchannel),
                        class,
                    ),
                    state,
                )
                .map_err(MaxwellEngineStreamError::dispatch)?;
                unreachable!("first excess CB word must report its precise range error");
            }
            return Ok(true);
        }
    } else if !inline || (class != compute::CLASS && class != inline_to_memory::CLASS) {
        return Ok(false);
    }
    if !inline {
        return Ok(false);
    }
    let mut cursor = 0;
    while cursor < data.len() {
        let tail = data.slice(cursor..);
        let (operation, count) = if class == threed::CLASS {
            let (profile, state) = channel.three_d_with_profile_mut();
            let (upload, count) = threed::embedded_words(profile, state, subchannel, tail)
                .map_err(MaxwellEngineStreamError::dispatch)?;
            (PendingEngineOperation::InlineToMemory(upload), count)
        } else if class == compute::CLASS {
            let (upload, count) =
                compute::data_words(class, channel.compute_mut(), subchannel, tail)
                    .map_err(MaxwellEngineStreamError::dispatch)?;
            (PendingEngineOperation::ComputeInlineToMemory(upload), count)
        } else {
            let (upload, count) =
                inline_to_memory::data_words(channel.inline_to_memory_mut(), subchannel, tail)
                    .map_err(MaxwellEngineStreamError::dispatch)?;
            (PendingEngineOperation::InlineToMemory(upload), count)
        };
        consume(MaxwellEngineEvent {
            instances: None,
            operation,
            three_d: channel.three_d(),
            compute: Some(channel.compute()),
            inline_words: Some(tail.slice(..count)),
        })
        .map_err(MaxwellEngineStreamError::Consumer)?;
        cursor += count;
    }
    Ok(true)
}
