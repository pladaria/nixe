//! Non-incrementing DMA calls consume borrowed argument arrays directly.
//! Exceptional shadow replay uses the same scalar engine entry as Eden.
use super::*;
use crate::MaxwellPushbufferSubchannel;

#[derive(Clone, Copy)]
pub(super) enum Port {
    ConstantBuffer,
    Embedded,
    Indices,
}

pub(super) const fn port(method: GpuMethodId) -> Option<Port> {
    match method.0 {
        0x2390..=0x23fc => Some(Port::ConstantBuffer),
        0x1b4 => Some(Port::Embedded),
        0x1300 | 0x15e8 | 0x15ec => Some(Port::Indices),
        _ => None,
    }
}

pub(super) fn call_multi_method<L: crate::MaxwellFrontend, H: crate::MaxwellCommandHost>(
    channel: &mut MaxwellGpuChannel,
    class: GpuClassId,
    subchannel: MaxwellPushbufferSubchannel,
    method: GpuMethodId,
    data: crate::MaxwellDmaArguments<'_>,
    port: Port,
    execution: &mut crate::execution::ChannelExecution<'_, L, H>,
) -> Result<bool, MaxwellEngineStreamError> {
    let cb = matches!(port, Port::ConstantBuffer);
    let inline = matches!(port, Port::Embedded);
    let indices = matches!(port, Port::Indices);
    if class == threed::CLASS {
        let state = channel.three_d_mut();
        if state.pending_notification.is_some()
            || (!cb
                && state.mme().shadow_ram_control().value()
                    == Some(&MaxwellThreeDMmeShadowRamControl::MethodReplay))
        {
            return Ok(false);
        }
    } else if !inline || (class != compute::CLASS && class != inline_to_memory::CLASS) {
        return Ok(false);
    }
    if class == threed::CLASS {
        let (profile, state) = channel.three_d_with_profile_mut();
        if indices {
            threed::inline_index_words(state, method, subchannel, data)
                .map_err(MaxwellEngineStreamError::dispatch)?;
            return Ok(true);
        }
        if cb {
            // ProcessCBMultiData consumes the literal packet directly, including
            // shadow replay mode; it does not write the data port's shadow word.
            let count = threed::process_constant_buffer_data(
                state,
                data.write_source(),
                execution,
                || data.source(0, method, subchannel),
            )?;
            if count < data.len() {
                threed::call_method(
                    profile,
                    method,
                    data.argument_at(count),
                    state,
                    execution,
                    || data.source(count, method, subchannel),
                )?;
                unreachable!("first excess CB word must report its precise range error");
            }
            return Ok(true);
        }
    }
    if !inline {
        return Ok(false);
    }
    let mut cursor = 0;
    while cursor < data.len() {
        let tail = data.slice(cursor..);
        let count = if class == threed::CLASS {
            let (_, state) = channel.three_d_with_profile_mut();
            let (upload, count) = threed::embedded_words(state, method, subchannel, tail)
                .map_err(MaxwellEngineStreamError::dispatch)?;
            execution.inline_to_memory(upload, Some(tail.slice(..count)))?;
            count
        } else if class == compute::CLASS {
            let (upload, count) =
                compute::data_words(channel.compute_mut(), method, subchannel, tail)
                    .map_err(MaxwellEngineStreamError::dispatch)?;
            execution.compute_inline(upload, Some(tail.slice(..count)))?;
            count
        } else {
            let (upload, count) = inline_to_memory::data_words(
                channel.inline_to_memory_mut(),
                method,
                subchannel,
                tail,
            )
            .map_err(MaxwellEngineStreamError::dispatch)?;
            execution.inline_to_memory(upload, Some(tail.slice(..count)))?;
            count
        };
        cursor += count;
    }
    Ok(true)
}
