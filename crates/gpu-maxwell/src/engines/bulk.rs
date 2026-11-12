//! Consume bulk ports admitted by the command owner, borrowing their literal data.
use super::*;

pub(crate) fn execute_multi<L: crate::MaxwellFrontend, H: crate::MaxwellCommandHost>(
    channel: &mut MaxwellGpuChannel,
    engine: Engine,
    source: MaxwellMethodSource,
    data: crate::MaxwellDmaArguments<'_>,
    execution: &mut crate::execution::ChannelExecution<'_, L, H>,
) -> Result<(), MaxwellEngineStreamError> {
    let method = source.method();
    let subchannel = source.subchannel();
    if engine == Engine::ThreeD {
        threed::draw::before_write(channel.three_d_mut(), method, execution)?;
    }
    match (engine, method.0) {
        (Engine::ThreeD, 0x1300 | 0x15e8 | 0x15ec) => {
            threed::inline_index_words(channel.three_d_mut(), method, subchannel, data)?;
            return Ok(());
        }
        (Engine::ThreeD, 0x2390..=0x23fc) => {
            let (profile, state) = channel.three_d_with_profile_mut();
            // Literal CB data does not write the port's shadow word.
            let count = threed::process_constant_buffer_data(
                state,
                data.write_source(),
                execution,
                || source,
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
            return Ok(());
        }
        (Engine::ThreeD | Engine::Compute | Engine::InlineToMemory, 0x1b4) => {}
        _ => unreachable!("unadmitted bulk port"),
    }
    let mut cursor = 0;
    while cursor < data.len() {
        let tail = data.slice(cursor..);
        let count = if engine == Engine::ThreeD {
            let (_, state) = channel.three_d_with_profile_mut();
            let (upload, count) = threed::embedded_words(state, method, subchannel, tail)
                .map_err(MaxwellEngineStreamError::dispatch)?;
            execution.inline_to_memory(upload, Some(tail.slice(..count)))?;
            count
        } else if engine == Engine::Compute {
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
    Ok(())
}
