//! Assemble hardware instance sequences before resource capture and native work.
use super::*;
use crate::MaxwellMethodSource;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct InstanceRun {
    trigger: MaxwellThreeDOperationTrigger,
    first: u32,
    count: u32,
}

// Eden's draw manager likewise defers instance sequences until another method
// consumes them. Resource/state writes and observable operations flush first.
// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/engines/draw_manager.cpp
pub(in crate::engines) fn before_method<E>(
    state: &mut MaxwellThreeDFrontendState,
    source: MaxwellMethodSource,
    consume: &mut impl FnMut(
        PendingEngineOperation,
        &MaxwellThreeDState,
        Option<(u32, u32)>,
    ) -> Result<(), E>,
) -> Result<(), E> {
    let Some(run) = state.instance_run else {
        return Ok(());
    };
    let method = source.method().0;
    let argument = source.argument();
    let continuing = state.mme().shadow_ram_control().value()
        != Some(&MaxwellThreeDMmeShadowRamControl::MethodReplay)
        && match method {
            0x1614 => argument == 0,
            0x1618 => MaxwellThreeDBegin::parse(argument).is_some_and(|next| {
                next.instance() == MaxwellThreeDBeginInstance::Subsequent
                    && state
                        .operation_state()
                        .vertex_input()
                        .primitive()
                        .begin()
                        .value()
                        .is_some_and(|prior| {
                            next.topology() == prior.topology()
                                && next.split_mode() == prior.split_mode()
                                && next.preserve_primitive_id() == prior.preserve_primitive_id()
                        })
            }),
            0x0d78 | 0x17e0 => {
                let same = match run.trigger {
                    MaxwellThreeDOperationTrigger::DrawVertexArray { vertex_count, .. } => {
                        method == 0x0d78 && argument == vertex_count
                    }
                    MaxwellThreeDOperationTrigger::DrawIndexBuffer { index_count, .. } => {
                        method == 0x17e0 && argument == index_count
                    }
                    _ => false,
                };
                same && run.first.checked_add(run.count)
                    == Some(
                        state
                            .operation_state()
                            .vertex_input()
                            .primitive()
                            .instance_index(),
                    )
            }
            _ => false,
        };
    if !continuing {
        flush(state, consume)?;
    } else if method == 0x1618 {
        state.complete_draw_end();
    }
    Ok(())
}

pub(in crate::engines) fn operation<E>(
    state: &mut MaxwellThreeDFrontendState,
    operation: PendingEngineOperation,
    consume: &mut impl FnMut(
        PendingEngineOperation,
        &MaxwellThreeDState,
        Option<(u32, u32)>,
    ) -> Result<(), E>,
) -> Result<(), E> {
    if let PendingEngineOperation::ThreeD(trigger) = operation
        && trigger.is_draw()
        && state.operation_state().inline_index().is_none()
    {
        if let Some(run) = &mut state.instance_run {
            run.count += 1;
        } else {
            state.instance_run = Some(InstanceRun {
                trigger,
                first: state
                    .operation_state()
                    .vertex_input()
                    .primitive()
                    .instance_index(),
                count: 1,
            });
        }
        return Ok(());
    }
    flush(state, consume)?;
    consume(operation, state.operation_state(), None)
}

pub(in crate::engines) fn flush<E>(
    state: &mut MaxwellThreeDFrontendState,
    consume: &mut impl FnMut(
        PendingEngineOperation,
        &MaxwellThreeDState,
        Option<(u32, u32)>,
    ) -> Result<(), E>,
) -> Result<(), E> {
    if let Some(run) = state.instance_run.take() {
        consume(
            PendingEngineOperation::ThreeD(run.trigger),
            state.operation_state(),
            Some((run.first, run.count)),
        )?;
        state.complete_draw_end();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;
    use nixe_gpu::MappingGeneration;
    use nixe_gpu::{FrontendSubmissionId, GpuVirtualAddress};
    use std::convert::Infallible;

    fn source() -> MaxwellMethodSource {
        let decoded = decode_maxwell_pushbuffer([Ok(MaxwellPushbufferWord::new(
            (4 << 29) | (0x3800 / 4),
            MaxwellGpfifoSourceLocation {
                channel: MaxwellChannelId::new(1),
                frontend: FrontendSubmissionId::new(1),
                entry_index: 0,
                word_offset: 0,
                pushbuffer: GpuVirtualAddress::try_new(0x1000, 40).unwrap(),
                mapping: MaxwellMappingId::new(1),
                generation: MappingGeneration::new(1),
            },
        ))])
        .unwrap();
        let MaxwellDecodedPacket::Methods(packet) = &decoded.packets()[0] else {
            panic!()
        };
        MaxwellMethodSource::decoded(
            packet.methods().first().unwrap(),
            packet.subchannel().unwrap(),
        )
    }

    fn execute(methods: &[(u32, u32)]) -> Vec<(u32, u32, u32, u32)> {
        let mut state = MaxwellThreeDFrontendState::default();
        let mut observed = Vec::new();
        let mut consume = |operation, state: &MaxwellThreeDState, instances: Option<(u32, u32)>| {
            let PendingEngineOperation::ThreeD(trigger) = operation else {
                panic!("draw")
            };
            let count = match trigger {
                MaxwellThreeDOperationTrigger::DrawVertexArray { vertex_count, .. } => vertex_count,
                _ => panic!("vertex draw"),
            };
            assert!(
                state.vertex_input().primitive().active_begin().is_some(),
                "draw capture precedes its deferred END"
            );
            let (first, instances) = instances.unwrap();
            observed.push((
                first,
                instances,
                count,
                state
                    .vertex_input()
                    .assembly()
                    .global_base_instance_index()
                    .value()
                    .copied()
                    .unwrap_or(0),
            ));
            Ok::<_, Infallible>(())
        };
        let mut host = MmeDispatchHost {
            profile: &SWITCH_1_GM20B_PROFILE,
            source: source(),
            candidate: &mut state,
            consume: &mut consume,
            _error: std::marker::PhantomData,
        };
        for &(method, value) in methods {
            match host.emit_method((method / 4) as u16, value) {
                Ok(()) => {}
                Err(MaxwellThreeDMmePreflightError::Dispatch(error)) => panic!("{error}"),
                Err(MaxwellThreeDMmePreflightError::Consumer(never)) => match never {},
            }
        }
        flush(host.candidate, host.consume).unwrap();
        assert!(
            !host
                .candidate
                .operation_state()
                .vertex_input()
                .primitive()
                .is_open()
        );
        observed
    }

    #[test]
    fn hardware_instances_are_one_resource_capture_and_draw() {
        let mut methods = vec![(0x1618, 4), (0x0d78, 3), (0x1614, 0)];
        for _ in 0..63 {
            methods.extend([(0x1618, 1 << 26 | 4), (0x0d78, 3), (0x1614, 0)]);
        }
        assert_eq!(execute(&methods), [(0, 64, 3, 0)]);
    }

    #[test]
    fn state_changes_reset_geometry_and_restarted_sequences_flush_before_mutation() {
        assert_eq!(
            execute(&[
                (0x1618, 4),
                (0x0d78, 3),
                (0x1614, 0),
                (0x1618, 1 << 26 | 4),
                (0x0d78, 3),
                (0x1614, 0),
                (0x1438, 9),
                (0x1618, 1 << 26 | 4),
                (0x0d78, 6),
                (0x1614, 0),
                (0x1618, 4),
                (0x0d78, 3),
                (0x1614, 0),
            ]),
            [(0, 2, 3, 0), (2, 1, 6, 9), (0, 1, 3, 9)]
        );
    }
}
