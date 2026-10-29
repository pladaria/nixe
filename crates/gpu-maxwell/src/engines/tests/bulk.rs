//! Differential traces exercise the actual production span dispatcher against
//! scalar delivery, including sources, cursor transitions and packed indices.
use super::*;

type Writes = Vec<(u64, u32)>;
pub(super) fn transfer(
    channel: &mut MaxwellGpuChannel,
    decoded: &MaxwellDecodedPushbuffer,
    scalar: bool,
) -> (Writes, usize, Result<(), MaxwellEngineDispatchError>) {
    let mut writes = Vec::new();
    let mut spans = 0;
    let mut mme = None;
    let mut parameters = Vec::new();
    let mut consume = async |event: MaxwellEngineEvent<'_>| {
        let (address, offset, value) = match event.operation {
            PendingEngineOperation::ThreeDInlineConstantBuffer(u) => {
                (u.address().get(), u.offset(), u.value())
            }
            PendingEngineOperation::InlineToMemory(u) => (u.address().get(), u.offset(), u.value()),
            PendingEngineOperation::ComputeInlineToMemory(u) => {
                (u.address().get(), u.offset(), u.value())
            }
            _ => panic!("expected a transfer"),
        };
        spans += 1;
        if let Some(words) = event.inline_words {
            for (i, word) in words.iter().enumerate() {
                writes.push((address + u64::from(offset) + i as u64 * 4, word.argument()));
            }
        } else {
            writes.push((address + u64::from(offset), value));
        }
        Ok::<_, std::convert::Infallible>(())
    };
    let packet = decoded.packets()[0].view();
    let packets: Vec<_> = if scalar {
        let crate::pushbuffer::packet::PacketView::Methods {
            header,
            subchannel,
            methods,
            ..
        } = packet
        else {
            panic!("method packet")
        };
        (0..methods.len())
            .map(|index| crate::pushbuffer::packet::PacketView::Methods {
                header,
                subchannel,
                methods: methods.slice(index..index + 1),
                last: index + 1 == methods.len(),
            })
            .collect()
    } else {
        vec![packet]
    };
    let result = packets
        .into_iter()
        .try_for_each(|packet| {
            crate::test_memory::ready(stream_maxwell_engine_entry(
                channel,
                &mut crate::pushbuffer::packet::CommandStream::fixture(packet),
                None,
                &mut mme,
                &mut parameters,
                &mut consume,
            ))
        })
        .map_err(|error| match error {
            MaxwellEngineStreamError::Dispatch(error) => *error,
            MaxwellEngineStreamError::Consumer(never) => match never {},
            MaxwellEngineStreamError::Decode(error) => panic!("invalid command fixture: {error}"),
        });
    (writes, spans, result)
}

fn cb() -> MaxwellGpuChannel {
    let mut channel = three_d_channel();
    for (m, v) in [(0x2380, 256), (0x2384, 0), (0x2388, 0x4000), (0x238c, 0)] {
        program_three_d(&mut channel, m, v);
    }
    channel
}

#[test]
fn constant_buffer_packet_modes_preserve_bytes_and_only_update_one_cursor() {
    let values: Vec<u32> = (0..24).map(|n| 0x12345600 + n).collect();
    for packet in [
        non_incrementing_packet_on_subchannel(0, 0x2390 / 4, &values),
        incrementing_packet_on_subchannel(0, 0x2390 / 4, &values),
        increment_once_packet(0x238c / 4, &[vec![0], values.clone()].concat()),
    ] {
        let mut fast = cb();
        let mut scalar = fast.clone();
        let a = transfer(&mut fast, &packet, false);
        let b = transfer(&mut scalar, &packet, true);
        a.2.unwrap();
        b.2.unwrap();
        assert_eq!(a.0, b.0);
        assert_eq!(a.1, 1);
        assert_eq!(b.1, values.len());
        assert_eq!(
            fast.three_d().shader_bindings().constant_buffer_offset(),
            scalar.three_d().shader_bindings().constant_buffer_offset()
        );
        assert_eq!(
            fast.three_d_mut().raw_register_value(GpuMethodId(0x2390)),
            scalar.three_d_mut().raw_register_value(GpuMethodId(0x2390))
        );
    }
}

#[test]
fn constant_buffer_excess_reports_exact_first_word_and_preserves_valid_prefix() {
    let mut fast = cb();
    program_three_d(&mut fast, 0x2380, 8);
    let mut scalar = fast.clone();
    let packet = non_incrementing_packet_on_subchannel(0, 0x2390 / 4, &[1, 2, 3, 4]);
    let a = transfer(&mut fast, &packet, false);
    let b = transfer(&mut scalar, &packet, true);
    assert_eq!(a.0, b.0);
    assert_eq!(a.0, vec![(0x4000, 1), (0x4004, 2)]);
    assert_eq!(a.2, b.2);
    assert!(
        matches!(a.2, Err(MaxwellEngineDispatchError::InvalidMethodEncoding {source,..}) if source.location().word_offset==3 && source.argument()==3)
    );
    assert_eq!(
        fast.three_d().shader_bindings().constant_buffer_offset(),
        scalar.three_d().shader_bindings().constant_buffer_offset()
    );
}

#[test]
fn equal_constant_buffer_offset_restarts_upload_and_shadow_replay_remains_executable() {
    let mut channel = cb();
    let packet = non_incrementing_packet_on_subchannel(0, 0x2390 / 4, &[10, 20, 30]);
    transfer(&mut channel, &packet, false).2.unwrap();
    program_three_d(&mut channel, 0x238c, 0);
    let (writes, _, result) = transfer(&mut channel, &packet, false);
    result.unwrap();
    assert_eq!(writes[0], (0x4000, 10));
    program_three_d(&mut channel, 0x124, 3);
    program_three_d(&mut channel, 0x238c, 0);
    let (writes, spans, result) = transfer(&mut channel, &packet, false);
    result.unwrap();
    assert_eq!(spans, 3);
    assert_eq!(writes, vec![(0x4000, 30), (0x4004, 30), (0x4008, 30)]);
}

#[test]
fn embedded_and_compute_uploads_preserve_partial_transfers_and_completion() {
    for (subchannel, class) in [
        (0, SWITCH_1_GM20B_PROFILE.classes().three_d()),
        (1, SWITCH_1_GM20B_PROFILE.classes().compute()),
    ] {
        let mut fast = channel();
        dispatch_first(
            &mut fast,
            &incrementing_packet_on_subchannel(subchannel, 0, &[class.0]),
        )
        .unwrap();
        dispatch_first(
            &mut fast,
            &incrementing_packet_on_subchannel(subchannel, 0x180 / 4, &[32, 1, 0, 0x8000]),
        )
        .unwrap();
        dispatch_first(
            &mut fast,
            &non_incrementing_packet_on_subchannel(subchannel, 0x1b0 / 4, &[1]),
        )
        .unwrap();
        let mut scalar = fast.clone();
        for values in [&[1, 2, 3][..], &[4, 5, 6, 7, 8][..]] {
            let packet = non_incrementing_packet_on_subchannel(subchannel, 0x1b4 / 4, values);
            let a = transfer(&mut fast, &packet, false);
            let b = transfer(&mut scalar, &packet, true);
            a.2.unwrap();
            b.2.unwrap();
            assert_eq!(a.0, b.0);
            assert_eq!(a.1, 1);
        }
        let packet = non_incrementing_packet_on_subchannel(subchannel, 0x1b4 / 4, &[9, 10]);
        assert_eq!(
            transfer(&mut fast, &packet, false).2,
            transfer(&mut scalar, &packet, true).2
        );
    }
}

#[test]
fn standalone_block_linear_spans_follow_gob_permutation_and_nonzero_origin() {
    let mut fast = channel();
    let class = SWITCH_1_GM20B_PROFILE.classes().inline_to_memory();
    dispatch_first(
        &mut fast,
        &incrementing_packet_on_subchannel(2, 0, &[class.0]),
    )
    .unwrap();
    dispatch_first(
        &mut fast,
        &incrementing_packet_on_subchannel(2, 0x180 / 4, &[64, 2, 0, 0x1000, 0]),
    )
    .unwrap();
    dispatch_first(
        &mut fast,
        &incrementing_packet_on_subchannel(2, 0x194 / 4, &[0, 128, 16, 1, 0, 16, 8]),
    )
    .unwrap();
    dispatch_first(
        &mut fast,
        &non_incrementing_packet_on_subchannel(2, 0x1b0 / 4, &[0]),
    )
    .unwrap();
    let mut scalar = fast.clone();
    let values: Vec<_> = (0..32).collect();
    let packet = non_incrementing_packet_on_subchannel(2, 0x1b4 / 4, &values);
    let a = transfer(&mut fast, &packet, false);
    let b = transfer(&mut scalar, &packet, true);
    a.2.unwrap();
    b.2.unwrap();
    assert_eq!(a.0, b.0);
    assert!(a.1 <= 8 && b.1 == 32);
    assert_eq!(a.0[0].0, 0x1000 + 1056);
    assert_eq!(a.0[4].0, 0x1000 + 1280);
}

#[test]
fn packed_inline_indices_are_owned_by_each_end_and_do_not_require_guest_index_storage() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x1618, 4);
    let mut captures = Vec::new();
    let mut mme = None;
    let mut parameters = Vec::new();
    for packet in [
        non_incrementing_packet_on_subchannel(0, 0x15ec / 4, &[0x00020001, 0x00040003]),
        non_incrementing_packet_on_subchannel(0, 0x1300 / 4, &[0x08070605, 0x0c0b0a09]),
        packet(0x1614 / 4, 0),
    ] {
        crate::test_memory::ready(stream_maxwell_engine_entry(
            &mut channel,
            &mut crate::pushbuffer::packet::CommandStream::fixture(packet.packets()[0].view()),
            None,
            &mut mme,
            &mut parameters,
            &mut async |event| {
                if let PendingEngineOperation::ThreeD(
                    MaxwellThreeDOperationTrigger::DrawIndexBuffer { index_count, .. },
                ) = event.operation
                {
                    assert_eq!(index_count, 12);
                    assert!(
                        event
                            .three_d
                            .vertex_input()
                            .primitive()
                            .active_begin()
                            .is_some()
                    );
                    captures.push(Arc::clone(event.three_d.inline_index().unwrap()));
                } else {
                    panic!("unexpected inline operation");
                }
                Ok::<_, std::convert::Infallible>(())
            },
        ))
        .unwrap_or_else(|_| panic!("inline stream failed"));
    }
    let expected: Vec<u8> = (1_u32..=12).flat_map(u32::to_le_bytes).collect();
    assert_eq!(&*captures[0], expected);
    assert!(channel.three_d().inline_index().is_none());
    assert!(
        channel
            .three_d()
            .vertex_input()
            .primitive()
            .active_begin()
            .is_none()
    );
    program_three_d(&mut channel, 0x1618, 4);
    program_three_d(&mut channel, 0x15e8, 99);
    let second = dispatch_method(&mut channel, 0x1614 / 4, 0).unwrap();
    let second = &second.operations()[0]
        .state()
        .inline_index()
        .unwrap()
        .clone();
    assert_eq!(&**second, 99_u32.to_le_bytes());
    assert_eq!(&*captures[0], expected);
}

#[test]
fn repeated_begin_executes_instance_transition_without_rebuilding_unchanged_parameters() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x1618, 4);
    program_three_d(&mut channel, 0x1618, 4 | (1 << 26));
    let before = channel.three_d().draw_state_identity();
    program_three_d(&mut channel, 0x1618, 4 | (1 << 26));
    assert_eq!(
        channel
            .three_d()
            .vertex_input()
            .primitive()
            .instance_index(),
        2
    );
    assert!(before.matches_parameters(channel.three_d()));
}

#[test]
fn inline_restart_requires_real_state_and_retains_a_normalized_immutable_stream() {
    let mut channel = three_d_channel();
    program_three_d(&mut channel, 0x1618, 4);
    program_three_d(&mut channel, 0x1644, 1);
    dispatch_first(
        &mut channel,
        &non_incrementing_packet_on_subchannel(0, 0x15e8 / 4, &[1, 0xff, 2]),
    )
    .unwrap();
    assert!(matches!(
        dispatch_method(&mut channel, 0x1614 / 4, 0),
        Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
            method_name: "INLINE_INDEX",
            ..
        })
    ));
    program_three_d(&mut channel, 0x1648, 0xff);
    let operation = dispatch_method(&mut channel, 0x1614 / 4, 0).unwrap();
    let bytes = operation.operations()[0]
        .state()
        .inline_index()
        .unwrap()
        .clone();
    assert_eq!(
        &*bytes,
        [1_u32, u32::MAX, 2]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>()
    );
    assert!(matches!(
        dispatch_method(&mut channel, 0x15e8 / 4, 1),
        Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
            method_name: "INLINE_INDEX",
            ..
        })
    ));
}
