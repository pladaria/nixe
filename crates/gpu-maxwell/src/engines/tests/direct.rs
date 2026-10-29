use super::*;
use crate::pushbuffer::packet::PacketView;
use std::convert::Infallible;

#[test]
fn one_entry_consumes_many_packets_and_large_payloads_before_the_ordered_effect() {
    let mut channel = three_d_channel();
    let values = vec![1.0_f32.to_bits(); 8191];
    let mut words = vec![(3 << 29) | (8191 << 16) | (0x1518 / 4)];
    words.extend(values);
    for (method, value) in [(0x1618, 4), (0xd78, 3), (0x1614, 0), (0x1438, 9)] {
        words.extend([(1 << 29) | (1 << 16) | (method / 4), value]);
    }
    let mut stream = crate::pushbuffer::packet::CommandStream::fixture_words(
        word(0, 0).unwrap().location(),
        &words,
    );
    let mut draws = Vec::new();
    crate::test_memory::ready(stream_maxwell_engine_entry(
        &mut channel,
        &mut stream,
        None,
        &mut None,
        &mut Vec::new(),
        &mut async |event| {
            assert!(matches!(event.operation, PendingEngineOperation::ThreeD(_)));
            assert_eq!(event.instances, Some((0, 1)));
            assert_eq!(
                event.three_d.raster().point_size().raw(),
                Some(1.0_f32.to_bits())
            );
            draws.push(event.instances);
            Ok::<_, Infallible>(())
        },
    ))
    .unwrap_or_else(|_| panic!("command entry"));
    assert_eq!(draws.len(), 1);
    assert!(stream.needs_entry());
    stream
        .finish(&mut crate::MaxwellCommandSourceScratch::default())
        .unwrap();
}

#[test]
fn repeated_condition_commands_flush_prior_instances_and_execute_in_order() {
    let mut channel = three_d_channel();
    let mut mme = None;
    let mut parameters = Vec::new();
    let mut observed = Vec::new();
    let mut consume = async |event: MaxwellEngineEvent<'_>| {
        match event.operation {
            PendingEngineOperation::ThreeD(_) => observed.push((1, event.instances.unwrap().1)),
            PendingEngineOperation::ThreeDRenderCondition(source) => {
                observed.push((source.method().0, source.argument()))
            }
            _ => panic!("unexpected operation"),
        }
        Ok::<_, Infallible>(())
    };
    for (method, value) in [
        (0x1558, 1),
        (0x1618, 4),
        (0xd78, 3),
        (0x1614, 0),
        (0x1558, 1),
        (0x1558, 1),
        (0x1944, 2),
    ] {
        let packet = packet(method / 4, value);
        crate::test_memory::ready(stream_maxwell_engine_entry(
            &mut channel,
            &mut crate::pushbuffer::packet::CommandStream::fixture(packet.packets()[0].view()),
            None,
            &mut mme,
            &mut parameters,
            &mut consume,
        ))
        .unwrap_or_else(|_| panic!("dispatch"));
    }
    crate::test_memory::ready(finish_maxwell_engine_stream(&mut channel, &mut consume)).unwrap();
    assert_eq!(
        observed,
        [(0x1558, 1), (1, 1), (0x1558, 1), (0x1558, 1), (0x1944, 2)]
    );
}

#[test]
fn instance_runs_cross_packets_and_flush_before_state_and_waits() {
    let mut channel = three_d_channel();
    let mut mme = None;
    let mut parameters = Vec::new();
    let mut observed = Vec::new();
    let mut consume = async |event: MaxwellEngineEvent<'_>| {
        match event.operation {
            PendingEngineOperation::ThreeD(_) => {
                assert!(
                    event
                        .three_d
                        .vertex_input()
                        .primitive()
                        .active_begin()
                        .is_some()
                );
                observed.push(event.instances.unwrap().1);
            }
            PendingEngineOperation::ThreeDSynchronization(_) => observed.push(0),
            _ => panic!("unexpected operation"),
        }
        Ok::<_, Infallible>(())
    };
    for (method, value) in [
        (0x1618, 4),
        (0xd78, 3),
        (0x1614, 0),
        (0x08, 123),
        (0x192c, 0),
        (0x1618, (1 << 26) | 4),
        (0xd78, 3),
        (0x1614, 0),
        (0x1438, 9),
        (0x1618, 4),
        (0xd78, 3),
        (0x1614, 0),
        (0x110, 0),
    ] {
        let packet = packet(method / 4, value);
        crate::test_memory::ready(stream_maxwell_engine_entry(
            &mut channel,
            &mut crate::pushbuffer::packet::CommandStream::fixture(packet.packets()[0].view()),
            None,
            &mut mme,
            &mut parameters,
            &mut consume,
        ))
        .unwrap_or_else(|_| panic!("dispatch"));
    }
    crate::test_memory::ready(finish_maxwell_engine_stream(&mut channel, &mut consume)).unwrap();
    assert_eq!(observed, [2, 1, 0]);
    assert!(!channel.three_d().vertex_input().primitive().is_open());
}

#[test]
fn mme_parameters_continue_across_spans_and_macro_end_consumes_pending_draw() {
    let mut channel = three_d_channel();
    let code = [
        1 | (2 << 8),
        1 | (2 << 4) | ((0xd78 / 4) << 14),
        (4 << 4) | (1 << 7) | (2 << 11),
        0x11,
    ];
    load_mme_program(&mut channel, 2, &code);
    program_three_d(&mut channel, 0x1618, 4);
    let packet = incrementing_packet((0x3800 + 2 * 8) / 4, &[0xdead_beef, 7]);
    let PacketView::Methods {
        header,
        subchannel,
        methods,
        ..
    } = packet.packets()[0].view()
    else {
        panic!()
    };
    let mut mme = None;
    let mut parameters = Vec::new();
    let mut draws = Vec::new();
    for index in 0..2 {
        crate::test_memory::ready(stream_maxwell_engine_entry(
            &mut channel,
            &mut crate::pushbuffer::packet::CommandStream::fixture(PacketView::Methods {
                header,
                subchannel,
                methods: methods.slice(index..index + 1),
                last: index == 1,
            }),
            None,
            &mut mme,
            &mut parameters,
            &mut async |event| {
                let PendingEngineOperation::ThreeD(
                    MaxwellThreeDOperationTrigger::DrawVertexArray { vertex_count, .. },
                ) = event.operation
                else {
                    panic!()
                };
                draws.push((vertex_count, event.instances));
                Ok::<_, Infallible>(())
            },
        ))
        .unwrap_or_else(|_| panic!("dispatch"));
        if index == 0 {
            assert!(draws.is_empty());
            assert_eq!(parameters, [0xdead_beef]);
        }
    }
    assert_eq!(draws, [(7, Some((0, 1)))]);
    assert!(mme.is_none());
    assert!(parameters.is_empty());
    crate::test_memory::ready(finish_maxwell_engine_stream(
        &mut channel,
        &mut async |_| -> Result<(), Infallible> { panic!("macro already consumed draw") },
    ))
    .unwrap();
}

#[test]
fn mme_parameter_packet_modes_execute_one_complete_macro_call() {
    let code = [
        1 | (2 << 8),
        1 | (2 << 4) | ((0xd78 / 4) << 14),
        (4 << 4) | (1 << 7) | (2 << 11),
        0x11,
    ];
    let method = (0x3800 + 2 * 8) / 4;
    for packet in [
        incrementing_packet(method, &[0xdead_beef, 7]),
        increment_once_packet(method, &[0xdead_beef, 7]),
        non_incrementing_packet_on_subchannel(0, method, &[0xdead_beef, 7]),
    ] {
        let mut channel = three_d_channel();
        load_mme_program(&mut channel, 2, &code);
        program_three_d(&mut channel, 0x1618, 4);
        let mut mme = None;
        let mut parameters = Vec::new();
        let mut draws = Vec::new();
        crate::test_memory::ready(stream_maxwell_engine_entry(
            &mut channel,
            &mut crate::pushbuffer::packet::CommandStream::fixture(packet.packets()[0].view()),
            None,
            &mut mme,
            &mut parameters,
            &mut async |event| {
                let PendingEngineOperation::ThreeD(
                    MaxwellThreeDOperationTrigger::DrawVertexArray { vertex_count, .. },
                ) = event.operation
                else {
                    panic!("expected one macro draw")
                };
                draws.push((vertex_count, event.instances));
                Ok::<_, Infallible>(())
            },
        ))
        .unwrap_or_else(|_| panic!("macro dispatch failed"));
        assert_eq!(draws, [(7, Some((0, 1)))]);
        assert!(mme.is_none());
        assert!(parameters.is_empty());
    }
}

#[test]
fn suspended_macro_send_resumes_without_repeating_prior_effects() {
    use std::{
        cell::{Cell, RefCell},
        future::{Future, poll_fn},
        task::{Context, Poll, Waker},
    };
    let mut channel = three_d_channel();
    for (method, value) in [(0x2380, 8), (0x2384, 0), (0x2388, 0x4000), (0x238c, 0)] {
        program_three_d(&mut channel, method, value);
    }
    load_mme_program(
        &mut channel,
        2,
        &[
            1 | (2 << 8),
            1 | (2 << 4) | ((0x2390 / 4) << 14),
            (4 << 4) | (2 << 11),
            (4 << 4) | (1 << 7) | (2 << 11),
            0x11,
        ],
    );
    let input = incrementing_packet((0x3800 + 2 * 8) / 4, &[0, 7]);
    let ready = Cell::new(false);
    let offsets = RefCell::new(Vec::new());
    let mut mme = None;
    let mut parameters = Vec::new();
    let mut consume = async |event: MaxwellEngineEvent<'_>| {
        let PendingEngineOperation::ThreeDInlineConstantBuffer(upload) = event.operation else {
            panic!("CB write");
        };
        assert_eq!(upload.value(), 7);
        offsets.borrow_mut().push(upload.offset());
        if upload.offset() == 4 {
            poll_fn(|_| {
                if ready.get() {
                    Poll::Ready(())
                } else {
                    Poll::Pending
                }
            })
            .await;
        }
        Ok::<_, Infallible>(())
    };
    let mut stream = crate::pushbuffer::packet::CommandStream::fixture(input.packets()[0].view());
    let future = stream_maxwell_engine_entry(
        &mut channel,
        &mut stream,
        None,
        &mut mme,
        &mut parameters,
        &mut consume,
    );
    let mut future = std::pin::pin!(future);
    let mut context = Context::from_waker(Waker::noop());
    for _ in 0..3 {
        assert!(future.as_mut().poll(&mut context).is_pending());
        assert_eq!(*offsets.borrow(), [0, 4]);
    }
    ready.set(true);
    assert!(matches!(
        future.as_mut().poll(&mut context),
        Poll::Ready(Ok(()))
    ));
    assert_eq!(*offsets.borrow(), [0, 4]);
}
