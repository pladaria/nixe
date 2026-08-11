use std::sync::{Arc, Barrier};
use std::thread;

use nixe_runtime::{EventObject, ExternalEvent, ExternalEventInbox, ExternalEventSource};
use nixe_scheduler::{GuestThreadId, WakeGeneration, WakeToken};

#[test]
fn concurrent_device_producers_are_published_in_one_monotonic_order() {
    const PRODUCERS: usize = 24;
    let inbox = ExternalEventInbox::bounded(PRODUCERS).unwrap();
    let sender = inbox.sender();
    let start = Arc::new(Barrier::new(PRODUCERS + 1));
    let workers: Vec<_> = (0..PRODUCERS)
        .map(|index| {
            let sender = sender.clone();
            let start = Arc::clone(&start);
            thread::spawn(move || {
                start.wait();
                let source = match index % 3 {
                    0 => ExternalEventSource::GpuCompletion,
                    1 => ExternalEventSource::Display,
                    _ => ExternalEventSource::Input,
                };
                let token = WakeToken {
                    thread: GuestThreadId::new(index as u64 + 1),
                    generation: WakeGeneration::new(1),
                };
                match source {
                    ExternalEventSource::GpuCompletion => sender.gpu_completion(token),
                    ExternalEventSource::Display => sender.display_event(token),
                    ExternalEventSource::Input => sender.input_event(token),
                    _ => unreachable!(),
                }
                .unwrap();
            })
        })
        .collect();
    start.wait();
    for worker in workers {
        worker.join().unwrap();
    }

    let mut previous = 0;
    for _ in 0..PRODUCERS {
        let event = inbox.try_recv_sequenced().unwrap().unwrap();
        assert!(event.sequence.get() > previous);
        previous = event.sequence.get();
    }
    assert_eq!(previous, PRODUCERS as u64);
}

#[test]
fn typed_event_object_overrides_the_generic_wait_fallback_source() {
    let inbox = ExternalEventInbox::bounded(1).unwrap();
    let token = WakeToken {
        thread: GuestThreadId::new(1),
        generation: WakeGeneration::new(1),
    };
    let (writable, readable) = EventObject::create_pair_with_source(ExternalEventSource::Display);
    inbox
        .sender()
        .watch_readable_event(readable, None, token, ExternalEventSource::Device)
        .unwrap();
    writable.signal();
    assert_eq!(
        inbox.try_recv_sequenced().unwrap().unwrap().event,
        ExternalEvent::Wake {
            source: ExternalEventSource::Display,
            token,
        }
    );
}
