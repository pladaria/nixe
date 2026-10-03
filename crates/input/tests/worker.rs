use std::thread;
use std::time::{Duration, Instant};

use nixe_input::InputWorker;

#[test]
fn sdl_main_thread_owner_outlives_worker_sampling_and_guest_reader() {
    // A separate test process gives SDL a single owner, independently of the
    // backend unit tests. This needs no connected controller or video surface.
    let sdl = sdl3::init().unwrap();
    let input = InputWorker::unmapped(&sdl).unwrap();
    let mut reader = input.reader();
    let mut reader = thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(15);
        while reader.take_latest().unwrap().is_none() {
            assert!(
                Instant::now() < deadline,
                "SDL input did not publish a state"
            );
            thread::sleep(Duration::from_millis(1));
        }
        reader
    })
    .join()
    .unwrap();
    drop(input);
    assert!(reader.take_latest().is_err());
    // Input shutdown leaves the shared SDL context available to another subsystem.
    let events = sdl.event().unwrap();
    drop(events);
}
