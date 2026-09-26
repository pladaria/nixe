use std::thread;
use std::time::{Duration, Instant};

use nixe_input::InputWorker;

#[test]
fn sdl_initializes_and_polls_on_the_input_worker() {
    // A separate test process gives SDL a single owner, independently of the
    // backend unit tests. This needs no connected controller or video surface.
    let mut input = InputWorker::unmapped().unwrap();
    let deadline = Instant::now() + Duration::from_secs(15);
    while input.take_latest().unwrap().is_none() {
        assert!(
            Instant::now() < deadline,
            "SDL input did not publish a state"
        );
        thread::sleep(Duration::from_millis(1));
    }
}
