//! Short physical actuator check: cargo run -p nixe-input --example vibration.

use nixe_input::{InputWorker, VibrationSide, VibrationValue};
use std::thread;
use std::time::{Duration, Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sdl = sdl3::init()?;
    let mut input = InputWorker::unmapped(&sdl)?;
    let output = input.vibration_output().expect("SDL actuator output");
    let deadline = Instant::now() + Duration::from_secs(5);
    let controller = loop {
        if let Some(sample) = input.take_latest()?
            && let Some(controller) = sample.state
        {
            break controller;
        }
        if Instant::now() >= deadline {
            return Err("no connected controller".into());
        }
        thread::sleep(Duration::from_millis(5));
    };
    println!("Testing {} ({:?})", controller.name, controller.kind);
    output.select_controller(Some(controller.id))?;
    for side in [VibrationSide::Left, VibrationSide::Right] {
        println!("{side:?}: 160 Hz / 320 Hz, force 0.2, 300 ms");
        output.send(
            side,
            VibrationValue {
                low_amplitude: 0.2,
                low_frequency: 160.0,
                high_amplitude: 0.2,
                high_frequency: 320.0,
            },
        )?;
        let deadline = Instant::now() + Duration::from_millis(300);
        while Instant::now() < deadline {
            input.take_latest()?;
            thread::sleep(Duration::from_millis(5));
        }
        output.stop()?;
        thread::sleep(Duration::from_millis(100));
        input.take_latest()?;
    }
    println!("Actuator output accepted; motors stopped.");
    Ok(())
}
