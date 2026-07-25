use std::fmt::Write as _;
use std::io::{self, IsTerminal, Write as _};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

use nixe_input::{
    Axis, Button, ControllerState, InputManager, MotionSensor, MotionVector, sdl::SdlInputBackend,
};

const POLL_INTERVAL: Duration = Duration::from_millis(16);

pub fn run() -> Result<(), String> {
    if !io::stdout().is_terminal() {
        return Err("input display requires an interactive terminal".to_owned());
    }

    let interrupted = install_interrupt_handler()?;
    let backend = SdlInputBackend::new().map_err(|error| error.to_string())?;
    let mut input = InputManager::new(backend);
    let _terminal = TerminalSession::enter()?;
    let mut previous = None;

    while !interrupted.load(Ordering::Acquire) {
        let state = input.read_input().map_err(|error| error.to_string())?;
        if previous.as_ref() != Some(&state) {
            render(&state)?;
            previous = Some(state);
        }
        thread::sleep(POLL_INTERVAL);
    }
    Ok(())
}

fn install_interrupt_handler() -> Result<Arc<AtomicBool>, String> {
    let interrupted = Arc::new(AtomicBool::new(false));
    let signal_flag = Arc::clone(&interrupted);
    ctrlc::set_handler(move || signal_flag.store(true, Ordering::Release))
        .map_err(|error| format!("cannot install Ctrl+C handler: {error}"))?;
    Ok(interrupted)
}

fn render(state: &Option<ControllerState>) -> Result<(), String> {
    let contents = format_state(state);
    let mut stdout = io::stdout().lock();
    write!(stdout, "\x1b[H{contents}\x1b[J")
        .and_then(|()| stdout.flush())
        .map_err(|error| format!("cannot update input display: {error}"))
}

fn format_state(state: &Option<ControllerState>) -> String {
    let Some(state) = state else {
        return "Waiting for a gamepad...\n\nPress Ctrl+C to exit.\n".to_owned();
    };

    let mut output = String::new();
    let _ = writeln!(output, "Device: {}", state.name);
    let _ = writeln!(output, "Type: {}\n", state.kind.identifier());
    let _ = writeln!(output, "Buttons");
    for button in Button::ALL {
        let _ = writeln!(
            output,
            "  {:<27} {}",
            button.identifier(),
            pressed(state.buttons.contains(button))
        );
    }
    let _ = writeln!(output, "\nAxes");
    for axis in Axis::ALL {
        write_axis(&mut output, axis, state);
    }

    let _ = writeln!(output, "\nSensors");
    for sensor in MotionSensor::ALL {
        write_sensor(&mut output, sensor, sensor_value(state, sensor));
    }
    let _ = writeln!(output, "\nPress Ctrl+C to exit.");
    output
}

const fn pressed(value: bool) -> &'static str {
    if value { "⬤" } else { "◯" }
}

fn normalize_axis(value: i16) -> f32 {
    if value < 0 {
        f32::from(value) / 32_768.0
    } else {
        f32::from(value) / 32_767.0
    }
}

fn normalize_trigger(value: u16) -> f32 {
    f32::from(value) / f32::from(u16::MAX)
}

fn write_axis(output: &mut String, axis: Axis, state: &ControllerState) {
    match axis {
        Axis::LeftX => write_signed_axis(output, axis, state.left_stick.x),
        Axis::LeftY => write_signed_axis(output, axis, state.left_stick.y),
        Axis::RightX => write_signed_axis(output, axis, state.right_stick.x),
        Axis::RightY => write_signed_axis(output, axis, state.right_stick.y),
        Axis::LeftTrigger => write_trigger(output, axis, state.triggers.left),
        Axis::RightTrigger => write_trigger(output, axis, state.triggers.right),
    }
}

fn write_signed_axis(output: &mut String, axis: Axis, value: i16) {
    let _ = writeln!(
        output,
        "  {:<27} {:+6} ({:+.3})",
        axis.identifier(),
        value,
        normalize_axis(value)
    );
}

fn write_trigger(output: &mut String, axis: Axis, value: u16) {
    let _ = writeln!(
        output,
        "  {:<27} {:5} ({:.3})",
        axis.identifier(),
        value,
        normalize_trigger(value)
    );
}

const fn sensor_value(state: &ControllerState, sensor: MotionSensor) -> Option<MotionVector> {
    match sensor {
        MotionSensor::Gyroscope => state.motion.gyroscope,
        MotionSensor::Accelerometer => state.motion.accelerometer,
        MotionSensor::LeftGyroscope => state.motion.left_gyroscope,
        MotionSensor::RightGyroscope => state.motion.right_gyroscope,
        MotionSensor::LeftAccelerometer => state.motion.left_accelerometer,
        MotionSensor::RightAccelerometer => state.motion.right_accelerometer,
    }
}

fn write_sensor(output: &mut String, sensor: MotionSensor, value: Option<MotionVector>) {
    let name = sensor.identifier();
    match value {
        Some(value) => {
            let _ = writeln!(
                output,
                "  {name:<27} x {:+9.4}  y {:+9.4}  z {:+9.4}",
                value.x, value.y, value.z
            );
        }
        None => {
            let _ = writeln!(output, "  {name:<27} unsupported");
        }
    }
}

struct TerminalSession;

impl TerminalSession {
    fn enter() -> Result<Self, String> {
        let mut stdout = io::stdout().lock();
        write!(stdout, "\x1b[2J\x1b[H\x1b[?25l")
            .and_then(|()| stdout.flush())
            .map_err(|error| format!("cannot initialize input display: {error}"))?;
        Ok(Self)
    }
}

impl Drop for TerminalSession {
    fn drop(&mut self) {
        let _ = writeln!(io::stdout(), "\x1b[?25h").and_then(|()| io::stdout().flush());
    }
}

#[cfg(test)]
mod tests {
    use nixe_input::{
        ButtonSet, ControllerId, ControllerKind, DPadState, FaceButtonLabels, MotionState,
        StickState, TriggerState,
    };

    use super::*;

    #[test]
    fn formats_waiting_state() {
        assert!(format_state(&None).starts_with("Waiting for a gamepad"));
    }

    #[test]
    fn formats_labels_controls_and_unavailable_sensors() {
        let mut buttons = ButtonSet::default();
        buttons.set(Button::South, true);
        let state = ControllerState {
            id: ControllerId::new(1),
            name: "Test pad".to_owned(),
            kind: ControllerKind::Standard,
            buttons,
            button_labels: FaceButtonLabels::default(),
            dpad: DPadState::default(),
            left_stick: StickState { x: -32768, y: 0 },
            right_stick: StickState::default(),
            triggers: TriggerState {
                left: 0,
                right: u16::MAX,
            },
            motion: MotionState::default(),
        };

        let output = format_state(&Some(state));
        assert!(output.contains("Device: Test pad"));
        assert!(output.contains("Type: standard"));
        assert!(output.contains("south                       ⬤"));
        assert!(output.contains("east                        ◯"));
        assert!(output.contains("leftx"));
        assert!(output.contains("-1.000"));
        assert!(output.contains("65535 (1.000)"));
        assert!(output.contains("gyroscope                   unsupported"));
    }
}
