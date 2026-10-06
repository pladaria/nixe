//! Exercise the real worker and SDL output ABI without a physical actuator.

use std::ffi::c_void;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use nixe_input::{InputWorker, VibrationSide, VibrationValue};
use sdl3_sys::joystick::{
    SDL_AttachVirtualJoystick, SDL_DetachVirtualJoystick, SDL_JOYSTICK_TYPE_GAMEPAD,
    SDL_VirtualJoystickDesc,
};

#[derive(Default)]
struct Effects {
    packets: Mutex<Vec<Vec<u8>>>,
    reject: AtomicBool,
}

unsafe extern "C" fn rumble(_: *mut c_void, _: u16, _: u16) -> bool {
    true
}

unsafe extern "C" fn send_effect(userdata: *mut c_void, data: *const c_void, size: i32) -> bool {
    // SDL retains userdata only until detach; the boxed Effects outlives it.
    let effects = unsafe { &*userdata.cast::<Effects>() };
    let packet = unsafe { std::slice::from_raw_parts(data.cast::<u8>(), size as usize) };
    effects.packets.lock().unwrap().push(packet.to_vec());
    if effects.reject.load(Ordering::Acquire) {
        unsafe {
            sdl3_sys::error::SDL_SetError(c"virtual actuator rejected packet".as_ptr());
        }
        return false;
    }
    true
}

fn wait_for(effects: &Effects, predicate: impl Fn(&[u8]) -> bool) {
    let deadline = Instant::now() + Duration::from_secs(3);
    while !effects
        .packets
        .lock()
        .unwrap()
        .iter()
        .any(|packet| predicate(packet))
    {
        assert!(
            Instant::now() < deadline,
            "SDL actuator packet was not delivered"
        );
        thread::sleep(Duration::from_millis(2));
    }
}

#[test]
fn native_waveforms_refresh_and_stop_on_retarget_and_worker_shutdown() {
    // Isolate this process from physical devices: the test must never vibrate
    // someone's real controller. SDL's virtual driver remains available.
    sdl3::hint::set("SDL_JOYSTICK_HIDAPI", "0");
    sdl3::hint::set("SDL_JOYSTICK_LINUX_CLASSIC", "0");
    sdl3::hint::set("SDL_JOYSTICK_LINUX_JOYSTICK", "0");
    let sdl = sdl3::init().unwrap();
    let _gamepads = sdl.gamepad().unwrap();
    let effects = Box::<Effects>::default();
    let mut description = SDL_VirtualJoystickDesc::new();
    description.r#type = SDL_JOYSTICK_TYPE_GAMEPAD.0 as u16;
    description.vendor_id = 0x057e;
    description.product_id = 0x2009;
    description.naxes = 6;
    description.nbuttons = 16;
    description.button_mask = 0xffff;
    description.axis_mask = 0x3f;
    description.name = c"Nixe virtual Switch Pro".as_ptr();
    description.userdata = (&*effects as *const Effects).cast_mut().cast();
    description.Rumble = Some(rumble);
    description.SendEffect = Some(send_effect);
    let instance = unsafe { SDL_AttachVirtualJoystick(&description) };
    assert_ne!(instance.0, 0);
    let mut input = InputWorker::unmapped(&sdl).unwrap();
    let output = input.vibration_output().unwrap();
    let deadline = Instant::now() + Duration::from_secs(3);
    let controller = loop {
        if let Some(sample) = input.take_latest().unwrap()
            && let Some(controller) = sample.state
        {
            assert_eq!(&*controller.name, "Nixe virtual Switch Pro");
            break controller;
        }
        assert!(Instant::now() < deadline);
        thread::sleep(Duration::from_millis(2));
    };
    output.select_controller(Some(controller.id)).unwrap();
    output
        .send(
            VibrationSide::Left,
            VibrationValue {
                low_amplitude: 0.5,
                low_frequency: 160.0,
                high_amplitude: 0.5,
                high_frequency: 320.0,
            },
        )
        .unwrap();
    let active = [0x10, 0, 0, 0x89, 0x40, 0x62, 0, 1, 0x40, 0x40];
    let stopped = [0x10, 0, 0, 1, 0x40, 0x40, 0, 1, 0x40, 0x40];
    wait_for(&effects, |packet| packet == active);
    // A sustained waveform must be refreshed even without more guest writes.
    thread::sleep(Duration::from_millis(120));
    assert!(
        effects
            .packets
            .lock()
            .unwrap()
            .iter()
            .filter(|packet| packet.as_slice() == active)
            .count()
            >= 2
    );
    effects.packets.lock().unwrap().clear();
    output.select_controller(None).unwrap();
    wait_for(&effects, |packet| packet == stopped);
    effects.packets.lock().unwrap().clear();
    output.select_controller(Some(controller.id)).unwrap();
    output
        .send(
            VibrationSide::Left,
            VibrationValue {
                low_amplitude: 0.5,
                low_frequency: 160.0,
                high_amplitude: 0.5,
                high_frequency: 320.0,
            },
        )
        .unwrap();
    wait_for(&effects, |packet| packet == active);
    effects.packets.lock().unwrap().clear();
    drop(input);
    wait_for(&effects, |packet| packet == stopped);

    // An actuator failure must reach the consumer through the existing worker
    // failure path, and shutdown must attempt to stop a partially sent effect.
    effects.packets.lock().unwrap().clear();
    effects.reject.store(true, Ordering::Release);
    let mut input = InputWorker::unmapped(&sdl).unwrap();
    let output = input.vibration_output().unwrap();
    let deadline = Instant::now() + Duration::from_secs(3);
    let controller = loop {
        if let Some(sample) = input.take_latest().unwrap()
            && let Some(controller) = sample.state
        {
            break controller;
        }
        assert!(Instant::now() < deadline);
        thread::sleep(Duration::from_millis(2));
    };
    output.select_controller(Some(controller.id)).unwrap();
    output
        .send(
            VibrationSide::Left,
            VibrationValue {
                low_amplitude: 0.5,
                low_frequency: 160.0,
                high_amplitude: 0.5,
                high_frequency: 320.0,
            },
        )
        .unwrap();
    loop {
        if let Err(error) = input.take_latest() {
            assert!(
                error
                    .to_string()
                    .contains("virtual actuator rejected packet")
            );
            break;
        }
        assert!(Instant::now() < deadline);
        thread::sleep(Duration::from_millis(2));
    }
    drop(input);
    wait_for(&effects, |packet| packet == stopped);
    assert!(unsafe { SDL_DetachVirtualJoystick(instance) });
}
