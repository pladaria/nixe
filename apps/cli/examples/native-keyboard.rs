//! Native software-keyboard smoke test using the normal Horizon GPU owner.
//! Run with a real GPU/window and validation layers; Enter prints the submission.
use std::sync::{Arc, atomic::AtomicBool};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut window = nixe_video_winit::WindowFrontend::new(Arc::new(AtomicBool::new(false)))?;
    let (device, client) = window.create_vulkan_device(nixe_gpu_vulkan::DeviceOptions {
        validation: nixe_gpu_vulkan::ValidationMode::Enabled,
        ..Default::default()
    })?;
    let monitor = device.validation_monitor();
    let mut renderer = nixe_gpu_vulkan::NativeGpuExecution::new(
        device,
        nixe_gpu_maxwell::MAXWELL_SHADER_NAMESPACE,
        nixe_gpu_vulkan::cache::CacheConfiguration::default(),
        128 * 1024 * 1024,
    )?;
    renderer.attach_presentation(client)?;
    let frontend = nixe_gpu_maxwell::MaxwellNativeFrontend::new(renderer.shader_translation_pool());
    let video =
        nixe_horizon::VideoSystem::with_gpu_frontend(Some(window.mailbox()), renderer, frontend);
    let control = window.control();
    let host = window.keyboard_host();
    let session = host.open(
        nixe_applets::keyboard::KeyboardConfig {
            kind: nixe_applets::keyboard::KeyboardType::Ascii,
            ok: "Accept".into(),
            header: "Native Vulkan keyboard".into(),
            subtitle: "Presentation / egui smoke test".into(),
            guide: "Type text and press Enter".into(),
            initial: "Nixe".into(),
            cursor_at_end: true,
            minimum: 1,
            maximum: 80,
            password: false,
            multiline: false,
            cancel_disabled: false,
            text_check: false,
            utf8: true,
            disabled: 0,
            optional_symbols: [None, None],
        },
        move |submission| {
            println!("{submission:?}");
            control.worker_finished();
        },
    )?;
    window.run()?;
    session.finish();
    host.close(&session);
    drop(video);
    monitor.check()?;
    Ok(())
}
