//! Opt-in native interoperability and emitted patch-shader proof, using production
//! device creation and production neutral submissions. No guest-specific replacements.
#![cfg(not(target_os = "macos"))]

#[path = "../test-support/hardware.rs"]
mod hardware;

#[path = "native_interop/builtins.rs"]
mod builtins;
#[cfg(target_os = "linux")]
#[path = "native_interop/capture.rs"]
mod capture;
#[path = "native_interop/measure.rs"]
mod measure;

#[global_allocator]
static ALLOCATOR: measure::CountingAllocator = measure::CountingAllocator;

#[test]
#[cfg(not(debug_assertions))]
#[ignore = "release-only CPU/allocator measurement; select this test explicitly, see README"]
fn native_warm_path_measurements() {
    if hardware::native_capabilities(true).is_none() {
        return;
    }
    production::benchmark();
}

#[test]
#[cfg(not(debug_assertions))]
#[ignore = "release-only cold/persisted submission measurement; set NIXE_TEST_NATIVE_CACHE_DIR, see README"]
fn native_pipeline_cache_measurements() {
    if hardware::native_capabilities(false).is_none() {
        return;
    }
    production::benchmark_pipeline_cache();
}
#[path = "native_interop/default_control.rs"]
mod default_control;
#[path = "native_interop/device.rs"]
mod device;
#[path = "native_interop/emitted.rs"]
mod emitted;
#[path = "native_interop/native.rs"]
mod native;
#[path = "native_interop/normal.rs"]
mod normal;
#[path = "native_interop/production.rs"]
mod production;

use device::Context;
use nixe_gpu::{BackendInstanceId, BackendResourceHandle, BackendResourceKind};
use normal::{NormalPipelines, Resources};
use std::collections::HashMap;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

const SIZE: u32 = 32;
static ACCEPTANCE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
#[ignore = "requires Vulkan tessellation and Khronos core/sync validation; see native_interop/README.md"]
fn wgpu_native_polygon_facing() {
    let _guard = ACCEPTANCE_LOCK.lock().unwrap();
    if hardware::native_capabilities(false).is_none() {
        return;
    }
    device::enable_validation();
    production::check_culling();
    device::assert_validation_clean();
}

#[test]
#[ignore = "requires native rectangular/smooth lines and Khronos core/sync validation; see native_interop/README.md"]
fn wgpu_native_wireframe_rasterization() {
    let _guard = ACCEPTANCE_LOCK.lock().unwrap();
    if hardware::native_capabilities(true).is_none() {
        return;
    }
    device::enable_validation();
    production::check_wireframe();
    device::assert_validation_clean();
}

#[test]
#[ignore = "requires Vulkan tessellation, Khronos core/sync validation, glslangValidator and spirv-val; see native_interop/README.md"]
fn wgpu_native_tessellation_interoperability() {
    let _guard = ACCEPTANCE_LOCK.lock().unwrap();
    if hardware::native_capabilities(false).is_none() {
        return;
    }
    device::enable_validation();
    production::check();
    {
        let ctx = Context::new(true);
        emitted::check(&ctx);
        default_control::check(&ctx);
        let normal = NormalPipelines::new(&ctx);
        let pipeline = native::Pipeline::new(&ctx);
        let completions = Arc::new(AtomicUsize::new(0));
        let mut oracles = Vec::new();
        let mut retired = Vec::new();
        // Prototype cache only: full neutral handle identity, not a slot-only key.
        let mut cache = HashMap::new();
        // Enqueue several generations without polling/waiting between them.
        // Reusing the same logical slot replaces its backing and descriptors;
        // old owners survive only through the completion retention below.
        for generation in 0..6 {
            let r = Resources::new(&ctx);
            let bindings = native::Bindings::new(
                pipeline.clone(),
                &r.buffer,
                [&r.color, &r.depth, &r.image],
                [&r.color_views[1], &r.depth_views[1], &r.image_view],
            );
            let key = BackendResourceHandle::new(
                BackendInstanceId::new(1),
                5,
                generation + 1,
                BackendResourceKind::Image,
            );
            assert!(cache.insert(key, bindings.clone()).is_none());
            assert!(Arc::ptr_eq(cache.get(&key).unwrap(), &bindings));
            let stale =
                BackendResourceHandle::new(key.instance(), key.slot(), generation, key.kind());
            assert!(!cache.contains_key(&stale));
            retired.push(Arc::downgrade(&bindings));
            let mut before = ctx.device.create_command_encoder(&Default::default());
            r.initialize(&mut before);
            r.upload(&ctx, &mut before);
            let prior_draw = generation % 2 == 0;
            if prior_draw {
                normal.draw(&r, &mut before, true);
            }
            let compute = generation % 3 == 0;
            if compute {
                normal.produce(&ctx, &r, &mut before);
            }
            r.handoff(&mut before);
            let first = cache.get(&key).unwrap().encode(&ctx, 2);

            // Same-use native -> native, followed by normal writes to the TCS/TES
            // inputs and another native draw. This tests WAR as well as RAW.
            let second = bindings.encode(&ctx, 1);
            let mut middle = ctx.device.create_command_encoder(&Default::default());
            if compute {
                r.upload(&ctx, &mut middle);
            } else {
                normal.produce(&ctx, &r, &mut middle);
            }
            r.handoff(&mut middle);
            let third = bindings.encode(&ctx, 1);
            let mut after = ctx.device.create_command_encoder(&Default::default());
            normal.draw(&r, &mut after, false);
            let pixels = normal.sample(&ctx, &r, &mut after, 1);
            let untouched = normal.sample(&ctx, &r, &mut after, 0);

            // Exercise both one ordered submit and a boundary between native
            // segments. No vkQueueSubmit, interop copy, or CPU handoff wait.
            if generation < 3 {
                ctx.queue.submit([
                    before.finish(),
                    first,
                    second,
                    middle.finish(),
                    third,
                    after.finish(),
                ]);
            } else {
                ctx.queue.submit([before.finish(), first]);
                let retained = bindings.clone();
                ctx.queue.on_submitted_work_done(move || drop(retained));
                ctx.queue
                    .submit([second, middle.finish(), third, after.finish()]);
            }
            let count = completions.clone();
            ctx.queue.on_submitted_work_done(move || {
                drop(bindings);
                count.fetch_add(1, Ordering::SeqCst);
            });
            cache.remove(&key); // logical eviction does not destroy in-flight uses
            // Drop logical resource owners before waiting. Native descriptors,
            // framebuffer and their wgpu backing remain alive via the callback.
            drop(r);
            oracles.push((pixels, untouched, prior_draw, compute));
        }
        let retired_pipeline = Arc::downgrade(&pipeline);
        assert!(cache.is_empty());
        drop(pipeline); // cache eviction with submitted uses still retained
        for (pixels, untouched, prior_draw, final_green) in oracles {
            normal::check(&ctx, pixels, |x| {
                if x >= SIZE * 3 / 4 {
                    [0, 0, 255, 0]
                } else if prior_draw && x < SIZE / 4 {
                    [255, 0, 0, 32]
                } else if final_green {
                    [0, 255, 0, 64]
                } else {
                    [0, 0, 255, 64]
                }
            });
            normal::check(&ctx, untouched, |_| [255, 0, 255, 191]);
        }
        assert_eq!(completions.load(Ordering::SeqCst), 6);
        assert!(retired.iter().all(|r| r.upgrade().is_none()));
        assert!(retired_pipeline.upgrade().is_none());
    }
    {
        // Native feature disabled: ordinary wgpu draws must remain usable.
        let ctx = Context::new(false);
        let normal = NormalPipelines::new(&ctx);
        let r = Resources::new(&ctx);
        let mut encoder = ctx.device.create_command_encoder(&Default::default());
        r.initialize(&mut encoder);
        normal.draw(&r, &mut encoder, true);
        let pixels = normal.sample(&ctx, &r, &mut encoder, 1);
        ctx.queue.submit([encoder.finish()]);
        normal::check(&ctx, pixels, |x| {
            if x < SIZE / 4 {
                [255, 0, 0, 32]
            } else {
                [0, 0, 0, 255]
            }
        });
    }
    device::assert_validation_clean();
}
