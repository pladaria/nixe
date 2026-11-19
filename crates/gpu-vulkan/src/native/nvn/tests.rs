use super::*;
use nixe_memory::{AllocationRange, Machine, device_memory::DeviceMemory};

mod support {
    use crate as nixe_gpu_vulkan;
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/support/device.rs"
    ));
}

#[test]
fn counter_comparison_snapshots_device_writes_in_queue_order()
-> Result<(), Box<dyn std::error::Error>> {
    let Some(device) = support::device_for_test(ApiPreference::Auto) else {
        return Ok(());
    };
    let conditional = device.profile().conditional_rendering;
    let validation = device.validation_monitor();
    let machine = Machine::new(16 << 20)?;
    let allocation = machine.account(4 << 20)?.allocate(8192)?;
    let memory = DeviceMemory::new(machine, 32);
    let address = 0x10004;
    let framebuffer = 0x11000;
    let _mapping = memory.map(
        0x10000,
        Arc::from([AllocationRange::new(allocation.pin(), 0, 8192)?]),
        true,
    )?;
    let mut gpu = NativeGpuExecution::new(
        memory.clone(),
        device,
        CacheConfiguration::default(),
        64 << 20,
        BufferDownloadPolicy::OnRead,
    )?;
    let target = Texture {
        address: framebuffer,
        width: 8,
        height: 4,
        depth: 1,
        stride: 32,
        flags: 0x10,
        format: 0x25,
        levels: 1,
        target: 1,
        samples: 1,
        swizzle: [2, 3, 4, 5],
        ..Default::default()
    };
    let counters = |a: u64, b: u64| Operation::Update {
        address,
        bytes: bytemuck::cast_slice(&[a, 0, b]).into(),
    };
    let compare = |mode| Operation::RenderEnable { mode, address };
    let clear = |color, scissor| Operation::Clear {
        target,
        color,
        scissor,
        mask: 15,
    };
    let full = [0, 0, 8, 4];
    let red = [1., 0., 0., 1.];
    let green = [0., 1., 0., 1.];
    let list = |operations| nixe_nvn::Submission {
        operations,
        ..Default::default()
    };
    gpu.submit_nvn(
        1,
        list(vec![
            counters(7, (1u64 << 32) | 7),
            compare(1),
            counters(7, 7),
            clear(green, full),
            compare(1),
            clear(red, full),
        ]),
    )?;
    if conditional {
        assert!(matches!(gpu.state.nvn.conditions.draw(1), Some(Some(_))));
    }
    gpu.submit_nvn(2, list(vec![clear(red, [4, 0, 4, 4])]))?;
    gpu.submit_nvn(1, list(vec![clear(red, full)]))?;
    gpu.residency.access().flush_read_span(framebuffer, 128)?;
    let mut pixels = [0; 128];
    memory
        .reader()
        .read_block_unsafe(framebuffer, &mut pixels)?;
    for row in pixels.chunks_exact(32) {
        for (x, pixel) in row.chunks_exact(4).enumerate() {
            assert_eq!(
                pixel,
                if x < 4 {
                    [0, 255, 0, 255]
                } else {
                    [255, 0, 0, 255]
                }
            );
        }
    }
    drop(gpu);
    validation.check()?;
    Ok(())
}
