//! Real-device compiler checks, outside the production execution path.
use crate::*;
use ash::vk;
use nixe_gpu::{ShaderSourceLocation, ShaderStage, ssa::*};
use std::{sync::Arc, time::Duration};
mod support {
    use crate as nixe_gpu_vulkan;
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/support/device.rs"
    ));
}

struct Kernel {
    owner: Arc<crate::device::DeviceOwner>,
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    set_layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    buffer: AllocatedBuffer,
}
impl Kernel {
    fn new(device: &VulkanDevice, module: &spirv::Module, words: usize) -> Self {
        if let Some(validator) = std::env::var_os("NIXE_SPIRV_VAL") {
            static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let serial = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "nixe-native-ssa-{}-{serial}.spv",
                std::process::id()
            ));
            std::fs::write(
                &path,
                module
                    .words
                    .iter()
                    .flat_map(|w| w.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
            let output = std::process::Command::new(validator)
                .args(["--target-env", "vulkan1.2"])
                .arg(&path)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            std::fs::remove_file(path).unwrap();
        }
        assert_eq!(module.bindings.len(), 1);
        assert_eq!(
            module.bindings[0].kind,
            nixe_gpu::ShaderResourceKind::StorageBuffer
        );
        let raw = &device.owner.device;
        let buffer = device
            .allocate_buffer(
                (words * 4) as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                MemoryLocation::GpuToCpu,
                AllocationMode::Managed,
                "SSA shader fixture",
            )
            .unwrap();
        // SAFETY: all create-info backing slices remain alive through each call.
        let (pipeline, layout, set_layout, pool, set) = unsafe {
            let bindings = [vk::DescriptorSetLayoutBinding::default()
                .binding(module.bindings[0].binding)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)];
            let set_layout = raw
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
                .unwrap();
            let layouts = [set_layout];
            let layout = raw
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts),
                    None,
                )
                .unwrap();
            let shader = raw
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo::default().code(&module.words),
                    None,
                )
                .unwrap();
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(shader)
                .name(c"main");
            let pipeline = raw
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default()
                        .stage(stage)
                        .layout(layout)],
                    None,
                )
                .unwrap()[0];
            raw.destroy_shader_module(shader, None);
            let sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)];
            let pool = raw
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(1)
                        .pool_sizes(&sizes),
                    None,
                )
                .unwrap();
            let set = raw
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(pool)
                        .set_layouts(&layouts),
                )
                .unwrap()[0];
            let info = [vk::DescriptorBufferInfo::default()
                .buffer(buffer.handle)
                .range(buffer.size())];
            raw.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(module.bindings[0].binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&info)],
                &[],
            );
            (pipeline, layout, set_layout, pool, set)
        };
        Self {
            owner: Arc::clone(&device.owner),
            pipeline,
            layout,
            set_layout,
            pool,
            set,
            buffer,
        }
    }
    fn execute(&mut self, device: &mut VulkanDevice, input: &[u32]) -> Vec<u32> {
        let bytes: Vec<_> = input.iter().flat_map(|v| v.to_le_bytes()).collect();
        // Previous dispatch completed; host access excludes every in-flight atom.
        unsafe { self.buffer.write_range(0, &bytes) }.unwrap();
        let slot = device.submission.begin().unwrap();
        let command = device.submission.slots[slot].command;
        // Queue submission makes flushed host writes available. The readback
        // dependency and targeted timeline wait make shader writes host-visible.
        unsafe {
            let raw = &device.owner.device;
            raw.cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            raw.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[self.set],
                &[],
            );
            raw.cmd_dispatch(command, 1, 1, 1);
            let barriers = [vk::BufferMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::HOST)
                .dst_access_mask(vk::AccessFlags2::HOST_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(self.buffer.handle)
                .size(self.buffer.size())];
            device.dispatch.barrier(
                command,
                &vk::DependencyInfo::default().buffer_memory_barriers(&barriers),
            );
        }
        let work = device.flush().unwrap().unwrap();
        assert!(device.wait_for_work(work, Duration::from_secs(10)).unwrap());
        let bytes = unsafe { self.buffer.read_range(0, self.buffer.size()) }.unwrap();
        bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }
}
impl Drop for Kernel {
    fn drop(&mut self) {
        // execute waits for the only submission referencing these handles.
        unsafe {
            let d = &self.owner.device;
            d.destroy_pipeline(self.pipeline, None);
            d.destroy_pipeline_layout(self.layout, None);
            d.destroy_descriptor_pool(self.pool, None);
            d.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}
fn options(device: &VulkanDevice) -> spirv::Options {
    let f = &device.profile().float_controls;
    spirv::Options {
        shader: nixe_gpu::SpirvShaderOptions {
            depth_clip_negative_one_to_one: false,
            vertex_position_transform: None,
            input_control_points: 0,
            tessellation_mode: None,
            float32: nixe_gpu::SpirvFloat32Capabilities {
                denorm_preserve: f.float32_denorm_preserve,
                rounding_mode_rte: f.float32_round_to_nearest_even,
                signed_zero_inf_nan_preserve: f.float32_signed_zero_inf_nan_preserve,
                fused_multiply_add: f.fma_float32,
            },
            float64: nixe_gpu::SpirvFloat64Capabilities {
                enabled: device.profile().core.shader_float64 != 0,
                rounding_mode_rte: f.float64_round_to_nearest_even,
                signed_zero_inf_nan_preserve: f.float64_signed_zero_inf_nan_preserve,
            },
        },
        integer64: device.profile().core.shader_int64 != 0,
        subgroup_quad: false,
        shared_memory_bytes: 0,
        local_memory_bytes: 0,
    }
}
fn pc() -> ShaderSourceLocation {
    ShaderSourceLocation::new(0x50)
}
fn constant(b: &mut FunctionBuilder, block: BlockId, bits: u32) -> Operand {
    b.emit(block, pc(), Op::Constant(Constant::u32(bits)), vec![])[0].into()
}
fn scalar_kernel(op: Op, arguments: &[Type], results: &[Type]) -> Function {
    let mut b = FunctionBuilder::new(ShaderStage::Compute);
    let block = b.entry();
    let mut args = Vec::new();
    for (index, ty) in arguments.iter().copied().enumerate() {
        let address = constant(&mut b, block, index as u32 * 4);
        let load = b.emit(
            block,
            pc(),
            Op::Load {
                space: MemorySpace::Storage,
                resource: Some(ResourceId::new(42)),
                address_type: Type::U32,
                ty: Type::U32,
            },
            vec![address],
        )[0];
        let arg = if ty == Type::U32 {
            load
        } else if ty == Type::Bool {
            let zero = constant(&mut b, block, 0);
            b.emit(
                block,
                pc(),
                Op::IntegerCompare {
                    ty: Type::U32,
                    comparison: nixe_gpu::ShaderIntegerComparison::NotEqual,
                },
                vec![load.into(), zero],
            )[0]
        } else {
            b.emit(
                block,
                pc(),
                Op::Bitcast {
                    from: Type::U32,
                    to: ty,
                },
                vec![load.into()],
            )[0]
        };
        args.push(arg.into());
    }
    let values = b.emit(block, pc(), op.clone(), args);
    for (index, (&value, ty)) in values.iter().zip(results.iter().copied()).enumerate() {
        let value = if ty == Type::U32 {
            value
        } else if ty == Type::Bool {
            let one = constant(&mut b, block, 1);
            let zero = constant(&mut b, block, 0);
            b.emit(
                block,
                pc(),
                Op::Select(Type::U32),
                vec![value.into(), one, zero],
            )[0]
        } else {
            b.emit(
                block,
                pc(),
                Op::Bitcast {
                    from: ty,
                    to: Type::U32,
                },
                vec![value.into()],
            )[0]
        };
        let address = constant(&mut b, block, 16 + index as u32 * 4);
        b.emit(
            block,
            pc(),
            Op::Store {
                space: MemorySpace::Storage,
                resource: Some(ResourceId::new(42)),
                address_type: Type::U32,
                ty: Type::U32,
            },
            vec![address, value.into()],
        );
    }
    b.terminate(block, pc(), Terminator::Return);
    let mut f = b.finish().unwrap();
    f.workgroup_size = Some([1, 1, 1]);
    f.optimize().unwrap();
    f
}
#[test]
fn ssa_rounding_and_conversions_execute_on_real_gpu() {
    let Some(mut device) = support::device_for_test(ApiPreference::Auto) else {
        return;
    };
    let validation = device.validation_monitor();
    use nixe_gpu::ShaderRoundingMode as R;
    let inputs = [
        0, 0x80000000, 1, 0x80000001, 0x3effffff, 0x3f000000, 0x3f000001, 0x3fc00000, 0x40200000,
        0xbfc00000, 0xc0200000, 0x4affffff, 0x4b000001, 0x4effffff, 0x4f000000, 0xcf000000,
        0xcf000001, 0x4f7fffff, 0x4f800000, 0x7f800000, 0xff800000, 0x7fc00000,
    ];
    for rounding in [
        R::NearestEven,
        R::TowardZero,
        R::TowardNegative,
        R::TowardPositive,
    ] {
        let round = |f: f32| match rounding {
            R::NearestEven => f.round_ties_even(),
            R::TowardZero => f.trunc(),
            R::TowardNegative => f.floor(),
            R::TowardPositive => f.ceil(),
        };
        for (op, signed, integral) in [
            (
                Op::RoundIntegral {
                    rounding,
                    flush_denormals: false,
                },
                false,
                true,
            ),
            (
                Op::FloatToInteger {
                    to: Type::U32,
                    bits: 32,
                    rounding,
                    flush_denormals: false,
                },
                false,
                false,
            ),
            (
                Op::FloatToInteger {
                    to: Type::I32,
                    bits: 32,
                    rounding,
                    flush_denormals: false,
                },
                true,
                false,
            ),
        ] {
            let module = spirv::emit(
                &scalar_kernel(
                    op,
                    &[Type::F32],
                    &[if integral {
                        Type::F32
                    } else if signed {
                        Type::I32
                    } else {
                        Type::U32
                    }],
                ),
                options(&device),
            )
            .unwrap();
            let mut kernel = Kernel::new(&device, &module, 5);
            for bits in inputs {
                let f = round(f32::from_bits(bits));
                let expected = if integral {
                    f.to_bits()
                } else if signed {
                    (f as i32) as u32
                } else {
                    f as u32
                };
                assert_eq!(
                    kernel.execute(&mut device, &[bits, 0, 0, 0, 0])[4],
                    expected,
                    "{rounding:?} {bits:08x}"
                );
            }
        }
    }
    drop(device);
    validation.check().unwrap();
}
#[path = "../../gpu/src/shader/half_arithmetic.rs"]
mod half_reference;
fn expand_half(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = (bits >> 10) & 31;
    let fraction = bits & 1023;
    match exponent {
        0 => sign * f32::from(fraction) * 2.0f32.powi(-24),
        31 => {
            if fraction == 0 {
                sign * f32::INFINITY
            } else {
                f32::NAN
            }
        }
        e => sign * (1.0 + f32::from(fraction) / 1024.0) * 2.0f32.powi(i32::from(e) - 15),
    }
}
#[test]
fn ssa_half_fma_avoids_double_rounding_on_real_gpu() {
    let Some(mut device) = support::device_for_test(ApiPreference::Auto) else {
        return;
    };
    let validation = device.validation_monitor();
    let module = spirv::emit(
        &scalar_kernel(
            Op::HalfFma { saturate: false },
            &[Type::F32; 3],
            &[Type::U32],
        ),
        options(&device),
    )
    .unwrap();
    let mut kernel = Kernel::new(&device, &module, 5);
    let mut cases = vec![
        (0x3e00, 0x3c01, 0x8001),
        (0x3e00, 0x3c03, 1),
        (0x4b80, 0x6c44, 0x8001),
        (0x4b80, 0x6c44, 1),
        (1, 0x3800, 0),
        (1, 0x3800, 1),
        (0x8000, 0x3c00, 0x8000),
        (0x7c00, 0, 0),
        (0x7bff, 0x3c00, 0xfbff),
    ];
    let mut random = 0x4561234u32;
    for _ in 0..512 {
        let mut next = || {
            random ^= random << 13;
            random ^= random >> 17;
            random ^= random << 5;
            random as u16
        };
        cases.push((next(), next(), next()));
    }
    for (a, b, c) in cases {
        let [a, b, c] = [a, b, c].map(expand_half);
        let expected = u32::from(half_reference::round_half(
            f64::from(a) * f64::from(b) + f64::from(c),
        ));
        let actual = kernel.execute(&mut device, &[a.to_bits(), b.to_bits(), c.to_bits(), 0, 0])[4];
        if expected & 0x7fff > 0x7c00 {
            assert!(actual & 0x7fff > 0x7c00);
        } else {
            assert_eq!(actual, expected, "{a:?} * {b:?} + {c:?}");
        }
    }
    drop(kernel);
    drop(device);
    validation.check().unwrap();
}
fn control_kernel() -> Function {
    let mut b = FunctionBuilder::new(ShaderStage::Compute);
    let entry = b.entry();
    let header = b.block(pc());
    let choose = b.block(pc());
    let yes = b.block(pc());
    let no = b.block(pc());
    let exit = b.block(pc());
    let counter = b.variable(Type::U32, InitialValue::Constant(Constant::u32(0)));
    let sum = b.variable(Type::U32, InitialValue::Constant(Constant::u32(0)));
    let zero = constant(&mut b, entry, 0);
    let limit = b.emit(
        entry,
        pc(),
        Op::Load {
            space: MemorySpace::Storage,
            resource: Some(ResourceId::new(42)),
            address_type: Type::U32,
            ty: Type::U32,
        },
        vec![zero],
    )[0];
    let one = constant(&mut b, entry, 1);
    b.terminate(entry, pc(), Terminator::Jump(header));
    let cond = b.emit(
        header,
        pc(),
        Op::IntegerCompare {
            ty: Type::U32,
            comparison: nixe_gpu::ShaderIntegerComparison::Less,
        },
        vec![counter.into(), limit.into()],
    )[0];
    b.terminate(
        header,
        pc(),
        Terminator::Branch {
            condition: cond.into(),
            when_true: choose,
            when_false: exit,
        },
    );
    let parity = b.emit(
        choose,
        pc(),
        Op::Binary {
            op: Binary::And,
            ty: Type::U32,
            float_control: nixe_gpu::ShaderFloatControl::PRECISE,
        },
        vec![counter.into(), one],
    )[0];
    let cond = b.emit(
        choose,
        pc(),
        Op::IntegerCompare {
            ty: Type::U32,
            comparison: nixe_gpu::ShaderIntegerComparison::NotEqual,
        },
        vec![parity.into(), zero],
    )[0];
    b.terminate(
        choose,
        pc(),
        Terminator::Branch {
            condition: cond.into(),
            when_true: yes,
            when_false: no,
        },
    );
    for (block, amount) in [(yes, 3), (no, 7)] {
        let amount = constant(&mut b, block, amount);
        let next = b.emit(
            block,
            pc(),
            Op::Binary {
                op: Binary::Add,
                ty: Type::U32,
                float_control: nixe_gpu::ShaderFloatControl::PRECISE,
            },
            vec![sum.into(), amount],
        )[0];
        b.assign(block, pc(), sum, next.into());
        let next = b.emit(
            block,
            pc(),
            Op::Binary {
                op: Binary::Add,
                ty: Type::U32,
                float_control: nixe_gpu::ShaderFloatControl::PRECISE,
            },
            vec![counter.into(), one],
        )[0];
        b.assign(block, pc(), counter, next.into());
        b.terminate(block, pc(), Terminator::Jump(header));
    }
    let address = constant(&mut b, exit, 16);
    b.emit(
        exit,
        pc(),
        Op::Store {
            space: MemorySpace::Storage,
            resource: Some(ResourceId::new(42)),
            address_type: Type::U32,
            ty: Type::U32,
        },
        vec![address, sum.into()],
    );
    b.terminate(exit, pc(), Terminator::Return);
    let mut f = b.finish().unwrap();
    f.workgroup_size = Some([1, 1, 1]);
    f.optimize().unwrap();
    f
}
#[test]
fn ssa_native_branches_loop_phis_and_memory_execute_on_real_gpu() {
    let Some(mut device) = support::device_for_test(ApiPreference::Auto) else {
        return;
    };
    let validation = device.validation_monitor();
    let module = spirv::emit(&control_kernel(), options(&device)).unwrap();
    let mut kernel = Kernel::new(&device, &module, 5);
    for limit in [0, 1, 2, 3, 9, 128] {
        let expected = (0..limit)
            .map(|i| if i & 1 == 0 { 7 } else { 3 })
            .sum::<u32>();
        assert_eq!(
            kernel.execute(&mut device, &[limit, 0, 0, 0, 0])[4],
            expected
        );
    }
    drop(kernel);
    drop(device);
    validation.check().unwrap();
}
#[test]
fn ssa_rtz_daz_ftz_multiply_is_exact_on_real_gpu() {
    let Some(mut device) = support::device_for_test(ApiPreference::Auto) else {
        return;
    };
    let validation = device.validation_monitor();
    let control = nixe_gpu::ShaderFloatControl::new(
        nixe_gpu::ShaderRoundingMode::TowardZero,
        nixe_gpu::ShaderNanMode::Propagate,
        true,
        true,
        false,
    );
    let f = scalar_kernel(
        Op::Binary {
            op: Binary::Multiply,
            ty: Type::F32,
            float_control: control,
        },
        &[Type::F32; 2],
        &[Type::F32],
    );
    let mut opts = options(&device);
    // This integer lowering needs neither float64 nor native float controls.
    opts.shader.float32 = Default::default();
    opts.shader.float64 = Default::default();
    let module = spirv::emit(&f, opts).unwrap();
    let mut kernel = Kernel::new(&device, &module, 5);
    let special = [
        0, 0x80000000, 1, 0x807fffff, 0x00800000, 0x80800000, 0x3fc00000, 0x3f800001, 0x3f7fffff,
        0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000, 0x7fc12345, 0xffa54321,
    ];
    let mut cases: Vec<_> = special
        .iter()
        .flat_map(|&a| special.iter().map(move |&b| (a, b)))
        .collect();
    let mut random = 0x97a61739u32;
    for _ in 0..512 {
        let mut next = || {
            random ^= random << 13;
            random ^= random >> 17;
            random ^= random << 5;
            random
        };
        cases.push((next(), next()));
    }
    for (a, b) in cases {
        let daz = |bits: u32| {
            if bits & 0x7f800000 == 0 {
                bits & 0x80000000
            } else {
                bits
            }
        };
        let aa = daz(a);
        let bb = daz(b);
        let product = f64::from(f32::from_bits(aa)) * f64::from(f32::from_bits(bb));
        let expected = if a & 0x7fffffff > 0x7f800000 {
            a | 0x400000
        } else if b & 0x7fffffff > 0x7f800000 {
            b | 0x400000
        } else if product.is_nan() {
            0x7fc00000
        } else {
            let rounded = (product as f32).to_bits();
            let magnitude = rounded & 0x7fffffff;
            let magnitude = if f64::from(f32::from_bits(magnitude)) > product.abs() {
                magnitude - 1
            } else {
                magnitude
            };
            (rounded & 0x80000000) | if magnitude < 0x800000 { 0 } else { magnitude }
        };
        let actual = kernel.execute(&mut device, &[a, b, 0, 0, 0])[4];
        assert_eq!(actual, expected, "a={a:08x} b={b:08x}");
    }
    drop(kernel);
    drop(device);
    validation.check().unwrap();
}
#[test]
fn ssa_fma_is_fused_and_preserves_ftz_sign_on_real_gpu() {
    let Some(mut device) = support::device_for_test(ApiPreference::Auto) else {
        return;
    };
    let validation = device.validation_monitor();
    let control = nixe_gpu::ShaderFloatControl::new(
        nixe_gpu::ShaderRoundingMode::NearestEven,
        nixe_gpu::ShaderNanMode::Propagate,
        true,
        true,
        false,
    );
    let f = scalar_kernel(
        Op::Fma {
            ty: Type::F32,
            float_control: control,
        },
        &[Type::F32; 3],
        &[Type::F32],
    );
    if !device.profile().float_controls.fma_float32 {
        assert!(
            spirv::emit(&f, options(&device))
                .unwrap_err()
                .detail
                .contains("VK_KHR_shader_fma")
        );
        return;
    }
    let module = spirv::emit(&f, options(&device)).unwrap();
    let mut kernel = Kernel::new(&device, &module, 5);
    for (a, b, c) in [
        (0x3f800001, 0x3f7ffffe, 0xbf800000),
        (0x7f7fffff, 0x40000000, 0xff7fffff),
        (0x00800000, 0x3effffff, 0),
        (0x80800000, 0x3effffff, 0x80000000),
        (0x00800000, 0x3effffff, 0x00800000),
        (0x80000000, 0x3f800000, 0x80000000),
    ] {
        let mut expected = f32::from_bits(a)
            .mul_add(f32::from_bits(b), f32::from_bits(c))
            .to_bits();
        if expected & 0x7f800000 == 0 {
            expected &= 0x80000000;
        }
        assert_eq!(
            kernel.execute(&mut device, &[a, b, c, 0, 0])[4],
            expected,
            "{a:08x} * {b:08x} + {c:08x}"
        );
    }
    drop(kernel);
    drop(device);
    validation.check().unwrap();
}
#[test]
fn ssa_integer_to_float_rounding_executes_on_real_gpu() {
    let Some(mut device) = support::device_for_test(ApiPreference::Auto) else {
        return;
    };
    let validation = device.validation_monitor();
    use nixe_gpu::ShaderRoundingMode as R;
    for from in [Type::U32, Type::I32] {
        for rounding in [
            R::NearestEven,
            R::TowardZero,
            R::TowardNegative,
            R::TowardPositive,
        ] {
            let module = spirv::emit(
                &scalar_kernel(
                    Op::Convert {
                        from,
                        to: Type::F32,
                        rounding,
                        flush_denormals: false,
                    },
                    &[from],
                    &[Type::F32],
                ),
                options(&device),
            )
            .unwrap();
            let mut kernel = Kernel::new(&device, &module, 5);
            for input in [
                0, 1, 0xffffffff, 0x80000000, 0x7fffffff, 0x01000001, 0x01000003, 0xfeffffff,
                0xfefffffd, 0x12345678,
            ] {
                let exact = if from == Type::U32 {
                    f64::from(input)
                } else {
                    f64::from(input as i32)
                };
                let nearest = exact as f32;
                let mut expected = nearest;
                match rounding {
                    R::TowardZero if (nearest as f64).abs() > exact.abs() => {
                        expected = if exact < 0.0 {
                            nearest.next_up()
                        } else {
                            nearest.next_down()
                        }
                    }
                    R::TowardNegative if nearest as f64 > exact => expected = nearest.next_down(),
                    R::TowardPositive if (nearest as f64) < exact => expected = nearest.next_up(),
                    _ => {}
                }
                assert_eq!(
                    kernel.execute(&mut device, &[input, 0, 0, 0, 0])[4],
                    expected.to_bits(),
                    "{from:?} {rounding:?} {input:08x}"
                );
            }
        }
    }
    drop(device);
    validation.check().unwrap();
}
#[test]
fn ssa_shared_memory_barrier_exchanges_values_between_invocations() {
    let Some(mut device) = support::device_for_test(ApiPreference::Auto) else {
        return;
    };
    let validation = device.validation_monitor();
    let mut b = FunctionBuilder::new(ShaderStage::Compute);
    let entry = b.entry();
    let lane = b.emit(
        entry,
        pc(),
        Op::Builtin {
            builtin: nixe_gpu::ShaderComputeBuiltin::LocalInvocationIndex,
            component: 0,
        },
        vec![],
    )[0];
    let one = constant(&mut b, entry, 1);
    let four = constant(&mut b, entry, 4);
    let initial = constant(&mut b, entry, 17);
    let binary = |b: &mut FunctionBuilder, op, a, z| {
        b.emit(
            entry,
            pc(),
            Op::Binary {
                op,
                ty: Type::U32,
                float_control: nixe_gpu::ShaderFloatControl::PRECISE,
            },
            vec![a, z],
        )[0]
    };
    let address = binary(&mut b, Binary::Multiply, lane.into(), four);
    let value = binary(&mut b, Binary::Add, lane.into(), initial);
    b.emit(
        entry,
        pc(),
        Op::Store {
            space: MemorySpace::Shared,
            resource: None,
            ty: Type::U32,
            address_type: Type::U32,
        },
        vec![address.into(), value.into()],
    );
    b.emit(
        entry,
        pc(),
        Op::Barrier {
            execution: BarrierScope::Workgroup,
            memory: BarrierScope::Workgroup,
            spaces: BarrierMemory::SHARED,
        },
        vec![],
    );
    let other = binary(&mut b, Binary::Xor, lane.into(), one);
    let other_address = binary(&mut b, Binary::Multiply, other.into(), four);
    let value = b.emit(
        entry,
        pc(),
        Op::Load {
            space: MemorySpace::Shared,
            resource: None,
            ty: Type::U32,
            address_type: Type::U32,
        },
        vec![other_address.into()],
    )[0];
    let output = constant(&mut b, entry, 16);
    let output = binary(&mut b, Binary::Add, address.into(), output);
    b.emit(
        entry,
        pc(),
        Op::Store {
            space: MemorySpace::Storage,
            resource: Some(ResourceId::new(42)),
            ty: Type::U32,
            address_type: Type::U32,
        },
        vec![output.into(), value.into()],
    );
    b.terminate(entry, pc(), Terminator::Return);
    let mut f = b.finish().unwrap();
    f.workgroup_size = Some([2, 1, 1]);
    f.optimize().unwrap();
    let mut options = options(&device);
    options.shared_memory_bytes = 8;
    let module = spirv::emit(&f, options).unwrap();
    let mut kernel = Kernel::new(&device, &module, 6);
    assert_eq!(&kernel.execute(&mut device, &[0; 6])[4..], [18, 17]);
    drop(kernel);
    drop(device);
    validation.check().unwrap();
}
#[test]
fn ssa_minmax_orders_signed_zero_and_preserves_subnormal_bits() {
    let Some(mut device) = support::device_for_test(ApiPreference::Auto) else {
        return;
    };
    let validation = device.validation_monitor();
    let module = spirv::emit(
        &scalar_kernel(
            Op::FloatMinMax {
                float_control: nixe_gpu::ShaderFloatControl::PRECISE,
            },
            &[Type::F32, Type::F32, Type::Bool],
            &[Type::F32],
        ),
        options(&device),
    )
    .unwrap();
    let mut kernel = Kernel::new(&device, &module, 5);
    for (a, b, min, expected) in [
        (0x80000000, 0, 1, 0x80000000),
        (0, 0x80000000, 1, 0x80000000),
        (0x80000000, 0, 0, 0),
        (0, 0x80000000, 0, 0),
        (1, 2, 1, 1),
        (1, 2, 0, 2),
        (0x7fc00000, 0x3f800000, 1, 0x3f800000),
        (0x3f800000, 0x7fc00000, 0, 0x3f800000),
    ] {
        assert_eq!(kernel.execute(&mut device, &[a, b, min, 0, 0])[4], expected);
    }
    drop(kernel);
    drop(device);
    validation.check().unwrap();
}

// Both entries and both exits carry different SSA values. A bounded cycle lets
// execution check static CFG splitting/funnels, rather than only SPIR-V validity.
fn irregular_control_kernel(irreducible: bool) -> Function {
    let mut b = FunctionBuilder::new(ShaderStage::Compute);
    let entry = b.entry();
    let a = b.block(pc());
    let z = b.block(pc());
    let exit_a = b.block(pc());
    let exit_z = b.block(pc());
    let join = b.block(pc());
    let counter = b.variable(Type::U32, InitialValue::Constant(Constant::u32(0)));
    let sum = b.variable(Type::U32, InitialValue::Constant(Constant::u32(0)));
    let load = |b: &mut FunctionBuilder, offset| {
        let address = constant(b, entry, offset);
        b.emit(
            entry,
            pc(),
            Op::Load {
                space: MemorySpace::Storage,
                resource: Some(ResourceId::new(42)),
                address_type: Type::U32,
                ty: Type::U32,
            },
            vec![address],
        )[0]
    };
    let limit = load(&mut b, 0);
    let selector = load(&mut b, 4);
    let zero = constant(&mut b, entry, 0);
    let one = constant(&mut b, entry, 1);
    let compare = |b: &mut FunctionBuilder, block, comparison, left, right| {
        b.emit(
            block,
            pc(),
            Op::IntegerCompare {
                ty: Type::U32,
                comparison,
            },
            vec![left, right],
        )[0]
    };
    let cond = compare(
        &mut b,
        entry,
        nixe_gpu::ShaderIntegerComparison::NotEqual,
        selector.into(),
        zero,
    );
    b.terminate(
        entry,
        pc(),
        if irreducible {
            Terminator::Branch {
                condition: cond.into(),
                when_true: a,
                when_false: z,
            }
        } else {
            Terminator::Jump(a)
        },
    );
    let add = |b: &mut FunctionBuilder, block, left, right| {
        b.emit(
            block,
            pc(),
            Op::Binary {
                op: Binary::Add,
                ty: Type::U32,
                float_control: nixe_gpu::ShaderFloatControl::PRECISE,
            },
            vec![left, right],
        )[0]
    };
    for (block, next, exit, amount) in [(a, z, exit_a, 3), (z, a, exit_z, 7)] {
        let amount = constant(&mut b, block, amount);
        let value = add(&mut b, block, sum.into(), amount);
        b.assign(block, pc(), sum, value.into());
        let value = add(&mut b, block, counter.into(), one);
        b.assign(block, pc(), counter, value.into());
        let cond = compare(
            &mut b,
            block,
            nixe_gpu::ShaderIntegerComparison::Less,
            counter.into(),
            limit.into(),
        );
        b.terminate(
            block,
            pc(),
            Terminator::Branch {
                condition: cond.into(),
                when_true: next,
                when_false: exit,
            },
        );
    }
    for (block, amount) in [(exit_a, 100), (exit_z, 200)] {
        let amount = constant(&mut b, block, amount);
        let value = add(&mut b, block, sum.into(), amount);
        b.assign(block, pc(), sum, value.into());
        b.terminate(block, pc(), Terminator::Jump(join));
    }
    let address = constant(&mut b, join, 8);
    b.emit(
        join,
        pc(),
        Op::Store {
            space: MemorySpace::Storage,
            resource: Some(ResourceId::new(42)),
            address_type: Type::U32,
            ty: Type::U32,
        },
        vec![address, sum.into()],
    );
    b.terminate(join, pc(), Terminator::Return);
    let mut function = b.finish().unwrap();
    function.workgroup_size = Some([1, 1, 1]);
    function.optimize().unwrap();
    function
}
#[test]
fn ssa_irreducible_entries_and_multiple_exits_preserve_values_on_real_gpu() {
    let Some(mut device) = support::device_for_test(ApiPreference::Auto) else {
        return;
    };
    let validation = device.validation_monitor();
    for irreducible in [false, true] {
        let module = spirv::emit(&irregular_control_kernel(irreducible), options(&device)).unwrap();
        let mut kernel = Kernel::new(&device, &module, 3);
        for selector in [0, 1] {
            for limit in [0u32, 1, 2, 3, 4, 9, 128] {
                let start_a = !irreducible || selector != 0;
                let count = limit.max(1);
                let is_a = |i| (i % 2 == 0) == start_a;
                let expected = (0..count).map(|i| if is_a(i) { 3 } else { 7 }).sum::<u32>()
                    + if is_a(count - 1) { 100 } else { 200 };
                assert_eq!(
                    kernel.execute(&mut device, &[limit, selector, 0])[2],
                    expected,
                    "irreducible={irreducible}, selector={selector}, limit={limit}"
                );
            }
        }
    }
    drop(device);
    validation.check().unwrap();
}
#[test]
fn gradual_underflow_and_mixed_float_controls_are_exact_without_denorm_preserve32() {
    let Some(mut device) = support::device_for_test(ApiPreference::Auto) else {
        return;
    };
    let monitor = device.validation_monitor();
    let mut opts = options(&device);
    // Exercise the exact capability lowering even on a host with native support.
    opts.shader.float32.denorm_preserve = false;
    assert!(
        opts.shader.float64.enabled
            && opts.shader.float64.rounding_mode_rte
            && opts.shader.float64.signed_zero_inf_nan_preserve
    );
    let add = [
        ([1, 0, 0], 1),
        ([0x80000001, 0, 0], 0x80000001),
        ([0x7fffff, 1, 0], 0x800000),
        ([0x800000, 0x807fffff, 0], 1),
        ([0x3f800000, 1, 0], 0x3f800000),
    ];
    let multiply = [
        ([1, 1, 0], 0),
        ([1, 0x3f000000, 0], 0),
        ([3, 0x3f000000, 0], 2),
        ([0x7fffff, 0x3f800000, 0], 0x7fffff),
        ([0x800000, 0x3f7fffff, 0], 0x800000),
        ([0x80000003, 0x3f000000, 0], 0x80000002),
        ([1, 0x7f800000, 0], 0x7f800000),
    ];
    let fma = [
        ([0, 0x3f800000, 1], 1),
        ([0x80800000, 0x3f800000, 0x800001], 1),
        ([0x3f800001, 0x3fc00000, 0x80000001], 0x3fc00001),
        ([0x3f800001, 0x3fc00000, 1], 0x3fc00002),
        ([3, 0x3f000000, 0x80000001], 0),
        ([3, 0x3f000000, 1], 2),
        ([0x800000, 0x3f7fffff, 0x80000001], 0x7ffffe),
        ([0x80000001, 0x3f000000, 0x80000000], 0x80000000),
    ];
    for (kind, cases) in [
        (0, add.as_slice()),
        (1, multiply.as_slice()),
        (2, fma.as_slice()),
    ] {
        let control = nixe_gpu::ShaderFloatControl::PRECISE;
        let op = if kind == 2 {
            Op::Fma {
                ty: Type::F32,
                float_control: control,
            }
        } else {
            Op::Binary {
                op: if kind == 0 {
                    Binary::Add
                } else {
                    Binary::Multiply
                },
                ty: Type::F32,
                float_control: control,
            }
        };
        let args = if kind == 2 { 3 } else { 2 };
        let function = scalar_kernel(op, &vec![Type::F32; args], &[Type::F32]);
        let module = spirv::emit(&function, opts).unwrap();
        let mut kernel = Kernel::new(&device, &module, 8);
        for &(input, expected) in cases {
            let mut words = [0; 8];
            words[..3].copy_from_slice(&input);
            let result = kernel.execute(&mut device, &words);
            assert_eq!(result[4], expected, "kind={kind} inputs={input:08x?}");
        }
    }
    for (daz, ftz, input, expected) in [
        (true, false, [0x800001, 0x80800000], 1),
        (false, true, [0x800000, 0x807fffff], 0),
        (false, true, [0x7fffff, 1], 0x800000),
        (true, false, [1, 0], 0),
    ] {
        let control = nixe_gpu::ShaderFloatControl::new(
            nixe_gpu::ShaderRoundingMode::NearestEven,
            nixe_gpu::ShaderNanMode::Propagate,
            ftz,
            daz,
            false,
        );
        let function = scalar_kernel(
            Op::Binary {
                op: Binary::Add,
                ty: Type::F32,
                float_control: control,
            },
            &[Type::F32; 2],
            &[Type::F32],
        );
        let module = spirv::emit(&function, opts).unwrap();
        let mut kernel = Kernel::new(&device, &module, 8);
        let mut words = [0; 8];
        words[..2].copy_from_slice(&input);
        let result = kernel.execute(&mut device, &words);
        assert_eq!(
            result[4], expected,
            "DAZ={daz} FTZ={ftz} inputs={input:08x?}"
        );
    }
    monitor.check().unwrap();
}
#[test]
fn inverse_math_preserves_subnormal_inputs_and_outputs() {
    let Some(mut device) = support::device_for_test(ApiPreference::Auto) else {
        return;
    };
    let validation = device.validation_monitor();
    let mut opts = options(&device);
    opts.shader.float32.denorm_preserve = false;
    for function in [MathFunction::Reciprocal, MathFunction::ReciprocalSqrt] {
        let code = scalar_kernel(
            Op::Math {
                function,
                accuracy: nixe_gpu::ShaderMathAccuracy::Approximate,
                float_control: nixe_gpu::ShaderFloatControl::PRECISE,
            },
            &[Type::F32],
            &[Type::F32],
        );
        let module = spirv::emit(&code, opts).unwrap();
        let mut kernel = Kernel::new(&device, &module, 8);
        for input in [
            1_u32, 3, 0x7fffff, 0x800000, 0x7f7fffff, 0x80000000, 0x7f800000,
        ] {
            let mut words = [0; 8];
            words[0] = input;
            let actual = kernel.execute(&mut device, &words)[4];
            let x = f64::from(f32::from_bits(input));
            let expected = (if function == MathFunction::Reciprocal {
                1. / x
            } else {
                1. / x.sqrt()
            }) as f32;
            if function == MathFunction::Reciprocal || !expected.is_finite() {
                assert_eq!(actual, expected.to_bits(), "{function:?} input={input:08x}");
            } else {
                assert!(
                    actual.abs_diff(expected.to_bits()) <= 1,
                    "approximate rsqrt input={input:08x} actual={actual:08x} reference={:08x}",
                    expected.to_bits()
                );
            }
        }
    }
    validation.check().unwrap();
}
#[test]
fn approximate_sfus_preserve_tiny_values_without_native_denorm_support() {
    let Some(mut device) = support::device_for_test(ApiPreference::Auto) else {
        return;
    };
    let validation = device.validation_monitor();
    let mut opts = options(&device);
    opts.shader.float32.denorm_preserve = false;
    // No FP64 is needed for these normalizations.
    opts.shader.float64.enabled = false;
    for function in [
        nixe_gpu::ShaderSpecialFunction::Sine,
        nixe_gpu::ShaderSpecialFunction::Cosine,
        nixe_gpu::ShaderSpecialFunction::Exp2,
        nixe_gpu::ShaderSpecialFunction::Log2,
        nixe_gpu::ShaderSpecialFunction::SquareRoot,
    ] {
        let code = scalar_kernel(
            Op::Math {
                function: MathFunction::Special(function),
                accuracy: nixe_gpu::ShaderMathAccuracy::Approximate,
                float_control: nixe_gpu::ShaderFloatControl::PRECISE,
            },
            &[Type::F32],
            &[Type::F32],
        );
        let module = spirv::emit(&code, opts).unwrap();
        let mut kernel = Kernel::new(&device, &module, 8);
        for input in [1_u32, 3, 0x7fffff, 0x800000, 0] {
            let x = f64::from(f32::from_bits(input));
            let expected = (match function {
                nixe_gpu::ShaderSpecialFunction::Sine => x.sin(),
                nixe_gpu::ShaderSpecialFunction::Cosine => x.cos(),
                nixe_gpu::ShaderSpecialFunction::Exp2 => x.exp2(),
                nixe_gpu::ShaderSpecialFunction::Log2 => x.log2(),
                nixe_gpu::ShaderSpecialFunction::SquareRoot => x.sqrt(),
            }) as f32;
            let mut words = [0; 8];
            words[0] = input;
            let actual = kernel.execute(&mut device, &words)[4];
            if matches!(
                function,
                nixe_gpu::ShaderSpecialFunction::Sine
                    | nixe_gpu::ShaderSpecialFunction::Cosine
                    | nixe_gpu::ShaderSpecialFunction::Exp2
            ) || !expected.is_finite()
            {
                assert_eq!(actual, expected.to_bits(), "{function:?} input={input:08x}");
            } else {
                assert!(
                    actual.abs_diff(expected.to_bits()) <= 1,
                    "approximate {function:?} input={input:08x} actual={actual:08x} reference={:08x}",
                    expected.to_bits()
                );
            }
        }
        if function == nixe_gpu::ShaderSpecialFunction::Exp2 {
            for input in [-150_f32, -149., -148.5, -127., -126.] {
                let expected = f64::from(input).exp2() as f32;
                let mut words = [0; 8];
                words[0] = input.to_bits();
                let actual = kernel.execute(&mut device, &words)[4];
                assert!(
                    actual.abs_diff(expected.to_bits()) <= 1,
                    "approximate exp2({input}) actual={actual:08x} reference={:08x}",
                    expected.to_bits()
                );
            }
        }
    }
    validation.check().unwrap();
}
