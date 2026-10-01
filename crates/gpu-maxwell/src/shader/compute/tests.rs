use super::*;
use nixe_gpu::{ShaderComputeBuiltin, ShaderStage};
use nixe_memory::MemoryPermissions;

fn bytes(words: &[u64]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

#[test]
fn headerless_program_uses_qmd_geometry_and_shared_instruction_decoder() {
    let (allocation, address_space, address) = super::super::test_support::mapped_memory();
    // CTAID.X, TID.X, ISCADD and EXIT. No graphics header precedes the bundle.
    allocation
        .write(
            0,
            &bytes(&[
                0,
                0xf0c8_0000_0257_0000,
                0xf0c8_0000_0217_0001,
                0x5c18_0280_0017_0000,
                0,
                0xe300_0000_0007_000f,
                0,
                0,
            ]),
        )
        .unwrap();
    let module = translate_compute_program(
        &address_space,
        &CanonicalWriteBatch::new(),
        address,
        2,
        [32, 2, 3],
    )
    .unwrap();
    let ir = module.module.ir().ir();
    assert_eq!(ir.stage(), ShaderStage::Compute);
    assert_eq!(ir.workgroup_size(), Some([32, 2, 3]));
    assert!(ir.inputs().is_empty() && ir.outputs().is_empty());
    assert!(matches!(
        ir.instructions()[0].operation(),
        ShaderOperation::LoadComputeBuiltin32 {
            builtin: ShaderComputeBuiltin::WorkgroupId,
            component: 0,
            ..
        }
    ));
    assert_eq!(ir.instructions()[0].source().byte_offset(), 8);
    assert!(
        ir.instructions().iter().any(|instruction| matches!(
            instruction.operation(),
            ShaderOperation::ShiftLeft32 { .. }
        ))
    );
    let wgsl = nixe_gpu::lower_shader_ir_to_wgsl(module.module.ir()).unwrap();
    assert!(wgsl.source().contains("@workgroup_size(32, 2, 3)"));
}

#[test]
fn compute_system_registers_are_typed_and_validate_selectors_and_registers() {
    for (base, builtin) in [
        (0x21_u64, ShaderComputeBuiltin::LocalInvocationId),
        (0x25, ShaderComputeBuiltin::WorkgroupId),
    ] {
        for component in 0..3 {
            let encoding = 0xf0c8_0000_0007_0000 | ((base + u64::from(component)) << 20);
            assert!(
                matches!(system_register(8, encoding, 1).unwrap(), ShaderOperation::LoadComputeBuiltin32 {
                destination, builtin: actual, component: actual_component
            } if destination == ShaderRegister::new(0) && actual == builtin && actual_component == component)
            );
            assert!(system_register(8, encoding | (1 << 8), 1).is_err());
            assert!(system_register(8, encoding | 1, 1).is_err());
        }
    }
    for selector in [0x20, 0x24, 0x28, 0xff] {
        assert!(matches!(
            system_register(8, 0xf0c8_0000_0007_0000 | (selector << 20), 1),
            Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
                stage: MaxwellShaderStage::Compute,
                ..
            })
        ));
    }
}

#[test]
fn headerless_code_reads_canonical_staged_bytes_through_a_gpu_va_alias() {
    let (allocation, mut address_space, address) = super::super::test_support::mapped_memory();
    let backing = allocation
        .backing_range(MemoryPermissions::READ_WRITE)
        .unwrap();
    let alias = address_space
        .map(crate::MaxwellMapRequest {
            allocation: crate::MaxwellAllocationId::new(1),
            backing: backing.clone(),
            backing_offset: 0,
            size: 0x1000,
            allocation_alignment: 0x1000,
            page_size: 0,
            kind: 0,
            cacheable: false,
            permissions: MemoryPermissions::READ_WRITE,
            fixed_offset: None,
        })
        .unwrap()
        .offset()
        .get();
    assert_ne!(address, alias);
    let mut writes = CanonicalWriteBatch::new();
    writes
        .stage(
            &backing,
            0,
            &bytes(&[0, 0xf0c8_0000_0217_0000, 0xe300_0000_0007_000f, 0]),
        )
        .unwrap();
    for entry in [address, alias] {
        assert!(translate_compute_program(&address_space, &writes, entry, 1, [32, 1, 1]).is_ok());
    }
    let mut unchanged = [0xff; 32];
    allocation.read(0, &mut unchanged).unwrap();
    assert_eq!(unchanged, [0; 32]);
}

#[test]
fn compute_rejects_conditional_exit_instead_of_truncating_the_program() {
    let (allocation, address_space, address) = super::super::test_support::mapped_memory();
    allocation
        .write(0, &bytes(&[0, 0xe300_0000_0000_000f, 0, 0]))
        .unwrap();
    assert!(matches!(
        translate_compute_program(
            &address_space,
            &CanonicalWriteBatch::new(),
            address,
            1,
            [32, 1, 1]
        ),
        Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            instruction_offset: 8,
            detail: "conditional EXIT requires complete shader control-flow discovery",
            ..
        })
    ));
}

pub(super) fn sinewave_kernel() -> MaxwellComputeProgram {
    // UAM output from the public deko3d example's compute kernel. Keeping
    // the full instruction stream checks shared decoding and relocation of
    // all eight global stores; it is not a replacement kernel.
    // https://github.com/switchbrew/switch-examples/blob/master/graphics/deko3d/deko_examples/source/sinewave.glsl
    let words = [
        0x007f9800e5e00701,
        0xf0c8000002570000,
        0xf0c8000002170001,
        0x5c18028000170000,
        0x001f9800fcc00701,
        0x5cb8000000070a01,
        0x4c98078000370002,
        0x391802fffff70202,
        0x007f8402e5a0072f,
        0x5cb8000000270a02,
        0x5080000000470202,
        0x5c68000000270101,
        0x001f9800fc2007e1,
        0x0103f0000007f002,
        0x0103f8000007f003,
        0x010bf8000007f004,
        0x001f9800fcc207e1,
        0x3280024000070104,
        0x4c58000800870105,
        0x1e040c90fdb70505,
        0x003f8400e1a007e6,
        0x5c90000000570005,
        0x5080000000170505,
        0x4c68000800970505,
        0x001f9800fc2207e1,
        0x3848000000570006,
        0x3818028001070000,
        0x4c98078800070007,
        0x001f9800fc2007e6,
        0x4c59000800470707,
        0x5180008800070707,
        0x4c98078800170008,
        0x001f9800fc2007e6,
        0x4c59000800570808,
        0x5180008800170808,
        0x4c98078800270009,
        0x001f9800fc2007e6,
        0x4c59000800670909,
        0x5180008800270909,
        0x4c9807880037000a,
        0x001f9800fc2007e6,
        0x4c59000800770a0a,
        0x5180008800370a0a,
        0x4c10800005070000,
        0x000784001c2007e6,
        0x4c1008000517ff01,
        0xeedc200000070007,
        0xeedc200000470008,
        0x01ff98007c4002e1,
        0xeedc200000870009,
        0xeedc200000c7000a,
        0x4c10800005070600,
        0x000784001c2007e6,
        0x4c1008000517ff01,
        0xeedc200000070004,
        0xeedc200000470005,
        0x07ffbc007c2002e1,
        0xeedc200000870002,
        0xeedc200000c70003,
        0xe30000000007000f,
    ];
    let (allocation, address_space, address) = super::super::test_support::mapped_memory();
    allocation.write(0, &bytes(&words)).unwrap();
    translate_compute_program(
        &address_space,
        &CanonicalWriteBatch::new(),
        address,
        11,
        [32, 1, 1],
    )
    .unwrap()
}

#[test]
fn sinewave_kernel_translates_all_stores_with_dispatch_bounded_storage() {
    let program = sinewave_kernel();
    assert_eq!(program.global_buffers.buffers.len(), 1);
    let buffer = &program.global_buffers.buffers[0];
    assert_eq!(
        (buffer.constant_buffer, buffer.byte_offset, buffer.binding),
        (0, 0x140, 32)
    );
    assert_eq!(
        program.global_buffers.byte_extents([8, 1, 1], [32, 1, 1]),
        [8192]
    );
    assert_eq!(
        program.global_buffers.byte_extents([16, 1, 1], [32, 1, 1]),
        [16384]
    );
    assert_eq!(
        program
            .module
            .ir()
            .ir()
            .instructions()
            .iter()
            .filter(|i| matches!(
                i.operation(),
                ShaderOperation::StoreStorageBuffer32 { binding: 32, .. }
            ))
            .count(),
        8
    );
    let wgsl = nixe_gpu::lower_shader_ir_to_wgsl(program.module.ir()).unwrap();
    super::super::test_support::validate_wgsl(&wgsl);
}
