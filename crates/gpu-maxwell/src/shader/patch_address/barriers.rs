//! UAM 1.1.0 output for the adjacent, original barriers.tesc test source.
//! Reproduce: uam -s tess_ctrl -r barriers.bin -o barriers.dksh barriers.tesc;
//! nvdisasm -b SM53 -ndf -hex barriers.bin. Schedule words/padding omitted.
//! UAM removes all three GLSL barriers, relying on Maxwell warp lockstep:
//! https://github.com/devkitPro/uam/blob/master/mesa-imported/codegen/nv50_ir_lowering_nvc0.cpp#L831-L835
use super::*;

pub(super) fn binary() -> MaxwellShaderBinary {
    let mut header = [0; 20];
    header[..6].copy_from_slice(&[
        0x00060861, 0x10000000, 0x03000000, 0, 0x001ff000, 0xf0000000,
    ]);
    header[13] = 0x000ff000;
    binary_from_words(
        header,
        &[
            [0xf0c8000001170000, 0x5b6403800ff70007, 0x0103e8000000f000],
            [0xeff07f808300ff00, 0xefd87f818307ff02, 0xf0c8000001170000],
            [0x5b6403800ff70007, 0x0103f4000000f000, 0xeff07f808300ff00],
            [0xefd87f818307ff01, 0x5c98078000270000, 0x0103f8000007f003],
            [0xeff1ff800807ff00, 0xf0c8000001170000, 0x5b6403800ff70007],
            [0x0103f8000000f000, 0xeff07f808000ff00, 0xeff07f808040ff00],
            [0xeff07f808080ff00, 0xeff07f808100ff00, 0xf0c8000001170000],
            [0xf0c8000001d70001, 0x384700000ff70102, 0x3800000081070101],
            [0x5b00000000270100, 0x5b007fa800270102, 0x5b30001800270100],
            [0xefd0000000070004, 0xefd982000707ff00, 0xeff1ff800707ff00],
            [0xe30000000007000f, 0, 0],
        ],
    )
}

#[test]
fn compiled_patch_reads_restore_raw_war_ordering() {
    let chain = graphics_chain_with_control(binary());
    let tc = chain[1].ir();
    let barriers: Vec<_> = tc
        .instructions()
        .iter()
        .filter(|i| matches!(i.operation(), ShaderOperation::PatchBarrier))
        .map(|i| (i.source().byte_offset(), i.predicate()))
        .collect();
    assert_eq!(
        barriers,
        [0x30, 0x58, 0x68].map(|pc| (pc, ShaderPredicate::Always))
    );
    // 0x58 is a predicated writer, but its WAR rendezvous is unconditional.
    // Every invocation must finish reading the first value before it changes.
    let shaders = native_chain_from_ir(chain, true);
    assert_eq!(shaders.output_control_points(), 3);
}

#[test]
fn patch_output_load_rejects_unrepresented_modes_and_addresses() {
    for encoding in [
        0xefd87f818307ff02 ^ (1 << 32), // .P without .O
        0xefd87f818307ff02 ^ (1 << 8),  // indexed attribute
        0xefd87f818307ff02 ^ (1 << 39), // vertex operand for patch load
        0xefd87f818307ff02 ^ (1 << 20), // unaligned slot
        0xefd87f818187ff02,             // reserved patch padding slot 6
        0xefd87f818300ff02,             // conditional output read
    ] {
        let mut program = binary();
        program.bundles[1].instructions[1] = encoding;
        assert!(
            translate_shader_binary(&program, 5, &BTreeMap::new()).is_err(),
            "{encoding:#018x}"
        );
    }
}

#[test]
fn ordinary_demo_control_shader_needs_no_patch_output_barriers() {
    let chain = graphics_chain();
    assert!(
        chain[1]
            .ir()
            .instructions()
            .iter()
            .all(|i| { !matches!(i.operation(), ShaderOperation::PatchBarrier) })
    );
}
