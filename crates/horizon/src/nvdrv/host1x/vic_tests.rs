use super::super::Mapping;
use super::super::video::gob_offset;
use super::*;
use crate::nvdrv::nvmap::{NvMapAllocationMetadata, NvMapCpuMapping, NvMapObjects, NvMapOwner};
use nixe_memory::{AddressSpaceId, GuestVirtualAddress};
use std::collections::BTreeMap;
use std::sync::Arc;

#[test]
fn nv12_conversion_uses_signed_guest_matrix_and_writes_the_actual_surface() {
    let machine = crate::test_memory::machine();
    let allocation = machine.account(0x20000).unwrap().allocate(0x20000).unwrap();
    let input = nixe_memory::AllocationRange::new(allocation.pin(), 0, 0x20000).unwrap();
    let mut nvmap = NvMapObjects::default();
    let owner = NvMapOwner::new(1);
    let handle = nvmap.create(owner, 0x20000).unwrap();
    nvmap
        .allocate(
            owner,
            handle,
            NvMapAllocationMetadata::new(0, 0, 4096, 0),
            NvMapCpuMapping::new(AddressSpaceId::new(1), GuestVirtualAddress::new(0x1000)),
            Arc::from([input.clone()]),
        )
        .unwrap();
    let object = nvmap
        .object_snapshot_by_owned_handle(owner, handle)
        .unwrap();
    let mappings = Arc::new(BTreeMap::from([(
        0x1000,
        Mapping {
            physical: nixe_gpu::physical::PhysicalMapping::new(object.ranges().unwrap()).unwrap(),
            object,
            references: 1,
        },
    )]));
    let mut config = vec![0; 0x610];
    let put = |bytes: &mut [u8], at: usize, value: u32| {
        bytes[at..at + 4].copy_from_slice(&value.to_le_bytes())
    };
    let width = 128;
    let height = 16;
    put(&mut config, 0x10, 4); // composited alpha
    put(&mut config, 0x18, (width - 1) << 16);
    put(&mut config, 0x1c, (height - 1) << 16);
    put(&mut config, 0x20, 31 | (1 << 11) | (1 << 15)); // ABGR block-linear, two GOBs
    put(&mut config, 0x24, (width - 1) | ((height - 1) << 14));
    put(&mut config, 0x28, (width - 1) | ((height - 1) << 14));
    let slot = 0x90;
    put(&mut config, slot, 0x101);
    put(&mut config, slot + 16, 1023 << 10);
    put(&mut config, slot + 20, 1023 | (1 << 10));
    put(&mut config, slot + 0x24, (width - 1) << 16);
    put(&mut config, slot + 0x2c, (height - 1) << 16);
    put(&mut config, slot + 0x30, (width - 1) << 16);
    put(&mut config, slot + 0x34, (height - 1) << 16);
    put(&mut config, slot + 0x40, 68 | (1 << 11) | (1 << 15));
    put(&mut config, slot + 0x44, (width - 1) | ((height - 1) << 14));
    put(&mut config, slot + 0x48, (width - 1) | ((height - 1) << 14));
    put(
        &mut config,
        slot + 0x4c,
        (width / 2 - 1) | ((height / 2 - 1) << 14),
    );
    // R=Y, G=U-128, B=V. This deliberately tests a signed matrix translation.
    for col in 0..4 {
        let values = match col {
            0 => [256i64, 0, 0],
            1 => [0, 256, 0],
            2 => [0, 0, 256],
            _ => [0, -512 * 256, 0],
        };
        let mut value = 0u64;
        for (row, v) in values.into_iter().enumerate() {
            value |= ((v as u64) & 0xfffff) << (row * 20);
        }
        if col == 1 {
            value |= 1 << 63;
        }
        config[slot + 0x60 + col * 8..slot + 0x68 + col * 8].copy_from_slice(&value.to_le_bytes());
    }
    put(&mut config, slot + 0xa0, 1023);
    put(&mut config, slot + 0xa4, 0x2050);
    machine
        .write_resolved(&[input.slice(0, config.len() as u64).unwrap()], &config)
        .unwrap();
    let mut luma = vec![0; 2048];
    let mut uv = vec![0; 2048];
    for y in 0..16 {
        for x in 0..128 {
            luma[gob_offset(x, y, 128, 1)] = (x + y) as u8;
        }
    }
    for y in 0..8 {
        for x in (0..128).step_by(2) {
            uv[gob_offset(x, y, 128, 1)] = 144;
            uv[gob_offset(x + 1, y, 128, 1)] = 192;
        }
    }
    machine
        .write_resolved(&[input.slice(0x1000, luma.len() as u64).unwrap()], &luma)
        .unwrap();
    machine
        .write_resolved(&[input.slice(0x2000, uv.len() as u64).unwrap()], &uv)
        .unwrap();
    let mut registers = vec![0; 0x446];
    registers[0x1c2] = 0x10;
    registers[0x100] = 0x20;
    registers[0x101] = 0x30;
    registers[0x1c8] = 0x100;
    Vic::default()
        .execute(&registers, &mappings, None, &mut BTreeMap::new())
        .unwrap();
    let mut result = vec![0; 8192];
    crate::test_memory::read(
        &[input.slice(0xf000, result.len() as u64).unwrap()],
        &mut result,
    );
    for y in 0..16 {
        for x in 0..128 {
            let pixel =
                std::array::from_fn::<_, 4, _>(|c| result[gob_offset(x * 4 + c, y, 512, 1)]);
            assert_eq!(pixel, [(x + y) as u8, 16, 192, 255], "({x},{y})");
        }
    }
    // An enabled identity gamut matrix takes the general compositor path.
    // Its output must match the opaque single-pass conversion byte for byte.
    for col in 0..4 {
        let value =
            if col < 3 { 256u64 << (col * 20) } else { 0 } | if col == 1 { 1 << 63 } else { 0 };
        config[slot + 0x80 + col * 8..slot + 0x88 + col * 8].copy_from_slice(&value.to_le_bytes());
    }
    machine
        .write_resolved(&[input.slice(0, config.len() as u64).unwrap()], &config)
        .unwrap();
    Vic::default()
        .execute(&registers, &mappings, None, &mut BTreeMap::new())
        .unwrap();
    let mut general = vec![0; result.len()];
    crate::test_memory::read(
        &[input.slice(0xf000, general.len() as u64).unwrap()],
        &mut general,
    );
    assert_eq!(general, result);
    // NV12 output stores the composited Y/U/V values directly. Exercise odd
    // visible extents, larger allocations and both linear and tiled padding.
    registers[0x1c9] = 0x180;
    let visible_width = width - 3;
    let visible_height = height - 1;
    for (kind, block) in [(0, 0), (1, 1), (1, 2)] {
        put(&mut config, 0x20, 68 | (kind << 11) | (block << 15));
        put(
            &mut config,
            0x24,
            (visible_width - 1) | ((visible_height - 1) << 14),
        );
        put(&mut config, 0x28, (width - 1) | ((height + 1 - 1) << 14));
        put(&mut config, 0x2c, (width / 2 - 1) | ((height - 1) << 14));
        let output = Surface::decode(&config[0x20..0x30], 1).unwrap();
        let chroma = chroma_surface(&output, word(&config, 0x2c)).unwrap();
        for (surface, offset) in [(&output, 0xf000), (&chroma, 0x17000)] {
            machine
                .write_resolved(
                    &[input.slice(offset, surface.length() as u64).unwrap()],
                    &vec![0xa5; surface.length()],
                )
                .unwrap();
        }
        machine
            .write_resolved(&[input.slice(0, config.len() as u64).unwrap()], &config)
            .unwrap();
        Vic::default()
            .execute(&registers, &mappings, None, &mut BTreeMap::new())
            .unwrap();
        for (is_chroma, surface, offset) in [(false, &output, 0xf000), (true, &chroma, 0x17000)] {
            let mut actual = vec![0; surface.length()];
            crate::test_memory::read(
                &[input.slice(offset, actual.len() as u64).unwrap()],
                &mut actual,
            );
            let mut expected = vec![0xa5; actual.len()];
            for y in 0..surface.height {
                for x in 0..surface.width {
                    expected[surface.offset(x, y)] = if !is_chroma {
                        (x + y) as u8
                    } else if x % 2 == 0 {
                        16
                    } else {
                        192
                    };
                }
            }
            assert_eq!(
                actual, expected,
                "kind={kind} block={block} chroma={is_chroma}"
            );
        }
    }
    // Reject fractional/scaled rectangles rather than copying the wrong source area.
    put(&mut config, slot + 0x24, ((width - 1) << 16) + 1);
    machine
        .write_resolved(&[input.slice(0, config.len() as u64).unwrap()], &config)
        .unwrap();
    assert!(
        Vic::default()
            .execute(&registers, &mappings, None, &mut BTreeMap::new())
            .unwrap_err()
            .contains("subpixel")
    );
}

#[test]
fn matrices_preserve_negative_coefficients_and_disabled_identity() {
    let mut wire = [0u8; 32];
    // R = -G + 400, G = R, B = B. Shift 0, coefficients in S12.8.
    let columns = [
        256u64 << 20,
        ((-256i64 as u64) & 0xfffff) | (1 << 63),
        256u64 << 40,
        400u64 * 256,
    ];
    for (i, col) in columns.into_iter().enumerate() {
        wire[i * 8..i * 8 + 8].copy_from_slice(&col.to_le_bytes());
    }
    assert_eq!(
        Matrix::decode(&wire).apply([100, 80, 200, 1023]),
        [320, 100, 200, 1023]
    );
    wire[15] &= 0x7f;
    assert_eq!(
        Matrix::decode(&wire).apply([100, 80, 200, 1023]),
        [100, 80, 200, 1023]
    );
}

#[test]
fn nv12_product_tables_preserve_exact_signed_matrix_rounding_and_clamp() {
    for matrix in [
        Matrix {
            coefficients: [
                [152618, 0, 209194, -228942],
                [152618, -51348, -106556, 139237],
                [152618, 264403, 0, -284313],
            ],
            shift: 9,
            enabled: true,
        },
        Matrix {
            coefficients: [
                [-256, 512, 0, 256],
                [256, -257, 128, -8192],
                [128, 256, -512, 100000],
            ],
            shift: 1,
            enabled: true,
        },
        Matrix {
            coefficients: [[0; 4]; 3],
            shift: 0,
            enabled: false,
        },
    ] {
        let table = Nv12Matrix::new(matrix);
        for y in 0..=255_u8 {
            for u in [0, 1, 64, 127, 128, 255] {
                for v in [0, 1, 128, 254, 255] {
                    let expected =
                        matrix.apply([i64::from(y) * 4, i64::from(u) * 4, i64::from(v) * 4, 1023]);
                    for (minimum, maximum) in [(0, 1023), (75, 900)] {
                        assert_eq!(
                            table.rgb([y, u, v], minimum, maximum),
                            std::array::from_fn(
                                |i| (expected[i].clamp(minimum, maximum) >> 2) as u8
                            )
                        );
                    }
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn nv12_simd_matches_scalar_signed_matrices_crops_clamp_and_channel_order() {
    if !std::is_x86_feature_detected!("sse4.1") {
        return;
    }
    let uv = [0, 255, 127, 128, 255, 0, 1, 254];
    for shift in [0, 1, 9, 15] {
        for enabled in [false, true] {
            let matrix = Matrix {
                coefficients: [
                    [524287, -524288, 12345, -524288],
                    [-524288, 524287, -54321, 524287],
                    [152618, 264403, 0, -284313],
                ],
                shift,
                enabled,
            };
            let scalar = Nv12Matrix::new(matrix);
            for value in 0..=255u8 {
                let luma = [value, value.wrapping_add(11), 0, 255, 127, 128, 64, 192];
                for x in [0, 1, 3] {
                    for (minimum, maximum) in [(0, 1023), (75, 900)] {
                        for bgra in [false, true] {
                            // SAFETY: runtime SSE4.1 check above; complete four-pixel sources.
                            let actual = unsafe {
                                rgba4(matrix, &luma, &uv, x, minimum, maximum, 197, bgra)
                            };
                            let mut expected = [0; 16];
                            for i in 0..4 {
                                let sx = x + i;
                                let mut rgb = scalar.rgb(
                                    [luma[sx], uv[sx / 2 * 2], uv[sx / 2 * 2 + 1]],
                                    minimum,
                                    maximum,
                                );
                                if bgra {
                                    rgb.swap(0, 2);
                                }
                                expected[i * 4..i * 4 + 3].copy_from_slice(&rgb);
                                expected[i * 4 + 3] = 197;
                            }
                            assert_eq!(
                                actual, expected,
                                "shift={shift} enabled={enabled} x={x} bgra={bgra}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn opaque_nv12_plane_copy_preserves_crop_pitch_and_tiled_padding() {
    let machine = crate::test_memory::machine();
    let allocation = machine.account(0x10000).unwrap().allocate(0x10000).unwrap();
    let input = nixe_memory::AllocationRange::new(allocation.pin(), 0, 0x10000).unwrap();
    let mut nvmap = NvMapObjects::default();
    let owner = NvMapOwner::new(1);
    let handle = nvmap.create(owner, 0x10000).unwrap();
    nvmap
        .allocate(
            owner,
            handle,
            NvMapAllocationMetadata::new(0, 0, 4096, 0),
            NvMapCpuMapping::new(AddressSpaceId::new(1), GuestVirtualAddress::new(0x1000)),
            Arc::from([input.clone()]),
        )
        .unwrap();
    let object = nvmap
        .object_snapshot_by_owned_handle(owner, handle)
        .unwrap();
    let mappings = Arc::new(BTreeMap::from([(
        0x1000,
        Mapping {
            physical: nixe_gpu::physical::PhysicalMapping::new(object.ranges().unwrap()).unwrap(),
            object,
            references: 1,
        },
    )]));
    let source: Vec<u8> = (0..128 * 64).map(|i| (i * 17 + i / 128) as u8).collect();
    for (kind, block_height) in [(0, 0), (1, 1), (1, 3)] {
        for width in [66, 67] {
            let output = Surface {
                width,
                height: 37,
                pitch: 128,
                rows: 48,
                kind,
                block_height,
                format: 68,
            };
            machine
                .write_resolved(std::slice::from_ref(&input), &vec![0xa5; 0x10000])
                .unwrap();
            write_plane(&output, &source, 128, [4, 6], &mappings, 0x2000).unwrap();
            let mut actual = vec![0; output.length()];
            crate::test_memory::read(
                &[input.slice(0x1000, actual.len() as u64).unwrap()],
                &mut actual,
            );
            let mut expected = vec![0xa5; output.length()];
            for y in 0..output.height {
                for x in 0..output.width {
                    expected[output.offset(x, y)] = source[(y + 6) * 128 + x + 4];
                }
            }
            assert_eq!(
                actual, expected,
                "kind={kind} block-height={block_height} width={width}"
            );
        }
    }
}

#[test]
fn visible_surface_runs_cover_pixels_once_in_physical_order() {
    for (kind, block_height) in [(0, 0), (1, 0), (1, 1), (1, 2), (1, 4)] {
        for bpp in [1, 4] {
            for width in [1, 15, 16, 17, 63, 64, 65, 128] {
                for height in [1, 7, 8, 9, 16, 17, 37] {
                    let plane = Surface {
                        width,
                        height,
                        pitch: (width * bpp).next_multiple_of(64) + 64,
                        rows: height + 3,
                        kind,
                        block_height,
                        format: 68,
                    };
                    let spans = plane.written_spans(bpp);
                    assert!(spans.windows(2).all(|pair| pair[0].end < pair[1].start));
                    let mut actual = vec![false; plane.length()];
                    for span in spans {
                        for byte in &mut actual[span] {
                            assert!(!*byte);
                            *byte = true;
                        }
                    }
                    let mut expected = vec![false; plane.length()];
                    for y in 0..height {
                        for x in 0..width * bpp {
                            expected[plane.offset(x, y)] = true;
                        }
                    }
                    assert_eq!(
                        actual, expected,
                        "kind={kind} block-height={block_height} bpp={bpp} extent={width}x{height}"
                    );
                }
            }
        }
    }
    let full = Surface {
        width: 1280,
        height: 720,
        pitch: 5120,
        rows: 720,
        kind: 1,
        block_height: 4,
        format: 31,
    };
    assert_eq!(full.written_spans(4).len(), 80);
}
