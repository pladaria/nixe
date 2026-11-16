use super::*;

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
