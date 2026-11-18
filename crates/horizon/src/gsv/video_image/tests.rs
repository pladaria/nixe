use crate::video_codec::{PlanarChroma, Plane};
use nixe_memory::{AllocationRange, CapturedMemory, Machine, PAGE_SIZE};

fn check_publication(
    width: usize,
    height: usize,
    source_padding: usize,
    fragmented: bool,
    planar: bool,
) {
    let mut preference = [0u32; 16];
    preference[1] = 1;
    preference[2] = ((width - 1) | ((height - 1) << 16)) as u32;
    let layout = super::Layout::new(preference).unwrap();
    let chroma_width = width.div_ceil(2);
    let chroma_rows = height.div_ceil(2);
    let luma_stride = width + source_padding;
    let u_stride = chroma_width + source_padding + 1;
    let v_stride = chroma_width + source_padding;
    let packed_stride = 2 * chroma_width + source_padding;
    let y = (0..(luma_stride * (height - 1) + width))
        .map(|i| (i * 37 + 3) as u8)
        .collect::<Vec<_>>();
    let u = (0..(u_stride * (chroma_rows - 1) + chroma_width))
        .map(|i| (i * 5 + 1) as u8)
        .collect::<Vec<_>>();
    let v = (0..(v_stride * (chroma_rows - 1) + chroma_width))
        .map(|i| (i * 3 + 71) as u8)
        .collect::<Vec<_>>();
    let mut packed = vec![0xd5; packed_stride * (chroma_rows - 1) + 2 * chroma_width];
    for row in 0..chroma_rows {
        for x in 0..chroma_width {
            packed[row * packed_stride + x * 2] = u[row * u_stride + x];
            packed[row * packed_stride + x * 2 + 1] = v[row * v_stride + x];
        }
    }
    let base = 7;
    let chroma_base = base + layout.chroma;
    let end = chroma_base + layout.pitch * chroma_rows + 15;
    let mut expected = vec![0xae; end];
    for (begin, width, rows, stride, input) in [
        (base, width, height, luma_stride, &y),
        (
            chroma_base,
            2 * chroma_width,
            chroma_rows,
            packed_stride,
            &packed,
        ),
    ] {
        for row in 0..rows {
            let destination = begin + row * layout.pitch;
            expected[destination..destination + width]
                .copy_from_slice(&input[row * stride..row * stride + width]);
            expected[destination + width..destination + layout.pitch].fill(0);
        }
    }
    let machine = Machine::new(2 << 20).unwrap();
    let account = machine.account(2 << 20).unwrap();
    // Split visible bytes, padding and an interleaved UV pair across physical extents.
    let mut cuts = vec![0, end];
    if fragmented {
        cuts.extend(
            [
                base + 1,
                base + 3,
                base + layout.pitch - 1,
                chroma_base - 1,
                chroma_base + 1,
                chroma_base + 3,
                chroma_base + layout.pitch - 1,
            ]
            .into_iter()
            .filter(|&n| n < end),
        );
    }
    cuts.sort_unstable();
    cuts.dedup();
    let capture = CapturedMemory::new(cuts.windows(2).map(|a| {
        let n = (a[1] - a[0]) as u64;
        let alloc = account.allocate(n.next_multiple_of(PAGE_SIZE)).unwrap();
        AllocationRange::new(alloc.pin(), 0, n).unwrap()
    }))
    .unwrap();
    machine
        .write_captured(&capture, 0, &vec![0xae; end])
        .unwrap()
        .into_ready()
        .unwrap();
    let luma_bytes = layout.pitch * height;
    machine
        .publish_captured(
            &capture,
            &[
                (base as u64, luma_bytes),
                (chroma_base as u64, layout.pitch * chroma_rows),
            ],
            move |at, dst| {
                let (plane, width, start) = if at < luma_bytes {
                    (
                        Plane::Packed {
                            data: &y,
                            stride: luma_stride,
                        },
                        width,
                        at,
                    )
                } else if planar {
                    (
                        Plane::Chroma(PlanarChroma {
                            u: &u,
                            v: &v,
                            u_stride,
                            v_stride,
                        }),
                        2 * chroma_width,
                        at - luma_bytes,
                    )
                } else {
                    (
                        Plane::Packed {
                            data: &packed,
                            stride: packed_stride,
                        },
                        2 * chroma_width,
                        at - luma_bytes,
                    )
                };
                super::write_plane(plane, width, layout.pitch, start, dst);
            },
        )
        .unwrap()
        .into_ready()
        .unwrap();
    let mut actual = vec![0; end];
    machine
        .read_captured_into(&capture, 0, &mut actual)
        .unwrap()
        .into_ready()
        .unwrap();
    if let Some(offset) = (0..end).find(|&i| actual[i] != expected[i]) {
        panic!(
            "pixel mismatch at {offset} width={width} height={height} source_padding={source_padding} fragmented={fragmented} planar={planar}"
        );
    }
}

#[test]
fn nv12_publication_matches_pixels_padding_and_untouched_bytes() {
    for width in [1, 2, 5, 63, 64, 65, 257, 960] {
        for height in [1, 3, 4, 6] {
            for padding in [0, 13] {
                for fragmented in [false, true] {
                    for planar in [false, true] {
                        check_publication(width, height, padding, fragmented, planar);
                    }
                }
            }
        }
    }
}
