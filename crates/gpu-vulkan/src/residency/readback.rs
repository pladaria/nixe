//! Bounded, versioned row windows amortize neighboring canonical CPU faults.
//! Missing windows share staging batches; only authoritative bytes are published.
use super::image::ImagePiece;
use super::*;
use rustc_hash::FxHashMap;
use std::collections::VecDeque;
#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub(super) struct WindowKey {
    image: ImageId,
    version: u64,
    mip: u32,
    layer: u32,
    aspect: u32,
    row: u32,
}
struct WindowPlane {
    plane: ImagePlane,
    end: u64,
    row_bytes: u64,
    rows: u32,
    blocks: u32,
    window_shift: u32,
}
impl WindowPlane {
    fn new(plane: ImagePlane, limit: u64) -> Self {
        let row_bytes = plane.row_bytes();
        let rows = plane.rows();
        // Leave staging space for overlapping producers/aspects. Power-of-two
        // windows preserve locality and make row selection a shift per run.
        let capacity = (limit / 4 / row_bytes).max(1).min(u64::from(rows));
        Self {
            plane,
            end: plane.end().unwrap(),
            row_bytes,
            rows,
            blocks: plane.extent.width.div_ceil(plane.block_extent.width),
            window_shift: capacity.ilog2(),
        }
    }
}
use super::cpu_readback::Scatter;
#[derive(Default)]
pub(super) struct ImageReadbacks {
    windows: FxHashMap<WindowKey, Box<[u8]>>,
    order: VecDeque<WindowKey>,
    bytes: usize,
    pub(super) pending: FxHashMap<WindowKey, Vec<(CpuVisibilityRequest, Vec<Scatter>)>>,
}
impl VulkanResidency {
    pub(super) fn record_cached_image_authorities(
        &mut self,
        request: CpuVisibilityRequest,
        id: ImageId,
        writes: &[Authority],
    ) -> Result<bool, VulkanError> {
        let entry = self.image(id)?;
        let limit = self.device.submission.configuration.arena_bytes;
        // Exact regions remain necessary for partial images and rows wider than
        // staging. Prefetch never consumes undefined texels.
        if !entry.discarded.is_empty()
            || entry
                .planes
                .iter()
                .any(|p| entry.storage_plane(*p).row_bytes() > limit)
        {
            return Ok(false);
        }
        let version = entry.version;
        let planes: Vec<_> = entry
            .planes
            .iter()
            .map(|&plane| WindowPlane::new(entry.storage_plane(plane), limit))
            .collect();
        let mut needed = FxHashMap::default();
        for write in writes {
            image_chunks(
                id,
                version,
                &planes,
                write,
                |key, piece, source, destination, length| {
                    needed
                        .entry(key)
                        .or_insert_with(|| (piece, Vec::new()))
                        .1
                        .push(Scatter {
                            source: source as usize,
                            destination: destination as usize,
                            length: length as usize,
                        });
                },
            )?;
        }
        let bytes: u64 = needed.values().map(|(p, _)| p.length).sum();
        // A page whose projected native working set cannot fit the bounded
        // cache uses exact bounded regions rather than thrashing the directory.
        if bytes > limit.saturating_mul(2) || needed.len() > 32 {
            return Ok(false);
        }
        let mut missing = Vec::new();
        for (key, (piece, copies)) in needed {
            if let Some(bytes) = self.image_readbacks.windows.get(&key) {
                let target = &mut self.cpu_readbacks.get_mut(&request).unwrap().bytes;
                for copy in copies {
                    target[copy.destination..copy.destination + copy.length]
                        .copy_from_slice(&bytes[copy.source..copy.source + copy.length]);
                }
            } else {
                self.cpu_readbacks
                    .get_mut(&request)
                    .unwrap()
                    .waiting_windows += 1;
                match self.image_readbacks.pending.entry(key) {
                    std::collections::hash_map::Entry::Occupied(mut entry) => {
                        entry.get_mut().push((request, copies));
                    }
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        entry.insert(vec![(request, copies)]);
                        missing.push((Some(key), piece, Vec::new()));
                    }
                }
            }
        }
        missing.sort_unstable_by_key(|(_, piece, _)| piece.logical);
        self.record_image_reads(request, id, missing)?;
        Ok(true)
    }
}
impl ImageReadbacks {
    pub(super) fn install(&mut self, key: WindowKey, bytes: Box<[u8]>, limit: usize) {
        if let Some(old) = self.windows.remove(&key) {
            self.bytes -= old.len();
            self.order.retain(|old| *old != key);
        }
        while self.bytes + bytes.len() > limit || self.windows.len() >= 32 {
            let Some(old) = self.order.pop_front() else {
                return;
            };
            self.bytes -= self.windows.remove(&old).unwrap().len();
        }
        self.bytes += bytes.len();
        self.order.push_back(key);
        self.windows.insert(key, bytes);
    }
}

fn image_chunks(
    id: ImageId,
    version: u64,
    planes: &[WindowPlane],
    write: &Authority,
    mut visit: impl FnMut(WindowKey, ImagePiece, u64, u64, u64),
) -> Result<(), VulkanError> {
    let mut copied = 0;
    while copied < write.end - write.start {
        let logical = write.offset + copied;
        let window = planes
            .iter()
            .find(|p| logical >= p.plane.offset && logical < p.end)
            .ok_or(VulkanError::InvalidResource(
                "image producer crosses a missing plane",
            ))?;
        let p = window.plane;
        let row = ((logical - p.offset) / p.row_pitch) as u32;
        let x = (logical - p.offset) % p.row_pitch;
        if x >= window.row_bytes {
            return Err(VulkanError::InvalidResource(
                "image producer crosses row padding",
            ));
        }
        let first = row >> window.window_shift << window.window_shift;
        let rows = (1u32 << window.window_shift).min(window.rows - first);
        let piece = ImagePiece {
            plane: p,
            row: first,
            rows,
            first: 0,
            blocks: window.blocks,
            skip: 0,
            logical: p.offset + u64::from(first) * p.row_pitch,
            length: window.row_bytes * u64::from(rows),
        };
        let key = WindowKey {
            image: id,
            version,
            mip: p.mip_level,
            layer: p.array_layer,
            aspect: p.aspect.as_raw(),
            row: first,
        };
        let source = u64::from(row - first) * window.row_bytes + x;
        let available = if p.row_pitch == window.row_bytes {
            piece.length - source
        } else {
            window.row_bytes - x
        };
        let length = available.min(write.end - write.start - copied);
        visit(key, piece, source, write.start + copied, length);
        copied += length;
    }
    Ok(())
}
