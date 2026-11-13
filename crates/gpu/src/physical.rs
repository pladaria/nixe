//! Immutable physical resource mappings and compact byte-exact image footprints.
use nixe_memory::{
    AllocationRange, ArenaOffset, MemoryError, PhysicalMapping, PhysicalSpan, WeakAllocation,
};
#[cfg(test)]
use std::hash::{Hash, Hasher};
use std::sync::Arc;

pub type PageSelection = Vec<(WeakAllocation, std::ops::Range<u64>)>;

pub fn pin_selected_pages(
    mut selected: PageSelection,
) -> Result<Vec<AllocationRange>, MemoryError> {
    selected.sort_unstable_by_key(|(source, bounds)| {
        (source.machine(), source.identity(), bounds.start)
    });
    let mut merged: PageSelection = Vec::new();
    for (source, bounds) in selected {
        if let Some((previous, last)) = merged.last_mut()
            && previous.machine() == source.machine()
            && previous.identity() == source.identity()
            && bounds.start <= last.end
        {
            last.end = last.end.max(bounds.end);
        } else {
            merged.push((source, bounds));
        }
    }
    merged
        .into_iter()
        .map(|(source, bounds)| {
            let pin = source
                .upgrade_range(bounds.start, bounds.end - bounds.start)
                .ok_or(MemoryError::Invalid("selected physical pages have retired"))?;
            AllocationRange::new(pin, bounds.start, bounds.end - bounds.start)
        })
        .collect()
}

/// Build protection/retirement pins once per physical allocation run. Inputs
/// keep every selected page live until the resulting union has been retained.
pub fn pin_union(ranges: &[AllocationRange]) -> Result<Vec<AllocationRange>, MemoryError> {
    let bounds = ranges
        .iter()
        .enumerate()
        .map(|(index, range)| {
            let start = range.offset();
            let end = start + range.bytes();
            (index, start, end)
        })
        .collect();
    pin_intervals(ranges, bounds)
}
pub(crate) fn pin_intervals(
    ranges: &[AllocationRange],
    mut bounds: Vec<(usize, u64, u64)>,
) -> Result<Vec<AllocationRange>, MemoryError> {
    let identity = |index: usize| {
        (
            ranges[index].allocation().machine().identity(),
            ranges[index].allocation().identity(),
        )
    };
    bounds.sort_unstable_by_key(|(index, start, _)| (identity(*index), *start));
    let mut merged: Vec<(usize, u64, u64)> = Vec::new();
    for (index, start, end) in bounds {
        if let Some((_, _, last)) = merged
            .last_mut()
            .filter(|(previous, _, last)| identity(*previous) == identity(index) && *last >= start)
        {
            *last = (*last).max(end);
        } else {
            merged.push((index, start, end));
        }
    }
    merged
        .into_iter()
        .map(|(index, start, end)| {
            let allocation = ranges[index].allocation().downgrade();
            let pin = allocation
                .upgrade_range(start, end - start)
                .ok_or(MemoryError::Invalid("captured physical union has retired"))?;
            AllocationRange::new(pin, start, end - start)
        })
        .collect()
}

impl Footprint {
    /// Select allocation coordinates; publication retains their union once.
    pub fn observed_pages(
        &self,
        mapping: &PhysicalMapping,
        selected: &mut PageSelection,
    ) -> Result<(), MemoryError> {
        let written = self;
        if mapping.machine() != written.mapping.machine() {
            return Err(MemoryError::Invalid(
                "observed input and producer name different machines",
            ));
        }
        // A footprint of this mapping already identifies each physical span.
        // Selecting its union is valid even when logical spans repeat RAM.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/device_memory_manager.inc
        if std::ptr::eq(mapping, written.mapping.as_ref()) {
            for (index, observed) in mapping.spans().iter().enumerate() {
                Self::select_observed_pages(mapping, observed, written, index, selected)?;
            }
            return Ok(());
        }
        if let [span] = written.mapping.spans() {
            for observed in mapping.spans() {
                if observed.allocation == span.allocation {
                    Self::select_observed_pages(mapping, observed, written, 0, selected)?;
                }
            }
            return Ok(());
        }
        // Different views can reorder or repeat physical bytes. Reuse the
        // interval index for this query, without retaining another cache index.
        let mut spans = nixe_memory::IntervalIndex::default();
        for (index, span) in written.mapping.spans().iter().enumerate() {
            let begin = span.logical.max(written.start);
            let end = (span.logical + span.bytes).min(written.end);
            if begin < end {
                spans.insert(
                    span.start.get() + begin - span.logical,
                    span.start.get() + end - span.logical,
                    index,
                )?;
            }
        }
        for observed in mapping.spans() {
            let mut result = Ok(());
            spans.overlapping(
                observed.start.get(),
                observed.start.get() + observed.bytes,
                |&index| {
                    if result.is_ok()
                        && written.mapping.spans()[index].allocation == observed.allocation
                    {
                        result = Self::select_observed_pages(
                            mapping, observed, written, index, selected,
                        );
                    }
                },
            );
            result?;
        }
        Ok(())
    }
    fn select_observed_pages(
        mapping: &PhysicalMapping,
        observed: &PhysicalSpan,
        written: &Footprint,
        index: usize,
        selected: &mut PageSelection,
    ) -> Result<(), MemoryError> {
        let span = &written.mapping.spans()[index];
        let logical_start = span.logical.max(written.start);
        let logical_end = (span.logical + span.bytes).min(written.end);
        if logical_start >= logical_end {
            return Ok(());
        }
        let begin = observed
            .start
            .get()
            .max(span.start.get() + logical_start - span.logical);
        let end = (observed.start.get() + observed.bytes)
            .min(span.start.get() + logical_end - span.logical);
        if begin >= end {
            return Ok(());
        }
        for page in
            (begin & !(nixe_memory::PAGE_SIZE - 1)..end).step_by(nixe_memory::PAGE_SIZE as usize)
        {
            let start = page.max(begin);
            let stop = (page + nixe_memory::PAGE_SIZE).min(end);
            let mut hit = false;
            written.visit(index, ArenaOffset::new(start), stop - start, |_| {
                hit = true;
                false
            })?;
            if !hit {
                continue;
            }
            let offset = (observed.allocation_offset() + start - observed.start.get())
                & !(nixe_memory::PAGE_SIZE - 1);
            let source = mapping.allocation(observed);
            if let Some((previous, bounds)) = selected.last_mut()
                && previous.identity() == source.identity()
                && bounds.end == offset
            {
                bounds.end += nixe_memory::PAGE_SIZE;
            } else {
                selected.push((source.clone(), offset..offset + nixe_memory::PAGE_SIZE));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ByteRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ImageStorage {
    Pitch {
        row_pitch: u64,
    },
    BlockLinear {
        block_height_log2: u8,
        block_depth_log2: u8,
    },
    BlockLinearAmpere {
        block_height_log2: u8,
        block_depth_log2: u8,
    },
}
impl ImageStorage {
    /// @SPECULATIVE: native GOB byte order follows observed Switch 2 surfaces;
    /// the complete platform layout is not publicly specified.
    /// https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/video_core/textures/decoders.cpp
    #[inline]
    fn gob_offset(self, offset: u64) -> u64 {
        if matches!(self, Self::BlockLinearAmpere { .. }) {
            let difference = ((offset >> 5) ^ (offset >> 6)) & 1;
            offset ^ (difference << 5) ^ (difference << 6)
        } else {
            offset
        }
    }
}
/// Coordinates are compressed blocks or samples when the image uses those
/// representations. Each aspect names only its actual bytes in a packed block.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ImageFootprint {
    pub depth_slice: u32,
    pub base: u64,
    pub columns: u32,
    pub rows: u32,
    pub packed_bytes: u32,
    pub aspect_offset: u32,
    pub aspect_bytes: u32,
    pub storage: ImageStorage,
    pub native_base: u64,
    pub native_element_bytes: u32,
    pub native_row_pitch: u64,
    pub rect: ByteRect,
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum WriteDomain {
    Buffer { start: u64, bytes: u64 },
    Image(ImageFootprint),
}
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Footprint {
    mapping: Arc<PhysicalMapping>,
    domain: WriteDomain,
    start: u64,
    end: u64,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ByteIntersection {
    pub start: ArenaOffset,
    pub bytes: u64,
    pub native: u64,
}
impl WriteDomain {
    /// Copy encoded storage into native byte order from the selected memory reader.
    pub fn capture(
        self,
        mut read: impl FnMut(u64, &mut [u8]) -> Result<(), MemoryError>,
        native_start: u64,
        destination: &mut [u8],
    ) -> Result<(), MemoryError> {
        if let WriteDomain::Image(image) = self
            && image.has_complete_gob()
        {
            return image.capture_gobs(&mut read, native_start, destination);
        }
        self.for_each_copy(|logical, bytes, native| {
            read(
                logical,
                capture_destination(destination, native_start, native, bytes)?,
            )
        })
    }
    fn for_each_copy(
        self,
        mut copy: impl FnMut(u64, u64, u64) -> Result<(), MemoryError>,
    ) -> Result<(), MemoryError> {
        match self {
            WriteDomain::Buffer { start, bytes } => copy(start, bytes, start),
            WriteDomain::Image(image) => {
                let packed = u64::from(image.packed_bytes);
                let full_element = image.aspect_bytes == image.packed_bytes
                    && image.native_element_bytes == image.packed_bytes;
                let left = u64::from(image.rect.x) * packed;
                let right = u64::from(image.rect.x + image.rect.width) * packed;
                if full_element
                    && (left | right).is_multiple_of(16)
                    && let ImageStorage::BlockLinear {
                        block_height_log2,
                        block_depth_log2,
                    }
                    | ImageStorage::BlockLinearAmpere {
                        block_height_log2,
                        block_depth_log2,
                    } = image.storage
                {
                    // Resolve a GOB's base once, then copy its four fixed-width
                    // units directly. Row bits are already included in offset.
                    // https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/video_core/textures/decoders.cpp#L80-L107
                    let offsets = if matches!(image.storage, ImageStorage::BlockLinearAmpere { .. })
                    {
                        [0, 64, 256, 320]
                    } else {
                        [0, 32, 256, 288]
                    };
                    let stride = 512u64 << (block_height_log2 + block_depth_log2);
                    let first_gob = left / 64;
                    let first_unit = (left % 64 / 16) as usize;
                    let first_end = ((right - first_gob * 64).min(64) / 16) as usize;
                    let gobs = left.div_ceil(64)..right.div_ceil(64);
                    let units = right / 16;
                    for y in image.rect.y..image.rect.y + image.rect.height {
                        let row = image.base + image.offset(0, u64::from(y));
                        let native_row = image.native_base + u64::from(y) * image.native_row_pitch;
                        // Only a cropped first GOB starts at a nonzero unit.
                        // The remaining GOBs use the same zero-origin loop as
                        // complete planes, without recalculating that crop.
                        if first_unit != 0 {
                            let base = row + first_gob * stride;
                            for (unit, &offset) in offsets[first_unit..first_end].iter().enumerate()
                            {
                                copy(base + offset, 16, native_row + left + unit as u64 * 16)?;
                            }
                        }
                        for gob in gobs.clone() {
                            let base = row + gob * stride;
                            let count = (units - gob * 4).min(4) as usize;
                            for (unit, &offset) in offsets[..count].iter().enumerate() {
                                copy(base + offset, 16, native_row + gob * 64 + unit as u64 * 16)?;
                            }
                        }
                    }
                    return Ok(());
                }
                for y in image.rect.y..image.rect.y + image.rect.height {
                    let row = image.base + image.offset(0, u64::from(y));
                    let native_row = image.native_base + u64::from(y) * image.native_row_pitch;
                    let address = |x: u64| match image.storage {
                        ImageStorage::Pitch { .. } => row + x,
                        ImageStorage::BlockLinear {
                            block_height_log2,
                            block_depth_log2,
                        }
                        | ImageStorage::BlockLinearAmpere {
                            block_height_log2,
                            block_depth_log2,
                        } => {
                            row + (x >> 6) * (512u64 << (block_height_log2 + block_depth_log2))
                                + ((x & 32) << 3)
                                + image.storage.gob_offset((x & 16) << 1)
                                + (x & 15)
                        }
                    };
                    let contiguous = |x: u64, bytes: u64| match image.storage {
                        ImageStorage::Pitch { .. } => bytes,
                        ImageStorage::BlockLinear { .. }
                        | ImageStorage::BlockLinearAmpere { .. } => bytes.min(16 - (x & 15)),
                    };
                    if full_element {
                        let mut x = left;
                        while x < right {
                            let count = contiguous(x, right - x);
                            copy(address(x), count, native_row + x)?;
                            x += count;
                        }
                    } else {
                        for column in image.rect.x..image.rect.x + image.rect.width {
                            let mut lane = u64::from(image.aspect_offset);
                            let end = lane + u64::from(image.aspect_bytes);
                            while lane < end {
                                let x = u64::from(column) * packed + lane;
                                let count = contiguous(x, end - lane);
                                copy(
                                    address(x),
                                    count,
                                    native_row
                                        + u64::from(column) * u64::from(image.native_element_bytes)
                                        + lane
                                        - u64::from(image.aspect_offset),
                                )?;
                                lane += count;
                            }
                        }
                    }
                }
                Ok(())
            }
        }
    }
}
impl ImageFootprint {
    fn has_complete_gob(self) -> bool {
        self.aspect_bytes == self.packed_bytes
            && self.native_element_bytes == self.packed_bytes
            && self.has_complete_storage_gob()
    }

    fn has_complete_storage_gob(self) -> bool {
        let left = u64::from(self.rect.x) * u64::from(self.packed_bytes);
        let right = left + u64::from(self.rect.width) * u64::from(self.packed_bytes);
        matches!(
            self.storage,
            ImageStorage::BlockLinear { .. } | ImageStorage::BlockLinearAmpere { .. }
        ) && self.packed_bytes.is_power_of_two()
            && self.packed_bytes <= 16
            && left.next_multiple_of(64) + 64 <= right
            && u64::from(self.rect.y).next_multiple_of(8) + 8
                <= u64::from(self.rect.y) + u64::from(self.rect.height)
    }

    fn capture_gobs(
        self,
        read: &mut impl FnMut(u64, &mut [u8]) -> Result<(), MemoryError>,
        native_start: u64,
        destination: &mut [u8],
    ) -> Result<(), MemoryError> {
        let (ImageStorage::BlockLinear { .. } | ImageStorage::BlockLinearAmpere { .. }) =
            self.storage
        else {
            unreachable!("complete GOBs require block-linear storage");
        };
        let packed = u64::from(self.packed_bytes);
        let element = u64::from(self.native_element_bytes);
        let full_element = self.aspect_bytes == self.packed_bytes
            && self.native_element_bytes == self.packed_bytes;
        let left = u64::from(self.rect.x) * u64::from(self.packed_bytes);
        let right = left + u64::from(self.rect.width) * u64::from(self.packed_bytes);
        let bottom = u64::from(self.rect.y + self.rect.height);
        let mut y = u64::from(self.rect.y);
        let mut gob = [0; 512];
        while y < bottom {
            let lines = if y.is_multiple_of(8) && bottom - y >= 8 {
                8
            } else {
                1
            };
            let mut x = left;
            while x < right {
                let width = (right - x).min(64 - x % 64);
                if lines == 8 && width == 64 {
                    // Capture packed pixels once, then extract native aspects
                    // from owned bytes. The caller admits the whole GOB.
                    read(self.base + self.offset(x, y), &mut gob)?;
                    for row in 0..8 {
                        let native = self.native_base
                            + (y + row) * self.native_row_pitch
                            + x / packed * element;
                        let target = capture_destination(
                            destination,
                            native_start,
                            native,
                            64 / packed * element,
                        )?;
                        let row_offset = (((row & 6) << 5) | ((row & 1) << 4)) as usize;
                        if full_element {
                            for (column, target) in target.chunks_exact_mut(16).enumerate() {
                                let at = row_offset + ((column & 2) << 7) + ((column & 1) << 5);
                                let at = self.storage.gob_offset(at as u64) as usize;
                                target.copy_from_slice(&gob[at..at + 16]);
                            }
                        } else {
                            for (column, target) in
                                target.chunks_exact_mut(element as usize).enumerate()
                            {
                                let x = column * packed as usize;
                                let at = row_offset
                                    + ((x & 32) << 3)
                                    + ((x & 16) << 1)
                                    + (x & 15)
                                    + self.aspect_offset as usize;
                                let at = self.storage.gob_offset(at as u64) as usize;
                                let bytes = self.aspect_bytes as usize;
                                target[..bytes].copy_from_slice(&gob[at..at + bytes]);
                            }
                        }
                    }
                } else {
                    // Cropped edges read only their demanded aspect bytes.
                    let edge = Self {
                        rect: ByteRect {
                            x: (x / packed) as u32,
                            y: y as u32,
                            width: (width / packed) as u32,
                            height: lines as u32,
                        },
                        ..self
                    };
                    WriteDomain::Image(edge).for_each_copy(|logical, bytes, native| {
                        read(
                            logical,
                            capture_destination(destination, native_start, native, bytes)?,
                        )
                    })?;
                }
                x += width;
            }
            y += lines;
        }
        Ok(())
    }

    fn physical_lanes(self, span: &PhysicalSpan) -> Option<(u32, u32)> {
        let packed = self.packed_bytes;
        if !packed.is_power_of_two()
            || packed > 16
            || matches!(self.storage, ImageStorage::Pitch { row_pitch } if !row_pitch.is_multiple_of(u64::from(packed)))
        {
            return None;
        }
        let origin = span
            .start
            .get()
            .wrapping_sub(span.logical)
            .wrapping_add(self.base);
        let mut lanes = 0;
        for lane in self.aspect_offset..self.aspect_offset + self.aspect_bytes {
            lanes |= 1 << (origin.wrapping_add(u64::from(lane)) & u64::from(packed - 1));
        }
        Some((packed, lanes))
    }
    fn offset(self, x: u64, y: u64) -> u64 {
        match self.storage {
            ImageStorage::Pitch { row_pitch } => {
                (u64::from(self.depth_slice) * u64::from(self.rows) + y) * row_pitch + x
            }
            ImageStorage::BlockLinear {
                block_height_log2,
                block_depth_log2,
            }
            | ImageStorage::BlockLinearAmpere {
                block_height_log2,
                block_depth_log2,
            } => {
                // GOB X/block-row strides include all Z slices even for a
                // 2D plane; only the selected slice owns physical bytes.
                // https://github.com/eden-emulator/mirror/blob/master/src/video_core/textures/decoders.cpp
                let height = 8u64 << block_height_log2;
                let row_bytes = u64::from(self.columns) * u64::from(self.packed_bytes);
                let slice = u64::from(self.depth_slice);
                let slice_bytes = u64::from(self.rows).div_ceil(height)
                    * row_bytes.div_ceil(64)
                    * (512u64 << (block_height_log2 + block_depth_log2));
                self.storage.gob_offset(
                    (slice >> block_depth_log2) * slice_bytes
                        + (slice % (1u64 << block_depth_log2)) * (512u64 << block_height_log2)
                        + (y / height)
                            * row_bytes.div_ceil(64)
                            * (512 << (block_height_log2 + block_depth_log2))
                        + (x / 64) * (512 << (block_height_log2 + block_depth_log2))
                        + (y % height / 8) * 512
                        + (x % 64 / 32) * 256
                        + (y % 8 / 2) * 64
                        + (x % 32 / 16) * 32
                        + (y % 2) * 16
                        + x % 16,
                )
            }
        }
    }
    fn same_geometry(self, other: Self) -> bool {
        self.base == other.base
            && self.depth_slice == other.depth_slice
            && self.packed_bytes == other.packed_bytes
            && self.storage == other.storage
            && match self.storage {
                ImageStorage::Pitch { .. } => self.depth_slice == 0 || self.rows == other.rows,
                ImageStorage::BlockLinear {
                    block_height_log2,
                    block_depth_log2,
                }
                | ImageStorage::BlockLinearAmpere {
                    block_height_log2,
                    block_depth_log2,
                } => {
                    (u64::from(self.columns) * u64::from(self.packed_bytes)).div_ceil(64)
                        == (u64::from(other.columns) * u64::from(other.packed_bytes)).div_ceil(64)
                        && (self.depth_slice < (1 << block_depth_log2)
                            || self.rows.div_ceil(8 << block_height_log2)
                                == other.rows.div_ceil(8 << block_height_log2))
                }
            }
    }
    fn dimensions(self) -> Result<(u64, u64), MemoryError> {
        if self.columns == 0
            || self.rows == 0
            || self.packed_bytes == 0
            || self.aspect_bytes == 0
            || self
                .aspect_offset
                .checked_add(self.aspect_bytes)
                .is_none_or(|end| end > self.packed_bytes)
            || self.rect.width == 0
            || self.rect.height == 0
            || self
                .rect
                .x
                .checked_add(self.rect.width)
                .is_none_or(|end| end > self.columns)
            || self
                .rect
                .y
                .checked_add(self.rect.height)
                .is_none_or(|end| end > self.rows)
        {
            return Err(MemoryError::Invalid(
                "invalid image footprint dimensions/aspect/rectangle",
            ));
        }
        let row_bytes = u64::from(self.columns) * u64::from(self.packed_bytes);
        let size = match self.storage {
            ImageStorage::Pitch { row_pitch } if row_pitch >= row_bytes => row_pitch
                .checked_mul((u64::from(self.depth_slice) + 1) * u64::from(self.rows) - 1)
                .and_then(|size| size.checked_add(row_bytes)),
            ImageStorage::BlockLinear {
                block_height_log2,
                block_depth_log2,
            }
            | ImageStorage::BlockLinearAmpere {
                block_height_log2,
                block_depth_log2,
            } if block_height_log2 <= 5 && block_depth_log2 <= 5 => {
                let height = 8u64 << block_height_log2;
                let padded = row_bytes
                    .div_ceil(64)
                    .checked_mul(512 << (block_height_log2 + block_depth_log2))
                    .and_then(|row| row.checked_mul(u64::from(self.rows).div_ceil(height)))
                    .and_then(|slice| {
                        slice.checked_mul((u64::from(self.depth_slice) >> block_depth_log2) + 1)
                    });
                padded.map(|_| {
                    let x = row_bytes - 1;
                    let y = u64::from(self.rows - 1);
                    self.offset(x, y) + 1
                })
            }
            _ => return Err(MemoryError::Invalid("unsupported image footprint storage")),
        }
        .ok_or(MemoryError::Invalid(
            "image footprint storage size overflows",
        ))?;
        let native_row = u64::from(self.columns) * u64::from(self.native_element_bytes);
        if self.native_element_bytes < self.aspect_bytes
            || self.native_row_pitch < native_row
            || self
                .native_row_pitch
                .checked_mul(u64::from(self.rows))
                .and_then(|size| self.native_base.checked_add(size))
                .is_none()
        {
            return Err(MemoryError::Invalid(
                "image footprint native range overflows",
            ));
        }
        Ok((row_bytes, size))
    }
    fn coordinates(self, offset: u64) -> (u64, u64, u64) {
        match self.storage {
            ImageStorage::Pitch { row_pitch } => (
                offset % row_pitch,
                if offset / (u64::from(self.rows) * row_pitch) == u64::from(self.depth_slice) {
                    offset / row_pitch % u64::from(self.rows)
                } else {
                    u64::from(self.rows)
                },
                row_pitch - offset % row_pitch,
            ),
            ImageStorage::BlockLinear {
                block_height_log2,
                block_depth_log2,
            }
            | ImageStorage::BlockLinearAmpere {
                block_height_log2,
                block_depth_log2,
            } => {
                let row_bytes = u64::from(self.columns) * u64::from(self.packed_bytes);
                let block_bytes = 512u64 << (block_height_log2 + block_depth_log2);
                let pitch = row_bytes.div_ceil(64) * block_bytes;
                let slice_bytes = u64::from(self.rows).div_ceil(8u64 << block_height_log2) * pitch;
                let slice_group = offset / slice_bytes;
                let in_slice_group = offset % slice_bytes;
                let block_row = in_slice_group / pitch;
                let in_row = in_slice_group % pitch;
                let within_block = in_row % block_bytes;
                let plane_bytes = 512u64 << block_height_log2;
                let slice = (slice_group << block_depth_log2) + within_block / plane_bytes;
                if slice != u64::from(self.depth_slice) {
                    // The remaining Z slices are not part of this 2D plane.
                    return (
                        0,
                        u64::from(self.rows),
                        plane_bytes - within_block % plane_bytes,
                    );
                }
                let gob = self.storage.gob_offset(in_row % 512);
                let x = (in_row / block_bytes) * 64
                    + (gob / 256) * 32
                    + ((gob / 32) & 1) * 16
                    + gob % 16;
                let y = block_row * (8 << block_height_log2)
                    + (within_block % plane_bytes / 512) * 8
                    + ((gob / 64) & 3) * 2
                    + ((gob / 16) & 1);
                (x, y, 16 - gob % 16)
            }
        }
    }
}
impl Footprint {
    pub fn new(
        mapping: Arc<PhysicalMapping>,
        domain: WriteDomain,
    ) -> Result<Arc<Self>, MemoryError> {
        let (base, bytes) = match domain {
            WriteDomain::Buffer { start, bytes } if bytes != 0 => (start, bytes),
            WriteDomain::Image(image) => (image.base, image.dimensions()?.1),
            _ => return Err(MemoryError::Invalid("empty resource write domain")),
        };
        if base
            .checked_add(bytes)
            .is_none_or(|end| end > mapping.bytes())
        {
            return Err(MemoryError::Invalid(
                "resource footprint exceeds its physical mapping",
            ));
        }
        let (start, end) = match domain {
            WriteDomain::Buffer { start, bytes } => (start, start + bytes),
            WriteDomain::Image(image) => {
                let first_x = u64::from(image.rect.x) * u64::from(image.packed_bytes)
                    + u64::from(image.aspect_offset);
                let last_x = u64::from(image.rect.x + image.rect.width - 1)
                    * u64::from(image.packed_bytes)
                    + u64::from(image.aspect_offset + image.aspect_bytes - 1);
                (
                    image.base + image.offset(first_x, u64::from(image.rect.y)),
                    image.base
                        + image.offset(last_x, u64::from(image.rect.y + image.rect.height - 1))
                        + 1,
                )
            }
        };
        Ok(Arc::new(Self {
            mapping,
            domain,
            start,
            end,
        }))
    }
    pub fn mapping(&self) -> &Arc<PhysicalMapping> {
        &self.mapping
    }
    pub fn domain(&self) -> WriteDomain {
        self.domain
    }
    pub fn bounds(&self) -> std::ops::Range<u64> {
        self.start..self.end
    }
    /// Publish a completed native plane into retained guest RAM. Resolve and
    /// validate a contiguous destination once, then swizzle directly into it.
    /// Fragmented/sparse mappings publish their actual spans in logical order.
    /// https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/video_core/texture_cache/util.cpp#L532-L588
    pub fn publish_resource(
        &self,
        writer: &nixe_memory::ResourceWriter<'_>,
        native_start: u64,
        source: &[u8],
    ) -> Result<(), MemoryError> {
        if writer.machine() != self.mapping.machine() {
            return Err(MemoryError::Invalid(
                "resource publication names another machine",
            ));
        }
        // Footprint construction validated all layout arithmetic. Check the
        // complete native interval once, before either publication traversal.
        let (begin, end) = match self.domain {
            WriteDomain::Buffer { start, bytes } => (start, start + bytes),
            WriteDomain::Image(image) => (
                image.native_base
                    + u64::from(image.rect.y) * image.native_row_pitch
                    + u64::from(image.rect.x) * u64::from(image.native_element_bytes),
                image.native_base
                    + u64::from(image.rect.y + image.rect.height - 1) * image.native_row_pitch
                    + u64::from(image.rect.x + image.rect.width - 1)
                        * u64::from(image.native_element_bytes)
                    + u64::from(image.aspect_bytes),
            ),
        };
        if native_start > begin || end - native_start > source.len() as u64 {
            return Err(MemoryError::Invalid(
                "publication source does not contain its native plane",
            ));
        }
        let spans = self.mapping.spans();
        let first = spans.partition_point(|span| span.logical + span.bytes <= self.start);
        let spans = &spans[first..];
        if let Some(extent) = self.contiguous_destination(spans) {
            let destination = writer.region(extent)?;
            let start = self.start;
            // Capture immutable bases by value so native RAM stores cannot
            // force reloads of borrowed metadata for every copy unit.
            return self.domain.for_each_copy(move |logical, bytes, native| {
                let offset = (native - native_start) as usize;
                // SAFETY: the validated layout emits only this footprint's
                // bytes, within its retained contiguous destination and the
                // native interval checked above. RAM never becomes a slice.
                unsafe {
                    destination.write_unchecked(
                        logical - start,
                        source.get_unchecked(offset..offset + bytes as usize),
                    );
                }
                Ok(())
            });
        }
        // Resolve each physical fragment once. Visiting only covered bytes
        // preserves padding, packed aspects, holes and later logical aliases
        // without a scratch copy of the enclosing subresource.
        for (index, span) in spans
            .iter()
            .take_while(|span| span.logical < self.end)
            .enumerate()
        {
            let begin = self.start.max(span.logical);
            let end = self.end.min(span.logical + span.bytes);
            if begin >= end {
                continue;
            }
            let extent = nixe_memory::Extent {
                start: ArenaOffset::new(span.start.get() + begin - span.logical),
                bytes: end - begin,
            };
            let destination = writer.region(extent)?;
            self.visit(first + index, extent.start, extent.bytes, move |copy| {
                let offset = (copy.native - native_start) as usize;
                // SAFETY: visit clips every copy to the selected retained
                // fragment and to the validated image rectangle/aspect. The
                // native bounds above cover the corresponding source bytes.
                unsafe {
                    destination.write_unchecked(
                        copy.start.get() - extent.start.get(),
                        source.get_unchecked(offset..offset + copy.bytes as usize),
                    );
                }
                true
            })?;
        }
        Ok(())
    }
    fn contiguous_destination(&self, spans: &[PhysicalSpan]) -> Option<nixe_memory::Extent> {
        let span = spans.first()?;
        if span.logical > self.start {
            return None;
        }
        let start = span.start.get() + self.start - span.logical;
        let mut cursor = self.start;
        // Physically adjacent runs can belong to separate allocations. Their
        // retained union still admits one destination, while holes or aliases
        // require the fragment traversal.
        for span in spans {
            if span.logical > cursor
                || span.start.get() + cursor - span.logical != start + cursor - self.start
            {
                return None;
            }
            cursor = self.end.min(span.logical + span.bytes);
            if cursor == self.end {
                return Some(nixe_memory::Extent {
                    start: ArenaOffset::new(start),
                    bytes: self.end - self.start,
                });
            }
        }
        None
    }
    pub fn covers(&self, other: &Self) -> bool {
        if !self.mapping.same_addressing(&other.mapping) {
            return false;
        }
        match (self.domain, other.domain) {
            (
                WriteDomain::Buffer {
                    start: a,
                    bytes: an,
                },
                WriteDomain::Buffer {
                    start: b,
                    bytes: bn,
                },
            ) => a <= b && a + an >= b + bn,
            (WriteDomain::Image(a), WriteDomain::Image(b)) if a.same_geometry(b) => {
                a.aspect_offset <= b.aspect_offset
                    && a.aspect_offset + a.aspect_bytes >= b.aspect_offset + b.aspect_bytes
                    && a.rect.x <= b.rect.x
                    && a.rect.y <= b.rect.y
                    && a.rect.x + a.rect.width >= b.rect.x + b.rect.width
                    && a.rect.y + a.rect.height >= b.rect.y + b.rect.height
            }
            _ => false,
        }
    }
    pub fn intersects(&self, other: &Self) -> bool {
        if self.mapping.machine() != other.mapping.machine() {
            return false;
        }
        if self.mapping.is_injective()
            && other.mapping.is_injective()
            && self.mapping.same_addressing(&other.mapping)
        {
            if self.start >= other.end || other.start >= self.end {
                return false;
            }
            match (self.domain, other.domain) {
                (
                    WriteDomain::Buffer {
                        start: a,
                        bytes: an,
                    },
                    WriteDomain::Buffer {
                        start: b,
                        bytes: bn,
                    },
                ) => return a < b + bn && b < a + an,
                (WriteDomain::Buffer { start, bytes }, WriteDomain::Image(_))
                    if start <= other.start && start + bytes >= other.end =>
                {
                    return true;
                }
                (WriteDomain::Image(_), WriteDomain::Buffer { start, bytes })
                    if start <= self.start && start + bytes >= self.end =>
                {
                    return true;
                }
                (WriteDomain::Image(a), WriteDomain::Image(b)) if a.same_geometry(b) => {
                    return a.aspect_offset < b.aspect_offset + b.aspect_bytes
                        && b.aspect_offset < a.aspect_offset + a.aspect_bytes
                        && a.rect.x < b.rect.x + b.rect.width
                        && b.rect.x < a.rect.x + a.rect.width
                        && a.rect.y < b.rect.y + b.rect.height
                        && b.rect.y < a.rect.y + a.rect.height;
                }
                _ => {}
            }
        }
        for (a_index, a) in self.mapping.spans().iter().enumerate() {
            let a_begin = a.logical.max(self.start);
            let a_end = (a.logical + a.bytes).min(self.end);
            if a_begin >= a_end {
                continue;
            }
            for (b_index, b) in other.mapping.spans().iter().enumerate() {
                if a.allocation != b.allocation {
                    continue;
                }
                if let (WriteDomain::Image(left), WriteDomain::Image(right)) =
                    (self.domain, other.domain)
                    && let (Some((left_size, left_lanes)), Some((right_size, right_lanes))) =
                        (left.physical_lanes(a), right.physical_lanes(b))
                    && left_size == right_size
                    && left_lanes & right_lanes == 0
                {
                    continue;
                }
                let b_begin = b.logical.max(other.start);
                let b_end = (b.logical + b.bytes).min(other.end);
                if b_begin >= b_end {
                    continue;
                }
                let begin =
                    (a.start.get() + a_begin - a.logical).max(b.start.get() + b_begin - b.logical);
                let end =
                    (a.start.get() + a_end - a.logical).min(b.start.get() + b_end - b.logical);
                if begin >= end {
                    continue;
                }
                let mut found = false;
                self.visit(a_index, ArenaOffset::new(begin), end - begin, |span| {
                    other
                        .visit(b_index, span.start, span.bytes, |_| {
                            found = true;
                            false
                        })
                        .expect("valid footprint intersection");
                    !found
                })
                .expect("valid footprint intersection");
                if found {
                    return true;
                }
            }
        }
        false
    }
    /// Import actual bytes in destination order, directly into owned or mapped
    /// upload storage. Unlike physical intersection queries, deswizzling writes
    /// adjacent output runs, also on write-combined staging memory.
    /// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/textures/decoders.cpp
    pub fn capture_resource(
        &self,
        reader: &nixe_memory::ResourceReader<'_>,
        native_start: u64,
        destination: &mut [u8],
    ) -> Result<(), MemoryError> {
        if reader.machine() != self.mapping.machine() {
            return Err(MemoryError::Invalid(
                "resource capture names another machine",
            ));
        }
        // Read packed GOBs only when every mapped byte they can touch is
        // retained. Physical fragmentation does not change the decoder; holes
        // retain the ordinary zero-fill behavior.
        let packed_image = match self.domain {
            WriteDomain::Image(image) if image.has_complete_storage_gob() => {
                let left = u64::from(image.rect.x) * u64::from(image.packed_bytes);
                let right =
                    u64::from(image.rect.x + image.rect.width) * u64::from(image.packed_bytes);
                let begin = image.base + image.offset(left, u64::from(image.rect.y));
                let end = image.base
                    + image.offset(right - 1, u64::from(image.rect.y + image.rect.height - 1))
                    + 1;
                let spans = self.mapping.spans();
                let first = spans.partition_point(|span| span.logical + span.bytes <= begin);
                let retained = spans[first..]
                    .iter()
                    .take_while(|span| span.logical < end)
                    .all(|span| {
                        let start = begin.max(span.logical);
                        let stop = end.min(span.logical + span.bytes);
                        reader
                            .region(nixe_memory::Extent {
                                start: ArenaOffset::new(span.start.get() + start - span.logical),
                                bytes: stop - start,
                            })
                            .is_ok()
                    });
                retained.then_some(image)
            }
            _ => None,
        };
        let mut region: Option<(std::ops::Range<u64>, nixe_memory::ResourceRead<'_>)> = None;
        let mut read = |mut logical: u64, mut destination: &mut [u8]| {
            while !destination.is_empty() {
                if let Some((range, read)) = &region
                    && range.contains(&logical)
                {
                    let count = (destination.len() as u64).min(range.end - logical) as usize;
                    let (target, rest) = destination.split_at_mut(count);
                    read.copy(logical - range.start, target)?;
                    destination = rest;
                    logical += count as u64;
                    continue;
                }
                let spans = self.mapping.spans();
                let index = spans.partition_point(|span| span.logical + span.bytes <= logical);
                let Some(span) = spans.get(index) else {
                    destination.fill(0);
                    break;
                };
                if logical < span.logical {
                    let count = destination.len().min((span.logical - logical) as usize);
                    let (zero, rest) = destination.split_at_mut(count);
                    zero.fill(0);
                    logical += count as u64;
                    destination = rest;
                    continue;
                }
                let start = span.start.get() + logical - span.logical;
                let regions = reader.regions();
                let index = regions.partition_point(|extent| extent.start.get() <= start);
                let extent = index.checked_sub(1).map(|index| regions[index]).ok_or(
                    MemoryError::Invalid("resource capture has no retained input"),
                )?;
                let begin = span.start.get().max(extent.start.get());
                let end = (span.start.get() + span.bytes).min(extent.start.get() + extent.bytes);
                if start < begin || start >= end {
                    return Err(MemoryError::Invalid(
                        "resource capture has no retained input",
                    ));
                }
                let read = reader.region(nixe_memory::Extent {
                    start: ArenaOffset::new(begin),
                    bytes: end - begin,
                })?;
                region = Some((
                    span.logical + begin - span.start.get()..span.logical + end - span.start.get(),
                    read,
                ));
            }
            Ok(())
        };
        if let Some(image) = packed_image {
            image.capture_gobs(&mut read, native_start, destination)
        } else {
            self.domain.capture(read, native_start, destination)
        }
    }
    /// The index supplies the candidate mapping span. This routine proves only
    /// the actual covered intersection, never the resource's bounding interval.
    pub fn visit(
        &self,
        span_index: usize,
        start: ArenaOffset,
        bytes: u64,
        mut visit: impl FnMut(ByteIntersection) -> bool,
    ) -> Result<(), MemoryError> {
        let span = self
            .mapping
            .spans()
            .get(span_index)
            .ok_or(MemoryError::Invalid("resource footprint span is invalid"))?;
        let end = start
            .get()
            .checked_add(bytes)
            .ok_or(MemoryError::Invalid("physical footprint query overflows"))?;
        let begin = start.get().max(span.start.get());
        let stop = end.min(span.start.get() + span.bytes);
        if begin >= stop {
            return Ok(());
        }
        let logical = span.logical + begin - span.start.get();
        match self.domain {
            WriteDomain::Buffer { start, bytes } => {
                let low = logical.max(start);
                let high = (logical + stop - begin).min(start + bytes);
                if low < high {
                    visit(ByteIntersection {
                        start: ArenaOffset::new(begin + low - logical),
                        bytes: high - low,
                        native: low,
                    });
                }
            }
            WriteDomain::Image(image) => {
                let mut at = logical.max(self.start);
                let end = (logical + stop - begin).min(self.end);
                let mut gob_base = u64::MAX;
                let mut gob_origin = (0, 0, 0);
                while at < end {
                    let offset = at - image.base;
                    let (x, y, continuous) = match image.storage {
                        ImageStorage::Pitch { .. } => image.coordinates(offset),
                        ImageStorage::BlockLinear { .. }
                        | ImageStorage::BlockLinearAmpere { .. } => {
                            // Resolve block and slice strides once per 512-byte GOB.
                            // Its 16-byte runs use only the intra-GOB bit permutation,
                            // as Eden's texture decoder, rather than repeating dynamic
                            // divisions for every run (or every packed aspect lane).
                            // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/textures/decoders.cpp
                            let base = offset & !511;
                            if base != gob_base {
                                gob_base = base;
                                gob_origin = image.coordinates(base);
                            }
                            let physical_within = offset & 511;
                            let within = image.storage.gob_offset(physical_within);
                            if gob_origin.2 > 16 {
                                (0, gob_origin.1, gob_origin.2 - physical_within)
                            } else {
                                (
                                    gob_origin.0
                                        + ((within >> 3) & 32)
                                        + ((within >> 1) & 16)
                                        + (within & 15),
                                    gob_origin.1 + ((within >> 5) & 6) + ((within >> 4) & 1),
                                    16 - (within & 15),
                                )
                            }
                        }
                    };
                    let packed = u64::from(image.packed_bytes);
                    let column = x / packed;
                    let lane = x % packed;
                    let mut length = continuous.min(end - at);
                    let in_rect = column >= u64::from(image.rect.x)
                        && column < u64::from(image.rect.x + image.rect.width)
                        && y >= u64::from(image.rect.y)
                        && y < u64::from(image.rect.y + image.rect.height);
                    let aspect_start = u64::from(image.aspect_offset);
                    let aspect_end = aspect_start + u64::from(image.aspect_bytes);
                    if in_rect && lane >= aspect_start && lane < aspect_end {
                        if image.aspect_bytes == image.packed_bytes
                            && image.native_element_bytes == image.packed_bytes
                        {
                            length =
                                length.min(u64::from(image.rect.x + image.rect.width) * packed - x);
                        } else {
                            length = length.min(aspect_end - lane);
                        }
                        if !visit(ByteIntersection {
                            start: ArenaOffset::new(begin + at - logical),
                            bytes: length,
                            native: image.native_base
                                + y * image.native_row_pitch
                                + column * u64::from(image.native_element_bytes)
                                + lane
                                - aspect_start,
                        }) {
                            return Ok(());
                        }
                    } else if !in_rect {
                        // A whole pitch-linear row can be skipped when it lies
                        // outside the rectangle. A tiled run is at most 16 bytes.
                        if y >= u64::from(image.rect.y)
                            && y < u64::from(image.rect.y + image.rect.height)
                            && column < u64::from(image.rect.x)
                        {
                            length = length.min(u64::from(image.rect.x) * packed - x);
                        }
                    } else {
                        length = length.min(if lane < aspect_start {
                            aspect_start - lane
                        } else {
                            packed - lane
                        });
                    }
                    at += length;
                }
            }
        }
        Ok(())
    }
}

fn capture_destination(
    destination: &mut [u8],
    native_start: u64,
    native: u64,
    bytes: u64,
) -> Result<&mut [u8], MemoryError> {
    let start = native
        .checked_sub(native_start)
        .and_then(|offset| usize::try_from(offset).ok())
        .ok_or(MemoryError::Invalid(
            "resource capture destination overflows",
        ))?;
    let end = start
        .checked_add(bytes as usize)
        .ok_or(MemoryError::Invalid(
            "resource capture destination overflows",
        ))?;
    destination
        .get_mut(start..end)
        .ok_or(MemoryError::Invalid("resource capture exceeds destination"))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn streaming_capture_preserves_fragmented_and_aliased_mapping_order() {
        use nixe_memory::{Machine, PAGE_SIZE};
        let machine = Machine::new(4 * PAGE_SIZE).unwrap();
        let allocation = machine
            .account(4 * PAGE_SIZE)
            .unwrap()
            .allocate(4 * PAGE_SIZE)
            .unwrap();
        let range = AllocationRange::new(allocation.pin(), 0, 4 * PAGE_SIZE).unwrap();
        let bytes: Vec<_> = (0..4 * PAGE_SIZE)
            .map(|i| (i / PAGE_SIZE + i % 251) as u8)
            .collect();
        machine
            .write_resolved(std::slice::from_ref(&range), &bytes)
            .unwrap();
        let mapping = PhysicalMapping::new(&[
            range.slice(2 * PAGE_SIZE, PAGE_SIZE).unwrap(),
            range.slice(0, PAGE_SIZE).unwrap(),
            range.slice(2 * PAGE_SIZE, PAGE_SIZE).unwrap(),
        ])
        .unwrap();
        let mut snapshot = vec![0; (2 * PAGE_SIZE + 6) as usize];
        mapping
            .capture_stream(PAGE_SIZE - 3, &mut snapshot)
            .unwrap_err();
        snapshot.resize((PAGE_SIZE + 6) as usize, 0);
        mapping
            .capture_stream(PAGE_SIZE - 3, &mut snapshot)
            .unwrap();
        let expected: Vec<_> = bytes[(3 * PAGE_SIZE - 3) as usize..(3 * PAGE_SIZE) as usize]
            .iter()
            .chain(&bytes[..PAGE_SIZE as usize])
            .chain(&bytes[(2 * PAGE_SIZE) as usize..(2 * PAGE_SIZE + 3) as usize])
            .copied()
            .collect();
        assert_eq!(snapshot, expected);
        let footprint = Footprint::new(
            mapping.clone(),
            WriteDomain::Buffer {
                start: PAGE_SIZE - 3,
                bytes: snapshot.len() as u64,
            },
        )
        .unwrap();
        let mut pages = Vec::new();
        footprint.observed_pages(&mapping, &mut pages).unwrap();
        let inputs = pin_selected_pages(pages).unwrap();
        let mut imported = vec![0x7a; (3 * PAGE_SIZE) as usize];
        machine
            .read_resources(&inputs, |reader| {
                footprint.capture_resource(reader, 0, &mut imported)
            })
            .unwrap();
        assert_eq!(
            &imported[(PAGE_SIZE - 3) as usize..(2 * PAGE_SIZE + 3) as usize],
            &expected
        );
        assert!(
            imported[..(PAGE_SIZE - 3) as usize]
                .iter()
                .all(|byte| *byte == 0x7a)
        );
        assert!(
            imported[(2 * PAGE_SIZE + 3) as usize..]
                .iter()
                .all(|byte| *byte == 0x7a)
        );
        drop(inputs);

        drop(range);
        drop(allocation);
        assert_eq!(machine.statistics().allocated_bytes, 0);
        assert!(
            mapping
                .capture_stream(PAGE_SIZE - 3, &mut snapshot)
                .is_err()
        );
    }

    #[test]
    fn cropped_aliases_share_tiled_addressing_without_claiming_other_rows_or_aspects() {
        use nixe_memory::{Machine, PAGE_SIZE};
        let machine = Machine::new(2 * PAGE_SIZE).unwrap();
        let allocation = machine
            .account(2 * PAGE_SIZE)
            .unwrap()
            .allocate(2 * PAGE_SIZE)
            .unwrap();
        let mapping = |bytes| {
            PhysicalMapping::new(&[AllocationRange::new(allocation.pin(), 0, bytes).unwrap()])
                .unwrap()
        };
        let image = |mapping, columns, rows, rect, aspect_offset, aspect_bytes| {
            Footprint::new(
                mapping,
                WriteDomain::Image(ImageFootprint {
                    depth_slice: 0,
                    base: 0,
                    columns,
                    rows,
                    packed_bytes: 4,
                    aspect_offset,
                    aspect_bytes,
                    storage: ImageStorage::BlockLinear {
                        block_height_log2: 1,
                        block_depth_log2: 0,
                    },
                    native_base: 0,
                    native_element_bytes: 4,
                    native_row_pitch: u64::from(columns) * 4,
                    rect,
                }),
            )
            .unwrap()
        };
        let full = image(
            mapping(2 * PAGE_SIZE),
            31,
            33,
            ByteRect {
                x: 0,
                y: 0,
                width: 31,
                height: 33,
            },
            0,
            4,
        );
        let crop = image(
            mapping(PAGE_SIZE),
            17,
            13,
            ByteRect {
                x: 0,
                y: 0,
                width: 17,
                height: 13,
            },
            0,
            3,
        );
        assert!(full.covers(&crop));
        assert!(!crop.covers(&full));
        assert!(full.intersects(&crop));
        assert!(crop.intersects(&full));
        let right = image(
            mapping(2 * PAGE_SIZE),
            31,
            33,
            ByteRect {
                x: 20,
                y: 0,
                width: 11,
                height: 13,
            },
            0,
            4,
        );
        let below = image(
            mapping(2 * PAGE_SIZE),
            31,
            33,
            ByteRect {
                x: 0,
                y: 20,
                width: 17,
                height: 13,
            },
            0,
            4,
        );
        let stencil = image(
            mapping(PAGE_SIZE),
            17,
            13,
            ByteRect {
                x: 0,
                y: 0,
                width: 17,
                height: 13,
            },
            3,
            1,
        );
        for disjoint in [&right, &below, &stencil] {
            assert!(!crop.intersects(disjoint));
            assert!(!disjoint.intersects(&crop));
            assert!(!crop.covers(disjoint));
        }
        let shifted =
            PhysicalMapping::new(&[
                AllocationRange::new(allocation.pin(), PAGE_SIZE, PAGE_SIZE).unwrap()
            ])
            .unwrap();
        let shifted = image(
            shifted,
            17,
            13,
            ByteRect {
                x: 0,
                y: 0,
                width: 17,
                height: 13,
            },
            0,
            3,
        );
        assert!(!crop.intersects(&shifted));
        assert!(!crop.covers(&shifted));
    }
    use nixe_memory::{Machine, PAGE_SIZE};

    #[test]
    fn volume_slice_projection_matches_gob_oracle_across_depth_groups() {
        use std::collections::BTreeMap;
        let machine = Machine::new(16 * PAGE_SIZE).unwrap();
        let allocation = machine
            .account(16 * PAGE_SIZE)
            .unwrap()
            .allocate(16 * PAGE_SIZE)
            .unwrap();
        let mapping =
            PhysicalMapping::new(&[
                AllocationRange::new(allocation.pin(), 0, 16 * PAGE_SIZE).unwrap()
            ])
            .unwrap();
        let data: Vec<_> = (0..16 * PAGE_SIZE)
            .map(|byte| (byte.wrapping_mul(13) ^ (byte >> 7)) as u8)
            .collect();
        machine
            .write_resolved(&mapping.pin(0, mapping.bytes()).unwrap(), &data)
            .unwrap();
        let origin = mapping.spans()[0].start.get();
        for (bh, bd) in [(0, 0), (0, 3), (1, 2)] {
            for z in 0..7 {
                let image = Footprint::new(
                    mapping.clone(),
                    WriteDomain::Image(ImageFootprint {
                        depth_slice: z,
                        base: 13,
                        columns: 23,
                        rows: 21,
                        packed_bytes: 4,
                        aspect_offset: 1,
                        aspect_bytes: 3,
                        storage: ImageStorage::BlockLinear {
                            block_height_log2: bh,
                            block_depth_log2: bd,
                        },
                        native_base: 0,
                        native_element_bytes: 4,
                        native_row_pitch: 92,
                        rect: ByteRect {
                            x: 3,
                            y: 2,
                            width: 17,
                            height: 13,
                        },
                    }),
                )
                .unwrap();
                let mut actual = BTreeMap::new();
                image
                    .visit(0, mapping.spans()[0].start, mapping.bytes(), |part| {
                        for byte in 0..part.bytes {
                            assert!(
                                actual
                                    .insert(part.native + byte, part.start.get() + byte)
                                    .is_none()
                            );
                        }
                        true
                    })
                    .unwrap();
                let mut expected = BTreeMap::new();
                let h = 1u64 << bh;
                let d = 1u64 << bd;
                for y in 2u64..15 {
                    for x in 3u64..20 {
                        for lane in 1..4 {
                            let raw_x = x * 4 + lane;
                            let block = 512 * h * d;
                            let address = (u64::from(z) / d) * 21u64.div_ceil(8 * h) * 2 * block
                                + (y / (8 * h) * 2 + raw_x / 64) * block
                                + (u64::from(z) % d * h + y / 8 % h) * 512
                                + (raw_x & 15)
                                + ((y & 1) << 4)
                                + ((raw_x & 16) << 1)
                                + ((y & 6) << 5)
                                + ((raw_x & 32) << 3);
                            expected.insert(y * 92 + x * 4 + lane - 1, origin + 13 + address);
                        }
                    }
                }
                assert_eq!(actual, expected, "block={bh}/{bd}, slice={z}");
                let mut selected = Vec::new();
                image.observed_pages(&mapping, &mut selected).unwrap();
                let inputs = pin_selected_pages(selected).unwrap();
                let mut captured = vec![0xcd; 92 * 21];
                machine
                    .read_resources(&inputs, |reader| {
                        image.capture_resource(reader, 0, &mut captured)
                    })
                    .unwrap();
                let mut wanted = vec![0xcd; captured.len()];
                for (&native, &physical) in &expected {
                    wanted[native as usize] = data[(physical - origin) as usize];
                }
                assert_eq!(captured, wanted, "capture bh={bh} bd={bd} z={z}");
            }
        }
    }

    #[test]
    fn cached_mappings_do_not_pin_ram_and_repeated_physical_bytes_keep_logical_order() {
        let machine = Machine::new(2 * PAGE_SIZE).unwrap();
        let account = machine.account(2 * PAGE_SIZE).unwrap();
        let allocation = account.allocate(PAGE_SIZE).unwrap();
        let id = allocation.identity();
        let ranges = [
            AllocationRange::new(allocation.pin(), 17, 9).unwrap(),
            AllocationRange::new(allocation.pin(), 19, 3).unwrap(),
        ];
        let mapping = PhysicalMapping::new(&ranges).unwrap();
        assert_eq!(mapping.spans().len(), 2);
        assert!(!mapping.is_injective());
        let first = Footprint::new(
            Arc::clone(&mapping),
            WriteDomain::Buffer { start: 0, bytes: 3 },
        )
        .unwrap();
        let repeated = Footprint::new(
            Arc::clone(&mapping),
            WriteDomain::Buffer { start: 9, bytes: 3 },
        )
        .unwrap();
        assert!(first.intersects(&repeated));
        let retained = mapping.pin(7, 4).unwrap();
        assert_eq!(
            retained
                .iter()
                .map(|r| (r.offset(), r.bytes()))
                .collect::<Vec<_>>(),
            [(24, 2), (19, 2)]
        );
        drop(ranges);
        drop(allocation);
        assert_eq!(machine.statistics().retained_bytes, PAGE_SIZE);
        drop(retained);
        assert_eq!(machine.statistics().allocated_bytes, 0);
        assert!(mapping.pin(0, 1).is_err());
        let replacement = account.allocate(PAGE_SIZE).unwrap();
        assert_ne!(replacement.identity(), id);
        assert!(mapping.pin(0, 1).is_err());
        assert_eq!(mapping.spans()[0].allocation, id);
    }
}

#[cfg(test)]
mod submapping_tests {
    use super::*;
    use nixe_memory::{Machine, PAGE_SIZE};
    #[test]
    fn buffer_expansion_stops_at_retired_holes_and_keeps_fragment_order() {
        use nixe_memory::{AddressSpace, AddressWidth, GuestVirtualAddress, MemoryPermissions};
        let machine = Machine::new(4 * PAGE_SIZE).unwrap();
        let allocation = machine
            .account(4 * PAGE_SIZE)
            .unwrap()
            .allocate(4 * PAGE_SIZE)
            .unwrap();
        let full =
            PhysicalMapping::new(&[
                AllocationRange::new(allocation.pin(), 0, 4 * PAGE_SIZE).unwrap()
            ])
            .unwrap();
        let space = AddressSpace::new(machine.clone(), AddressWidth::Bits32).unwrap();
        let base = GuestVirtualAddress::new(PAGE_SIZE);
        space
            .map(
                base,
                allocation,
                0,
                4 * PAGE_SIZE,
                MemoryPermissions::READ_WRITE,
            )
            .unwrap();
        space
            .unmap(base.checked_add(PAGE_SIZE).unwrap(), PAGE_SIZE)
            .unwrap();
        let left = full
            .slice(16, 32)
            .unwrap()
            .expanded_buffer(1024 * 1024)
            .unwrap();
        assert_eq!(left.bytes(), PAGE_SIZE);
        let right = full
            .slice(2 * PAGE_SIZE + 16, 32)
            .unwrap()
            .expanded_buffer(1024 * 1024)
            .unwrap();
        assert_eq!(right.bytes(), 2 * PAGE_SIZE);
        assert_eq!(
            right.containing_offset(&full, 3 * PAGE_SIZE + 10..3 * PAGE_SIZE + 15),
            Some(PAGE_SIZE + 10)
        );
        let fragmented =
            PhysicalMapping::join(&[left.clone(), right.clone(), left.clone()]).unwrap();
        assert!(Arc::ptr_eq(
            &fragmented,
            &fragmented.expanded_buffer(1024 * 1024).unwrap()
        ));
        let reordered = PhysicalMapping::join(&[right, left]).unwrap();
        assert_eq!(
            fragmented.containing_offset(&reordered, 0..3 * PAGE_SIZE),
            Some(PAGE_SIZE)
        );
        assert_eq!(
            fragmented.containing_offset(&reordered, PAGE_SIZE + 10..2 * PAGE_SIZE + 10),
            Some(2 * PAGE_SIZE + 10)
        );
        assert_eq!(
            fragmented.containing_offset(&full, PAGE_SIZE - 10..2 * PAGE_SIZE + 10),
            None
        );
        assert_eq!(
            fragmented.containing_offset(&reordered, 0..4 * PAGE_SIZE),
            None
        );
        assert_eq!(fragmented.containing_offset(&reordered, 10..10), None);
        assert!(
            full.slice(PAGE_SIZE, 16)
                .unwrap()
                .expanded_buffer(PAGE_SIZE)
                .is_err()
        );
    }

    #[test]
    fn located_span_pins_check_bounds_and_retain_only_the_requested_page() {
        let machine = Machine::new(3 * PAGE_SIZE).unwrap();
        let allocation = machine
            .account(3 * PAGE_SIZE)
            .unwrap()
            .allocate(3 * PAGE_SIZE)
            .unwrap();
        let mapping = PhysicalMapping::new(&[
            AllocationRange::new(allocation.pin(), 0, PAGE_SIZE).unwrap(),
            AllocationRange::new(allocation.pin(), 2 * PAGE_SIZE, PAGE_SIZE).unwrap(),
        ])
        .unwrap();
        let start = mapping.spans()[1].start;
        assert!(mapping.pin_span(2, start, 1).is_err());
        assert!(mapping.pin_span(0, start, 1).is_err());
        assert!(mapping.pin_span(1, start, 0).is_err());
        assert!(mapping.pin_span(1, start, PAGE_SIZE + 1).is_err());
        assert!(mapping.pin_span(1, ArenaOffset::new(u64::MAX), 2).is_err());
        let pin = mapping
            .pin_span(1, ArenaOffset::new(start.get() + 17), 4)
            .unwrap()
            .unwrap();
        assert_eq!(pin.offset(), 2 * PAGE_SIZE + 17);
        assert_eq!(pin.bytes(), 4);
        drop(allocation);
        assert_eq!(machine.statistics().allocated_bytes, PAGE_SIZE);
        assert!(
            mapping
                .pin_span(0, mapping.spans()[0].start, 1)
                .unwrap()
                .is_none()
        );
        drop(pin);
        assert!(mapping.pin_span(1, start, 1).unwrap().is_none());
    }
    #[test]
    fn mapping_construction_and_join_deduplicate_interleaved_allocation_identities() {
        let machine = Machine::new(64 * PAGE_SIZE).unwrap();
        let account = machine.account(64 * PAGE_SIZE).unwrap();
        let allocations: Vec<_> = (0..64)
            .map(|_| account.allocate(PAGE_SIZE).unwrap())
            .collect();
        let ranges: Vec<_> = (0..256)
            .map(|n| AllocationRange::new(allocations[(n * 17) % 64].pin(), 7, 53).unwrap())
            .collect();
        let direct = PhysicalMapping::new(&ranges).unwrap();
        let parts: Vec<_> = ranges
            .iter()
            .map(|r| PhysicalMapping::new(std::slice::from_ref(r)).unwrap())
            .collect();
        let joined = PhysicalMapping::join(&parts).unwrap();
        assert_eq!(*direct, *joined);
        assert_eq!(direct.allocations().len(), 64);
        assert_eq!(joined.allocations().len(), 64);
        assert!(!joined.is_injective());
        for (index, range) in ranges.iter().enumerate() {
            let pinned = joined.pin(index as u64 * 53, 53).unwrap();
            assert_eq!(
                pinned[0].allocation().identity(),
                range.allocation().identity()
            );
            assert_eq!(pinned[0].offset(), 7);
        }
    }
    #[test]
    fn selected_pages_match_written_bytes_across_views_aliases_and_image_holes() {
        use std::collections::BTreeSet;
        let machine = Machine::new(16 * PAGE_SIZE).unwrap();
        let allocation = machine
            .account(16 * PAGE_SIZE)
            .unwrap()
            .allocate(16 * PAGE_SIZE)
            .unwrap();
        let range = |offset, bytes| AllocationRange::new(allocation.pin(), offset, bytes).unwrap();
        let mapping = PhysicalMapping::new(&[
            range(4 * PAGE_SIZE, PAGE_SIZE),
            range(0, PAGE_SIZE),
            range(4 * PAGE_SIZE, PAGE_SIZE),
            range(8 * PAGE_SIZE, PAGE_SIZE),
        ])
        .unwrap();
        let other = PhysicalMapping::new(&[
            range(8 * PAGE_SIZE + 7, 37),
            range(4 * PAGE_SIZE + 19, 2 * PAGE_SIZE),
            range(13, PAGE_SIZE - 13),
            range(4 * PAGE_SIZE, 23),
        ])
        .unwrap();
        let continuous = PhysicalMapping::new(&[range(0, 16 * PAGE_SIZE)]).unwrap();
        let mut domains = vec![
            WriteDomain::Buffer {
                start: 0,
                bytes: mapping.bytes(),
            },
            WriteDomain::Buffer {
                start: 2 * PAGE_SIZE + 23,
                bytes: 17,
            },
            WriteDomain::Buffer {
                start: PAGE_SIZE - 1,
                bytes: PAGE_SIZE + 2,
            },
        ];
        for storage in [
            ImageStorage::Pitch { row_pitch: 1024 },
            ImageStorage::BlockLinear {
                block_height_log2: 1,
                block_depth_log2: 0,
            },
            ImageStorage::BlockLinear {
                block_height_log2: 0,
                block_depth_log2: 3,
            },
            ImageStorage::BlockLinear {
                block_height_log2: 1,
                block_depth_log2: 2,
            },
        ] {
            domains.push(WriteDomain::Image(ImageFootprint {
                depth_slice: 0,
                base: 0,
                columns: 17,
                rows: 13,
                packed_bytes: 4,
                aspect_offset: 3,
                aspect_bytes: 1,
                storage,
                native_base: 0,
                native_element_bytes: 1,
                native_row_pitch: 17,
                rect: ByteRect {
                    x: 3,
                    y: 1,
                    width: 11,
                    height: 10,
                },
            }));
        }
        for domain in domains {
            let written = Footprint::new(mapping.clone(), domain).unwrap();
            let mut written_bytes = BTreeSet::new();
            for (index, span) in mapping.spans().iter().enumerate() {
                written
                    .visit(index, span.start, span.bytes, |part| {
                        written_bytes.extend(part.start.get()..part.start.get() + part.bytes);
                        true
                    })
                    .unwrap();
            }
            for observed in [&mapping, &other, &continuous] {
                let mut expected = BTreeSet::new();
                for span in observed.spans() {
                    for byte in span.start.get()..span.start.get() + span.bytes {
                        if written_bytes.contains(&byte) {
                            expected.insert((
                                span.allocation,
                                (span.allocation_offset() + byte - span.start.get()) / PAGE_SIZE,
                            ));
                        }
                    }
                }
                let mut pages = Vec::new();
                written.observed_pages(observed, &mut pages).unwrap();
                let actual: BTreeSet<_> = pages
                    .iter()
                    .flat_map(|(source, range)| {
                        (range.start / PAGE_SIZE..range.end / PAGE_SIZE)
                            .map(|page| (source.identity(), page))
                    })
                    .collect();
                assert_eq!(actual, expected, "domain={domain:?}, observed={observed:?}");
            }
        }
        let foreign = Machine::new(PAGE_SIZE)
            .unwrap()
            .account(PAGE_SIZE)
            .unwrap()
            .allocate(PAGE_SIZE)
            .unwrap();
        let foreign =
            PhysicalMapping::new(&[AllocationRange::new(foreign.pin(), 0, PAGE_SIZE).unwrap()])
                .unwrap();
        let written = Footprint::new(
            foreign,
            WriteDomain::Buffer {
                start: 0,
                bytes: PAGE_SIZE,
            },
        )
        .unwrap();
        assert!(written.observed_pages(&mapping, &mut Vec::new()).is_err());
    }
    #[test]
    fn a_weak_submapping_preserves_exact_byte_identity_without_retaining_ram() {
        let machine = Machine::new(2 * PAGE_SIZE).unwrap();
        let account = machine.account(2 * PAGE_SIZE).unwrap();
        let first = account.allocate(PAGE_SIZE).unwrap();
        let second = account.allocate(PAGE_SIZE).unwrap();
        let second_id = second.identity();
        let ranges = vec![
            AllocationRange::new(first.pin(), 0, PAGE_SIZE).unwrap(),
            AllocationRange::new(second.pin(), 0, PAGE_SIZE).unwrap(),
        ];
        let parent = PhysicalMapping::new(&ranges).unwrap();
        let view = parent.slice(PAGE_SIZE + 3, PAGE_SIZE - 6).unwrap();
        assert_eq!(view.spans()[0].allocation, second_id);
        assert_eq!(view.spans()[0].logical, 0);
        assert_eq!(view.spans()[0].bytes, PAGE_SIZE - 6);
        let pins = view.pin(0, PAGE_SIZE - 6).unwrap();
        assert_eq!(pins[0].offset(), 3);
        assert_eq!(*PhysicalMapping::new(&pins).unwrap(), *view);
        drop(pins);
        drop(ranges);
        drop(first);
        drop(second);
        drop(parent);
        assert_eq!(machine.statistics().allocated_bytes, 0);
        let metadata_only = view.slice(1, 2).unwrap();
        assert!(metadata_only.pin(0, 2).is_err());
        assert!(view.slice(0, 0).is_err());
        assert!(view.slice(u64::MAX, 2).is_err());
    }
    #[test]
    fn joining_views_preserves_alias_identity_hashing_and_exact_input_retention() {
        let machine = Machine::new(3 * PAGE_SIZE).unwrap();
        let allocation = machine
            .account(3 * PAGE_SIZE)
            .unwrap()
            .allocate(3 * PAGE_SIZE)
            .unwrap();
        let original =
            PhysicalMapping::new(&[
                AllocationRange::new(allocation.pin(), 0, 3 * PAGE_SIZE).unwrap()
            ])
            .unwrap();
        let joined = PhysicalMapping::join(&[
            original.slice(13, PAGE_SIZE - 13).unwrap(),
            original.slice(PAGE_SIZE, PAGE_SIZE + 9).unwrap(),
        ])
        .unwrap();
        let direct = original.slice(13, 2 * PAGE_SIZE - 4).unwrap();
        assert_eq!(*joined, *direct);
        let hash = |mapping: &PhysicalMapping| {
            let mut hash = rustc_hash::FxHasher::default();
            mapping.hash(&mut hash);
            hash.finish()
        };
        assert_eq!(hash(&joined), hash(&direct));
        let reversed = PhysicalMapping::join(&[
            original.slice(PAGE_SIZE, PAGE_SIZE).unwrap(),
            original.slice(0, PAGE_SIZE).unwrap(),
        ])
        .unwrap();
        assert_ne!(*joined, *reversed);
        let repeated = PhysicalMapping::join(&[direct.clone(), direct.clone()]).unwrap();
        assert!(!repeated.is_injective());
        let accepted = reversed.pin(PAGE_SIZE + 7, 1).unwrap();
        assert_eq!(accepted[0].offset(), 7);
        drop(allocation);
        assert_eq!(machine.statistics().allocated_bytes, PAGE_SIZE);
        drop(accepted);
        assert_eq!(machine.statistics().allocated_bytes, 0);
        assert!(joined.pin(0, 1).is_err());
        assert!(PhysicalMapping::join(&[]).is_err());
        let other_machine = Machine::new(PAGE_SIZE).unwrap();
        let other = other_machine
            .account(PAGE_SIZE)
            .unwrap()
            .allocate(PAGE_SIZE)
            .unwrap();
        let other_mapping =
            PhysicalMapping::new(&[AllocationRange::new(other.pin(), 0, PAGE_SIZE).unwrap()])
                .unwrap();
        assert!(PhysicalMapping::join(&[joined, other_mapping]).is_err());
    }
    #[test]
    fn slicing_an_aliased_mapping_recomputes_which_bytes_still_overlap() {
        let machine = Machine::new(PAGE_SIZE).unwrap();
        let allocation = machine
            .account(PAGE_SIZE)
            .unwrap()
            .allocate(PAGE_SIZE)
            .unwrap();
        let range = AllocationRange::new(allocation.pin(), 0, PAGE_SIZE).unwrap();
        let parent = PhysicalMapping::new(&[range.clone(), range]).unwrap();
        assert!(!parent.is_injective());
        assert!(parent.slice(PAGE_SIZE, PAGE_SIZE).unwrap().is_injective());
        assert!(
            parent
                .slice(PAGE_SIZE / 2, PAGE_SIZE)
                .unwrap()
                .is_injective()
        );
        assert!(!parent.slice(0, PAGE_SIZE + 1).unwrap().is_injective());
    }
}
