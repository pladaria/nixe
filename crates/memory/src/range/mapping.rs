//! Compact byte permutations. Retaining a view never expands its elements.
use super::*;

/// A bijective byte permutation within a rectangular tile. Tile storage is
/// dense; the table and its inverse are shared across all views of this layout.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CanonicalByteTile {
    width: u32,
    height: u32,
    offsets: Arc<[u32]>,
    inverse: Arc<[[u32; 2]]>,
    preserved_low_bits: u32,
    contiguous: Arc<[u32]>,
    inverse_contiguous: Arc<[u32]>,
}
impl CanonicalByteTile {
    pub fn new(width: u32, height: u32, offsets: Vec<u32>) -> Result<Self, CanonicalRangeError> {
        let count = width
            .checked_mul(height)
            .ok_or(CanonicalRangeError::RangeOverflow)? as usize;
        if count == 0 || offsets.len() != count {
            return Err(CanonicalRangeError::InvalidSubrange);
        }
        let mut inverse = vec![u32::MAX; count];
        for (logical, &physical) in offsets.iter().enumerate() {
            let entry = inverse
                .get_mut(physical as usize)
                .ok_or(CanonicalRangeError::InvalidSubrange)?;
            if *entry != u32::MAX {
                return Err(CanonicalRangeError::InvalidSubrange);
            }
            *entry = logical as u32;
        }
        let preserved_low_bits = offsets
            .iter()
            .enumerate()
            .map(|(i, &v)| ((i as u32) ^ v).trailing_zeros())
            .min()
            .unwrap_or(0);
        let mut contiguous = vec![1; count];
        for i in (0..count.saturating_sub(1)).rev() {
            if (i + 1) % width as usize != 0 && offsets[i] + 1 == offsets[i + 1] {
                contiguous[i] = contiguous[i + 1] + 1;
            }
        }
        let mut inverse_contiguous = vec![1; count];
        for i in (0..count.saturating_sub(1)).rev() {
            if inverse[i] + 1 == inverse[i + 1] && inverse[i] / width == inverse[i + 1] / width {
                inverse_contiguous[i] = inverse_contiguous[i + 1] + 1;
            }
        }
        Ok(Self {
            width,
            height,
            offsets: offsets.into(),
            inverse: inverse
                .into_iter()
                .map(|at| [at % width, at / width])
                .collect(),
            preserved_low_bits,
            contiguous: contiguous.into(),
            inverse_contiguous: inverse_contiguous.into(),
        })
    }
}
impl Hash for CanonicalByteTile {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.width.hash(state);
        self.height.hash(state);
        self.offsets.hash(state);
    }
}
/// An element-strided rectangular view, optionally tiled. Bytes within an
/// element wrap at source_element_bytes (for example an ignored transfer byte).
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct CanonicalByteLayout {
    pub base: u64,
    pub columns: u32,
    pub rows: u32,
    pub element_bytes: u32,
    pub source_element_bytes: u32,
    pub source_byte_offset: u32,
    pub source_row_pitch: u64,
    pub tile: Option<CanonicalByteTile>,
}
impl CanonicalByteLayout {
    fn address(&self, logical: u64) -> u64 {
        let row_bytes = u64::from(self.columns) * u64::from(self.element_bytes);
        let y = logical / row_bytes;
        let within = logical % row_bytes;
        let x = within / u64::from(self.element_bytes) * u64::from(self.source_element_bytes)
            + (within % u64::from(self.element_bytes) + u64::from(self.source_byte_offset))
                % u64::from(self.source_element_bytes);
        self.base
            + if let Some(tile) = &self.tile {
                let width = u64::from(tile.width);
                let height = u64::from(tile.height);
                let tiles = (u64::from(self.columns) * u64::from(self.source_element_bytes))
                    .div_ceil(width);
                (y / height * tiles + x / width) * width * height
                    + u64::from(tile.offsets[((y % height) * width + x % width) as usize])
            } else {
                y * self.source_row_pitch + x
            }
    }
    fn inverse_runs(&self, physical: u64, size: u64, mut visit: impl FnMut(u64, u64, u64)) {
        let stride = u64::from(self.source_element_bytes);
        let bytes = u64::from(self.element_bytes);
        let width_bytes = u64::from(self.columns) * stride;
        let mut project = |x: u64, y: u64, run: u64, consumed: u64| {
            if y < u64::from(self.rows) && x < width_bytes {
                let valid = run.min(width_bytes - x);
                if bytes == stride && self.source_byte_offset == 0 {
                    visit(y * u64::from(self.columns) * bytes + x, consumed, valid);
                } else {
                    let offset = u64::from(self.source_byte_offset);
                    let mut at = 0;
                    while at < valid {
                        let raw = (x + at) % stride;
                        let native = (raw + stride - offset) % stride;
                        let length = (valid - at).min(if raw < offset {
                            offset - raw
                        } else {
                            stride - raw
                        });
                        let mut byte = native;
                        while byte < bytes {
                            visit(
                                (y * u64::from(self.columns) + (x + at) / stride) * bytes + byte,
                                consumed + at,
                                length.min(bytes - byte),
                            );
                            byte += stride;
                        }
                        at += length;
                    }
                }
            }
        };
        // Locate tiles once per physical tile, rather than divide for every
        // 16-byte permutation run. Shared inverse entries already contain x/y.
        // Skip an excluded prefix in one step, independently of its byte length.
        let mut consumed = self.base.saturating_sub(physical).min(size);
        if let Some(tile) = &self.tile {
            let width = u64::from(tile.width);
            let height = u64::from(tile.height);
            let tile_bytes = width * height;
            let tiles = width_bytes.div_ceil(width);
            while consumed < size {
                let relative = physical + consumed - self.base;
                let tile_index = relative / tile_bytes;
                let base_x = tile_index % tiles * width;
                let base_y = tile_index / tiles * height;
                let mut cell = (relative % tile_bytes) as usize;
                let end = (size - consumed).min(tile_bytes - cell as u64) as usize + cell;
                while cell < end {
                    let [x, y] = tile.inverse[cell];
                    let run = (tile.inverse_contiguous[cell] as usize).min(end - cell);
                    project(
                        base_x + u64::from(x),
                        base_y + u64::from(y),
                        run as u64,
                        consumed,
                    );
                    consumed += run as u64;
                    cell += run;
                }
            }
        } else {
            while consumed < size {
                let relative = physical + consumed - self.base;
                let x = relative % self.source_row_pitch;
                let y = relative / self.source_row_pitch;
                let run = (self.source_row_pitch - x).min(size - consumed);
                project(x, y, run, consumed);
                consumed += run;
            }
        }
    }
    fn run(&self, at: u64, remaining: u64) -> (u64, u64) {
        let row_bytes = u64::from(self.columns) * u64::from(self.element_bytes);
        let y = at / row_bytes;
        let within = at % row_bytes;
        let byte = within % u64::from(self.element_bytes);
        let source_byte =
            (byte + u64::from(self.source_byte_offset)) % u64::from(self.source_element_bytes);
        let x = within / u64::from(self.element_bytes) * u64::from(self.source_element_bytes)
            + source_byte;
        let mut size = remaining.min(row_bytes - within);
        if self.element_bytes != self.source_element_bytes || self.source_byte_offset != 0 {
            size = size
                .min(u64::from(self.element_bytes) - byte)
                .min(u64::from(self.source_element_bytes) - source_byte);
        }
        if let Some(tile) = &self.tile {
            let cell =
                (y % u64::from(tile.height)) * u64::from(tile.width) + x % u64::from(tile.width);
            size = size.min(u64::from(tile.contiguous[cell as usize]));
        }
        (self.address(at), size)
    }
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum ByteMapping {
    Rect {
        source: CanonicalBackingRange,
        layout: CanonicalByteLayout,
    },
    Concat(Box<[CanonicalBackingRange]>),
    Slice {
        source: CanonicalBackingRange,
        offset: u64,
        size: u64,
    },
}
impl ByteMapping {
    pub(super) fn snapshot(
        &self,
        offset: u64,
        size: u64,
        out: &mut Vec<CanonicalBackingSegment>,
    ) -> Result<(), CanonicalRangeError> {
        match self {
            Self::Slice {
                source,
                offset: base,
                ..
            } => source.snapshot_subrange_into(base + offset, size, out)?,
            Self::Rect { source, layout } => {
                let mut at = offset;
                while at < offset + size {
                    let (physical, size) = layout.run(at, offset + size - at);
                    source.snapshot_subrange_into(physical, size, out)?;
                    at += size;
                }
            }
            Self::Concat(ranges) => {
                let mut base = 0;
                for range in ranges {
                    let start = offset.max(base);
                    let end = (offset + size).min(base + range.size());
                    if start < end {
                        range.snapshot_subrange_into(start - base, end - start, out)?;
                    }
                    base += range.size();
                }
            }
        }
        Ok(())
    }
}
impl CanonicalBackingRange {
    /// Retain a rectangular byte permutation without expanding its pixels.
    pub fn mapped(
        source: Self,
        mut layout: CanonicalByteLayout,
    ) -> Result<Self, CanonicalRangeError> {
        if layout.columns == 0
            || layout.rows == 0
            || layout.element_bytes == 0
            || layout.source_element_bytes == 0
            || layout.source_byte_offset >= layout.source_element_bytes
            || layout.source_row_pitch
                < u64::from(layout.columns) * u64::from(layout.source_element_bytes)
        {
            return Err(CanonicalRangeError::InvalidSubrange);
        }
        let size = u64::from(layout.columns)
            .checked_mul(u64::from(layout.rows))
            .and_then(|v| v.checked_mul(u64::from(layout.element_bytes)))
            .ok_or(CanonicalRangeError::RangeOverflow)?;
        let physical_size = if let Some(tile) = &layout.tile {
            let tile_bytes = u64::from(tile.width) * u64::from(tile.height);
            (u64::from(layout.columns) * u64::from(layout.source_element_bytes))
                .div_ceil(u64::from(tile.width))
                .checked_mul(u64::from(layout.rows).div_ceil(u64::from(tile.height)))
                .and_then(|v| v.checked_mul(tile_bytes))
        } else {
            u64::from(layout.rows - 1)
                .checked_mul(layout.source_row_pitch)
                .and_then(|v| {
                    v.checked_add(
                        u64::from(layout.columns) * u64::from(layout.source_element_bytes),
                    )
                })
        }
        .ok_or(CanonicalRangeError::RangeOverflow)?;
        if layout
            .base
            .checked_add(physical_size)
            .is_none_or(|end| end > source.size())
        {
            return Err(CanonicalRangeError::InvalidSubrange);
        }
        let source = source.logical_view(layout.base, physical_size)?;
        layout.base = 0;
        if layout.tile.is_none()
            && layout.element_bytes == layout.source_element_bytes
            && layout.source_byte_offset == 0
            && layout.source_row_pitch
                == u64::from(layout.columns) * u64::from(layout.element_bytes)
        {
            return Ok(source);
        }
        Self::with_mapping(size, ByteMapping::Rect { source, layout })
    }
    /// Concatenate retained views without cloning their expanded topology.
    pub fn concatenate(ranges: Vec<Self>) -> Result<Self, CanonicalRangeError> {
        if ranges.len() == 1 {
            return Ok(ranges.into_iter().next().unwrap());
        }
        let size = ranges.iter().try_fold(0u64, |size, r| {
            size.checked_add(r.size())
                .ok_or(CanonicalRangeError::RangeOverflow)
        })?;
        Self::with_mapping(size, ByteMapping::Concat(ranges.into()))
    }
    fn with_mapping(size: u64, mapping: ByteMapping) -> Result<Self, CanonicalRangeError> {
        if size == 0 {
            return Err(CanonicalRangeError::Empty);
        }
        let sources: Vec<_> = match &mapping {
            ByteMapping::Rect { source, .. } | ByteMapping::Slice { source, .. } => vec![source],
            ByteMapping::Concat(ranges) => ranges.iter().collect(),
        };
        let mut pages = BTreeMap::new();
        let mut gates = BTreeMap::new();
        for source in sources {
            for page in source.pages() {
                pages.entry(page.identity()).or_insert_with(|| page.clone());
            }
            for gate in source.execution_gates() {
                gates.entry(gate.identity()).or_insert_with(|| gate.clone());
            }
        }
        Ok(Self {
            size,
            layout: Arc::new(CanonicalRangeLayout {
                topology: std::sync::OnceLock::new(),
                mapping: Some(mapping),
                segment_runs: std::sync::OnceLock::new(),
                byte_mapping_hash: std::sync::OnceLock::new(),
                page_segments: std::sync::OnceLock::new(),
                pages: pages.into_values().collect(),
                gates: gates.into_values().collect(),
                owner: Mutex::new(None),
                visibility_epoch: std::sync::OnceLock::new(),
                clean_read_epoch: AtomicU64::new(u64::MAX),
                cpu_summary: Mutex::new(std::sync::Weak::new()),
            }),
        })
    }
    pub fn logical_view(&self, offset: u64, size: u64) -> Result<Self, CanonicalRangeError> {
        if size == 0 || offset.checked_add(size).is_none_or(|end| end > self.size()) {
            return Err(CanonicalRangeError::InvalidSubrange);
        }
        if offset == 0 && size == self.size() {
            return Ok(self.clone());
        }
        if self.layout.mapping.is_none() {
            return self.snapshot_subrange(offset, size);
        }
        if let Some(ByteMapping::Concat(ranges)) = &self.layout.mapping {
            let mut base = 0;
            for range in ranges {
                if offset >= base && offset + size <= base + range.size() {
                    return range.logical_view(offset - base, size);
                }
                base += range.size();
            }
        }
        Self::with_mapping(
            size,
            ByteMapping::Slice {
                source: self.clone(),
                offset,
                size,
            },
        )
    }
    /// Whether repeated physical bytes are possible (conservative for subviews).
    pub fn may_repeat_bytes(&self) -> bool {
        match &self.layout.mapping {
            Some(ByteMapping::Rect { source, layout }) => {
                layout.element_bytes > layout.source_element_bytes || source.may_repeat_bytes()
            }
            Some(ByteMapping::Slice { source, .. }) => source.may_repeat_bytes(),
            Some(ByteMapping::Concat(ranges)) => {
                let mut pages = BTreeSet::new();
                ranges.iter().any(|range| {
                    range.may_repeat_bytes()
                        || range.pages().any(|page| !pages.insert(page.identity()))
                })
            }
            None => {
                let index = self
                    .layout
                    .page_segments
                    .get_or_init(|| self.page_segment_index());
                index.values().any(|spans| {
                    if spans.len() < 2 {
                        return false;
                    }
                    let mut spans = spans
                        .iter()
                        .map(|&(_, start, size)| (start, start + size))
                        .collect::<Vec<_>>();
                    spans.sort_unstable();
                    spans.windows(2).any(|s| s[0].1 > s[1].0)
                })
            }
        }
    }
    fn page_segment_index(&self) -> CanonicalPageSegments {
        let mut index = BTreeMap::<_, Vec<_>>::new();
        let mut logical = 0;
        for segment in self.segments() {
            index.entry(segment.page()).or_default().push((
                logical,
                segment.offset(),
                segment.size(),
            ));
            logical += segment.size();
        }
        index
    }
    /// Exact mappings for one physical page. Only demanded pages are expanded;
    /// inverse tiling locates bytes directly without traversing the whole image.
    pub fn visit_page_mappings(&self, page: CanonicalPageId, visit: &mut dyn FnMut(u64, u64, u64)) {
        match &self.layout.mapping {
            Some(ByteMapping::Slice {
                source,
                offset,
                size,
            }) => source.visit_page_mappings(page, &mut |logical, physical, length| {
                let begin = logical.max(*offset);
                let end = (logical + length).min(offset + size);
                if begin < end {
                    visit(begin - offset, physical + begin - logical, end - begin);
                }
            }),
            Some(ByteMapping::Rect { source, layout }) => {
                source.visit_page_mappings(page, &mut |logical, physical, size| {
                    layout.inverse_runs(logical, size, |mapped, consumed, length| {
                        visit(mapped, physical + consumed, length)
                    });
                });
            }
            Some(ByteMapping::Concat(ranges)) => {
                let mut base = 0;
                for range in ranges {
                    range.visit_page_mappings(page, &mut |logical, physical, size| {
                        visit(base + logical, physical, size)
                    });
                    base += range.size();
                }
            }
            None => {
                let index = self
                    .layout
                    .page_segments
                    .get_or_init(|| self.page_segment_index());
                if let Some(spans) = index.get(&page) {
                    for &(logical, physical, size) in spans {
                        visit(logical, physical, size);
                    }
                }
            }
        }
    }
    /// A conservative physical byte-lane mask, used to prove packed aspects
    /// disjoint without constructing exact interval maps. Unknown means all lanes.
    pub fn page_byte_lanes(&self, page: CanonicalPageId, unit: u32, meaningful: u32) -> u64 {
        match &self.layout.mapping {
            Some(ByteMapping::Slice { source, offset, .. }) => {
                if offset % u64::from(unit) == 0 {
                    source.page_byte_lanes(page, unit, meaningful)
                } else {
                    u64::MAX
                }
            }
            Some(ByteMapping::Rect { source, layout }) => {
                let stride = layout.source_element_bytes;
                if !stride.is_power_of_two()
                    || stride > 16
                    || unit != layout.element_bytes
                    || layout.source_row_pitch % u64::from(stride) != 0
                    || layout
                        .tile
                        .as_ref()
                        .is_some_and(|tile| tile.preserved_low_bits < stride.trailing_zeros())
                {
                    return u64::MAX;
                }
                let mut mask = 0;
                source.visit_page_mappings(page, &mut |logical, physical, _| {
                    let phase = physical.wrapping_sub(logical).wrapping_add(layout.base)
                        % u64::from(stride);
                    for byte in 0..meaningful {
                        let lane = (phase + u64::from((layout.source_byte_offset + byte) % stride))
                            % u64::from(stride);
                        mask |= (u64::MAX / ((1u64 << stride) - 1)) << lane;
                    }
                });
                mask
            }
            Some(ByteMapping::Concat(_)) => u64::MAX,
            None => {
                if unit == meaningful || !unit.is_power_of_two() {
                    return u64::MAX;
                }
                let mut mask = 0;
                self.visit_page_mappings(page, &mut |logical, physical, size| {
                    for byte in 0..size.min(64) {
                        if (logical + byte) % u64::from(unit) < u64::from(meaningful) {
                            mask |= 1 << ((physical + byte) % 64);
                        }
                    }
                });
                mask
            }
        }
    }
    pub fn byte_mapping_key(&self) -> CanonicalByteMappingKey {
        CanonicalByteMappingKey(self.clone())
    }
    pub(super) fn byte_mapping_hash(&self) -> u64 {
        *self.layout.byte_mapping_hash.get_or_init(|| {
            let mut hash = std::collections::hash_map::DefaultHasher::new();
            self.size().hash(&mut hash);
            match &self.layout.mapping {
                None => {
                    0u8.hash(&mut hash);
                    for segment in self.segments() {
                        segment.page().hash(&mut hash);
                        segment.offset().hash(&mut hash);
                        segment.size().hash(&mut hash);
                    }
                }
                Some(ByteMapping::Rect { source, layout }) => {
                    1u8.hash(&mut hash);
                    source.byte_mapping_hash().hash(&mut hash);
                    layout.hash(&mut hash);
                }
                Some(ByteMapping::Concat(ranges)) => {
                    2u8.hash(&mut hash);
                    for range in ranges {
                        range.byte_mapping_hash().hash(&mut hash);
                    }
                }
                Some(ByteMapping::Slice {
                    source,
                    offset,
                    size,
                }) => {
                    3u8.hash(&mut hash);
                    source.byte_mapping_hash().hash(&mut hash);
                    offset.hash(&mut hash);
                    size.hash(&mut hash);
                }
            }
            hash.finish()
        })
    }
    /// Sufficient proof of equal ordered physical bytes without expanding
    /// permutations. Capture permissions/generations are not physical identity.
    /// Differently expressed permutations may conservatively fail this proof.
    pub fn same_byte_mapping(&self, other: &Self) -> bool {
        if self.size() != other.size() {
            return false;
        }
        if self.shares_layout(other) {
            return true;
        }
        match (&self.layout.mapping, &other.layout.mapping) {
            (
                Some(ByteMapping::Rect {
                    source: a,
                    layout: x,
                }),
                Some(ByteMapping::Rect {
                    source: b,
                    layout: y,
                }),
            ) => x == y && a.same_byte_mapping(b),
            (Some(ByteMapping::Concat(a)), Some(ByteMapping::Concat(b))) => {
                a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.same_byte_mapping(b))
            }
            (
                Some(ByteMapping::Slice {
                    source: a,
                    offset: x,
                    ..
                }),
                Some(ByteMapping::Slice {
                    source: b,
                    offset: y,
                    ..
                }),
            ) => x == y && a.same_byte_mapping(b),
            (None, None) => {
                self.segments().len() == other.segments().len()
                    && self.segments().iter().zip(other.segments()).all(|(a, b)| {
                        a.page() == b.page() && a.offset() == b.offset() && a.size() == b.size()
                    })
            }
            _ => false,
        }
    }
    /// Retained topology size, without forcing expansion of mapped byte views.
    pub fn metadata_bytes(&self) -> usize {
        std::mem::size_of::<CanonicalRangeLayout>()
            + self.layout.pages.len() * std::mem::size_of::<CanonicalBackingPage>()
            + self.layout.topology.get().map_or(0, |t| {
                t.segments.len() * (std::mem::size_of::<CanonicalBackingSegment>() + 8)
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CanonicalAllocation, MemoryPermissions};
    #[test]
    fn physical_page_overlap_is_conservative_and_independent_of_logical_order() {
        let allocation = CanonicalAllocation::zeroed(256, 64).unwrap();
        let whole = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let first = whole.logical_view(0, 16).unwrap();
        let same_page = whole.logical_view(32, 16).unwrap();
        let distant = whole.logical_view(128, 16).unwrap();
        assert!(first.may_share_pages(&same_page));
        assert!(!first.may_share_pages(&distant));
        let reordered = CanonicalBackingRange::new(vec![
            whole.segments()[3].clone(),
            whole.segments()[0].clone(),
        ])
        .unwrap();
        assert!(reordered.may_share_pages(&first));
        assert!(!reordered.may_share_pages(&distant));
        assert!(whole.may_share_pages(&reordered));
        assert!(reordered.may_share_pages(&whole));
    }
    #[test]
    fn inverse_tile_runs_match_forward_bytes_across_prefixes_and_partial_tiles() {
        let tile = CanonicalByteTile::new(3, 5, (0..15).map(|i| i * 7 % 15).collect()).unwrap();
        for tile in [None, Some(tile)] {
            for (element_bytes, source_byte_offset) in [(4, 0), (4, 1), (1, 3), (8, 1)] {
                let layout = CanonicalByteLayout {
                    base: 17,
                    columns: 7,
                    rows: 9,
                    element_bytes,
                    source_element_bytes: 4,
                    source_byte_offset,
                    source_row_pitch: 32,
                    tile: tile.clone(),
                };
                let length = u64::from(layout.columns * layout.rows * element_bytes);
                for start in (0..350u64).step_by(13) {
                    let end = start + 37;
                    let mut expected: Vec<_> = (0..length)
                        .filter_map(|logical| {
                            let physical = layout.address(logical);
                            (physical >= start && physical < end).then_some((logical, physical))
                        })
                        .collect();
                    let mut actual = Vec::new();
                    layout.inverse_runs(start, end - start, |logical, physical, size| {
                        actual.extend(
                            (0..size).map(|byte| (logical + byte, start + physical + byte)),
                        );
                    });
                    expected.sort_unstable();
                    actual.sort_unstable();
                    assert_eq!(actual, expected, "layout={layout:?} range={start}..{end}");
                }
            }
        }
    }
    #[test]
    fn mapped_ranges_reject_overflow_and_invalid_tile_permutations() {
        let allocation = CanonicalAllocation::zeroed(256, 64).unwrap();
        let source = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let layout = CanonicalByteLayout {
            base: 0,
            columns: 2,
            rows: 2,
            element_bytes: 4,
            source_element_bytes: 4,
            source_byte_offset: 1,
            source_row_pitch: 8,
            tile: None,
        };
        assert!(CanonicalByteTile::new(2, 2, vec![0, 1, 1, 3]).is_err());
        assert!(CanonicalByteTile::new(2, 2, vec![0, 1, 2, 4]).is_err());
        assert!(
            CanonicalBackingRange::mapped(
                source.clone(),
                CanonicalByteLayout {
                    source_row_pitch: u64::MAX,
                    rows: 3,
                    ..layout.clone()
                }
            )
            .is_err()
        );
        assert!(
            CanonicalBackingRange::mapped(
                source.clone(),
                CanonicalByteLayout {
                    columns: u32::MAX,
                    rows: u32::MAX,
                    element_bytes: 16,
                    source_element_bytes: 16,
                    source_row_pitch: u64::MAX,
                    ..layout.clone()
                }
            )
            .is_err()
        );
        let view = CanonicalBackingRange::mapped(source, layout).unwrap();
        assert_eq!(view.logical_view(4, 4).unwrap().size(), 4);
        assert!(view.logical_view(15, 2).is_err());
        assert!(view.layout.topology.get().is_none());
    }
    #[test]
    fn physical_mapping_proof_ignores_capture_permissions_without_expanding() {
        let allocation = CanonicalAllocation::zeroed(256, 64).unwrap();
        let source = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let mut segments = source.segments().to_vec();
        for segment in &mut segments {
            segment.permissions = MemoryPermissions::READ;
        }
        let read_only = CanonicalBackingRange::new(segments).unwrap();
        let layout = CanonicalByteLayout {
            base: 0,
            columns: 4,
            rows: 4,
            element_bytes: 1,
            source_element_bytes: 4,
            source_byte_offset: 2,
            source_row_pitch: 16,
            tile: None,
        };
        let a = CanonicalBackingRange::mapped(source, layout.clone()).unwrap();
        let b = CanonicalBackingRange::mapped(read_only, layout).unwrap();
        assert!(a.same_byte_mapping(&b));
        let key = a.byte_mapping_key();
        assert_eq!(key, b.byte_mapping_key());
        let hash = |key: &CanonicalByteMappingKey| {
            let mut hash = std::collections::hash_map::DefaultHasher::new();
            key.hash(&mut hash);
            hash.finish()
        };
        assert_eq!(hash(&key), hash(&b.byte_mapping_key()));
        allocation.write(0, &[0x37; 64]).unwrap();
        assert_eq!(
            key,
            a.byte_mapping_key(),
            "content changes do not change byte identity"
        );
        assert!(a.layout.topology.get().is_none());
        assert!(b.layout.topology.get().is_none());
        assert_ne!(a, b);
    }
    #[test]
    fn mapped_inverse_retains_repeated_aliases_and_sliced_concatenations() {
        let allocation = CanonicalAllocation::zeroed(128, 64).unwrap();
        let source = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let raw = CanonicalBackingRange::new(vec![
            source.segments()[1].clone(),
            source.segments()[0].clone(),
            source.segments()[1].clone(),
        ])
        .unwrap();
        let view = CanonicalBackingRange::mapped(
            raw,
            CanonicalByteLayout {
                base: 0,
                columns: 4,
                rows: 12,
                element_bytes: 1,
                source_element_bytes: 4,
                source_byte_offset: 2,
                source_row_pitch: 16,
                tile: None,
            },
        )
        .unwrap();
        assert!(view.may_repeat_bytes());
        let concat = CanonicalBackingRange::concatenate(vec![view.clone(), view.clone()]).unwrap();
        let slice = concat.logical_view(8, 64).unwrap();
        for page in slice.pages() {
            slice.visit_page_mappings(page.identity(), &mut |logical, physical, size| {
                for b in 0..size {
                    let bytes = slice.snapshot_subrange(logical + b, 1).unwrap();
                    assert_eq!(bytes.segments()[0].page(), page.identity());
                    assert_eq!(bytes.segments()[0].offset(), physical + b);
                }
            });
        }
        assert!(slice.layout.topology.get().is_none());
        assert!(view.layout.topology.get().is_none());
    }
}
