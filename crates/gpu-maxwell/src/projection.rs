//! Command-processor reads observe earlier transfers without staging whole pages.
use crate::{MaxwellMemoryCopyOperation, MaxwellResolvedRange};
use nixe_memory::{
    CanonicalBackingRange, CanonicalBackingSegment, CanonicalWriteBatch, CanonicalWriteBatchError,
};

#[derive(Default)]
pub(crate) struct MemoryProjection {
    entries: Vec<Entry>,
    pages: std::collections::HashMap<nixe_memory::CanonicalPageId, Vec<usize>>,
}
struct Entry {
    target: Vec<CanonicalBackingSegment>,
    value: Value,
}
enum Value {
    Inline {
        bytes: [u8; 8],
        size: usize,
    },
    Bytes(Box<[u8]>),
    Copy {
        source: CanonicalBackingRange,
        operation: Box<MaxwellMemoryCopyOperation>,
    },
}
pub(crate) fn canonical(
    range: &MaxwellResolvedRange,
) -> Result<CanonicalBackingRange, CanonicalWriteBatchError> {
    CanonicalBackingRange::new(canonical_segments(range)?)
        .map_err(|_| CanonicalWriteBatchError::IncompleteRange)
}
pub(crate) fn canonical_segments(
    range: &MaxwellResolvedRange,
) -> Result<Vec<CanonicalBackingSegment>, CanonicalWriteBatchError> {
    let mut segments = Vec::new();
    for segment in range.segments() {
        segment
            .mapping()
            .backing()
            .snapshot_subrange_into(segment.backing_offset(), segment.size(), &mut segments)
            .map_err(|_| CanonicalWriteBatchError::IncompleteRange)?;
    }
    Ok(segments)
}
// (read-relative offset, write-relative offset, length), including physical aliases.
fn intersections(
    read: &CanonicalBackingRange,
    offset: u64,
    size: u64,
    write: &[CanonicalBackingSegment],
) -> Vec<(usize, u64, usize)> {
    let mut result = Vec::new();
    let mut read_base = 0;
    for a in read.segments() {
        let start = offset.max(read_base);
        let end = (offset + size).min(read_base + a.size());
        if start < end {
            let physical_start = a.offset() + start - read_base;
            let physical_end = physical_start + end - start;
            let mut write_base = 0;
            for b in write {
                if a.page() == b.page() {
                    let from = physical_start.max(b.offset());
                    let to = physical_end.min(b.offset() + b.size());
                    if from < to {
                        result.push((
                            (start - offset + from - physical_start) as usize,
                            write_base + from - b.offset(),
                            (to - from) as usize,
                        ));
                    }
                }
                write_base += b.size();
            }
        }
        read_base += a.size();
        if read_base >= offset + size {
            break;
        }
    }
    result
}
impl MemoryProjection {
    fn insert(&mut self, entry: Entry) {
        let index = self.entries.len();
        let mut previous = None;
        for segment in &entry.target {
            let page = segment.page();
            if previous != Some(page) {
                let entries = self.pages.entry(page).or_default();
                if entries.last() != Some(&index) {
                    entries.push(index);
                }
                previous = Some(page);
            }
        }
        self.entries.push(entry);
    }
    fn candidates(
        &self,
        range: &CanonicalBackingRange,
        offset: u64,
        size: u64,
        limit: usize,
    ) -> Vec<usize> {
        let mut result = Vec::new();
        let mut base = 0;
        for segment in range.segments() {
            if offset < base + segment.size()
                && base < offset + size
                && let Some(entries) = self.pages.get(&segment.page())
            {
                result.extend(entries.iter().copied().take_while(|index| *index < limit));
            }
            base += segment.size();
            if base >= offset + size {
                break;
            }
        }
        result.sort_unstable();
        result.dedup();
        result
    }
    pub(crate) fn write(
        &mut self,
        target: &MaxwellResolvedRange,
        bytes: &[u8],
    ) -> Result<(), CanonicalWriteBatchError> {
        if target.size() != bytes.len() as u64 {
            return Err(CanonicalWriteBatchError::IncompleteRange);
        }
        let value = if bytes.len() <= 8 {
            let mut inline = [0; 8];
            inline[..bytes.len()].copy_from_slice(bytes);
            Value::Inline {
                bytes: inline,
                size: bytes.len(),
            }
        } else {
            Value::Bytes(bytes.into())
        };
        self.insert(Entry {
            target: canonical_segments(target)?,
            value,
        });
        Ok(())
    }
    pub(crate) fn copy(
        &mut self,
        operation: MaxwellMemoryCopyOperation,
        source: &MaxwellResolvedRange,
        destination: &MaxwellResolvedRange,
    ) -> Result<(), CanonicalWriteBatchError> {
        self.insert(Entry {
            target: canonical_segments(destination)?,
            value: Value::Copy {
                source: canonical(source)?,
                operation: Box::new(operation),
            },
        });
        Ok(())
    }
    // Each copy decreases the entry index. Iteration avoids recursive stack
    // growth when a command stream chains thousands of overlapping transfers.
    fn read_byte_before<'a>(
        &'a self,
        mut range: &'a CanonicalBackingRange,
        mut offset: u64,
        mut limit: usize,
    ) -> Result<u8, CanonicalWriteBatchError> {
        'resolve: loop {
            for index in self.candidates(range, offset, 1, limit).into_iter().rev() {
                let entry = &self.entries[index];
                for (_, from, _) in intersections(range, offset, 1, &entry.target)
                    .into_iter()
                    .rev()
                {
                    match &entry.value {
                        Value::Inline { bytes, .. } => return Ok(bytes[from as usize]),
                        Value::Bytes(bytes) => return Ok(bytes[from as usize]),
                        Value::Copy { source, operation } => match operation
                            .project_byte(from)
                            .map_err(|_| CanonicalWriteBatchError::RangeOverflow)?
                        {
                            Some(crate::engines::ProjectedCopyByte::Source(position)) => {
                                range = source;
                                offset = position;
                                limit = index;
                                continue 'resolve;
                            }
                            Some(crate::engines::ProjectedCopyByte::Constant(value)) => {
                                return Ok(value);
                            }
                            None => continue,
                        },
                    }
                }
            }
            let mut byte = [0];
            CanonicalWriteBatch::new().read_staged(range, offset, &mut byte)?;
            return Ok(byte[0]);
        }
    }
    fn read_before(
        &self,
        range: &CanonicalBackingRange,
        offset: u64,
        output: &mut [u8],
        limit: usize,
    ) -> Result<(), CanonicalWriteBatchError> {
        let end = offset
            .checked_add(output.len() as u64)
            .ok_or(CanonicalWriteBatchError::RangeOverflow)?;
        if end > range.size() {
            return Err(CanonicalWriteBatchError::OutOfBounds {
                offset,
                size: output.len() as u64,
                range_size: range.size(),
            });
        }
        if output.is_empty() {
            return Ok(());
        }
        if limit == 0 {
            return CanonicalWriteBatch::new().read_staged(range, offset, output);
        }
        let candidates = self.candidates(range, offset, output.len() as u64, limit);
        if candidates.is_empty() {
            return CanonicalWriteBatch::new().read_staged(range, offset, output);
        }
        let mut covered = vec![false; output.len()];
        for index in candidates.into_iter().rev() {
            let entry = &self.entries[index];
            for (to, from, size) in intersections(range, offset, output.len() as u64, &entry.target)
                .into_iter()
                .rev()
            {
                match &entry.value {
                    Value::Inline {
                        bytes,
                        size: length,
                    } => {
                        let bytes = &bytes[..*length];
                        for byte in 0..size {
                            if !covered[to + byte] {
                                output[to + byte] = bytes[from as usize + byte];
                                covered[to + byte] = true;
                            }
                        }
                    }
                    Value::Bytes(bytes) => {
                        for byte in 0..size {
                            if !covered[to + byte] {
                                output[to + byte] = bytes[from as usize + byte];
                                covered[to + byte] = true;
                            }
                        }
                    }
                    Value::Copy { source, operation } => {
                        for byte in 0..size {
                            if covered[to + byte] {
                                continue;
                            }
                            match operation
                                .project_byte(from + byte as u64)
                                .map_err(|_| CanonicalWriteBatchError::RangeOverflow)?
                            {
                                Some(crate::engines::ProjectedCopyByte::Source(position)) => {
                                    output[to + byte] =
                                        self.read_byte_before(source, position, index)?
                                }
                                Some(crate::engines::ProjectedCopyByte::Constant(value)) => {
                                    output[to + byte] = value
                                }
                                None => continue,
                            }
                            covered[to + byte] = true;
                        }
                    }
                }
            }
            if covered.iter().all(|covered| *covered) {
                return Ok(());
            }
        }
        let empty = CanonicalWriteBatch::new();
        let mut cursor = 0;
        while cursor < output.len() {
            if covered[cursor] {
                cursor += 1;
                continue;
            }
            let start = cursor;
            while cursor < output.len() && !covered[cursor] {
                cursor += 1;
            }
            empty.read_staged(range, offset + start as u64, &mut output[start..cursor])?;
        }
        Ok(())
    }
}
impl MemoryProjection {
    #[cfg(test)]
    pub(crate) fn stage(
        &mut self,
        range: &CanonicalBackingRange,
        offset: u64,
        bytes: &[u8],
    ) -> Result<(), CanonicalWriteBatchError> {
        let target = range
            .snapshot_subrange(offset, bytes.len() as u64)
            .map_err(|_| CanonicalWriteBatchError::RangeOverflow)?;
        self.insert(Entry {
            target: target.segments().to_vec(),
            value: Value::Bytes(bytes.into()),
        });
        Ok(())
    }
    pub(crate) fn read_staged(
        &self,
        range: &CanonicalBackingRange,
        offset: u64,
        output: &mut [u8],
    ) -> Result<(), CanonicalWriteBatchError> {
        self.read_before(range, offset, output, self.entries.len())
    }
    pub(crate) fn overlaps(
        &self,
        range: &CanonicalBackingRange,
        offset: u64,
        size: u64,
    ) -> Result<bool, CanonicalWriteBatchError> {
        if offset
            .checked_add(size)
            .is_none_or(|end| end > range.size())
        {
            return Err(CanonicalWriteBatchError::RangeOverflow);
        }
        Ok(self
            .candidates(range, offset, size, self.entries.len())
            .into_iter()
            .any(|index| {
                !intersections(range, offset, size, &self.entries[index].target).is_empty()
            }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nixe_memory::{CanonicalAllocation, MemoryPermissions};
    #[test]
    fn repeated_physical_pages_within_one_transfer_keep_the_last_logical_write() {
        let allocation = CanonicalAllocation::zeroed(4096, 4096).unwrap();
        let whole = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let byte = whole.snapshot_subrange(0, 1).unwrap();
        let mut projection = MemoryProjection::default();
        projection.insert(Entry {
            target: vec![byte.segments()[0].clone(), byte.segments()[0].clone()],
            value: Value::Bytes(vec![11, 22].into_boxed_slice()),
        });
        let mut output = [0];
        projection.read_staged(&whole, 0, &mut output).unwrap();
        assert_eq!(output, [22]);
        assert_eq!(projection.read_byte_before(&whole, 0, 1).unwrap(), 22);
    }

    #[test]
    fn projections_preserve_physical_alias_order_and_uncovered_bytes() {
        let allocation = CanonicalAllocation::zeroed(4096, 4096).unwrap();
        allocation.write(0, &[1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
        let whole = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let first = whole.snapshot_subrange(2, 4).unwrap();
        let alias = whole.snapshot_subrange(4, 4).unwrap();
        let mut projection = MemoryProjection::default();
        projection.insert(Entry {
            target: first.segments().to_vec(),
            value: Value::Bytes(vec![20, 30, 40, 50].into_boxed_slice()),
        });
        projection.insert(Entry {
            target: alias.segments().to_vec(),
            value: Value::Bytes(vec![60, 70, 80, 90].into_boxed_slice()),
        });
        let mut bytes = [0; 8];
        projection.read_staged(&whole, 0, &mut bytes).unwrap();
        assert_eq!(bytes, [1, 2, 20, 30, 60, 70, 80, 90]);
        let mut original = [0; 8];
        whole.read(0, &mut original).unwrap();
        assert_eq!(original, [1, 2, 3, 4, 5, 6, 7, 8]);
        assert!(projection.overlaps(&whole, 3, 1).unwrap());
        assert!(!projection.overlaps(&whole, 0, 2).unwrap());
    }
}
