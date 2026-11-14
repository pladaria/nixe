//! Coherent rectangular transfers retain both mappings through publication.
use super::*;
use nixe_gpu::physical::{Footprint, WriteDomain};
use nixe_memory::{AccessOperation, PhysicalMapping};

fn failure(error: impl std::fmt::Display) -> Box<str> {
    error.to_string().into_boxed_str()
}
impl<E: ampere::Rasterizer> Back<'_, E> {
    pub(super) fn copy_memory(&mut self, copy: &ampere::dma::Copy) -> Result<()> {
        let (source, destination) = {
            let mut space = self.memory.space.lock().unwrap();
            (
                space
                    .capture(copy.source.address, copy.source.bytes)
                    .map_err(failure)?,
                space
                    .capture(copy.destination.address, copy.destination.bytes)
                    .map_err(failure)?,
            )
        };
        let footprint = |capture: &GpuCapture, layout| {
            let ranges = capture.ranges.capture(0, capture.ranges.bytes())?;
            Footprint::new(PhysicalMapping::new(&ranges)?, WriteDomain::Image(layout))
        };
        let source_layout = footprint(&source, copy.source.layout).map_err(failure)?;
        let destination_layout =
            footprint(&destination, copy.destination.layout).map_err(failure)?;
        let bounds = source_layout.bounds();
        let ranges = source
            .ranges
            .capture(bounds.start, bounds.end - bounds.start)
            .map_err(failure)?;
        // Resolve the producer once for the whole source; never wait per GOB.
        let AccessResult::Read(bytes) = self
            .renderer
            .access_memory(
                ranges,
                AccessOperation::Read(vec![0; (bounds.end - bounds.start) as usize]),
            )
            .map_err(failure)?
        else {
            return Err("invalid DMA source read completion".into());
        };
        let pixels =
            u64::from(copy.source.layout.rect.width) * u64::from(copy.source.layout.rect.height);
        let write = if copy.source.layout == copy.destination.layout
            && bounds.end - bounds.start == pixels
        {
            nixe_gpu::physical::PackedWrite {
                ranges: vec![destination_layout.bounds()],
                bytes,
            }
        } else {
            destination_layout
                .repack_bytes(&source_layout, bounds.start, &bytes)
                .map_err(failure)?
        };
        let mut ranges = Vec::new();
        for run in &write.ranges {
            ranges.extend(
                destination
                    .ranges
                    .capture(run.start, run.end - run.start)
                    .map_err(failure)?,
            );
        }
        // One exact scattered write preserves padding, CPU notifications and
        // physical aliases. The source snapshot remains independent of overlap.
        let AccessResult::Written(bytes) = self
            .renderer
            .access_memory(ranges, AccessOperation::Write(write.bytes))
            .map_err(failure)?
        else {
            return Err("invalid DMA destination write completion".into());
        };
        let mut at = 0;
        for run in write.ranges {
            let end = at + (run.end - run.start) as usize;
            self.memory.waiters.written(
                &destination.ranges,
                run.start,
                &bytes[at..end],
                |source| read_owned(self.renderer, &source.ranges, 4),
            )?;
            at = end;
        }
        Ok(())
    }
}
