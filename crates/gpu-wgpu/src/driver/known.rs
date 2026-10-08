//! Ordered knowledge of physical upload bytes. Canonical ownership stays in memory.
use super::*;

impl WgpuBackendDriver {
    pub(super) fn record_ordered_writes(
        &mut self,
        accepted: &AcceptedBackendSubmission<'_>,
    ) -> Result<(), BackendDriverError> {
        let dependencies = accepted.resources();
        for operation in accepted.submission().operations() {
            // Distinct command positions order aliases even within one submission.
            let serial = self.take_resource_use()?;
            for access in operation.accesses() {
                if !access.scope().mode().writes() {
                    continue;
                }
                let target = access.target();
                let handle = dependency_handle(dependencies, target.dependency())?;
                invalidate_known_write(&self.visibility, self.resource_record(handle)?, target)?;
                record_device_write(self.resource_record_mut(handle)?, target, serial)?;
            }
            if let GpuCommand::UploadBuffer { destination, bytes } = operation.command() {
                let handle = dependency_handle(
                    dependencies,
                    ResourceDependency::Buffer(destination.buffer),
                )?;
                if let BackendResourceCreateInfo::Buffer {
                    view: Some(view), ..
                } = &self.resource_record(handle)?.immutable
                {
                    self.visibility
                        .update_known(
                            view.backing().range(),
                            destination
                                .range
                                .offset()
                                .checked_sub(view.buffer_offset())
                                .ok_or_else(|| unsupported("upload precedes backing"))?,
                            destination.range.size(),
                            Some(bytes),
                        )
                        .map_err(|error| BackendDriverError::failure(error.to_string()))?;
                }
            }
            if let GpuCommand::Clear(ClearOperation::Image {
                target,
                value: ClearValue::Color(_),
                ..
            }) = operation.command()
            {
                let handle =
                    dependency_handle(dependencies, ResourceDependency::Image(target.image))?;
                self.reclaim_cleared_retired_images(handle, *target)?;
            }
        }
        Ok(())
    }

    pub(super) fn unknown_writebacks(
        &self,
        demanded: &[DemandedWriteback],
    ) -> Result<Vec<DemandedWriteback>, BackendDriverError> {
        let mut output = Vec::with_capacity(demanded.len());
        for write in demanded {
            match *write {
                DemandedWriteback::Buffer(buffer) => {
                    let ranges = self
                        .visibility
                        .unknown_ranges(
                            buffer.page,
                            buffer.page_offset as usize,
                            buffer.range.size as usize,
                        )
                        .map_err(|error| BackendDriverError::failure(error.to_string()))?;
                    let unknown: usize = ranges.iter().map(|range| range.len()).sum();
                    nixe_trace::event(
                        "gpu.visibility_known_bytes",
                        0,
                        buffer.range.size - unknown as u64,
                    );
                    nixe_trace::event("gpu.visibility_buffer_demand_bytes", 0, buffer.range.size);
                    for range in ranges {
                        output.push(DemandedWriteback::Buffer(DemandedBufferWriteback {
                            page_offset: range.start as u64,
                            range: TransferRange {
                                offset: buffer.range.offset + range.start as u64
                                    - buffer.page_offset,
                                size: range.len() as u64,
                            },
                            ..buffer
                        }));
                    }
                }
                DemandedWriteback::Image { .. } => output.push(*write),
            }
        }
        Ok(output)
    }
}

fn invalidate_known_write(
    visibility: &WgpuVisibilityCoordinator,
    record: &ResourceRecord,
    target: nixe_gpu::AccessTarget,
) -> Result<(), BackendDriverError> {
    let result = match (target, &record.immutable) {
        (
            nixe_gpu::AccessTarget::Buffer { range, .. },
            BackendResourceCreateInfo::Buffer {
                view: Some(view), ..
            },
        ) => visibility.update_known(
            view.backing().range(),
            range
                .offset()
                .checked_sub(view.buffer_offset())
                .ok_or_else(|| unsupported("write precedes backing"))?,
            range.size(),
            None,
        ),
        (
            nixe_gpu::AccessTarget::Image { subresources, .. },
            BackendResourceCreateInfo::Image { .. },
        ) => {
            if let Some(content) = &record.content {
                for domain in &content.image_domains {
                    if image_subresources_overlap(domain.subresources, subresources) {
                        visibility
                            .update_known(&domain.backing, 0, domain.backing.size(), None)
                            .map_err(|error| BackendDriverError::failure(error.to_string()))?;
                    }
                }
            }
            Ok(())
        }
        _ => Ok(()),
    };
    result.map_err(|error| BackendDriverError::failure(error.to_string()))
}
