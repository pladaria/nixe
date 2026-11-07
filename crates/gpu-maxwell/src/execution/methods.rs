//! Direct engine memory and rasterizer calls. Only unavailable native
//! preparation has a continuation; method admission never produces commands.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/engines/maxwell_3d.cpp
use super::*;
use crate::{
    MaxwellComputeInlineToMemoryUpload, MaxwellComputeLaunch, MaxwellComputeOperationTrigger,
    MaxwellComputeState, MaxwellInlineToMemoryUpload, MaxwellThreeDState,
    MaxwellThreeDSynchronizationTrigger, MaxwellTwoDBlitOperation, MaxwellTwoDSolidOperation,
};

impl<L: MaxwellFrontend, H: MaxwellCommandHost> ChannelExecution<'_, L, H> {
    async fn wait_native(
        &mut self,
        source: MaxwellMethodSource,
        mut consume: impl FnMut(&mut Self) -> Result<bool, MaxwellSubmissionExecutionError>,
    ) -> Result<(), MaxwellSubmissionExecutionError> {
        let result = std::future::poll_fn(|_| {
            self.host.preparation_pending(true)?;
            match at_method(|| source, consume(self))? {
                true => std::task::Poll::Ready(Ok(())),
                false => std::task::Poll::Pending,
            }
        })
        .await;
        self.host.preparation_pending(false)?;
        result
    }
    pub(crate) fn notification(
        &mut self,
        address: u64,
        source: MaxwellMethodSource,
    ) -> Result<(), MaxwellSubmissionExecutionError> {
        let result = (|| {
            let target = self
                .address_space
                .borrow()
                .address(address)
                .map_err(MaxwellGpuAccessError::Address)
                .and_then(|address| self.address_space.borrow().resolve_range(address, 16))
                .map_err(|error| MaxwellSubmissionExecutionError::InlineAddress {
                    source,
                    error,
                })?;
            self.report(source, &target, 0, false, true, true)?;
            self.prior_work_pending = false;
            Ok(())
        })();
        at_method(|| source, result)
    }
    pub(crate) fn host_synchronization(
        &mut self,
        source: MaxwellMethodSource,
        operation: MaxwellHostSynchronizationKind,
    ) -> Result<(), MaxwellSubmissionExecutionError> {
        let result = (|| {
            match operation {
                MaxwellHostSynchronizationKind::SemaphoreRelease {
                    address,
                    payload,
                    short,
                    wait,
                } => {
                    let target = self
                        .address_space
                        .borrow()
                        .address(address)
                        .map_err(MaxwellGpuAccessError::Address)
                        .and_then(|address| {
                            self.address_space
                                .borrow()
                                .resolve_range(address, if short { 4 } else { 16 })
                        })
                        .map_err(|error| MaxwellSubmissionExecutionError::InlineAddress {
                            source,
                            error,
                        })?;
                    self.report(source, &target, payload, short, true, false)?;
                    if wait {
                        self.prior_work_pending = false;
                    }
                    Ok(())
                }
                MaxwellHostSynchronizationKind::WaitForIdle { .. } => {
                    // All engine work on this channel shares the ordered
                    // backend stream. Both scopes therefore drain its prior
                    // segment; they never complete another channel's work.
                    self.host.order_device(self.frontend)?;
                    self.prior_work_pending = false;
                    Ok(())
                }
                MaxwellHostSynchronizationKind::L2SysmemInvalidate { .. }
                | MaxwellHostSynchronizationKind::L2FlushDirty { .. } => {
                    // Puller::MemOpB consumes the per-core GPU dirty managers.
                    self.host.invalidate_gpu_cache();
                    Ok(())
                }
            }
        })();
        at_method(|| source, result)
    }
    pub(crate) fn compute_inline(
        &mut self,
        upload: MaxwellComputeInlineToMemoryUpload,
        inline_words: Option<crate::MaxwellDmaArguments<'_>>,
    ) -> Result<(), MaxwellSubmissionExecutionError> {
        let source = upload.source();
        let result = (|| {
            // Both FLUSH_DISABLE and FLUSH_ONLY use coherent canonical writes.
            // Backend execution captures preceding overlapping accesses before
            // these writes and commits them before the following operation;
            // write-only initialization commits before returning completion.
            // This also satisfies enabled SYSMEMBAR without an extra host wait
            // or per-upload flush command. Neither mode releases a semaphore.
            // https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/compute/clb1c0.h#L129-L165
            if let Some(words) = inline_words {
                self.push_inline_words(
                    upload.address().get(),
                    upload.offset(),
                    words,
                    words.len() * 4,
                    upload.source(),
                )?;
            } else {
                self.push_inline_write(
                    upload.address().get(),
                    upload.offset(),
                    upload.value(),
                    upload.source(),
                )?;
            }
            Ok(())
        })();
        at_method(|| source, result)
    }
    pub(crate) fn inline_to_memory(
        &mut self,
        upload: MaxwellInlineToMemoryUpload,
        inline_words: Option<crate::MaxwellDmaArguments<'_>>,
    ) -> Result<(), MaxwellSubmissionExecutionError> {
        let source = upload.source();
        let result = (|| {
            if let Some(words) = inline_words {
                self.push_inline_words(
                    upload.address().get(),
                    upload.offset(),
                    words,
                    (words.len() * 4).min(upload.remaining_bytes() as usize),
                    upload.source(),
                )?;
                return Ok(());
            }
            self.write(
                upload.address().get(),
                upload.offset(),
                &upload.value().to_le_bytes()[..upload.remaining_bytes().min(4) as usize],
                upload.source(),
            )?;
            Ok(())
        })();
        at_method(|| source, result)
    }
    pub(crate) fn copy_memory(
        &mut self,
        operation: MaxwellMemoryCopyOperation,
    ) -> Result<(), MaxwellSubmissionExecutionError> {
        let source = operation.source();
        let result = (|| {
            self.address_space
                .borrow_mut()
                .flush_caching(self.host)
                .map_err(|e| MaxwellSubmissionExecutionError::Native(e.to_string().into()))?;

            self.copy(operation)?;
            if let Some((address, payload)) = operation.semaphore_release {
                let target = resolve_inline_target(
                    &self.address_space.borrow(),
                    address,
                    0,
                    operation.source(),
                )?;
                self.report(operation.source(), &target, payload, true, true, false)?;
            }
            self.prior_work_pending = true;
            Ok(())
        })();
        at_method(|| source, result)
    }
    pub(crate) fn compute_synchronization(
        &mut self,
        operation: MaxwellComputeOperationTrigger,
    ) -> Result<(), MaxwellSubmissionExecutionError> {
        let source = operation.source();
        let result = (|| {
            let plan = lower_maxwell_compute_synchronization(operation, self.prior_work_pending);
            match plan {
                MaxwellComputeSynchronizationPlan::WaitForIdle { .. } => {
                    self.host.order_device(self.frontend)?;
                    self.prior_work_pending = false;
                }
                MaxwellComputeSynchronizationPlan::InvalidateShaderCaches { caches, .. }
                | MaxwellComputeSynchronizationPlan::InvalidateShaderCachesNoWfi { caches } => {
                    if matches!(
                        plan,
                        MaxwellComputeSynchronizationPlan::InvalidateShaderCaches { .. }
                    ) {
                        self.prior_work_pending = false;
                    }
                    self.host.cache_maintenance(
                        self.frontend,
                        CacheMaintenanceOperation::InvalidateShaderCaches {
                            instruction: caches.instruction(),
                            global_data: caches.global_data(),
                            constant: caches.constant(),
                        },
                    )?;
                }
            }
            Ok(())
        })();
        at_method(|| source, result)
    }
    pub(crate) fn bind_constant_buffer(
        &mut self,
        three_d: &MaxwellThreeDState,
        group: u8,
        slot: u8,
        origin: impl FnOnce() -> MaxwellMethodSource,
    ) -> Result<(), MaxwellSubmissionExecutionError> {
        let result = (|| {
            self.cache.borrow_mut().bind_graphics_uniform(
                group,
                slot,
                three_d.shader_bindings().groups()[group as usize].constant_buffers()
                    [slot as usize],
                &self.address_space.borrow(),
            )?;
            Ok(())
        })();
        at_method(origin, result)
    }
    pub(crate) fn render_condition(
        &mut self,
        three_d: &MaxwellThreeDState,
        source: MaxwellMethodSource,
    ) -> Result<(), MaxwellSubmissionExecutionError> {
        let result = (|| {
            self.address_space
                .borrow_mut()
                .flush_caching(self.host)
                .map_err(|e| MaxwellSubmissionExecutionError::Native(e.to_string().into()))?;
            self.cache.borrow_mut().update_render_condition(
                three_d,
                &self.address_space.borrow(),
                self.host,
            )?;
            Ok(())
        })();
        at_method(|| source, result)
    }
    pub(crate) fn three_d_synchronization(
        &mut self,
        three_d: &MaxwellThreeDState,
        trigger: MaxwellThreeDSynchronizationTrigger,
    ) -> Result<(), MaxwellSubmissionExecutionError> {
        let source = trigger.source();
        let result = (|| {
            let plan = lower_maxwell_three_d_synchronization(
                trigger,
                three_d,
                self.completion,
                self.prior_work_pending,
            )
            .map_err(MaxwellSubmissionExecutionError::ThreeDSynchronization)?;
            if let MaxwellThreeDSynchronizationPlan::IncrementSyncpoint {
                completion: reserved,
                ..
            } = plan
            {
                self.completion_signal_count = self.completion_signal_count.saturating_add(1);
                let expected = self.completion.map_or(0, ReservedTimelinePoint::increments);
                if self.completion_signal_count > expected {
                    return Err(MaxwellSubmissionExecutionError::DuplicateCompletionSignal {
                        reserved,
                        expected,
                        observed: self.completion_signal_count,
                    });
                }
                self.host.signal_syncpoint()?;
            }
            let drains_prior_work = matches!(
                plan,
                MaxwellThreeDSynchronizationPlan::DecompressUncompressedSurface { .. }
                    | MaxwellThreeDSynchronizationPlan::WaitForIdle { .. }
                    | MaxwellThreeDSynchronizationPlan::InvalidateShaderCaches { .. }
                    | MaxwellThreeDSynchronizationPlan::FlushPendingWrites { .. }
                    | MaxwellThreeDSynchronizationPlan::InvalidateTextureCache { .. }
                    | MaxwellThreeDSynchronizationPlan::ReportSemaphoreWrite(_)
                    | MaxwellThreeDSynchronizationPlan::IncrementSyncpoint { .. }
            );
            if matches!(
                plan,
                MaxwellThreeDSynchronizationPlan::InvalidateTextureCache { .. }
            ) {
                // Maxwell3D::invalidate_texture_data_cache: invalidation then WFI.
                self.host.invalidate_gpu_cache();
                self.host.order_device(self.frontend)?;
            }
            if let MaxwellThreeDSynchronizationPlan::ReportSemaphoreWrite(report) = plan {
                let short = report.structure_size()
                    == crate::MaxwellThreeDReportSemaphoreStructureSize::OneWord;
                let target = self
                    .address_space
                    .borrow()
                    .address(report.address().get())
                    .map_err(MaxwellGpuAccessError::Address)
                    .and_then(|address| {
                        self.address_space
                            .borrow()
                            .resolve_range(address, if short { 4 } else { 16 })
                    })
                    .map_err(|error| MaxwellSubmissionExecutionError::InlineAddress {
                        source: report.source(),
                        error,
                    })?;
                self.report(
                    report.source(),
                    &target,
                    report.payload(),
                    short,
                    report.is_fence(),
                    false,
                )?;
            } else if matches!(plan, MaxwellThreeDSynchronizationPlan::WaitForIdle { .. }) {
                self.host.order_device(self.frontend)?;
            } else if let Some(operation) = three_d_synchronization_operation(plan) {
                self.host.cache_maintenance(self.frontend, operation)?;
            }
            if drains_prior_work {
                self.prior_work_pending = false;
            }
            Ok(())
        })();
        at_method(|| source, result)
    }
    pub(crate) fn solid(
        &mut self,
        request: &MaxwellTwoDSolidOperation,
    ) -> Result<bool, MaxwellSubmissionExecutionError> {
        debug_assert!(self.preparing_blit.is_none() && self.preparing_solid.is_none());
        let result = self.solid_inner(request);
        let ready = at_method(|| request.source, result)?;
        if !ready {
            self.preparing_solid = Some(*request);
        }
        Ok(ready)
    }
    fn solid_inner(
        &mut self,
        request: &MaxwellTwoDSolidOperation,
    ) -> Result<bool, MaxwellSubmissionExecutionError> {
        let [x0, y0, x1, y1] = request.rectangle;
        if x0 < x1 && y0 < y1 {
            if !self.cache.borrow_mut().try_solid(
                request,
                &self.address_space.borrow(),
                self.frontend,
            )? {
                return Ok(false);
            }
            self.prior_work_pending = true;
        }
        Ok(true)
    }
    pub(crate) async fn wait_solid(
        &mut self,
        request: &MaxwellTwoDSolidOperation,
    ) -> Result<(), MaxwellSubmissionExecutionError> {
        self.wait_native(request.source, |this| this.solid_inner(request))
            .await
    }
    pub(crate) fn blit(
        &mut self,
        request: &MaxwellTwoDBlitOperation,
    ) -> Result<bool, MaxwellSubmissionExecutionError> {
        debug_assert!(self.preparing_blit.is_none() && self.preparing_solid.is_none());
        let result = self.blit_inner(request);
        let ready = at_method(|| request.source(), result)?;
        if !ready {
            self.preparing_blit = Some(*request);
        }
        Ok(ready)
    }
    fn blit_inner(
        &mut self,
        request: &MaxwellTwoDBlitOperation,
    ) -> Result<bool, MaxwellSubmissionExecutionError> {
        if !self.cache.borrow_mut().try_blit(
            request,
            &self.address_space.borrow(),
            self.frontend,
        )? {
            return Ok(false);
        }
        self.prior_work_pending = true;
        Ok(true)
    }
    pub(crate) async fn wait_blit(
        &mut self,
        request: &MaxwellTwoDBlitOperation,
    ) -> Result<(), MaxwellSubmissionExecutionError> {
        self.wait_native(request.source(), |this| this.blit_inner(request))
            .await
    }
    pub(crate) async fn wait_two_d(&mut self) -> Result<(), MaxwellSubmissionExecutionError> {
        if let Some(request) = self.preparing_blit.take() {
            debug_assert!(self.preparing_solid.is_none());
            self.wait_blit(&request).await
        } else {
            let request = self
                .preparing_solid
                .take()
                .expect("unavailable native 2D preparation");
            self.wait_solid(&request).await
        }
    }
    pub(crate) fn draw(
        &mut self,
        three_d: &MaxwellThreeDState,
        draw: &crate::MaxwellThreeDDrawState,
        origin: impl FnOnce() -> MaxwellMethodSource,
    ) -> Result<bool, MaxwellSubmissionExecutionError> {
        let result = self.draw_inner(three_d, draw);
        let ready = at_method(origin, result)?;
        if ready {
            self.tick_work()?;
        }
        Ok(ready)
    }
    fn draw_inner(
        &mut self,
        three_d: &MaxwellThreeDState,
        draw: &crate::MaxwellThreeDDrawState,
    ) -> Result<bool, MaxwellSubmissionExecutionError> {
        if !self.cache.borrow().render_enabled() {
            return Ok(true);
        }
        self.address_space
            .borrow_mut()
            .flush_caching(self.host)
            .map_err(|e| MaxwellSubmissionExecutionError::Native(e.to_string().into()))?;
        if !self.cache.borrow_mut().try_draw(
            three_d,
            draw,
            &self.address_space.borrow(),
            self.frontend,
            self.host,
        )? {
            return Ok(false);
        }
        self.prior_work_pending = true;
        Ok(true)
    }
    pub(crate) async fn wait_draw(
        &mut self,
        three_d: &MaxwellThreeDState,
        draw: &crate::MaxwellThreeDDrawState,
        source: MaxwellMethodSource,
    ) -> Result<(), MaxwellSubmissionExecutionError> {
        self.wait_native(source, |this| this.draw_inner(three_d, draw))
            .await?;
        self.tick_work()?;
        Ok(())
    }
    /// Clear consumes current registers directly; unlike shader preparation it
    /// has no unavailable-pipeline continuation. Eden exposes distinct Draw and
    /// Clear rasterizer calls rather than a provenance-bearing graphics command.
    /// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/engines/draw_manager.cpp#L75-L84
    pub(crate) fn clear(
        &mut self,
        three_d: &MaxwellThreeDState,
        origin: impl FnOnce() -> MaxwellMethodSource,
    ) -> Result<(), MaxwellSubmissionExecutionError> {
        let result = (|| {
            if !self.cache.borrow().render_enabled() {
                return Ok(());
            }
            self.address_space
                .borrow_mut()
                .flush_caching(self.host)
                .map_err(|e| MaxwellSubmissionExecutionError::Native(e.to_string().into()))?;
            self.cache
                .borrow_mut()
                .clear(three_d, &self.address_space.borrow(), self.frontend)?;
            self.prior_work_pending = true;
            Ok(())
        })();
        at_method(origin, result)
    }
    pub(crate) fn launch(
        &mut self,
        launch: MaxwellComputeLaunch,
        compute: &MaxwellComputeState,
    ) -> Result<bool, MaxwellSubmissionExecutionError> {
        debug_assert!(self.active_compute.is_none());
        let result = (|| {
            self.active_compute = Some(
                crate::engines::resolve_compute_launch(
                    &launch,
                    compute,
                    &self.address_space.borrow(),
                )
                .map_err(MaxwellSubmissionExecutionError::ComputeLaunch)?,
            );
            self.launch_inner()
        })();
        at_method(|| launch.source(), result)
    }
    fn launch_inner(&mut self) -> Result<bool, MaxwellSubmissionExecutionError> {
        self.address_space
            .borrow_mut()
            .flush_caching(self.host)
            .map_err(|e| MaxwellSubmissionExecutionError::Native(e.to_string().into()))?;
        if !self.cache.borrow_mut().try_launch(
            self.active_compute.as_ref().unwrap(),
            &self.address_space.borrow(),
            self.frontend,
            self.host,
        )? {
            return Ok(false);
        }
        self.active_compute = None;
        self.prior_work_pending = true;
        Ok(true)
    }
    pub(crate) async fn wait_launch(&mut self) -> Result<(), MaxwellSubmissionExecutionError> {
        let source = self
            .active_compute
            .as_ref()
            .expect("pending native dispatch")
            .source();
        self.wait_native(source, Self::launch_inner).await
    }
}
fn at_method<T>(
    origin: impl FnOnce() -> MaxwellMethodSource,
    result: Result<T, MaxwellSubmissionExecutionError>,
) -> Result<T, MaxwellSubmissionExecutionError> {
    result.map_err(|cause| MaxwellSubmissionExecutionError::AtMethod {
        source: origin(),
        cause: Box::new(cause),
    })
}
