//! Native recording identity shared by generation-specific draw preparation.
use super::*;

#[derive(Clone, Copy, Debug)]
#[repr(usize)]
pub(in crate::native) enum DrawMiss {
    Cold,
    Kind,
    State,
    Mapping,
    Operation,
    Submission,
    RenderPass,
    Texture,
    Content,
}

#[derive(Clone, Copy, Debug)]
#[repr(usize)]
pub(in crate::native) enum TextureMiss {
    Cold,
    Identity,
    Handle,
    Descriptor,
    Image,
    Unsupported,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct ReuseStatistics {
    pub eligible_draws: u64,
    pub draw_hits: u64,
    pub pipeline_hits: u64,
    pub texture_attempts: u64,
    pub texture_hits: u64,
    pub draw_misses: [u64; 9],
    pub texture_misses: [u64; 6],
}

#[derive(Clone, Copy)]
pub(in crate::native) struct DrawScope {
    channel: Option<u64>,
    memory: [u64; 3],
    revision: u64,
    batch: u64,
    tick: u64,
    render_pass: u64,
    targets: u64,
}
impl DrawScope {
    pub fn capture(execution: &NativeExecution<'_>, memory: [u64; 3]) -> Self {
        Self {
            channel: execution.state.current_context,
            memory,
            revision: execution.residency.texture_cache.binding_revision,
            batch: execution.state.statistics.batches + 1,
            tick: execution.residency.device().current_work().tick(),
            render_pass: execution
                .residency
                .device()
                .recording_statistics()
                .rendering_scopes,
            targets: execution
                .residency
                .texture_cache
                .bindings
                .channel
                .render_targets_serial,
        }
    }
    pub fn validate(
        &self,
        execution: &NativeExecution<'_>,
        pipeline: usize,
        memory: [u64; 3],
    ) -> Result<(), DrawMiss> {
        if self.channel != execution.state.current_context {
            return Err(DrawMiss::State);
        }
        if self.memory != memory
            || self.revision != execution.residency.texture_cache.binding_revision
        {
            return Err(DrawMiss::Mapping);
        }
        if self.batch != execution.state.statistics.batches {
            return Err(DrawMiss::Operation);
        }
        if self.tick != execution.residency.device().current_work().tick() {
            return Err(DrawMiss::Submission);
        }
        let device = execution.residency.device();
        let bank = &execution.residency.texture_cache.bindings.channel;
        if self.render_pass != device.recording_statistics().rendering_scopes
            || self.targets != bank.render_targets_serial
            || bank.invalidated_targets != 0
            || device.submission.recording.is_none_or(|slot| {
                let slot = &device.submission.slots[slot];
                slot.active_render_pass.is_none() || slot.bound_graphics != pipeline
            })
        {
            return Err(DrawMiss::RenderPass);
        }
        Ok(())
    }
}
