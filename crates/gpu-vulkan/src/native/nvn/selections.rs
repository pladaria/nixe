//! Queue-local native selections. The recorder synchronizes the contents of each
//! selected range at use, including when its address or descriptor is unchanged.
use super::*;
use nixe_nvn::{Bindings, Range};

type Epoch = (u64, u64);
type Geometry = Option<(Range, BufferRange, u64)>;

#[derive(Default)]
pub(super) struct Queue {
    epoch: Epoch,
    pub graphics: Program,
    pub compute: Program,
    pub index: Geometry,
    pub vertices: [Geometry; 32],
    selection: Option<Arc<nixe_nvn::Attachments>>,
    attachments: [Option<(Texture, ImageViewId)>; 9],
    framebuffer: Option<([u32; 2], Arc<PreparedTargets>)>,
}

impl Queue {
    pub(super) fn invalidate(&mut self, epoch: Epoch) {
        if self.epoch != epoch {
            self.epoch = epoch;
            self.vertices.fill(None);
            self.index = None;
            self.selection = None;
            self.attachments.fill(None);
            self.framebuffer = None;
        }
    }

    pub(super) fn targets(
        &mut self,
        native: &mut NativeExecution<'_>,
        draw: &nixe_nvn::Draw,
        extent: [u32; 2],
        epoch: Epoch,
    ) -> Result<Option<Arc<PreparedTargets>>, VulkanError> {
        if self
            .selection
            .as_ref()
            .is_none_or(|old| !Arc::ptr_eq(old, draw.attachments))
        {
            for (i, selection) in self.attachments.iter_mut().enumerate() {
                let texture = if i == 8 {
                    draw.attachments.depth
                } else {
                    draw.attachments.colors.get(i).copied().flatten()
                };
                if selection.as_ref().map(|s| s.0) != texture {
                    *selection = texture
                        .map(|t| resources::target_view(native, t).map(|v| (t, v)))
                        .transpose()?;
                    self.framebuffer = None;
                }
            }
            self.selection = Some(draw.attachments.clone());
        }
        if native.residency.texture_cache.binding_revision != epoch.1
            || native.residency.device_memory().generation() != epoch.0
        {
            return Ok(None);
        }
        if let Some((size, framebuffer)) = &self.framebuffer
            && *size == extent
        {
            return Ok(Some(framebuffer.clone()));
        }
        let views = self.attachments.map(|a| a.map(|s| s.1));
        let framebuffer = native.residency.framebuffer_for_views(views, extent)?;
        if native.residency.texture_cache.binding_revision != epoch.1
            || native.residency.device_memory().generation() != epoch.0
        {
            return Ok(None);
        }
        self.framebuffer = Some((extent, framebuffer.clone()));
        Ok(Some(framebuffer))
    }
}

pub(super) fn geometry(
    native: &mut NativeExecution<'_>,
    slot: &mut Geometry,
    range: Range,
    usage: vk::BufferUsageFlags,
) -> Result<BufferRange, VulkanError> {
    if let Some((bound, physical, revision)) = *slot
        && range == bound
        && revision == native.residency.texture_cache.binding_revision
    {
        return Ok(physical);
    }
    let physical =
        native
            .residency
            .device_buffer(range.address, range.size, usage, false, false)?;
    *slot = Some((
        range,
        physical,
        native.residency.texture_cache.binding_revision,
    ));
    Ok(physical)
}

#[derive(Default)]
pub(super) struct Program {
    pub program: Option<HostProgram>,
    slots: Vec<Binding>,
    epoch: Epoch,
}

enum Binding {
    Texture([u32; 2], bool),
    Buffer(Range),
    Driver(Arc<[u8]>),
}

impl NativeExecution<'_> {
    pub(super) fn nvn_bindings(
        &mut self,
        selected: &mut Program,
        pipeline: &shader::Program,
        bindings: &Bindings,
        groups: [u32; 3],
        first_instance: u32,
        samples: u32,
    ) -> Result<(), VulkanError> {
        let epoch = (
            self.residency.device_memory().generation(),
            self.residency.texture_cache.binding_revision,
        );
        if let Some(program) = &mut selected.program
            && program.pipeline.identity() != pipeline.native.identity()
            && program.pipeline.layout.key == pipeline.native.layout.key
        {
            // The new pipeline owns access scopes and Vulkan bind state. Equal
            // descriptor layouts can consume the already resolved physical slots.
            program.pipeline = pipeline.native.clone();
        }
        if selected
            .program
            .as_ref()
            .is_none_or(|p| p.pipeline.identity() != pipeline.native.identity())
        {
            selected.program = Some(self.residency.new_nvn_program(pipeline.native.clone())?);
            selected.slots.clear();
        } else if epoch != selected.epoch {
            selected.slots.clear();
        }
        selected.epoch = epoch;
        let program = selected.program.as_mut().unwrap();
        for (i, input) in pipeline.inputs.iter().enumerate() {
            let previous = selected.slots.get(i);
            let (key, binding) = match *input {
                shader::Input::Driver(stage, bank, size) => {
                    let binding = self.state.nvn.driver_banks.select(
                        (stage, bank),
                        size,
                        bindings,
                        groups,
                        first_instance,
                        samples,
                    );
                    let HostBinding::UniformBytes(bytes) = &binding else {
                        unreachable!()
                    };
                    if matches!(previous, Some(Binding::Driver(old)) if Arc::ptr_eq(old, bytes)) {
                        continue;
                    }
                    (Binding::Driver(bytes.clone()), binding)
                }
                shader::Input::Texture {
                    stage,
                    slot,
                    arrayed,
                } => {
                    let input = bindings
                        .sampled
                        .get(stage, slot)
                        .ok_or_else(|| error(format!("unbound texture {stage}:{slot}")))?;
                    let ids = input.descriptor_ids();
                    if matches!(previous, Some(Binding::Texture(old, a)) if *old == ids && *a == arrayed)
                    {
                        continue;
                    }
                    (
                        Binding::Texture(ids, arrayed),
                        resources::binding(self, input, arrayed)?,
                    )
                }
                shader::Input::Uniform(..)
                | shader::Input::Storage(..)
                | shader::Input::Constants(_) => {
                    let storage = matches!(input, shader::Input::Storage(..));
                    let range = match *input {
                        shader::Input::Uniform(stage, slot, size) => {
                            let mut range =
                                bindings.buffers.get(false, stage, slot).ok_or_else(|| {
                                    error(format!("unbound uniform slot {stage}:{slot}"))
                                })?;
                            range.size = range.size.min(size);
                            range
                        }
                        shader::Input::Storage(stage, slot) => bindings
                            .buffers
                            .get(true, stage, slot)
                            .ok_or_else(|| error(format!("unbound storage slot {stage}:{slot}")))?,
                        shader::Input::Constants(range) => range,
                        _ => unreachable!(),
                    };
                    if matches!(previous, Some(Binding::Buffer(old)) if *old == range) {
                        continue;
                    }
                    (Binding::Buffer(range), self.nvn_buffer(range, storage)?)
                }
            };
            self.residency.update_host_binding(program, i, &binding)?;
            if let Some(slot) = selected.slots.get_mut(i) {
                *slot = key;
            } else {
                selected.slots.push(key);
            }
        }
        Ok(())
    }
}
