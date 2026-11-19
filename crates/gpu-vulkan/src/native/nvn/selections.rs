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
    shader: Option<Rc<shader::Compiled>>,
    sampled: Option<Arc<nixe_nvn::SampledBindings>>,
    handles: Option<Arc<nixe_nvn::TextureHandles>>,
    driver: Option<Arc<nixe_nvn::BufferBindings>>,
    driver_args: ([u32; 3], u32, u32),
    epoch: Epoch,
}

enum Binding {
    Texture([u32; 2], bool),
    Image(u32, bool, nixe_shader_recompiler::types::ImageFormat),
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
        let dirty = bindings.buffers.take_dirty();
        let all = selected.slots.len() != pipeline.inputs.len()
            || selected
                .shader
                .as_ref()
                .is_none_or(|s| !Rc::ptr_eq(s, &pipeline.compiled))
            || pipeline.inputs.len() > 128;
        let mut mask = 0u128;
        if !all {
            mask = dirty.slots().map_or(u128::MAX, |slots| {
                slots.fold(0, |m, slot| {
                    m | pipeline
                        .dependencies
                        .buffers
                        .get(&slot)
                        .copied()
                        .unwrap_or(0)
                })
            });
        }
        fn changed<T>(old: &mut Option<Arc<T>>, new: &Arc<T>) -> bool {
            if old.as_ref().is_some_and(|v| Arc::ptr_eq(v, new)) {
                return false;
            }
            *old = Some(new.clone());
            true
        }
        if changed(&mut selected.sampled, &bindings.sampled) {
            mask |= pipeline.dependencies.sampled;
        }
        if changed(&mut selected.handles, &bindings.handles) {
            mask |= pipeline.dependencies.handles;
        }
        let args = (groups, first_instance, samples);
        if changed(&mut selected.driver, &bindings.buffers.driver) || selected.driver_args != args {
            mask |= pipeline.dependencies.driver;
            selected.driver_args = args;
        }
        if all {
            selected.shader = Some(pipeline.compiled.clone());
        }
        #[cfg(feature = "execution-trace")]
        {
            self.state.nvn.slots[0] += pipeline.inputs.len() as u64;
            self.state.nvn.slots[1] += if all {
                pipeline.inputs.len() as u64
            } else {
                (mask
                    & (u128::MAX
                        .checked_shr(128 - pipeline.inputs.len() as u32)
                        .unwrap_or(0)))
                .count_ones() as u64
            };
        }
        let mut index = 0;
        while index < pipeline.inputs.len() {
            let i = if all {
                index
            } else {
                if mask == 0 {
                    break;
                }
                let i = mask.trailing_zeros() as usize;
                mask &= mask - 1;
                if i >= pipeline.inputs.len() {
                    break;
                }
                i
            };
            index = i + 1;
            let input = &pipeline.inputs[i];
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
                shader::Input::Image {
                    stage,
                    slot,
                    ty,
                    format,
                } => {
                    use nixe_shader_recompiler::types::TextureType as T;
                    let input = bindings.sampled.get(stage, slot).ok_or_else(|| {
                        error(format!(
                            "unbound NVN storage image at stage {stage}, row {slot}"
                        ))
                    })?;
                    let id = input.descriptor_ids()[0];
                    let arrayed =
                        matches!(ty, T::ColorArray1D | T::ColorArray2D | T::ColorArrayCube);
                    if matches!(previous, Some(Binding::Image(old, a, f)) if (*old,*a,*f) == (id,arrayed,format))
                    {
                        continue;
                    }
                    let view = resources::descriptor_view(self, input, arrayed)?;
                    let view = self.residency.storage_image_view(view, format)?;
                    (
                        Binding::Image(id, arrayed, format),
                        HostBinding::Image(self.residency.host_image(view)?),
                    )
                }
                shader::Input::Texture { stage, slot, ty } => {
                    use nixe_shader_recompiler::types::TextureType as T;
                    let arrayed =
                        matches!(ty, T::ColorArray1D | T::ColorArray2D | T::ColorArrayCube);
                    // Each texture instruction selects a view of its declared dimension.
                    // An unrelated handle may still occupy an unused conditional slot.
                    let input = bindings.sampled.get(stage, slot).filter(|t| {
                        matches!(
                            (t.texture.target_kind(), ty),
                            (0 | 3, T::Color1D | T::ColorArray1D)
                                | (
                                    1 | 4 | 5 | 6 | 7,
                                    T::Color2D | T::Color2DRect | T::ColorArray2D
                                )
                                | (2, T::Color3D)
                                | (8 | 9, T::ColorCube | T::ColorArrayCube)
                                | (10, T::Buffer)
                        )
                    });
                    let ids = input.map_or([0; 2], |t| t.descriptor_ids());
                    if matches!(previous, Some(Binding::Texture(old, a)) if *old == ids && *a == arrayed)
                    {
                        continue;
                    }
                    (
                        Binding::Texture(ids, arrayed),
                        match input {
                            Some(t) => resources::binding(self, t, arrayed)?,
                            None => resources::null_binding(self)?,
                        },
                    )
                }
                shader::Input::Uniform(..)
                | shader::Input::Storage(..)
                | shader::Input::Constants(..) => {
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
                        shader::Input::Constants(stage, size) => {
                            let shader = &pipeline
                                .usage
                                .stages
                                .iter()
                                .find(|s| s.shader.stage == stage)
                                .expect("constant bank belongs to a linked stage")
                                .shader;
                            Range {
                                address: shader.address + u64::from(shader.constants),
                                size,
                            }
                        }
                        _ => unreachable!(),
                    };
                    if matches!(previous, Some(Binding::Buffer(old)) if *old == range) {
                        continue;
                    }
                    (Binding::Buffer(range), self.nvn_buffer(range, storage)?)
                }
            };
            #[cfg(feature = "execution-trace")]
            {
                self.state.nvn.slots[2] += 1;
            }
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
