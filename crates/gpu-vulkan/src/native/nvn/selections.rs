//! Queue-local native selections. The recorder synchronizes the contents of each
//! selected range at use, including when its address or descriptor is unchanged.
use super::*;
use nixe_nvn::{Bindings, Range};

type Epoch = (u64, u64);
type Geometry = Option<(Range, BufferRange, u64)>;

#[derive(Default)]
pub(super) struct Queue {
    epoch: Epoch,
    pub count_samples: bool,
    pub graphics: Program,
    pub compute: Program,
    pub index: Geometry,
    pub vertices: [Geometry; 32],
    selection: Option<Arc<nixe_nvn::Attachments>>,
    attachments: [Option<(Texture, ImageViewId)>; 9],
    framebuffer: Option<Framebuffer>,
}

struct Framebuffer {
    size: [u32; 2],
    targets: Arc<PreparedTargets>,
    rescaled: bool,
    scale_frame: u64,
}

impl Queue {
    pub(super) fn new(id: u64) -> Self {
        Self {
            graphics: Program {
                owner: id,
                ..Default::default()
            },
            compute: Program {
                owner: id,
                ..Default::default()
            },
            ..Default::default()
        }
    }

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
    ) -> Result<Option<(Arc<PreparedTargets>, bool)>, VulkanError> {
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
        let scaling = native.residency.resolution().active;
        let frame = native.residency.render_scale_frame();
        let views = self.attachments.map(|a| a.map(|s| s.1));
        if scaling {
            for &view in views.iter().flatten() {
                if !native.residency.prepare_scaling_view(view)? {
                    return Ok(None);
                }
            }
        }
        let recheck_scale = scaling
            && self
                .framebuffer
                .as_ref()
                .is_none_or(|fb| !fb.rescaled && fb.scale_frame != frame);
        let rescaled = if recheck_scale {
            let rescaled = native.residency.rescale_explicit_targets(&views)?;
            if native.residency.texture_cache.binding_revision != epoch.1 {
                // Promotion retires these views. Select the new physical views
                // before capturing either the framebuffer or the descriptors.
                return Ok(None);
            }
            rescaled
        } else {
            self.framebuffer.as_ref().is_some_and(|fb| fb.rescaled)
        };
        if let Some(fb) = &mut self.framebuffer
            && fb.size == extent
        {
            fb.scale_frame = frame;
            return Ok(Some((fb.targets.clone(), rescaled)));
        }
        let framebuffer = native.residency.framebuffer_for_views(views, extent)?;
        if native.residency.texture_cache.binding_revision != epoch.1
            || native.residency.device_memory().generation() != epoch.0
        {
            return Ok(None);
        }
        self.framebuffer = Some(Framebuffer {
            size: extent,
            targets: framebuffer.clone(),
            rescaled,
            scale_frame: frame,
        });
        Ok(Some((framebuffer, rescaled)))
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
    owner: u64,
    pool_bounds: Option<Arc<nixe_nvn::PoolBounds>>,
    pub program: Option<HostProgram>,
    slots: Vec<Binding>,
    shader: Option<Rc<shader::Compiled>>,
    sampled: Option<Arc<nixe_nvn::SampledBindings>>,
    handles: Option<Arc<nixe_nvn::TextureHandles>>,
    driver_revision: Option<u64>,
    driver_args: ([u32; 3], u32, u32),
    epoch: Epoch,
}

impl Program {
    pub(super) fn use_pools(&mut self, bounds: &Arc<nixe_nvn::PoolBounds>) {
        if self
            .pool_bounds
            .as_ref()
            .is_none_or(|old| !Arc::ptr_eq(old, bounds))
        {
            if self
                .slots
                .iter()
                .any(|s| matches!(s, Binding::Buffer(r) if r.size == 0 && r.address != 0))
            {
                self.slots.clear();
            }
            self.pool_bounds = Some(bounds.clone());
        }
    }

    pub(super) fn pool_remaining(&self, address: u64) -> Option<u64> {
        self.pool_bounds.as_ref()?.remaining(address)
    }
}

enum Binding {
    Texture([u32; 2], bool),
    Image(u32, bool, nixe_shader_recompiler::types::ImageFormat, bool),
    Buffer(Range),
    Driver(Arc<[u8]>),
    Texel(u32),
    PointerTable,
}

impl NativeExecution<'_> {
    pub(super) fn nvn_scaled_images(
        &mut self,
        selected: &Program,
        pipeline: &shader::Program,
    ) -> Result<Option<state::ShaderConstants>, VulkanError> {
        let program = selected.program.as_ref().unwrap();
        let mut constants = state::ShaderConstants::default();
        for &descriptor in &program.pipeline.image_bindings {
            let Some(view) = program.image_binding_view(descriptor) else {
                continue;
            };
            if !self.residency.prepare_scaling_view(view)? {
                return Ok(None);
            }
            if self.residency.image_is_rescaled(view) {
                match pipeline.inputs[descriptor] {
                    shader::Input::Texture { scale_index, .. } => {
                        constants.set_image_scaled(false, scale_index);
                    }
                    shader::Input::Image { scale_index, .. } => {
                        constants.set_image_scaled(true, scale_index);
                    }
                    _ => unreachable!("NVN image binding"),
                }
            }
        }
        Ok(Some(constants))
    }

    pub(super) fn nvn_bindings(
        &mut self,
        selected: &mut Program,
        pipeline: &shader::Program,
        bindings: &Bindings,
        descriptors: &shader::Descriptors,
        args: ([u32; 3], u32, u32),
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
        let mut mask = pipeline.dependencies.bindless;
        if !all {
            for (storage, stage, slot) in dirty.slots() {
                mask |= pipeline.dependencies.buffers[storage as usize][stage]
                    .get(slot)
                    .copied()
                    .unwrap_or(0);
            }
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
        let (groups, first_instance, samples) = args;
        let driver = bindings.buffers.driver_revision();
        if selected.driver_revision != Some(driver) || selected.driver_args != args {
            mask |= pipeline.dependencies.driver;
            selected.driver_revision = Some(driver);
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
                shader::Input::TexelBuffer { storage, .. } => {
                    let [id, _] = descriptors.get(pipeline.descriptors.bindings[i]);
                    if matches!(previous, Some(Binding::Texel(old)) if *old == id) {
                        continue;
                    }
                    let descriptor = bindings.bindless.unwrap().resolve([id, 0]);
                    let binding = if let Some(descriptor) = descriptor {
                        resources::texel_binding(self, descriptor.texture, storage)?
                    } else {
                        HostBinding::NullBuffer
                    };
                    (Binding::Texel(id), binding)
                }
                shader::Input::Push(block) => {
                    static ZERO: std::sync::OnceLock<Arc<[u8; 256]>> = std::sync::OnceLock::new();
                    let bytes = bindings
                        .buffers
                        .push_block(block as usize)
                        .unwrap_or_else(|| ZERO.get_or_init(|| Arc::new([0; 256])));
                    if matches!(previous, Some(Binding::Driver(old)) if std::ptr::eq(old.as_ptr(), bytes.as_ptr()))
                    {
                        continue;
                    }
                    let bytes: Arc<[u8]> = bytes.clone();
                    (
                        Binding::Driver(bytes.clone()),
                        HostBinding::UniformBytes(bytes),
                    )
                }
                shader::Input::Driver(stage, bank, size) => {
                    let compute = if stage == 5 && bank == 0 {
                        let shader = pipeline.usage.stages[0].shader;
                        (shader.isa == 1).then_some(shader.compute)
                    } else {
                        None
                    };
                    let binding = self.state.nvn.driver_banks.select(
                        (stage, bank, selected.owner, compute),
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
                    bindless,
                    ..
                } => {
                    use nixe_shader_recompiler::types::TextureType as T;
                    let arrayed =
                        matches!(ty, T::ColorArray1D | T::ColorArray2D | T::ColorArrayCube);
                    let writable = pipeline.native.resources[i].writes();
                    let dynamic;
                    let input = if bindless.is_some() {
                        let ids = descriptors.get(pipeline.descriptors.bindings[i]);
                        if matches!(previous, Some(Binding::Image(old, a, f, w)) if (*old, *a, *f, *w) == (ids[0], arrayed, format, writable)) {
                            continue;
                        }
                        dynamic = bindings.bindless.unwrap().resolve(ids);
                        dynamic.as_ref()
                    } else {
                        bindings.sampled.get(stage, slot)
                    }.ok_or_else(|| {
                        error(format!(
                            "unbound NVN storage image at stage {stage}, row {slot}"
                        ))
                    })?;
                    let id = input.descriptor_ids()[0];
                    if matches!(previous, Some(Binding::Image(old, a, f, w)) if (*old, *a, *f, *w) == (id, arrayed, format, writable))
                    {
                        continue;
                    }
                    let view = resources::descriptor_view(self, input, arrayed, writable)?;
                    let view = self.residency.storage_image_view(view, format)?;
                    (
                        Binding::Image(id, arrayed, format, writable),
                        HostBinding::Image(self.residency.host_image(view)?),
                    )
                }
                shader::Input::Texture {
                    stage,
                    slot,
                    ty,
                    bindless,
                    ..
                } => {
                    use nixe_shader_recompiler::types::TextureType as T;
                    let arrayed =
                        matches!(ty, T::ColorArray1D | T::ColorArray2D | T::ColorArrayCube);
                    // Each texture instruction selects a view of its declared dimension.
                    // An unrelated handle may still occupy an unused conditional slot.
                    let dynamic;
                    let input = if bindless.is_some() {
                        let ids = descriptors.get(pipeline.descriptors.bindings[i]);
                        if matches!(previous, Some(Binding::Texture(old, a)) if *old == ids && *a == arrayed)
                        {
                            continue;
                        }
                        dynamic = bindings.bindless.unwrap().resolve(ids);
                        dynamic.as_ref()
                    } else {
                        bindings.sampled.get(stage, slot)
                    };
                    let typed = input.filter(|t| {
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
                    if bindless.is_some() && input.is_some() && typed.is_none() {
                        return Err(error("NVN2 bindless texture target changed pipeline type"));
                    }
                    let ids = typed.map_or([0; 2], |t| t.descriptor_ids());
                    if matches!(previous, Some(Binding::Texture(old, a)) if *old == ids && *a == arrayed)
                    {
                        continue;
                    }
                    (
                        Binding::Texture(ids, arrayed),
                        match typed {
                            Some(t) => resources::binding(self, t, arrayed)?,
                            None => resources::null_binding(self)?,
                        },
                    )
                }
                shader::Input::PointerTable {
                    stage,
                    slot,
                    start,
                    end,
                    written,
                } => {
                    let range = bindings.buffers.get(true, stage, slot).unwrap_or_default();
                    (
                        Binding::PointerTable,
                        self.residency
                            .nvn_pointer_table(range, start, end, written)?,
                    )
                }
                shader::Input::Uniform(..)
                | shader::Input::Storage(..)
                | shader::Input::StoragePointer(..)
                | shader::Input::Constants(..) => {
                    let storage = matches!(
                        input,
                        shader::Input::Storage(..) | shader::Input::StoragePointer(..)
                    );
                    let range = match *input {
                        shader::Input::Uniform(stage, slot, size) => {
                            let mut range =
                                bindings.buffers.get(false, stage, slot).unwrap_or_default();
                            range.size = range.size.min(size);
                            range
                        }
                        // Reflected slots may be unused by the current shader invocation.
                        // The driver bank already represents absent ranges with a null address.
                        // https://github.com/seedds/nxnative/blob/a366365/dm/gfx/nvn/nvn_mtl.m#L13330-L13334
                        shader::Input::Storage(stage, slot) => {
                            bindings.buffers.get(true, stage, slot).unwrap_or_default()
                        }
                        shader::Input::StoragePointer(stage, key) => {
                            let address = u64::from_le_bytes(shader::read_cbuf(
                                &mut self.residency,
                                bindings,
                                stage,
                                key,
                            )?);
                            Range { address, size: 0 }
                        }
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
                    let binding = if storage && range.address != 0 {
                        self.nvn_storage_binding(
                            range,
                            selected.pool_bounds.as_deref(),
                            pipeline,
                            input,
                        )?
                    } else if range.size == 0 {
                        HostBinding::NullBuffer
                    } else {
                        self.nvn_buffer(range, storage)?
                    };
                    // Keep the original pointer so a later mapping invalidates a
                    // null binding even if the pointer value is unchanged.
                    (Binding::Buffer(range), binding)
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

    #[inline]
    pub(super) fn nvn_storage_binding(
        &mut self,
        range: Range,
        pools: Option<&nixe_nvn::PoolBounds>,
        pipeline: &shader::Program,
        input: &shader::Input,
    ) -> Result<HostBinding, VulkanError> {
        // A zero extent leaves the storage pointer live in the driver bank.
        // Vulkan requires a byte range for the backing descriptor.
        let size = if range.size != 0 {
            range.size
        } else if let Some(size) = pools.and_then(|p| p.remaining(range.address)) {
            size
        } else {
            return self.nvn_unmapped_storage(range.address, pipeline, input);
        };
        let limits = &self.residency.device().profile().limits;
        let prefix = range.address % limits.min_storage_buffer_offset_alignment;
        // Shader offsets are relative to the aligned descriptor base.
        self.nvn_buffer(
            Range {
                address: range.address - prefix,
                size: (size + prefix).min(u64::from(limits.max_storage_buffer_range)),
            },
            true,
        )
    }

    #[cold]
    fn nvn_unmapped_storage(
        &mut self,
        address: u64,
        pipeline: &shader::Program,
        input: &shader::Input,
    ) -> Result<HostBinding, VulkanError> {
        let stage = match *input {
            shader::Input::Storage(stage, _) | shader::Input::StoragePointer(stage, _) => stage,
            _ => unreachable!(),
        };
        let shader = &pipeline
            .usage
            .stages
            .iter()
            .find(|source| source.shader.stage == stage)
            .expect("storage pointer belongs to a linked stage")
            .shader;
        let memory = self.residency.device_memory();
        let mapped = memory.physical_address(address).is_some();
        if shader.isa != 1 || address >= 1 << memory.address_bits() || mapped {
            return Err(error(format_args!(
                "NVN storage address {address:#x} has no live pool extent (mapped={mapped}, shader={:#x}, input={input:?})",
                shader.address
            )));
        }
        // @SIMPLIFICATION An unsized pointer whose base page is unmapped uses a
        // null binding. Loads return zero and stores are discarded. Accesses
        // through offsets into later mapped pages require per-page translation.
        // https://github.com/seedds/nxnative/blob/a366365/dm/gfx/nvn/nvn_mtl.m#L13336-L13356
        if !self.state.nvn.reported_unmapped_storage {
            log::debug!(
                "NVN2 storage address {address:#x} is unmapped; using a null descriptor (shader={:#x}, input={input:?})",
                shader.address
            );
            self.state.nvn.reported_unmapped_storage = true;
        }
        Ok(HostBinding::NullBuffer)
    }
}
