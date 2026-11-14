//! Pipeline-local sampled selections. Validate live handles and descriptor bytes
//! before reusing physical views, including content and alias synchronization.
//! https://github.com/zortos293/Lichii/blob/646b5d17135a2b2701ccc7a959afd0bbbf2f6160/src/video_core/texture_cache/texture_cache.h
use super::reads::ReadSlot;
use super::*;
use crate::native::pipeline_resources::{HandleSource, ImageHandle, PipelineResources};
use crate::residency::execution::ShaderImage;
use nixe_shader_recompiler::{
    environment::Observations,
    types::{TexturePixelFormat, TextureType},
};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Handle {
    Constant {
        primary: ImageHandle,
        secondary: Option<ImageHandle>,
    },
    Immediate(u32),
}
struct Entry {
    stage: u8,
    handle: Handle,
    sampler: bool,
    changed: bool,
    primary: ReadSlot<4>,
    secondary: ReadSlot<4>,
    tic: ReadSlot<32>,
    tsc: ReadSlot<32>,
    indices: (u32, u32),
}
struct Check {
    entry: usize,
    texture: Option<TextureType>,
    format: Option<TexturePixelFormat>,
    integer: Option<bool>,
    matches: bool,
}
pub(super) struct SampledTextures {
    entries: Vec<Entry>,
    checks: Vec<Check>,
    pub images: smallvec::SmallVec<[ShaderImage; 16]>,
    identity: Option<[u64; 10]>,
    eligible: bool,
}
impl SampledTextures {
    pub fn new(
        resources: &PipelineResources,
        observations: &[(u8, Observations)],
    ) -> Result<Self, VulkanError> {
        let mut result = Self {
            entries: Vec::new(),
            checks: Vec::new(),
            images: Default::default(),
            identity: None,
            eligible: Self::eligible(resources),
        };
        for (stage, resources) in resources.stages[..5].iter().enumerate() {
            for image in &resources.images {
                let HandleSource::Direct { primary, secondary } = image.handle else {
                    return Err(compilation(
                        "native computed texture handles are not implemented",
                    ));
                };
                result.add_entry(
                    stage as u8,
                    Handle::Constant { primary, secondary },
                    image.sampler,
                );
            }
        }
        // Compilation can consume descriptors for instructions later eliminated by
        // optimization. Keep those dependencies even without a descriptor binding.
        for (stage, observations) in observations {
            for (key, old) in &observations.texture_handles {
                let handle = Handle::Constant {
                    primary: ImageHandle {
                        cbuf: key.index.0,
                        offset: key.offset.0,
                        shift: key.shift_left,
                    },
                    secondary: key.has_secondary.then_some(ImageHandle {
                        cbuf: key.sec_index.0,
                        offset: key.sec_offset.0,
                        shift: key.sec_shift_left,
                    }),
                };
                result.add_check(*stage, handle, *old, observations);
            }
            let immediate: std::collections::BTreeSet<_> = observations
                .texture_types
                .keys()
                .chain(observations.texture_pixel_formats.keys())
                .chain(observations.texture_integer_formats.keys())
                .copied()
                .collect();
            for old in immediate {
                if !observations
                    .texture_handles
                    .values()
                    .any(|value| *value == old)
                {
                    result.add_check(*stage, Handle::Immediate(old.0), old, observations);
                }
            }
        }
        Ok(result)
    }
    fn add_entry(&mut self, stage: u8, handle: Handle, sampler: bool) -> usize {
        let index = self.entries.len();
        self.entries.push(Entry {
            stage,
            handle,
            sampler,
            changed: true,
            primary: Default::default(),
            secondary: Default::default(),
            tic: Default::default(),
            tsc: Default::default(),
            indices: (0, 0),
        });
        index
    }
    fn add_check(
        &mut self,
        stage: u8,
        handle: Handle,
        old: nixe_shader_recompiler::types::TextureHandle,
        observations: &Observations,
    ) {
        let texture = observations.texture_types.get(&old).copied();
        let format = observations.texture_pixel_formats.get(&old).copied();
        let integer = observations.texture_integer_formats.get(&old).copied();
        if texture.is_none() && format.is_none() && integer.is_none() {
            return;
        }
        let entry = self
            .entries
            .iter()
            .position(|entry| entry.stage == stage && entry.handle == handle)
            .unwrap_or_else(|| self.add_entry(stage, handle, false));
        self.checks.push(Check {
            entry,
            texture,
            format,
            integer,
            matches: false,
        });
    }
    fn compatible(&mut self) -> Result<bool, VulkanError> {
        use nixe_gpu::nvidia::shader as classify;
        let mut compatible = true;
        for check in &mut self.checks {
            let entry = &self.entries[check.entry];
            if entry.changed {
                let words = std::array::from_fn(|i| {
                    u32::from_le_bytes(entry.tic.bytes[i * 4..i * 4 + 4].try_into().unwrap())
                });
                check.matches = true;
                if let Some(expected) = check.texture {
                    check.matches &=
                        classify::texture_type(words).map_err(compilation)? == expected;
                }
                if check.format.is_some() || check.integer.is_some() {
                    let format = classify::pixel_format(words).map_err(compilation)?;
                    check.matches &= check.format.is_none_or(|expected| format == expected)
                        && check
                            .integer
                            .is_none_or(|expected| classify::is_integer(format) == expected);
                }
            }
            compatible &= check.matches;
        }
        Ok(compatible)
    }

    fn identity(
        execution: &NativeExecution<'_>,
        memory: &impl RenderMemory,
        r: &Registers,
    ) -> [u64; 10] {
        [
            execution.state.current_context.unwrap(),
            memory.image_space_id(),
            memory.mapping_generation(),
            execution.residency.device_memory().generation(),
            execution.residency.texture_cache.binding_revision,
            r.address(0x1574, 0x1578),
            u64::from(r.get(0x157c)),
            r.address(0x155c, 0x1560),
            u64::from(r.get(0x1564)),
            u64::from(r.get(0x1234)),
        ]
    }
    pub fn eligible(resources: &PipelineResources) -> bool {
        resources.stages[..5].iter().all(|s| {
            s.images
                .iter()
                .all(|i| i.sampler && i.storage_format.is_none())
        })
    }
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        &mut self,
        execution: &mut NativeExecution<'_>,
        memory: &mut impl RenderMemory,
        r: &Registers,
        constants: &[[Option<(u64, u32)>; 32]; 5],
        groups: [Option<u8>; 5],
        resources: &PipelineResources,
    ) -> Result<Option<bool>, VulkanError> {
        let identity = Self::identity(execution, memory, r);
        let mut same = self.identity == Some(identity) && self.eligible;
        self.identity = None;
        let linked = textures::graphics_linked(r)?;
        execution.ampere_image_pools(memory.image_space_id(), r, false, linked);
        let tic_base = r.address(0x1574, 0x1578);
        let tsc_base = r.address(0x155c, 0x1560);
        for entry in &mut self.entries {
            let handle = match entry.handle {
                Handle::Constant { primary, secondary } => {
                    let group = groups[entry.stage as usize].expect("active descriptor stage");
                    let constants = &constants[group as usize];
                    same &= !entry.primary.update(
                        &mut execution.residency,
                        memory,
                        textures::constant_address(constants, primary)?,
                    )?;
                    let mut handle = u32::from_le_bytes(entry.primary.bytes) << primary.shift;
                    if let Some(secondary) = secondary {
                        same &= !entry.secondary.update(
                            &mut execution.residency,
                            memory,
                            textures::constant_address(constants, secondary)?,
                        )?;
                        handle |= u32::from_le_bytes(entry.secondary.bytes) << secondary.shift;
                    }
                    handle
                }
                Handle::Immediate(handle) => handle,
            };
            let (tic, tsc) = textures::indices(linked, handle);
            entry.indices = (tic, tsc);
            if tic_base == 0
                || tic > r.get(0x157c)
                || (entry.sampler
                    && (tsc_base == 0 || tsc > if linked { r.get(0x157c) } else { r.get(0x1564) }))
            {
                return Err(compilation(
                    "native texture handle exceeds descriptor pools",
                ));
            }
            entry.changed = entry.tic.update(
                &mut execution.residency,
                memory,
                tic_base
                    .checked_add(u64::from(tic) * 32)
                    .ok_or_else(|| compilation("descriptor address overflow"))?,
            )?;
            same &= !entry.changed;
            if entry.sampler {
                same &= !entry.tsc.update(
                    &mut execution.residency,
                    memory,
                    tsc_base
                        .checked_add(u64::from(tsc) * 32)
                        .ok_or_else(|| compilation("descriptor address overflow"))?,
                )?;
            }
        }
        if !self.compatible()? {
            return Ok(None);
        }
        same &= identity == Self::identity(execution, memory, r);
        if same {
            for image in &self.images {
                let Some(view) = image.view else {
                    same = false;
                    break;
                };
                if !execution.residency.sampled_image_ready(view)
                    && execution.prepare_shader_texture_view(view)? != view
                {
                    same = false;
                    break;
                }
                if identity != Self::identity(execution, memory, r)
                    || execution
                        .residency
                        .texture_cache
                        .bindings
                        .has_deleted_images
                {
                    same = false;
                    break;
                }
            }
        }
        if !same {
            self.images.clear();
            let mut index = 0;
            for (stage, group) in groups.into_iter().enumerate() {
                if group.is_none() {
                    continue;
                }
                for slot in &resources.stages[stage].images {
                    let entry = &self.entries[index];
                    index += 1;
                    let Some(image) = execution.resolve_ampere_texture(
                        memory,
                        slot,
                        false,
                        entry.indices.0,
                        entry.indices.1,
                        entry.tic.bytes,
                        slot.sampler.then_some(entry.tsc.bytes),
                    )?
                    else {
                        return Ok(Some(false));
                    };
                    self.images.push(image);
                }
            }
        }
        self.identity = Some(Self::identity(execution, memory, r));
        Ok(Some(same))
    }
}
