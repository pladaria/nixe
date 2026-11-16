//! Persistent pipeline binding slots consume channel-owned live descriptor records.
//! Update changed selections and synchronize contents independently of bindings.
//! https://github.com/zortos293/Lichii/blob/646b5d17135a2b2701ccc7a959afd0bbbf2f6160/src/video_core/texture_cache/texture_cache.h
use super::reads::ReadSlot;
use super::*;
use crate::native::pipeline_resources::{HandleSource, ImageHandle, PipelineResources};
use crate::native::sampled::{ImageBindings, SampledKey};
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
    tic_revision: u64,
    tsc_revision: u64,
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
    pub bindings: ImageBindings,
    eligible: bool,
}
impl SampledTextures {
    pub fn new<'b>(
        resources: &PipelineResources,
        observations: impl IntoIterator<Item = (u8, &'b Observations)>,
    ) -> Result<Self, VulkanError> {
        let mut result = Self {
            entries: Vec::new(),
            checks: Vec::new(),
            bindings: ImageBindings::new(resources),
            eligible: Self::eligible(resources),
        };
        for (stage, resources) in resources.stages.iter().enumerate() {
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
                result.add_check(stage, handle, *old, observations);
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
                    result.add_check(stage, Handle::Immediate(old.0), old, observations);
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
            tic_revision: 0,
            tsc_revision: 0,
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
    fn compatible(
        &mut self,
        execution: &NativeExecution<'_>,
        compute: bool,
    ) -> Result<bool, VulkanError> {
        use nixe_gpu::nvidia::shader as classify;
        let mut compatible = true;
        for check in &mut self.checks {
            let entry = &self.entries[check.entry];
            if entry.changed {
                let bytes = &execution
                    .residency
                    .texture_cache
                    .bindings
                    .channel
                    .image_descriptors[usize::from(compute)]
                .entries[entry.indices.0 as usize]
                    .bytes;
                let words = std::array::from_fn(|i| {
                    u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap())
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
        compute: bool,
    ) -> [u64; 6] {
        execution.sampled_scope(
            [
                memory.image_space_id(),
                memory.mapping_generation(),
                execution.residency.device_memory().generation(),
            ],
            compute,
        )
    }
    pub fn eligible(resources: &PipelineResources) -> bool {
        resources.stages.iter().all(|s| {
            s.texels.is_empty()
                && s.images
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
        self.prepare_bindings(
            execution,
            memory,
            |stage| &constants[groups[stage as usize].expect("active descriptor stage") as usize],
            resources,
            textures::graphics_linked(r)?,
            false,
        )
    }
    pub fn prepare_compute(
        &mut self,
        execution: &mut NativeExecution<'_>,
        memory: &mut impl RenderMemory,
        constants: &[Option<(u64, u32)>],
        resources: &PipelineResources,
        linked: bool,
    ) -> Result<Option<bool>, VulkanError> {
        self.prepare_bindings(execution, memory, |_| constants, resources, linked, true)
    }
    #[allow(clippy::too_many_arguments)]
    fn prepare_bindings<'b>(
        &mut self,
        execution: &mut NativeExecution<'_>,
        memory: &mut impl RenderMemory,
        constants: impl Fn(u8) -> &'b [Option<(u64, u32)>],
        resources: &PipelineResources,
        linked: bool,
        compute: bool,
    ) -> Result<Option<bool>, VulkanError> {
        let identity = Self::identity(execution, memory, compute);
        let force = self.bindings.begin(identity, self.eligible);
        let mut same = !force && self.eligible;
        for entry in &mut self.entries {
            let handle = match entry.handle {
                Handle::Constant { primary, secondary } => {
                    let constants = constants(entry.stage);
                    entry.primary.update(
                        &mut execution.residency,
                        memory,
                        textures::constant_address(constants, primary)?,
                    )?;
                    let mut handle = u32::from_le_bytes(entry.primary.bytes) << primary.shift;
                    if let Some(secondary) = secondary {
                        entry.secondary.update(
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
            let (_, revision) =
                textures::image_descriptor(&mut execution.residency, memory, compute, tic)?;
            entry.changed = force || entry.indices.0 != tic || entry.tic_revision != revision;
            entry.tic_revision = revision;
            if entry.sampler {
                let (_, revision) =
                    textures::sampler_descriptor(&mut execution.residency, memory, compute, tsc)?;
                entry.tsc_revision = revision;
            }
            entry.indices = (tic, tsc);
        }
        if !self.compatible(execution, compute)? {
            return Ok(None);
        }
        let force = force || identity != Self::identity(execution, memory, compute);
        let mut index = 0;
        for stage in &resources.stages {
            for slot in &stage.images {
                let entry = &self.entries[index];
                let key = SampledKey {
                    image: entry.indices.0,
                    sampler: slot.sampler.then_some(entry.indices.1),
                    image_revision: entry.tic_revision,
                    sampler_revision: entry.tsc_revision,
                };
                let Some(unchanged) =
                    self.bindings
                        .update(execution, index, key, force, |execution| {
                            execution.resolve_ampere_texture(
                                memory,
                                slot,
                                compute,
                                key.image,
                                key.sampler.unwrap_or(0),
                            )
                        })?
                else {
                    return Ok(Some(false));
                };
                same &= unchanged;
                index += 1;
            }
        }
        debug_assert_eq!(index, self.bindings.images.len());
        let current = Self::identity(execution, memory, compute);
        Ok(Some(self.bindings.finish(identity, current, same)))
    }
}
