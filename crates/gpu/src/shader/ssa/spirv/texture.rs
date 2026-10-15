//! Separate image/sampler descriptors; live logical identities acquire dense
//! set-zero bindings only after SSA optimization.
//! https://docs.vulkan.org/spec/latest/chapters/textures.html
use super::*;
use crate::ShaderResourceKind as K;

pub(super) struct Descriptor {
    variable: u32,
    ty: u32,
    kind: K,
}
impl Emitter<'_> {
    pub(super) fn reflect_texture(&mut self, instruction: &Instruction) -> Result<(), Error> {
        let (image, sampler, kind) = match instruction.op {
            Op::TextureSample {
                image,
                sampler,
                kind,
                ..
            } => (image, sampler, kind),
            Op::TextureLevels { image } => {
                let kind = self
                    .function
                    .resources
                    .iter()
                    .find(|r| r.id == image)
                    .ok_or_else(|| {
                        Error::at(
                            instruction.source,
                            "image query lacks a resource declaration",
                        )
                    })?
                    .kind;
                (image, None, kind)
            }
            _ => return Ok(()),
        };
        self.math.source = instruction.source;
        self.descriptor(image, kind)?;
        if let Some(sampler) = sampler {
            let kind = if matches!(
                instruction.op,
                Op::TextureSample {
                    mode: SampleMode::Compare | SampleMode::CompareLevelZero,
                    ..
                }
            ) {
                K::ComparisonSampler
            } else {
                K::Sampler
            };
            self.descriptor(sampler, kind)?;
        }
        Ok(())
    }
    fn descriptor(&mut self, resource: ResourceId, kind: K) -> Result<(), Error> {
        if let Some(old) = self.descriptors.get(&resource) {
            if old.kind != kind {
                return Err(self.error("logical resource has conflicting descriptor classes"));
            }
            return Ok(());
        }
        if self.buffers.contains_key(&resource) {
            return Err(self.error("image/sampler identity aliases a buffer descriptor"));
        }
        let declared = self
            .function
            .resources
            .iter()
            .find(|r| r.id == resource)
            .ok_or_else(|| self.error("sampled resource has no frontend declaration"))?;
        if declared.kind != kind {
            return Err(self.error("sampled operation disagrees with its declared resource class"));
        }
        let ty = match kind {
            K::Sampler | K::ComparisonSampler => self.math.b.type_sampler(),
            K::SampledImage
            | K::SampledImage2DArray
            | K::SampledDepthImage
            | K::SampledDepthImage2DArray
            | K::SampledImageCube => {
                let depth = u32::from(matches!(
                    kind,
                    K::SampledDepthImage | K::SampledDepthImage2DArray
                ));
                let array = u32::from(matches!(
                    kind,
                    K::SampledImage2DArray | K::SampledDepthImage2DArray
                ));
                self.math.b.type_image(
                    self.math.float,
                    if kind == K::SampledImageCube {
                        spv::Dim::DimCube
                    } else {
                        spv::Dim::Dim2D
                    },
                    depth,
                    array,
                    0,
                    1,
                    spv::ImageFormat::Unknown,
                    None,
                )
            }
            _ => return Err(self.error("unsupported sampled descriptor class")),
        };
        let ptr = self
            .math
            .b
            .type_pointer(None, spv::StorageClass::UniformConstant, ty);
        let variable = self
            .math
            .b
            .variable(ptr, None, spv::StorageClass::UniformConstant, None);
        let binding = self.binding(resource);
        self.math.b.decorate(
            variable,
            spv::Decoration::DescriptorSet,
            [SpvOperand::LiteralBit32(0)],
        );
        self.math.b.decorate(
            variable,
            spv::Decoration::Binding,
            [SpvOperand::LiteralBit32(binding)],
        );
        self.io.variables.push(variable);
        self.bindings.push(Binding {
            resource,
            binding,
            kind,
            readable: true,
            writable: false,
        });
        self.descriptors
            .insert(resource, Descriptor { variable, ty, kind });
        Ok(())
    }
    pub(super) fn texture(&mut self, inst: &Instruction, args: &[u32]) -> Result<(), Error> {
        let image = match inst.op {
            Op::TextureSample { image, .. } | Op::TextureLevels { image } => image,
            _ => unreachable!(),
        };
        let descriptor = self
            .descriptors
            .get(&image)
            .ok_or_else(|| self.error("live image descriptor is missing"))?;
        let image_ty = descriptor.ty;
        let image = self
            .math
            .b
            .load(image_ty, None, descriptor.variable, None, [])
            .map_err(|e| self.error(e.to_string()))?;
        if matches!(inst.op, Op::TextureLevels { .. }) {
            self.math.b.capability(spv::Capability::ImageQuery);
            let levels = self
                .math
                .b
                .image_query_levels(self.math.uint, None, image)
                .map_err(|e| self.error(e.to_string()))?;
            self.result(inst.results[0], levels);
            return Ok(());
        }
        let Op::TextureSample {
            sampler,
            kind,
            mode,
            offset,
            ..
        } = inst.op
        else {
            unreachable!()
        };
        let array = matches!(kind, K::SampledImage2DArray | K::SampledDepthImage2DArray);
        let count = if kind == K::SampledImageCube { 3 } else { 2 };
        let fetch = matches!(mode, SampleMode::Fetch(_));
        if fetch && kind == K::SampledImageCube {
            return Err(self.error("cube texel fetch is not a guest-supported operation"));
        }
        let scalar = if fetch {
            self.math.int
        } else {
            self.math.float
        };
        let mut coordinates = args[..count].to_vec();
        if array {
            let layer = if fetch {
                self.math.b.bitcast(self.math.int, None, args[count])
            } else {
                self.math
                    .b
                    .convert_u_to_f(self.math.float, None, args[count])
            }
            .map_err(|e| self.error(e.to_string()))?;
            coordinates.push(layer);
        }
        let coordinate_ty = self.math.b.type_vector(scalar, coordinates.len() as u32);
        let coordinate = self
            .math
            .b
            .composite_construct(coordinate_ty, None, coordinates)
            .map_err(|e| self.error(e.to_string()))?;
        let vector = self.math.b.type_vector(self.math.float, 4);
        let explicit = matches!(
            mode,
            SampleMode::Explicit
                | SampleMode::LevelZero
                | SampleMode::CompareLevelZero
                | SampleMode::Fetch(_)
        );
        let compare = matches!(mode, SampleMode::Compare | SampleMode::CompareLevelZero);
        if !explicit && self.function.stage != ShaderStage::Fragment {
            return Err(self.error("implicit derivatives require fragment stage"));
        }
        let mut flags = spv::ImageOperands::NONE;
        let mut operands = Vec::new();
        if explicit {
            let lod = match mode {
                SampleMode::Fetch(level) => self.math.b.constant_bit32(self.math.int, level),
                SampleMode::Explicit => args[count + usize::from(array)],
                _ => self.math.b.constant_bit32(self.math.float, 0),
            };
            flags |= spv::ImageOperands::LOD;
            operands.push(SpvOperand::IdRef(lod));
        }
        if let Some([x, y]) = offset {
            if kind == K::SampledImageCube {
                return Err(self.error("cube sample offsets are invalid"));
            }
            let ty = self.math.b.type_vector(self.math.int, 2);
            let x = self.math.b.constant_bit32(self.math.int, x as u32);
            let y = self.math.b.constant_bit32(self.math.int, y as u32);
            let offset = self.math.b.constant_composite(ty, [x, y]);
            flags |= spv::ImageOperands::CONST_OFFSET;
            operands.push(SpvOperand::IdRef(offset));
        }
        let result = if fetch {
            self.math
                .b
                .image_fetch(vector, None, image, coordinate, Some(flags), operands)
        } else {
            let sampler = sampler
                .and_then(|s| self.descriptors.get(&s))
                .ok_or_else(|| self.error("sample has no live sampler"))?;
            let sampler = self
                .math
                .b
                .load(sampler.ty, None, sampler.variable, None, [])
                .map_err(|e| self.error(e.to_string()))?;
            let sampled_ty = self.math.b.type_sampled_image(image_ty);
            let sampled = self
                .math
                .b
                .sampled_image(sampled_ty, None, image, sampler)
                .map_err(|e| self.error(e.to_string()))?;
            if compare {
                let reference = args[count + usize::from(array)];
                if explicit {
                    self.math.b.image_sample_dref_explicit_lod(
                        self.math.float,
                        None,
                        sampled,
                        coordinate,
                        reference,
                        flags,
                        operands,
                    )
                } else {
                    self.math.b.image_sample_dref_implicit_lod(
                        self.math.float,
                        None,
                        sampled,
                        coordinate,
                        reference,
                        if flags.is_empty() { None } else { Some(flags) },
                        operands,
                    )
                }
            } else if explicit {
                self.math
                    .b
                    .image_sample_explicit_lod(vector, None, sampled, coordinate, flags, operands)
            } else {
                self.math.b.image_sample_implicit_lod(
                    vector,
                    None,
                    sampled,
                    coordinate,
                    if flags.is_empty() { None } else { Some(flags) },
                    operands,
                )
            }
        }
        .map_err(|e| self.error(e.to_string()))?;
        let depth = matches!(kind, K::SampledDepthImage | K::SampledDepthImage2DArray);
        for component in 0..4 {
            let value = if compare {
                result
            } else if depth && component == 3 {
                self.math
                    .b
                    .constant_bit32(self.math.float, 1.0f32.to_bits())
            } else {
                self.math
                    .b
                    .composite_extract(
                        self.math.float,
                        None,
                        result,
                        [if depth { 0 } else { component as u32 }],
                    )
                    .map_err(|e| self.error(e.to_string()))?
            };
            self.result(inst.results[component], value);
        }
        Ok(())
    }
}
