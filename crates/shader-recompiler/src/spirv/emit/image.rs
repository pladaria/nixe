use super::image_operands::Operands;
use super::*;
use crate::{
    ir::TextureInstInfo,
    types::{ImageFormat, Stage, TextureType as T},
};
use spirv::{ImageOperands as M, Scope, StorageClass};
impl EmitContext<'_> {
    pub(super) fn image_instruction(&mut self, ir: &mut Ir, id: InstId) -> Result<Option<Id>> {
        let opcode = ir.inst(id)?.opcode();
        if matches!(opcode, O::IsTextureScaled | O::IsImageScaled) {
            return Ok(Some(self.is_scaled(
                ir,
                ir.inst(id)?.arg(0)?,
                opcode == O::IsImageScaled,
            )?));
        }
        if !matches!(
            opcode,
            O::ImageSampleImplicitLod
                | O::ImageSampleExplicitLod
                | O::ImageSampleDrefImplicitLod
                | O::ImageSampleDrefExplicitLod
                | O::ImageGather
                | O::ImageGatherDref
                | O::ImageFetch
                | O::ImageQueryDimensions
                | O::ImageQueryLod
                | O::ImageGradient
                | O::ImageRead
                | O::ImageWrite
        ) {
            return self.image_atomic(ir, id);
        }
        let info = TextureInstInfo(ir.inst(id)?.flags);
        let index = ir.inst(id)?.arg(0)?;
        let coord_value = ir.inst(id)?.arg(1)?;
        let mut coords = self.def(ir, coord_value)?;
        let f = self.types.f32[0];
        let f4 = self.types.f32[3];
        let u4 = self.types.u32[3];
        if opcode == O::ImageRead
            && info.image_format() == ImageFormat::Typeless
            && !self.profile.support_typeless_image_loads
        {
            return Ok(Some(self.builder.constant_null(u4)?));
        }
        if opcode == O::ImageQueryDimensions {
            let (image, _) = self.image_object(ir, info, index, false, true)?;
            let skip = match ir.resolve(ir.inst(id)?.arg(2)?)? {
                Value::U1(value) => value,
                _ => return Err(invariant("Mip query skip flag must be immediate")),
            };
            let def = self.image_definition(info, false)?;
            let uses_lod = !def.is_multisample && info.texture_type() != T::Buffer;
            let n = match info.texture_type() {
                T::Color1D | T::Buffer => 1,
                T::ColorArray2D | T::Color3D | T::ColorArrayCube => 3,
                _ => 2,
            };
            let size = if uses_lod {
                self.op(
                    Op::ImageQuerySizeLod,
                    self.types.u32[n - 1],
                    &[image, coords],
                )?
            } else {
                self.op(Op::ImageQuerySize, self.types.u32[n - 1], &[image])?
            };
            let mips = if skip {
                self.constants.u32_zero
            } else {
                self.op(Op::ImageQueryLevels, self.types.u32[0], &[image])?
            };
            let mut values = vec![size];
            for _ in n..3 {
                values.push(self.constants.u32_zero);
            }
            values.push(mips);
            return Ok(Some(self.op(Op::CompositeConstruct, u4, &values)?));
        }
        if opcode == O::ImageQueryLod {
            let (sampler, _) = self.image_object(ir, info, index, false, false)?;
            let lod = self.op(Op::ImageQueryLod, self.types.f32[1], &[sampler, coords])?;
            return Ok(Some(self.op(
                Op::CompositeConstruct,
                f4,
                &[lod, self.constants.f32_zero, self.constants.f32_zero],
            )?));
        }
        let storage = matches!(opcode, O::ImageRead | O::ImageWrite);
        let raw = storage || opcode == O::ImageFetch;
        let (object, integer) = self.image_object(ir, info, index, storage, raw)?;
        let dref = matches!(
            opcode,
            O::ImageSampleDrefImplicitLod | O::ImageSampleDrefExplicitLod
        );
        let ty = if dref {
            f
        } else if integer {
            u4
        } else {
            f4
        };
        let mut args = vec![object.0, coords.0];
        let mut operands = Operands::default();
        let mut required = false;
        let op = match opcode {
            O::ImageSampleImplicitLod
            | O::ImageSampleExplicitLod
            | O::ImageSampleDrefImplicitLod
            | O::ImageSampleDrefExplicitLod => {
                let implicit = matches!(
                    opcode,
                    O::ImageSampleImplicitLod | O::ImageSampleDrefImplicitLod
                );
                if dref {
                    args.push(self.def(ir, ir.inst(id)?.arg(2)?)?.0);
                }
                let bias_lod = self.def(ir, ir.inst(id)?.arg(if dref { 3 } else { 2 })?)?;
                let offset = ir.inst(id)?.arg(if dref { 4 } else { 3 })?;
                let actual_implicit = implicit && self.stage == Stage::Fragment;
                let lod = if implicit && !actual_implicit {
                    self.constants.f32_zero
                } else {
                    bias_lod
                };
                let clamp = implicit && info.has_lod_clamp() && (!dref || actual_implicit);
                operands = Operands::sample(
                    self,
                    ir,
                    actual_implicit && info.has_bias(),
                    !actual_implicit,
                    clamp,
                    lod,
                    offset,
                )?;
                required = !actual_implicit;
                match (dref, actual_implicit) {
                    (false, true) => Op::ImageSampleImplicitLod,
                    (false, false) => Op::ImageSampleExplicitLod,
                    (true, true) => Op::ImageSampleDrefImplicitLod,
                    (true, false) => Op::ImageSampleDrefExplicitLod,
                }
            }
            O::ImageGather | O::ImageGatherDref => {
                operands = Operands::gather(self, ir, ir.inst(id)?.arg(2)?, ir.inst(id)?.arg(3)?)?;
                if self.profile.need_gather_subpixel_offset {
                    let (image, _) = self.image_object(ir, info, index, false, true)?;
                    coords = self.gather_coords(info, image, coords)?;
                    args[1] = coords.0;
                }
                args.push(
                    if opcode == O::ImageGather {
                        self.const_u32(info.gather_component())?
                    } else {
                        self.def(ir, ir.inst(id)?.arg(4)?)?
                    }
                    .0,
                );
                if opcode == O::ImageGather {
                    Op::ImageGather
                } else {
                    Op::ImageDrefGather
                }
            }
            O::ImageFetch => {
                let offset = self.def(ir, ir.inst(id)?.arg(2)?)?;
                coords = self.fetch_coords(info, coords, offset)?;
                args[1] = coords.0;
                let lod = self.def(ir, ir.inst(id)?.arg(3)?)?;
                let ms = self.def(ir, ir.inst(id)?.arg(4)?)?;
                if info.texture_type() != T::Buffer && ms == Id::NONE && lod != Id::NONE {
                    operands.add(M::LOD, &[lod]);
                }
                if ms != Id::NONE {
                    operands.add(M::SAMPLE, &[ms]);
                }
                Op::ImageFetch
            }
            O::ImageGradient => {
                let derivatives = self.def(ir, ir.inst(id)?.arg(2)?)?;
                let offset = ir.inst(id)?.arg(3)?;
                let clamp = self.def(ir, ir.inst(id)?.arg(4)?)?;
                let n = info.num_derivatives() as usize;
                if !(1..=3).contains(&n) || derivatives == Id::NONE {
                    return Err(invariant("Invalid texture derivatives"));
                }
                let second = if n == 3 {
                    self.def(ir, offset)?
                } else {
                    Id::NONE
                };
                if n == 3 && second == Id::NONE {
                    return Err(invariant("Missing third texture derivative pair"));
                }
                let mut x = [Id::NONE; 3];
                let mut y = [Id::NONE; 3];
                for i in 0..n {
                    let vector = if i == 2 { second } else { derivatives };
                    let component = if i == 2 { 0 } else { i as u32 * 2 };
                    x[i] = self.extract(f, vector, component)?;
                    y[i] = self.extract(f, vector, component + 1)?;
                }
                let x = self.op(Op::CompositeConstruct, self.types.f32[n - 1], &x[..n])?;
                let y = self.op(Op::CompositeConstruct, self.types.f32[n - 1], &y[..n])?;
                operands.add(M::GRAD, &[x, y]);
                if n != 3 {
                    operands.offset(self, ir, offset, false)?;
                }
                if info.has_lod_clamp() {
                    operands.add(M::MIN_LOD, &[clamp]);
                }
                required = true;
                Op::ImageSampleExplicitLod
            }
            O::ImageRead => Op::ImageRead,
            O::ImageWrite => {
                let mut color = self.def(ir, ir.inst(id)?.arg(2)?)?;
                if !integer {
                    color = self.op(Op::Bitcast, f4, &[color])?;
                }
                args.push(color.0);
                self.builder.instruction(Op::ImageWrite, &args)?;
                return Ok(Some(Id::NONE));
            }
            _ => unreachable!(),
        };
        operands.append(&mut args, required);
        let mut result = self.image_result(ir, id, op, ty, &args)?;
        if cfg!(target_os = "android")
            && self.fix_bloom_effects
            && opcode == O::ImageSampleExplicitLod
            && !integer
        {
            let scale = self.const_f32(0.98)?;
            result = self.op(Op::VectorTimesScalar, f4, &[result, scale])?;
        }
        Ok(Some(if opcode == O::ImageRead && !integer {
            self.op(Op::Bitcast, u4, &[result])?
        } else if !storage && !dref && integer {
            self.op(Op::Bitcast, f4, &[result])?
        } else {
            result
        }))
    }
    fn image_atomic(&mut self, ir: &Ir, id: InstId) -> Result<Option<Id>> {
        let atomic = match ir.inst(id)?.opcode() {
            O::ImageAtomicIAdd32 => Op::AtomicIAdd,
            O::ImageAtomicSMin32 => Op::AtomicSMin,
            O::ImageAtomicUMin32 => Op::AtomicUMin,
            O::ImageAtomicSMax32 => Op::AtomicSMax,
            O::ImageAtomicUMax32 => Op::AtomicUMax,
            O::ImageAtomicAnd32 => Op::AtomicAnd,
            O::ImageAtomicOr32 => Op::AtomicOr,
            O::ImageAtomicXor32 => Op::AtomicXor,
            O::ImageAtomicExchange32 => Op::AtomicExchange,
            _ => return Ok(None),
        };
        if ir.resolve(ir.inst(id)?.arg(0)?)? != Value::U32(0) {
            return Err(unsupported("Image atomic indexing requires immediate zero"));
        }
        let info = TextureInstInfo(ir.inst(id)?.flags);
        let def = self.image_definition(info, true)?;
        let coords = self.def(ir, ir.inst(id)?.arg(1)?)?;
        let value = self.def(ir, ir.inst(id)?.arg(2)?)?;
        let pointer = self.op(
            Op::ImageTexelPointer,
            self.objects.image_u32,
            &[def.id, coords, self.constants.u32_zero],
        )?;
        let scope = self.const_u32(Scope::Device as u32)?;
        Ok(Some(self.op(
            atomic,
            self.types.u32[0],
            &[pointer, scope, self.constants.u32_zero, value],
        )?))
    }
    fn is_scaled(&mut self, ir: &Ir, index: Value, image: bool) -> Result<Id> {
        let u = self.types.u32[0];
        let bit = if self.profile.unified_descriptor_binding {
            let base = if image {
                self.image_scaling_index
            } else {
                self.texture_scaling_index
            };
            let member = self.const_u32(u32::from(image))?;
            let (word_index, bit_mask) = if let Value::U32(index) = ir.resolve(index)? {
                let index = index.wrapping_add(base);
                (
                    self.const_u32(index / 32)?,
                    self.const_u32(1 << (index % 32))?,
                )
            } else {
                let mut index = self.def(ir, index)?;
                if base != 0 {
                    let base = self.const_u32(base)?;
                    index = self.op(Op::IAdd, u, &[index, base])?;
                }
                let shift = self.const_u32(5)?;
                let word = self.op(Op::ShiftRightLogical, u, &[index, shift])?;
                let mask = self.const_u32(31)?;
                let bit = self.op(Op::BitwiseAnd, u, &[index, mask])?;
                let one = self.const_u32(1)?;
                (word, self.op(Op::ShiftLeftLogical, u, &[one, bit])?)
            };
            let pointer_ty = self.builder.type_pointer(StorageClass::PushConstant, u)?;
            let pointer = self.op(
                Op::AccessChain,
                pointer_ty,
                &[self.objects.rescaling_push_constants, member, word_index],
            )?;
            let word = self.op(Op::Load, u, &[pointer])?;
            self.op(Op::BitwiseAnd, u, &[word, bit_mask])?
        } else {
            let composite = self.op(
                Op::Load,
                self.types.f32[3],
                &[self.objects.rescaling_uniform_constant],
            )?;
            let mask = self.extract(self.types.f32[0], composite, u32::from(image))?;
            let mask = self.op(Op::Bitcast, u, &[mask])?;
            let index = self.def(ir, index)?;
            let shifted = self.op(Op::ShiftRightLogical, u, &[mask, index])?;
            let one = self.const_u32(1)?;
            self.op(Op::BitwiseAnd, u, &[shifted, one])?
        };
        self.op(
            Op::INotEqual,
            self.types.bool,
            &[bit, self.constants.u32_zero],
        )
    }
}
