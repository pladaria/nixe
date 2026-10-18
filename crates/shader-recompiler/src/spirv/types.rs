use super::{Builder, Id};
use crate::{
    CompileError, Profile, Result, artifact::ShaderInfo, diagnostics::FailureCategory, ir::Type,
};
use spirv::{Capability, StorageClass};

#[derive(Clone, Debug, Default)]
pub struct CommonTypes {
    pub void: Id,
    pub bool: Id,
    pub u8: Id,
    pub s8: Id,
    pub u16: Id,
    pub s16: Id,
    pub u64: Id,
    pub u32: [Id; 4],
    pub s32: [Id; 4],
    pub f16: [Id; 4],
    pub f32: [Id; 4],
    pub f64: [Id; 4],
    pub private_u32: Id,
    pub input_f32: Id,
    pub input_u32: Id,
    pub input_s32: Id,
    pub output_f32: Id,
    pub output_u32: Id,
}
fn vectors(b: &mut Builder, base: Id) -> Result<[Id; 4]> {
    Ok([
        base,
        b.type_vector(base, 2)?,
        b.type_vector(base, 3)?,
        b.type_vector(base, 4)?,
    ])
}
impl CommonTypes {
    pub fn define(b: &mut Builder, profile: &Profile, info: &ShaderInfo) -> Result<Self> {
        let void = b.type_void()?;
        let bool = b.type_bool()?;
        let f32 = b.type_float(32)?;
        let f32 = vectors(b, f32)?;
        let u32 = b.type_int(32, false)?;
        let u32 = vectors(b, u32)?;
        let s32 = b.type_int(32, true)?;
        let s32 = vectors(b, s32)?;
        let mut t = Self {
            void,
            bool,
            f32,
            u32,
            s32,
            private_u32: b.type_pointer(StorageClass::Private, u32[0])?,
            input_f32: b.type_pointer(StorageClass::Input, f32[0])?,
            input_u32: b.type_pointer(StorageClass::Input, u32[0])?,
            input_s32: b.type_pointer(StorageClass::Input, s32[0])?,
            output_f32: b.type_pointer(StorageClass::Output, f32[0])?,
            output_u32: b.type_pointer(StorageClass::Output, u32[0])?,
            ..Default::default()
        };
        if info.uses_int8 && profile.support_int8 {
            b.capability(Capability::Int8)?;
            t.u8 = b.type_int(8, false)?;
            t.s8 = b.type_int(8, true)?;
        }
        if info.uses_int16 && profile.support_int16 {
            b.capability(Capability::Int16)?;
            t.u16 = b.type_int(16, false)?;
            t.s16 = b.type_int(16, true)?;
        }
        if info.uses_int64 && profile.support_int64 {
            b.capability(Capability::Int64)?;
            t.u64 = b.type_int(64, false)?;
        }
        if info.uses_fp16 {
            b.capability(Capability::Float16)?;
            let base = b.type_float(16)?;
            t.f16 = vectors(b, base)?;
        }
        if info.uses_fp64 {
            b.capability(Capability::Float64)?;
            let base = b.type_float(64)?;
            t.f64 = vectors(b, base)?;
        }
        Ok(t)
    }
    pub fn get(&self, ty: Type) -> Result<Id> {
        let id = match ty {
            Type::Void => self.void,
            Type::U1 => self.bool,
            Type::U8 => self.u8,
            Type::U16 => self.u16,
            Type::U64 => self.u64,
            Type::U32 => self.u32[0],
            Type::U32x2 => self.u32[1],
            Type::U32x3 => self.u32[2],
            Type::U32x4 => self.u32[3],
            Type::F16 => self.f16[0],
            Type::F16x2 => self.f16[1],
            Type::F16x3 => self.f16[2],
            Type::F16x4 => self.f16[3],
            Type::F32 => self.f32[0],
            Type::F32x2 => self.f32[1],
            Type::F32x3 => self.f32[2],
            Type::F32x4 => self.f32[3],
            Type::F64 => self.f64[0],
            Type::F64x2 => self.f64[1],
            Type::F64x3 => self.f64[2],
            Type::F64x4 => self.f64[3],
            _ => Id::NONE,
        };
        if id == Id::NONE {
            Err(CompileError::new(
                FailureCategory::Unsupported,
                format!("SPIR-V type {ty:?} was not declared"),
            ))
        } else {
            Ok(id)
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub struct CommonConstants {
    pub true_value: Id,
    pub false_value: Id,
    pub u32_zero: Id,
    pub f32_zero: Id,
}
impl CommonConstants {
    pub fn define(b: &mut Builder, t: &CommonTypes) -> Result<Self> {
        Ok(Self {
            true_value: b.constant_bool(t.bool, true)?,
            false_value: b.constant_bool(t.bool, false)?,
            u32_zero: b.constant(t.u32[0], &[0])?,
            f32_zero: b.constant(t.f32[0], &[0])?,
        })
    }
}
