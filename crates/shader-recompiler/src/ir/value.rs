use super::{Attribute, Type};
use crate::{CompileError, Result, arena::InstId, diagnostics::FailureCategory};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Reg(pub u8);

impl Reg {
    pub const RZ: Self = Self(255);
    pub const fn index(self) -> usize {
        self.0 as usize
    }
    pub const fn is_aligned(self, alignment: u8) -> bool {
        self.0 == 255 || self.0.is_multiple_of(alignment)
    }
    pub fn offset(self, offset: i32) -> Result<Self> {
        if self == Self::RZ {
            return Ok(Self::RZ);
        }
        let result = i64::from(self.0) + i64::from(offset);
        if !(0..255).contains(&result) {
            return Err(CompileError::new(
                FailureCategory::Internal,
                format!("Register arithmetic out of range: R{} + {offset}", self.0),
            ));
        }
        Ok(Self(result as u8))
    }
}

pub const NUM_REGS: usize = 256;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Pred(u8);

impl Pred {
    pub const PT: Self = Self(7);
    pub fn new(index: u8) -> Result<Self> {
        if index <= 7 {
            Ok(Self(index))
        } else {
            Err(CompileError::new(
                FailureCategory::InvalidArgument,
                "Predicate index exceeds 7",
            ))
        }
    }
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Patch(u8);

#[allow(non_upper_case_globals)]
impl Patch {
    pub const TessellationLodLeft: Self = Self(0);
    pub const TessellationLodTop: Self = Self(1);
    pub const TessellationLodRight: Self = Self(2);
    pub const TessellationLodBottom: Self = Self(3);
    pub const TessellationLodInteriorU: Self = Self(4);
    pub const TessellationLodInteriorV: Self = Self(5);

    pub fn new(index: u8) -> Result<Self> {
        if index <= 125 {
            Ok(Self(index))
        } else {
            Err(CompileError::new(
                FailureCategory::InvalidArgument,
                "Patch index exceeds 125",
            ))
        }
    }

    pub const fn index(self) -> usize {
        self.0 as usize
    }
    pub const fn is_generic(self) -> bool {
        self.0 >= 6
    }
    pub fn generic_index(self) -> Result<u32> {
        if self.is_generic() {
            Ok(u32::from(self.0 - 6) / 4)
        } else {
            Err(CompileError::new(
                FailureCategory::InvalidArgument,
                "Tessellation LOD is not a generic patch",
            ))
        }
    }
    pub fn generic_element(self) -> Result<u32> {
        self.generic_index().map(|_| u32::from(self.0 - 6) % 4)
    }
}

/// Floating immediates retain exact bits, including signed zero and NaN payloads.
/// None is an absent operand; Undef is represented by its own IR instruction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Value {
    #[default]
    None,
    Inst(InstId),
    Reg(Reg),
    Pred(Pred),
    Attribute(Attribute),
    Patch(Patch),
    U1(bool),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    F32(u32),
    F64(u64),
}

impl Value {
    pub const fn f32(value: f32) -> Self {
        Self::F32(value.to_bits())
    }
    pub const fn f64(value: f64) -> Self {
        Self::F64(value.to_bits())
    }
    pub const fn is_empty(self) -> bool {
        matches!(self, Self::None)
    }
    pub const fn raw_type(self) -> Type {
        match self {
            Self::None => Type::Void,
            Self::Inst(_) => Type::Opaque,
            Self::Reg(_) => Type::Reg,
            Self::Pred(_) => Type::Pred,
            Self::Attribute(_) => Type::Attribute,
            Self::Patch(_) => Type::Patch,
            Self::U1(_) => Type::U1,
            Self::U8(_) => Type::U8,
            Self::U16(_) => Type::U16,
            Self::U32(_) => Type::U32,
            Self::U64(_) => Type::U64,
            Self::F32(_) => Type::F32,
            Self::F64(_) => Type::F64,
        }
    }
}

/// A scalar/vector type constraint checked once when a raw value enters a typed
/// emitter helper. The value's bits and node identity are never copied deeply.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TypedValue<const MASK: u32>(Value);

impl<const MASK: u32> TypedValue<MASK> {
    pub fn new(value: Value, ir: &super::Ir) -> Result<Self> {
        let ty = ir.value_type(value)?;
        if ty.0 & MASK == 0 {
            return Err(CompileError::new(
                FailureCategory::InvalidArgument,
                format!("Value type {ty:?} is incompatible with mask {MASK:#x}"),
            ));
        }
        Ok(Self(value))
    }
    pub const fn value(self) -> Value {
        self.0
    }
}

pub type U1 = TypedValue<{ Type::U1.0 }>;
pub type U8 = TypedValue<{ Type::U8.0 }>;
pub type U16 = TypedValue<{ Type::U16.0 }>;
pub type U32 = TypedValue<{ Type::U32.0 }>;
pub type U64 = TypedValue<{ Type::U64.0 }>;
pub type F16 = TypedValue<{ Type::F16.0 }>;
pub type F32 = TypedValue<{ Type::F32.0 }>;
pub type F64 = TypedValue<{ Type::F64.0 }>;
pub type U32U64 = TypedValue<{ Type::U32.0 | Type::U64.0 }>;
pub type F32F64 = TypedValue<{ Type::F32.0 | Type::F64.0 }>;
pub type F16F32F64 = TypedValue<{ Type::F16.0 | Type::F32.0 | Type::F64.0 }>;
pub type U8U16U32U64 = TypedValue<{ Type::U8.0 | Type::U16.0 | Type::U32.0 | Type::U64.0 }>;
pub type U16U32U64 = TypedValue<{ Type::U16.0 | Type::U32.0 | Type::U64.0 }>;
pub type UAny = U8U16U32U64;
