//! Semantic SSA used by the native compiler. Guest register names, instruction
//! encodings and reconvergence stacks never enter this representation.
//!
//! Construction/optimization happen on shader cache misses. The verifier is an
//! explicit diagnostic tool (also used in debug builds), never a draw-time pass.

pub mod artifact;
mod builder;
pub mod compile;
mod default_control;
mod dominance;
mod half;
mod optimize;
mod uniform;
pub use uniform::{UniformControl, UniformSelection};
mod verify;
mod vertex;
pub use default_control::{DefaultControl, default_control};
pub use vertex::scale_vertex_inputs;
pub mod spirv;

pub use builder::{FunctionBuilder, InitialValue, Operand, VariableId};
pub use dominance::Dominance;

use super::{
    ShaderComputeBuiltin, ShaderFloatComparison, ShaderFloatControl, ShaderIntegerComparison,
    ShaderInterfaceElement, ShaderIoLocation, ShaderMathAccuracy, ShaderQuadOperation,
    ShaderResourceKind, ShaderRoundingMode, ShaderSourceLocation, ShaderSpecialFunction,
};
use crate::ShaderStage;
use std::fmt;

macro_rules! id {
    ($name:ident) => {
        #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub struct $name(pub(crate) u32);
        impl $name {
            #[must_use]
            pub const fn index(self) -> usize {
                self.0 as usize
            }
        }
    };
}
id!(BlockId);
id!(ValueId);
id!(ResourceId);
impl ResourceId {
    #[must_use]
    pub const fn new(binding: u32) -> Self {
        Self(binding)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Type {
    Bool,
    U32,
    I32,
    F32,
    F16,
    U64,
    I64,
    F64,
    Effect,
}

impl Type {
    pub(crate) fn bits(self) -> Option<u32> {
        match self {
            Self::Bool => Some(1),
            Self::F16 => Some(16),
            Self::U32 | Self::I32 | Self::F32 => Some(32),
            Self::U64 | Self::I64 | Self::F64 => Some(64),
            Self::Effect => None,
        }
    }
    pub(crate) fn integer(self) -> bool {
        matches!(self, Self::U32 | Self::I32 | Self::U64 | Self::I64)
    }
}

/// Exact bits; no host floating point folding or implicit numerical defaults.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Constant {
    pub ty: Type,
    pub bits: u64,
}
impl Constant {
    #[must_use]
    pub const fn u32(bits: u32) -> Self {
        Self {
            ty: Type::U32,
            bits: bits as u64,
        }
    }
    #[must_use]
    pub const fn boolean(value: bool) -> Self {
        Self {
            ty: Type::Bool,
            bits: value as u64,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Binary {
    Add,
    Multiply,
    And,
    Or,
    Xor,
    ShiftLeft,
    ShiftRight,
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Unary {
    Not,
    Negate,
    Absolute,
    BitReverse,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MathFunction {
    Reciprocal,
    ReciprocalSqrt,
    Special(ShaderSpecialFunction),
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SampleMode {
    Implicit,
    Explicit,
    LevelZero,
    Compare,
    CompareLevelZero,
    Fetch(u32),
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MemorySpace {
    Constant,
    Storage,
    Shared,
    Local,
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum BarrierScope {
    Invocation,
    Workgroup,
    Device,
    Patch,
}

/// Memory classes whose writes a shader barrier makes visible. Execution and
/// memory scopes are independent; a shared-only rendezvous must not add a
/// global storage/image fence. ISA frontends choose this mask explicitly.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct BarrierMemory(u8);
impl BarrierMemory {
    pub const NONE: Self = Self(0);
    pub const SHARED: Self = Self(1);
    pub const STORAGE: Self = Self(2);
    pub const IMAGES: Self = Self(4);
    pub const OUTPUTS: Self = Self(8);
    pub(crate) fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }
}
impl std::ops::BitOr for BarrierMemory {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

/// Operand/result types are determined by the opcode. Scalar interpretations
/// use explicit bitcasts: a frontend's untyped word registers cannot leak into
/// shared transformations. Floating controls survive every pass unchanged.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum Op {
    Constant(Constant),
    Undefined(Type),
    Parameter {
        index: u32,
        ty: Type,
    },
    EffectStart,
    Copy(Type),
    Bitcast {
        from: Type,
        to: Type,
    },
    Unary {
        op: Unary,
        ty: Type,
    },
    Binary {
        op: Binary,
        ty: Type,
        float_control: ShaderFloatControl,
    },
    Fma {
        ty: Type,
        float_control: ShaderFloatControl,
    },
    AddCarry,
    /// Low binary16 lane, with signaling NaNs quieted and payload preserved.
    UnpackHalf,
    /// Exact binary16 bits in the low lane; upper lane is zero. No rounding.
    PackHalf,
    /// Single-rounded binary16 FMA; promoted inputs remain explicit F32 FMA.
    HalfFma {
        saturate: bool,
    },
    FloatMinMax {
        float_control: ShaderFloatControl,
    },
    Math {
        function: MathFunction,
        accuracy: ShaderMathAccuracy,
        float_control: ShaderFloatControl,
    },
    RoundIntegral {
        rounding: ShaderRoundingMode,
        flush_denormals: bool,
    },
    FloatToInteger {
        to: Type,
        bits: u32,
        rounding: ShaderRoundingMode,
        flush_denormals: bool,
    },
    Quad(ShaderQuadOperation),
    QuadSwizzleAdd {
        swizzle: u32,
        float_control: ShaderFloatControl,
    },
    Interpolate {
        location: ShaderIoLocation,
        component: u32,
        interpolation: super::ShaderInterpolation,
    },
    ControlPointLoad {
        location: ShaderIoLocation,
        component: u32,
        output: bool,
    },
    ControlPointStore {
        location: ShaderIoLocation,
        component: u32,
    },
    PatchOutputLoad {
        location: ShaderIoLocation,
        component: u32,
    },
    TextureSample {
        image: ResourceId,
        sampler: Option<ResourceId>,
        kind: ShaderResourceKind,
        mode: SampleMode,
        offset: Option<[i32; 2]>,
    },
    TextureLevels {
        image: ResourceId,
    },
    IntegerCompare {
        ty: Type,
        comparison: ShaderIntegerComparison,
    },
    FloatCompare {
        ty: Type,
        comparison: ShaderFloatComparison,
        flush_denormals: bool,
    },
    Convert {
        from: Type,
        to: Type,
        rounding: ShaderRoundingMode,
        flush_denormals: bool,
    },
    Select(Type),
    Builtin {
        builtin: ShaderComputeBuiltin,
        component: u32,
    },
    Input {
        location: ShaderIoLocation,
        component: u32,
        ty: Type,
    },
    Output {
        location: ShaderIoLocation,
        component: u32,
        ty: Type,
    },
    /// Explicit byte addressing. Width/alignment rules belong to the frontend.
    Load {
        address_type: Type,
        space: MemorySpace,
        resource: Option<ResourceId>,
        ty: Type,
    },
    Store {
        address_type: Type,
        space: MemorySpace,
        resource: Option<ResourceId>,
        ty: Type,
    },
    Barrier {
        execution: BarrierScope,
        memory: BarrierScope,
        spaces: BarrierMemory,
    },
}

impl Op {
    #[must_use]
    pub fn ordered(&self) -> bool {
        // Reads are ordered too: optimizers cannot move a live load across an
        // aliasing write/barrier. Unused nonvolatile reads may splice the token.
        matches!(
            self,
            Self::Load { .. }
                | Self::Store { .. }
                | Self::Output { .. }
                | Self::Barrier { .. }
                | Self::ControlPointLoad { output: true, .. }
                | Self::ControlPointStore { .. }
                | Self::PatchOutputLoad { .. }
                | Self::TextureSample { .. }
        )
    }

    pub(crate) fn signature(&self) -> (Vec<Type>, Vec<Type>) {
        use Op::*;
        let (mut args, mut results) = match *self {
            Constant(c) => (vec![], vec![c.ty]),
            Undefined(ty) | Parameter { ty, .. } => (vec![], vec![ty]),
            EffectStart => (vec![], vec![Type::Effect]),
            Copy(ty) | Unary { ty, .. } => (vec![ty], vec![ty]),
            Bitcast { from, to } | Convert { from, to, .. } => (vec![from], vec![to]),
            Binary { ty, .. } => (vec![ty, ty], vec![ty]),
            Fma { ty, .. } => (vec![ty, ty, ty], vec![ty]),
            AddCarry => (vec![Type::U32; 3], vec![Type::U32; 2]),
            UnpackHalf => (vec![Type::U32], vec![Type::F16]),
            PackHalf => (vec![Type::F16], vec![Type::U32]),
            HalfFma { .. } => (vec![Type::F16; 3], vec![Type::F16]),
            FloatMinMax { .. } => (vec![Type::F32, Type::F32, Type::Bool], vec![Type::F32]),
            Math { .. } | RoundIntegral { .. } => (vec![Type::F32], vec![Type::F32]),
            FloatToInteger { to, .. } => (vec![Type::F32], vec![to]),
            Quad(_) => (vec![Type::U32], vec![Type::U32]),
            QuadSwizzleAdd { .. } => (vec![Type::F32; 2], vec![Type::F32]),
            Interpolate { .. } => (vec![], vec![Type::F32]),
            ControlPointLoad { .. } => (vec![Type::U32], vec![Type::U32]),
            ControlPointStore { .. } => (vec![Type::U32; 2], vec![]),
            PatchOutputLoad { .. } => (vec![], vec![Type::U32]),
            TextureSample { kind, mode, .. } => {
                let coords = if kind == ShaderResourceKind::SampledImageCube {
                    3
                } else {
                    2
                };
                let mut inputs = vec![
                    if matches!(mode, SampleMode::Fetch(_)) {
                        Type::I32
                    } else {
                        Type::F32
                    };
                    coords
                ];
                if matches!(
                    kind,
                    ShaderResourceKind::SampledImage2DArray
                        | ShaderResourceKind::SampledDepthImage2DArray
                ) {
                    inputs.push(Type::U32);
                }
                if matches!(
                    mode,
                    SampleMode::Explicit | SampleMode::Compare | SampleMode::CompareLevelZero
                ) {
                    inputs.push(Type::F32);
                }
                (inputs, vec![Type::F32; 4])
            }
            TextureLevels { .. } => (vec![], vec![Type::U32]),
            IntegerCompare { ty, .. } | FloatCompare { ty, .. } => (vec![ty, ty], vec![Type::Bool]),
            Select(ty) => (vec![Type::Bool, ty, ty], vec![ty]),
            Builtin { .. } => (vec![], vec![Type::U32]),
            Input { ty, .. } => (vec![], vec![ty]),
            Output { ty, .. } => (vec![ty], vec![]),
            Load {
                ty, address_type, ..
            } => (vec![address_type], vec![ty]),
            Store {
                ty, address_type, ..
            } => (vec![address_type, ty], vec![]),
            Barrier { .. } => (vec![], vec![]),
        };
        if self.ordered() {
            args.push(Type::Effect);
            results.push(Type::Effect);
        }
        (args, results)
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Instruction {
    pub op: Op,
    pub arguments: Box<[ValueId]>,
    pub results: Box<[ValueId]>,
    pub source: ShaderSourceLocation,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Phi {
    pub result: ValueId,
    /// Exactly one incoming value per distinct CFG predecessor.
    pub incoming: Vec<(BlockId, ValueId)>,
    pub source: ShaderSourceLocation,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum Terminator<A = ValueId> {
    Jump(BlockId),
    Branch {
        condition: A,
        when_true: BlockId,
        when_false: BlockId,
    },
    /// Integer selection with an explicit default edge.
    Switch {
        selector: A,
        cases: Box<[(u32, BlockId)]>,
        default: BlockId,
    },
    Return,
    Discard,
    Unreachable,
}
impl<A> Terminator<A> {
    pub fn successors(&self) -> Vec<BlockId> {
        let mut edges = match self {
            Self::Jump(to) => vec![*to],
            Self::Branch {
                when_true,
                when_false,
                ..
            } => vec![*when_true, *when_false],
            Self::Switch { cases, default, .. } => {
                cases.iter().map(|(_, b)| *b).chain([*default]).collect()
            }
            Self::Return | Self::Discard | Self::Unreachable => vec![],
        };
        edges.sort_unstable();
        edges.dedup();
        edges
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Block {
    pub source: ShaderSourceLocation,
    pub phis: Vec<Phi>,
    pub instructions: Vec<Instruction>,
    pub terminator: Terminator,
    pub terminator_source: ShaderSourceLocation,
    /// Keeps the last ordered operation live, including at early exits.
    pub effect: ValueId,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Definition {
    Phi {
        block: BlockId,
    },
    Instruction {
        block: BlockId,
        index: u32,
        result: u32,
    },
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Value {
    pub ty: Type,
    pub definition: Definition,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Resource {
    pub id: ResourceId,
    pub kind: ShaderResourceKind,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Function {
    pub stage: ShaderStage,
    pub inputs: Vec<ShaderInterfaceElement>,
    pub outputs: Vec<ShaderInterfaceElement>,
    pub resources: Vec<Resource>,
    pub workgroup_size: Option<[u32; 3]>,
    /// Private bytes per invocation, belonging to this shader stage.
    pub local_memory_bytes: u32,
    pub tessellation_control_points: Option<u32>,
    pub blocks: Vec<Block>,
    pub values: Vec<Value>,
}
impl Function {
    #[must_use]
    pub const fn entry(&self) -> BlockId {
        BlockId(0)
    }
    pub fn verify(&self) -> Result<(), Error> {
        verify::verify(self)
    }
    pub fn optimize(&mut self) -> Result<(), Error> {
        optimize::optimize(self)
    }
    /// Refresh dense definitions after a frontend semantic transformation.
    /// Use before optimization/debug verification when operations were removed.
    pub fn recompact(&mut self) {
        optimize::compact(self);
    }
    /// Dense values and source PCs make failed fixtures/compiles inspectable.
    #[must_use]
    pub fn dump(&self) -> String {
        use std::fmt::Write;
        let mut out = format!("SSA {:?}\n", self.stage);
        for (index, block) in self.blocks.iter().enumerate() {
            writeln!(out, "b{index} pc={:#x}:", block.source.byte_offset()).unwrap();
            for phi in &block.phis {
                writeln!(
                    out,
                    "  v{}:{:?} = phi {:?} @ {:#x}",
                    phi.result.0,
                    self.values[phi.result.index()].ty,
                    phi.incoming,
                    phi.source.byte_offset()
                )
                .unwrap();
            }
            for instruction in &block.instructions {
                writeln!(
                    out,
                    "  {:?} = {:?} {:?} @ {:#x}",
                    instruction.results,
                    instruction.op,
                    instruction.arguments,
                    instruction.source.byte_offset()
                )
                .unwrap();
            }
            writeln!(
                out,
                "  {:?} effect=v{} @ {:#x}",
                block.terminator,
                block.effect.0,
                block.terminator_source.byte_offset()
            )
            .unwrap();
        }
        out
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Error {
    pub stage: Option<ShaderStage>,
    pub source: ShaderSourceLocation,
    pub detail: String,
}
impl Error {
    pub(crate) fn at(source: ShaderSourceLocation, detail: impl Into<String>) -> Self {
        Self {
            stage: None,
            source,
            detail: detail.into(),
        }
    }
}
impl Error {
    pub(crate) fn in_stage(mut self, stage: ShaderStage) -> Self {
        self.stage = Some(stage);
        self
    }
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(stage) = self.stage {
            write!(f, "{stage:?} ")?;
        }
        write!(
            f,
            "shader SSA at PC {:#x}: {}",
            self.source.byte_offset(),
            self.detail
        )
    }
}
impl std::error::Error for Error {}

#[cfg(test)]
mod tests;
