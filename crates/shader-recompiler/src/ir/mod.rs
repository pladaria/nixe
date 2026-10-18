//! Shared IR data vocabulary. Mutable nodes and algorithms remain compiler-internal.

mod attribute;
mod condition;
mod core;
mod emitter;
mod emitter_composite;
mod emitter_convert;
mod emitter_dispatch;
mod modifiers;
mod opcode;
mod types;
mod value;

pub use attribute::Attribute;
pub use condition::{Condition, FlowTest};
pub use core::{Block, Instruction, Ir, Program, SyntaxNode};
pub use emitter::Emitter;
pub use modifiers::{FmzMode, FpControl, FpRounding, TextureInstInfo};
pub use opcode::{Opcode, Signature};
pub use types::Type;
pub use value::*;
