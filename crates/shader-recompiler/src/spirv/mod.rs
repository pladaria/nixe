//! Binary SPIR-V construction and shader emission.
mod builder;
pub use builder::{Builder, DeferredPhi, Id, Section};
mod types;
pub use types::{CommonConstants, CommonTypes};
mod context;
mod interfaces;
pub use context::{
    BufferKind, EmitContext, ImageDefinition, InputGeneric, InputLoad, OutputGeneric,
    StoragePointers,
};
mod memory;

mod buffers;

mod images;

mod indirect;

mod atomics;

mod global;

mod emit;

pub use emit::{EmissionRoute, emission_route};
mod capabilities;
mod program;
pub use program::emit_spirv;
