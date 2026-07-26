//! Host-independent GPU contracts and diagnostics.
//!
//! Console frontends and host backends meet at this boundary without sharing
//! Horizon ABI, console packet formats, or concrete host graphics objects.

mod address;
mod diagnostics;

pub use address::{GpuVirtualAddress, GpuVirtualAddressError};
pub use diagnostics::{
    CpuVirtualAddress, GpfifoEntryIndex, GpuChannelId, GpuClassId, GpuMethodId,
    GraphicsAllocationId, GraphicsGapKind, SyncpointValue,
};
pub use nixe_memory::MappingGeneration;
