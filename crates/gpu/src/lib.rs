//! Host-independent GPU contracts and diagnostics.
//!
//! Console frontends and host backends meet at this boundary without sharing
//! Horizon ABI, console packet formats, or concrete host graphics objects.

mod diagnostics;

pub use diagnostics::{
    CpuVirtualAddress, GpfifoEntryIndex, GpuChannelId, GpuClassId, GpuMethodId, GpuVirtualAddress,
    GraphicsAllocationId, GraphicsGapKind, MappingGeneration, SyncpointValue,
};
