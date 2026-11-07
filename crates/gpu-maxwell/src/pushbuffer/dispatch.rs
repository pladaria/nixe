//! Channel puller state, direct subchannel binding and execution effects.
//!
//! Puller methods mutate channel state directly; ordinary calls return no
//! descriptive record. Packet provenance is retained only by effects and errors.

use std::fmt::{Display, Formatter};

use nixe_gpu::{FrontendSubmissionId, GpuClassId, GpuMethodId};

use crate::{
    MaxwellChannelFrontendState, MaxwellChannelId, MaxwellGpfifoSourceLocation, MaxwellGpuProfile,
    MaxwellPushbufferSubchannel,
};

/// Byte address of the PBDMA SetObject host method.
pub const MAXWELL_SET_OBJECT_METHOD: GpuMethodId = GpuMethodId(0);
/// Byte address of the legacy channel `MEM_OP_A` host method accepted by GM20B.
pub const MAXWELL_LEGACY_MEM_OP_A_METHOD: GpuMethodId = GpuMethodId(0x28);
/// Byte address of the legacy channel `MEM_OP_B` host method accepted by GM20B.
pub const MAXWELL_LEGACY_MEM_OP_B_METHOD: GpuMethodId = GpuMethodId(0x2c);

const SET_OBJECT_CLASS_MASK: u32 = 0xffff;
const SET_OBJECT_ENGINE_SHIFT: u32 = 16;
const SET_OBJECT_ENGINE_MASK: u32 = 0x1f;
const SET_OBJECT_DEFINED_MASK: u32 =
    SET_OBJECT_CLASS_MASK | (SET_OBJECT_ENGINE_MASK << SET_OBJECT_ENGINE_SHIFT);
const MEM_OP_A_DEFINED_MASK: u32 = 0xffff_fffc;
const MEM_OP_B_OPERAND_HIGH_MASK: u32 = 0xff;
const MEM_OP_B_OPERATION_SHIFT: u32 = 27;
const MEM_OP_B_OPERATION_MASK: u32 = 0x1f << MEM_OP_B_OPERATION_SHIFT;
const MEM_OP_B_DEFINED_MASK: u32 = MEM_OP_B_OPERATION_MASK | MEM_OP_B_OPERAND_HIGH_MASK;
const MEM_OP_B_L2_SYSMEM_INVALIDATE: u8 = 0x0e;
const MEM_OP_B_L2_FLUSH_DIRTY: u8 = 0x10;

/// Complete pointer-free source of one expanded Maxwell method write.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellMethodSource {
    // The command reader supplies the active channel and submission identity.
    location: MaxwellGpfifoSourceLocation,
    subchannel: MaxwellPushbufferSubchannel,
    method: GpuMethodId,
    argument: u32,
}

impl MaxwellMethodSource {
    pub(crate) const fn set_argument(&mut self, argument: u32) {
        self.argument = argument;
    }

    pub(crate) const fn dma(
        location: MaxwellGpfifoSourceLocation,
        subchannel: MaxwellPushbufferSubchannel,
        method: GpuMethodId,
        argument: u32,
    ) -> Self {
        Self {
            location,
            subchannel,
            method,
            argument,
        }
    }
    pub(crate) const fn emitted_by_mme(self, method: GpuMethodId, argument: u32) -> Self {
        Self {
            method,
            argument,
            ..self
        }
    }

    #[must_use]
    pub const fn channel(self) -> MaxwellChannelId {
        self.location.channel
    }

    #[must_use]
    pub const fn submission(self) -> FrontendSubmissionId {
        self.location.frontend
    }

    #[must_use]
    pub const fn location(self) -> MaxwellGpfifoSourceLocation {
        self.location
    }

    #[must_use]
    pub const fn subchannel(self) -> MaxwellPushbufferSubchannel {
        self.subchannel
    }

    #[must_use]
    pub const fn method(self) -> GpuMethodId {
        self.method
    }

    #[must_use]
    pub const fn argument(self) -> u32 {
        self.argument
    }
}

impl Display for MaxwellMethodSource {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{} {} {} argument={:#010x}",
            self.location, self.subchannel, self.method, self.argument
        )
    }
}

/// Verified synchronization selected by a channel host-method write.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellHostSynchronizationKind {
    /// Drain prior engine work before processing subsequent channel methods.
    WaitForIdle { all: bool },
    SemaphoreRelease {
        address: u64,
        payload: u32,
        short: bool,
        wait: bool,
    },
    /// Discard cached system-memory reads before later channel work proceeds.
    L2SysmemInvalidate { operand_high: u8 },
    /// Write back dirty device L2 data before later channel work proceeds.
    L2FlushDirty { operand_high: u8 },
}

/// Failure at the class-binding boundary, before engine method execution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MaxwellMethodDispatchError {
    InvalidSetObjectValue {
        source: MaxwellMethodSource,
    },
    UnsupportedSetObjectEngine {
        source: MaxwellMethodSource,
        engine: u8,
    },
    UnsupportedClassForSubchannel {
        source: MaxwellMethodSource,
        class: GpuClassId,
        expected: Option<GpuClassId>,
    },
    UnsupportedHostMethod {
        source: MaxwellMethodSource,
    },
    InvalidHostMethodValue {
        source: MaxwellMethodSource,
        defined_mask: u32,
    },
    UnsupportedHostMemoryOperation {
        source: MaxwellMethodSource,
        operation: u8,
    },
    UnboundSubchannel {
        source: MaxwellMethodSource,
    },
}

impl Display for MaxwellMethodDispatchError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidSetObjectValue { source } => write!(
                formatter,
                "SetObject argument sets reserved bits outside ClassId[15:0] and EngineId[20:16]: {source}"
            ),
            Self::UnsupportedSetObjectEngine { source, engine } => write!(
                formatter,
                "SetObject engine is not available in the Switch 1 profile: {source} engine-id={engine}"
            ),
            Self::UnsupportedClassForSubchannel {
                source,
                class,
                expected,
            } => {
                write!(
                    formatter,
                    "SetObject class is unsupported by the selected Switch 1 engine: {source} requested-{class}"
                )?;
                match expected {
                    Some(expected) => write!(formatter, " expected-{expected}"),
                    None => formatter.write_str(" subchannel-has-no-hardware-engine"),
                }
            }
            Self::UnsupportedHostMethod { source } => {
                write!(
                    formatter,
                    "Maxwell host method semantics are unavailable: {source}"
                )
            }
            Self::InvalidHostMethodValue {
                source,
                defined_mask,
            } => write!(
                formatter,
                "Maxwell host method argument sets bits outside its verified field mask: {source} defined-mask={defined_mask:#010x}"
            ),
            Self::UnsupportedHostMemoryOperation { source, operation } => write!(
                formatter,
                "Maxwell host memory-operation semantics are unavailable: {source} operation={operation:#04x}"
            ),
            Self::UnboundSubchannel { source } => {
                write!(
                    formatter,
                    "Maxwell method targets an unbound subchannel: {source} class=unbound"
                )
            }
        }
    }
}

impl std::error::Error for MaxwellMethodDispatchError {}

/// Apply a channel puller method and return only an actual pending effect.
/// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/engines/puller.cpp
pub(crate) fn call_puller_method(
    profile: &MaxwellGpuProfile,
    source: MaxwellMethodSource,
    frontend: &mut MaxwellChannelFrontendState,
) -> Result<Option<MaxwellHostSynchronizationKind>, MaxwellMethodDispatchError> {
    use MaxwellHostSynchronizationKind as Effect;
    match source.method().0 {
        0 => {
            if source.argument & !SET_OBJECT_DEFINED_MASK != 0 {
                return Err(MaxwellMethodDispatchError::InvalidSetObjectValue { source });
            }
            let engine_id =
                ((source.argument >> SET_OBJECT_ENGINE_SHIFT) & SET_OBJECT_ENGINE_MASK) as u8;
            if engine_id != 0 {
                return Err(MaxwellMethodDispatchError::UnsupportedSetObjectEngine {
                    source,
                    engine: engine_id,
                });
            }
            let engine = match source.subchannel.get() {
                0 => Some(crate::engines::Engine::ThreeD),
                1 => Some(crate::engines::Engine::Compute),
                2 => Some(crate::engines::Engine::InlineToMemory),
                3 => Some(crate::engines::Engine::TwoD),
                4 => Some(crate::engines::Engine::Copy),
                _ => None,
            };
            let classes = profile.classes();
            let expected = engine.map(|engine| match engine {
                crate::engines::Engine::ThreeD => classes.three_d(),
                crate::engines::Engine::Compute => classes.compute(),
                crate::engines::Engine::InlineToMemory => classes.inline_to_memory(),
                crate::engines::Engine::TwoD => classes.two_d(),
                crate::engines::Engine::Copy => classes.dma_copy(),
            });
            let class = GpuClassId(source.argument & SET_OBJECT_CLASS_MASK);
            if expected != Some(class) {
                return Err(MaxwellMethodDispatchError::UnsupportedClassForSubchannel {
                    source,
                    class,
                    expected,
                });
            }
            frontend.bind_subchannel(
                source.subchannel,
                engine.expect("verified subchannel engine"),
            );
            Ok(None)
        }
        0x08 => Ok(None),
        0x10..=0x18 => {
            let index = ((source.method().0 - 0x10) / 4) as usize;
            let mask = [0xff, 0xffff_fffc, u32::MAX][index];
            if source.argument & !mask != 0 {
                return Err(MaxwellMethodDispatchError::InvalidHostMethodValue {
                    source,
                    defined_mask: mask,
                });
            }
            frontend.set_host_semaphore(index, source.argument);
            Ok(None)
        }
        0x1c => {
            if source.argument & 0x1f != 2 || source.argument & !0x0110_001f != 0 {
                return Err(MaxwellMethodDispatchError::UnsupportedHostMethod { source });
            }
            let [high, low, payload] = frontend.host_semaphore();
            Ok(Some(Effect::SemaphoreRelease {
                address: (u64::from(high) << 32) | u64::from(low),
                payload,
                short: source.argument & (1 << 24) != 0,
                wait: source.argument & (1 << 20) == 0,
            }))
        }
        0x28 => {
            if source.argument & !MEM_OP_A_DEFINED_MASK != 0 {
                return Err(MaxwellMethodDispatchError::InvalidHostMethodValue {
                    source,
                    defined_mask: MEM_OP_A_DEFINED_MASK,
                });
            }
            frontend.set_legacy_mem_op_a(source.argument & MEM_OP_A_DEFINED_MASK);
            Ok(None)
        }
        0x2c => {
            if source.argument & !MEM_OP_B_DEFINED_MASK != 0 {
                return Err(MaxwellMethodDispatchError::InvalidHostMethodValue {
                    source,
                    defined_mask: MEM_OP_B_DEFINED_MASK,
                });
            }
            let operation =
                ((source.argument & MEM_OP_B_OPERATION_MASK) >> MEM_OP_B_OPERATION_SHIFT) as u8;
            match operation {
                MEM_OP_B_L2_SYSMEM_INVALIDATE => Ok(Some(Effect::L2SysmemInvalidate {
                    operand_high: (source.argument & MEM_OP_B_OPERAND_HIGH_MASK) as u8,
                })),
                MEM_OP_B_L2_FLUSH_DIRTY => Ok(Some(Effect::L2FlushDirty {
                    operand_high: (source.argument & MEM_OP_B_OPERAND_HIGH_MASK) as u8,
                })),
                operation => Err(MaxwellMethodDispatchError::UnsupportedHostMemoryOperation {
                    source,
                    operation,
                }),
            }
        }
        0x50 => {
            frontend.set_reference_count(source.argument);
            Ok(Some(Effect::WaitForIdle { all: true }))
        }
        0x78 => {
            if source.argument & !1 != 0 {
                return Err(MaxwellMethodDispatchError::InvalidHostMethodValue {
                    source,
                    defined_mask: 1,
                });
            }
            Ok(Some(Effect::WaitForIdle {
                all: source.argument != 0,
            }))
        }
        _ => Err(MaxwellMethodDispatchError::UnsupportedHostMethod { source }),
    }
}
