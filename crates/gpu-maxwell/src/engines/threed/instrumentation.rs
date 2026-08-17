//! Typed, source-preserving `MAXWELL_B` instrumentation annotations.

use crate::MaxwellMethodSource;

use super::MaxwellThreeDRegister;

/// An opaque 32-bit instrumentation annotation supplied by the guest driver.
///
/// NVIDIA publishes the complete value fields, but no rendering semantics for
/// their contents. Retaining the exact bits and source makes captures and
/// replay deterministic without treating instrumentation as raster state.
///
/// ABI source:
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L103-L107>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct MaxwellThreeDInstrumentationValue(u32);

impl MaxwellThreeDInstrumentationValue {
    #[must_use]
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }
}

/// One validated instrumentation-register transition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellThreeDInstrumentationStateWrite {
    Header {
        value: MaxwellThreeDInstrumentationValue,
        source: MaxwellMethodSource,
    },
    Data {
        value: MaxwellThreeDInstrumentationValue,
        source: MaxwellMethodSource,
    },
}

/// Last instrumentation header and data programmed on one 3D engine.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellThreeDInstrumentationState {
    header: MaxwellThreeDRegister<MaxwellThreeDInstrumentationValue>,
    data: MaxwellThreeDRegister<MaxwellThreeDInstrumentationValue>,
}

impl MaxwellThreeDInstrumentationState {
    #[must_use]
    pub const fn header(&self) -> &MaxwellThreeDRegister<MaxwellThreeDInstrumentationValue> {
        &self.header
    }

    #[must_use]
    pub const fn data(&self) -> &MaxwellThreeDRegister<MaxwellThreeDInstrumentationValue> {
        &self.data
    }

    pub(super) fn apply(&mut self, write: MaxwellThreeDInstrumentationStateWrite) {
        match write {
            MaxwellThreeDInstrumentationStateWrite::Header { value, source } => {
                self.header = MaxwellThreeDRegister::programmed(value.bits(), value, source);
            }
            MaxwellThreeDInstrumentationStateWrite::Data { value, source } => {
                self.data = MaxwellThreeDRegister::programmed(value.bits(), value, source);
            }
        }
    }
}
