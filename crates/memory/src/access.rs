//! Device-neutral access metadata.

use std::fmt::{Display, Formatter};

/// Read, write, and execute permissions observed through one virtual mapping.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct MemoryPermissions(u8);

impl MemoryPermissions {
    /// No access.
    pub const NONE: Self = Self(0);
    /// Read-only data.
    pub const READ: Self = Self(1 << 0);
    /// Write-only data.
    pub const WRITE: Self = Self(1 << 1);
    /// Instruction execution.
    pub const EXECUTE: Self = Self(1 << 2);
    /// Read/write data.
    pub const READ_WRITE: Self = Self(Self::READ.0 | Self::WRITE.0);
    /// Readable executable code.
    pub const READ_EXECUTE: Self = Self(Self::READ.0 | Self::EXECUTE.0);
    /// Writable executable memory, useful for coherency tests.
    pub const READ_WRITE_EXECUTE: Self = Self(Self::READ.0 | Self::WRITE.0 | Self::EXECUTE.0);

    /// Creates permissions from the portable read/write/execute bit set.
    #[must_use]
    pub const fn from_bits(bits: u8) -> Option<Self> {
        if bits & !Self::READ_WRITE_EXECUTE.0 == 0 {
            Some(Self(bits))
        } else {
            None
        }
    }

    /// Returns whether all requested permissions are present.
    #[must_use]
    pub const fn contains(self, permission: Self) -> bool {
        self.0 & permission.0 == permission.0
    }

    /// Returns the portable read/write/execute bit representation.
    #[must_use]
    pub const fn bits(self) -> u8 {
        self.0
    }
}

/// Stable identity assigned to a non-CPU memory consumer.
///
/// This is an emulator-side domain identifier, not a guest GPU channel,
/// backend queue, host pointer, or graphics-API object.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct NonCpuDeviceId(u64);

impl NonCpuDeviceId {
    /// Creates an identity from a value assigned by the owning subsystem.
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the owner-assigned numeric representation.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl Display for NonCpuDeviceId {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "device=0x{:016x}", self.0)
    }
}

/// Ordered point in one non-CPU device's visibility domain.
///
/// The value names when a transition must be true. It does not claim that a
/// guest fence, host submission, or host queue has completed.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct DeviceVisibilityPoint(u64);

impl DeviceVisibilityPoint {
    /// Creates a point from a value ordered by the device owner.
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the owner-assigned ordered value.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl Display for DeviceVisibilityPoint {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "visibility-point=0x{:016x}", self.0)
    }
}

/// Bytes a non-CPU device may observe or modify.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DeviceAccessKind {
    /// The device only observes existing contents.
    Read,
    /// The device modifies contents.
    Write,
    /// The device observes and modifies contents.
    ReadWrite,
}

impl DeviceAccessKind {
    /// Returns whether existing contents may be observed.
    #[must_use]
    pub const fn reads(self) -> bool {
        matches!(self, Self::Read | Self::ReadWrite)
    }

    /// Returns whether contents may be modified.
    #[must_use]
    pub const fn writes(self) -> bool {
        matches!(self, Self::Write | Self::ReadWrite)
    }
}

/// Complete non-CPU access declaration for a canonical backing range.
///
/// Device visibility names the point before which canonical contents and
/// unaffected bytes must be available to the device. CPU visibility names the
/// point after which device-produced contents may be made visible to the CPU.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DeviceAccessDeclaration {
    device: NonCpuDeviceId,
    kind: DeviceAccessKind,
    device_visible_at: DeviceVisibilityPoint,
    cpu_visible_at: Option<DeviceVisibilityPoint>,
}

impl DeviceAccessDeclaration {
    /// Creates a declaration whose visibility points exactly match its access.
    pub const fn new(
        device: NonCpuDeviceId,
        kind: DeviceAccessKind,
        device_visible_at: DeviceVisibilityPoint,
        cpu_visible_at: Option<DeviceVisibilityPoint>,
    ) -> Result<Self, DeviceAccessDeclarationError> {
        let reversed = match cpu_visible_at {
            Some(point) => point.0 < device_visible_at.0,
            None => false,
        };
        if kind.writes() != cpu_visible_at.is_some() || reversed {
            return Err(DeviceAccessDeclarationError);
        }
        Ok(Self {
            device,
            kind,
            device_visible_at,
            cpu_visible_at,
        })
    }

    /// Declares a device read.
    #[must_use]
    pub const fn read(device: NonCpuDeviceId, visible_at: DeviceVisibilityPoint) -> Self {
        Self {
            device,
            kind: DeviceAccessKind::Read,
            device_visible_at: visible_at,
            cpu_visible_at: None,
        }
    }

    /// Declares a device write.
    pub const fn write(
        device: NonCpuDeviceId,
        device_visible_at: DeviceVisibilityPoint,
        cpu_visible_at: DeviceVisibilityPoint,
    ) -> Result<Self, DeviceAccessDeclarationError> {
        Self::new(
            device,
            DeviceAccessKind::Write,
            device_visible_at,
            Some(cpu_visible_at),
        )
    }

    /// Declares a device read followed by a write.
    pub const fn read_write(
        device: NonCpuDeviceId,
        read_visible_at: DeviceVisibilityPoint,
        write_visible_at: DeviceVisibilityPoint,
    ) -> Result<Self, DeviceAccessDeclarationError> {
        Self::new(
            device,
            DeviceAccessKind::ReadWrite,
            read_visible_at,
            Some(write_visible_at),
        )
    }

    /// Returns the declared non-CPU consumer.
    #[must_use]
    pub const fn device(self) -> NonCpuDeviceId {
        self.device
    }

    /// Returns whether the device reads, writes, or performs both.
    #[must_use]
    pub const fn kind(self) -> DeviceAccessKind {
        self.kind
    }

    /// Returns when canonical input must be visible to the device.
    #[must_use]
    pub const fn device_visible_at(self) -> DeviceVisibilityPoint {
        self.device_visible_at
    }

    /// Returns when device output may become visible to the CPU.
    #[must_use]
    pub const fn cpu_visible_at(self) -> Option<DeviceVisibilityPoint> {
        self.cpu_visible_at
    }
}

/// A device declaration has missing, inapplicable, or reversed visibility.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeviceAccessDeclarationError;

impl Display for DeviceAccessDeclarationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("device access and ordered visibility points do not agree")
    }
}

impl std::error::Error for DeviceAccessDeclarationError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_access_requires_exact_visibility_points() {
        let device = NonCpuDeviceId::new(7);
        let point = DeviceVisibilityPoint::new(4);
        assert_eq!(
            DeviceAccessDeclaration::new(device, DeviceAccessKind::Read, point, None,),
            Ok(DeviceAccessDeclaration::read(device, point))
        );
        assert_eq!(
            DeviceAccessDeclaration::new(device, DeviceAccessKind::Write, point, Some(point),),
            DeviceAccessDeclaration::write(device, point, point)
        );
        assert_eq!(
            DeviceAccessDeclaration::new(device, DeviceAccessKind::Read, point, Some(point),),
            Err(DeviceAccessDeclarationError)
        );
        assert_eq!(
            DeviceAccessDeclaration::write(
                device,
                DeviceVisibilityPoint::new(5),
                DeviceVisibilityPoint::new(4),
            ),
            Err(DeviceAccessDeclarationError)
        );
    }
}
