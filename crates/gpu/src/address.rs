//! Typed guest GPU virtual addresses.

use std::fmt::{Display, Formatter};

/// A virtual address in a guest GPU address space.
///
/// The owning GPU frontend supplies its profile width at construction and at
/// every arithmetic boundary. The type deliberately implements neither `Add`
/// nor `Sub`, so address traversal cannot silently wrap or escape the selected
/// guest profile.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct GpuVirtualAddress(u64);

impl GpuVirtualAddress {
    /// Constructs an address after checking it against the selected profile
    /// width.
    pub const fn try_new(value: u64, address_bits: u8) -> Result<Self, GpuVirtualAddressError> {
        if !valid_address_width(address_bits) {
            return Err(GpuVirtualAddressError::InvalidAddressWidth { bits: address_bits });
        }
        if !fits_address_width(value, address_bits) {
            return Err(GpuVirtualAddressError::AddressOutOfRange {
                value,
                bits: address_bits,
            });
        }
        Ok(Self(value))
    }

    /// Returns the guest GPU address bit pattern.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }

    /// Adds a byte offset without wrapping or crossing the selected profile's
    /// virtual-address limit.
    pub const fn checked_add(
        self,
        byte_offset: u64,
        address_bits: u8,
    ) -> Result<Self, GpuVirtualAddressError> {
        if !valid_address_width(address_bits) {
            return Err(GpuVirtualAddressError::InvalidAddressWidth { bits: address_bits });
        }
        if !fits_address_width(self.0, address_bits) {
            return Err(GpuVirtualAddressError::AddressOutOfRange {
                value: self.0,
                bits: address_bits,
            });
        }
        let Some(value) = self.0.checked_add(byte_offset) else {
            return Err(GpuVirtualAddressError::ArithmeticOverflow);
        };
        Self::try_new(value, address_bits)
    }

    /// Subtracts a byte offset without wrapping below zero.
    pub const fn checked_sub(
        self,
        byte_offset: u64,
        address_bits: u8,
    ) -> Result<Self, GpuVirtualAddressError> {
        if !valid_address_width(address_bits) {
            return Err(GpuVirtualAddressError::InvalidAddressWidth { bits: address_bits });
        }
        if !fits_address_width(self.0, address_bits) {
            return Err(GpuVirtualAddressError::AddressOutOfRange {
                value: self.0,
                bits: address_bits,
            });
        }
        let Some(value) = self.0.checked_sub(byte_offset) else {
            return Err(GpuVirtualAddressError::ArithmeticOverflow);
        };
        Ok(Self(value))
    }

    /// Returns whether the address satisfies a non-zero power-of-two
    /// alignment.
    #[must_use]
    pub const fn is_aligned_to(self, alignment: u64) -> bool {
        alignment.is_power_of_two() && self.0 & (alignment - 1) == 0
    }
}

impl Display for GpuVirtualAddress {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "gpu-va=0x{:016x}", self.0)
    }
}

/// Failure to construct or traverse a profile-sized GPU virtual address.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuVirtualAddressError {
    InvalidAddressWidth { bits: u8 },
    AddressOutOfRange { value: u64, bits: u8 },
    ArithmeticOverflow,
}

impl Display for GpuVirtualAddressError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidAddressWidth { bits } => {
                write!(
                    formatter,
                    "GPU virtual-address width is invalid: bits={bits}"
                )
            }
            Self::AddressOutOfRange { value, bits } => write!(
                formatter,
                "GPU virtual address exceeds profile width: address=0x{value:016x} bits={bits}"
            ),
            Self::ArithmeticOverflow => {
                formatter.write_str("GPU virtual-address arithmetic overflowed")
            }
        }
    }
}

impl std::error::Error for GpuVirtualAddressError {}

const fn valid_address_width(address_bits: u8) -> bool {
    address_bits > 0 && address_bits <= u64::BITS as u8
}

const fn fits_address_width(value: u64, address_bits: u8) -> bool {
    address_bits == u64::BITS as u8 || value < (1_u64 << address_bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SWITCH_1_GPU_VA_BITS: u8 = 40;

    #[test]
    fn construction_enforces_the_profile_width() {
        assert_eq!(
            GpuVirtualAddress::try_new(0x00ff_ffff_ffff, SWITCH_1_GPU_VA_BITS)
                .expect("highest 40-bit address")
                .get(),
            0x00ff_ffff_ffff
        );
        assert_eq!(
            GpuVirtualAddress::try_new(0x0100_0000_0000, SWITCH_1_GPU_VA_BITS),
            Err(GpuVirtualAddressError::AddressOutOfRange {
                value: 0x0100_0000_0000,
                bits: SWITCH_1_GPU_VA_BITS,
            })
        );
        assert_eq!(
            GpuVirtualAddress::try_new(0, 0),
            Err(GpuVirtualAddressError::InvalidAddressWidth { bits: 0 })
        );
    }

    #[test]
    fn arithmetic_cannot_wrap_or_cross_the_profile_limit() {
        let address = GpuVirtualAddress::try_new(0x00ff_ffff_f000, SWITCH_1_GPU_VA_BITS).unwrap();
        assert_eq!(
            address.checked_add(0x1000, SWITCH_1_GPU_VA_BITS),
            Err(GpuVirtualAddressError::AddressOutOfRange {
                value: 0x0100_0000_0000,
                bits: SWITCH_1_GPU_VA_BITS,
            })
        );
        assert_eq!(
            GpuVirtualAddress::try_new(0, SWITCH_1_GPU_VA_BITS)
                .unwrap()
                .checked_sub(1, SWITCH_1_GPU_VA_BITS),
            Err(GpuVirtualAddressError::ArithmeticOverflow)
        );
    }

    #[test]
    fn formatting_is_stable_and_pointer_free() {
        let address = GpuVirtualAddress::try_new(0xabcdef, SWITCH_1_GPU_VA_BITS).unwrap();
        assert_eq!(address.to_string(), "gpu-va=0x0000000000abcdef");
    }
}
