//! A proven upper bound of stores relative to a constant-buffer pointer.
//! https://github.com/zortos293/Lichii/blob/646b5d17135a2b2701ccc7a959afd0bbbf2f6160/src/shader_recompiler/storage_write_extent.h

use serde::{Deserialize, Serialize};
#[derive(
    Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash,
)]
pub struct StorageWriteExtent {
    pub bounded: bool,
    pub guarded: bool,
    pub const_end: u32,
    pub local_stride: [u32; 3],
    pub group_stride: [u32; 3],
    pub guard_const: u32,
    pub guard_local_stride: [u32; 3],
    pub guard_group_stride: [u32; 3],
}
impl StorageWriteExtent {
    /// Unknown extents and indirect group counts retain the complete binding.
    /// Split multiplication operands must satisfy their 16-bit guard at dispatch.
    pub fn resolve(self, block: [u32; 3], grid: Option<[u32; 3]>) -> Option<u64> {
        if !self.bounded {
            return None;
        }
        let evaluate = |constant: u32, local: [u32; 3], group: [u32; 3]| {
            let mut total = u64::from(constant);
            for i in 0..3 {
                total =
                    total.checked_add(u64::from(local[i]) * u64::from(block[i].checked_sub(1)?))?;
                if group[i] != 0 {
                    total = total
                        .checked_add(u64::from(group[i]) * u64::from(grid?[i].checked_sub(1)?))?;
                }
                if total > 1 << 62 {
                    return None;
                }
            }
            Some(total)
        };
        if self.guarded
            && evaluate(
                self.guard_const,
                self.guard_local_stride,
                self.guard_group_stride,
            )? >= 0x10000
        {
            return None;
        }
        evaluate(self.const_end, self.local_stride, self.group_stride)
    }
    /// Component maxima cover all stores. One unknown store invalidates the proof.
    pub fn merge(&mut self, other: Self) {
        if !self.bounded || !other.bounded {
            *self = Self::default();
            return;
        }
        self.const_end = self.const_end.max(other.const_end);
        for i in 0..3 {
            self.local_stride[i] = self.local_stride[i].max(other.local_stride[i]);
            self.group_stride[i] = self.group_stride[i].max(other.group_stride[i]);
            self.guard_local_stride[i] =
                self.guard_local_stride[i].max(other.guard_local_stride[i]);
            self.guard_group_stride[i] =
                self.guard_group_stride[i].max(other.guard_group_stride[i]);
        }
        self.guarded |= other.guarded;
        self.guard_const = self.guard_const.max(other.guard_const);
    }
}
