//! Read-only Linux signal image. Never read the dispatcher's live registers.
//! Linux UAPI layouts (also available in the installed asm/sigcontext.h):
//! https://github.com/torvalds/linux/blob/master/arch/arm64/include/uapi/asm/sigcontext.h
//! https://github.com/torvalds/linux/blob/master/arch/x86/include/uapi/asm/sigcontext.h

use super::{FaultSlot, NativeFaultSite};
use std::{marker::PhantomData, ptr::NonNull, sync::atomic::Ordering};

/// Borrowed during dispatch or after escape, never across worker-slot reuse.
pub struct CapturedFault<'a> {
    pub(super) slot: NonNull<FaultSlot>,
    pub(super) site: Option<&'a NativeFaultSite>,
    pub(super) lifetime: PhantomData<&'a FaultSlot>,
}

impl CapturedFault<'_> {
    pub fn native_pc(&self) -> usize {
        unsafe { self.slot.as_ref() }
            .native_pc
            .load(Ordering::Relaxed)
    }

    pub fn fault_address(&self) -> usize {
        unsafe { self.slot.as_ref() }
            .fault_address
            .load(Ordering::Relaxed)
    }

    /// Fixed-stub/legacy dispatch only. After escape the registry may have been
    /// dropped; the execution owner must attribute native_pc under its epoch.
    pub fn site(&self) -> &NativeFaultSite {
        self.site
            .expect("site is only available during attributed dispatch")
    }

    fn context(&self) -> &libc::mcontext_t {
        unsafe { &(*(*self.slot.as_ref().context.get()).as_ptr()).uc_mcontext }
    }

    /// Architectural host numbering, not libc's greg array indices.
    pub fn integer(&self, index: u8) -> Option<u64> {
        #[cfg(target_arch = "x86_64")]
        {
            let indices = [
                libc::REG_RAX,
                libc::REG_RCX,
                libc::REG_RDX,
                libc::REG_RBX,
                libc::REG_RSP,
                libc::REG_RBP,
                libc::REG_RSI,
                libc::REG_RDI,
                libc::REG_R8,
                libc::REG_R9,
                libc::REG_R10,
                libc::REG_R11,
                libc::REG_R12,
                libc::REG_R13,
                libc::REG_R14,
                libc::REG_R15,
            ];
            Some(self.context().gregs[*indices.get(usize::from(index))? as usize] as u64)
        }
        #[cfg(target_arch = "aarch64")]
        {
            self.context().regs.get(usize::from(index)).copied()
        }
    }

    pub fn flags(&self) -> u64 {
        #[cfg(target_arch = "x86_64")]
        {
            self.context().gregs[libc::REG_EFL as usize] as u64
        }
        #[cfg(target_arch = "aarch64")]
        {
            self.context().pstate
        }
    }

    pub fn vector(&self, index: u8) -> Option<u128> {
        #[cfg(target_arch = "x86_64")]
        {
            let fp = unsafe { self.context().fpregs.as_ref()? };
            let words = fp._xmm.get(usize::from(index))?.element;
            Some(
                words
                    .into_iter()
                    .enumerate()
                    .fold(0, |bits, (i, word)| bits | (u128::from(word) << (32 * i))),
            )
        }
        #[cfg(target_arch = "aarch64")]
        {
            if index >= 32 {
                return None;
            }
            let fp = self.fpsimd()?;
            Some(u128::from_ne_bytes(
                fp[16 + usize::from(index) * 16..32 + usize::from(index) * 16]
                    .try_into()
                    .ok()?,
            ))
        }
    }

    /// Raw host [control,status]. x86 MXCSR is split like the FP owner save.
    pub fn fp(&self) -> Option<[u64; 2]> {
        #[cfg(target_arch = "x86_64")]
        {
            let mxcsr = unsafe { self.context().fpregs.as_ref()? }.mxcsr;
            Some([u64::from(mxcsr & !0x3f), u64::from(mxcsr & 0x3f)])
        }
        #[cfg(target_arch = "aarch64")]
        {
            let fp = self.fpsimd()?;
            Some([
                u64::from(u32::from_ne_bytes(fp[12..16].try_into().ok()?)),
                u64::from(u32::from_ne_bytes(fp[8..12].try_into().ok()?)),
            ])
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn fpsimd(&self) -> Option<&[u8]> {
        // libc keeps __reserved private. UAPI places the 4096-byte record area
        // on the next 16-byte boundary after pstate. FPSIMD must reside here,
        // not in extra_context; unknown extension records can be skipped.
        const OFFSET: usize =
            (std::mem::offset_of!(libc::mcontext_t, pstate) + 8).next_multiple_of(16);
        const _: () = assert!(OFFSET + 4096 <= std::mem::size_of::<libc::mcontext_t>());
        let bytes = unsafe {
            std::slice::from_raw_parts(
                std::ptr::from_ref(self.context()).cast::<u8>().add(OFFSET),
                4096,
            )
        };
        fpsimd_record(bytes)
    }
}

#[cfg(any(test, target_arch = "aarch64"))]
fn fpsimd_record(mut bytes: &[u8]) -> Option<&[u8]> {
    while bytes.len() >= 8 {
        let magic = u32::from_ne_bytes(bytes[..4].try_into().ok()?);
        let size = u32::from_ne_bytes(bytes[4..8].try_into().ok()?) as usize;
        if size < 16 || !size.is_multiple_of(16) || size > bytes.len() {
            return None;
        }
        if magic == 0x4650_8001 {
            return (size >= 528).then_some(&bytes[..size]);
        }
        bytes = &bytes[size..];
    }
    None
}

#[cfg(test)]
mod tests {
    use super::fpsimd_record;
    #[test]
    fn fpsimd_record_scan_is_bounded_and_skips_extensions() {
        let mut bytes = [0u8; 560];
        bytes[..4].copy_from_slice(&1u32.to_ne_bytes());
        bytes[4..8].copy_from_slice(&16u32.to_ne_bytes());
        bytes[16..20].copy_from_slice(&0x4650_8001u32.to_ne_bytes());
        bytes[20..24].copy_from_slice(&528u32.to_ne_bytes());
        assert_eq!(fpsimd_record(&bytes).unwrap().len(), 528);
        for size in [0u32, 8, 17, 512, 4096, u32::MAX] {
            bytes[20..24].copy_from_slice(&size.to_ne_bytes());
            assert!(fpsimd_record(&bytes).is_none());
        }
    }
}
