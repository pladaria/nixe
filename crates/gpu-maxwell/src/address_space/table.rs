//! Maxwell's small/big GPU page tables store compressed DAddr pages directly.
//! Types and physical continuity are separate packed arrays, as in MemoryManager.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/memory_manager.cpp
use nixe_memory::{Machine, SparseArray, device_memory::DeviceMemory};
use std::sync::{
    OnceLock,
    atomic::{AtomicU32, AtomicU64, Ordering},
};

const FREE: u64 = 0;
const RESERVED: u64 = 1;
const MAPPED: u64 = 2;
#[derive(Debug)]
struct Entries {
    bits: u32,
    mask: u64,
    pages: SparseArray<AtomicU32>,
    types: SparseArray<AtomicU64>,
    continuous: Option<SparseArray<AtomicU64>>,
}
impl Entries {
    fn new(machine: &Machine, address_bits: u8, bits: u32, big: bool) -> Self {
        let count = 1usize << (u32::from(address_bits) - bits);
        // SAFETY: atomic integer entries have valid zero representations and no Drop.
        unsafe {
            Self {
                bits,
                mask: (1 << bits) - 1,
                pages: SparseArray::new(machine, count),
                types: SparseArray::new(machine, count.div_ceil(32)),
                continuous: big.then(|| SparseArray::new(machine, count.div_ceil(64))),
            }
        }
    }
    #[inline]
    fn kind(&self, index: usize) -> u64 {
        (self.types[index / 32].load(Ordering::Acquire) >> (2 * (index % 32))) & 3
    }
    fn set_kind(&self, index: usize, kind: u64) {
        let shift = 2 * (index % 32);
        self.types[index / 32]
            .fetch_update(Ordering::Release, Ordering::Relaxed, |value| {
                Some((value & !(3 << shift)) | (kind << shift))
            })
            .unwrap();
    }
    fn set_continuous(&self, index: usize, continuous: bool) {
        let bits = self.continuous.as_ref().expect("big page continuity");
        let mask = 1 << (index % 64);
        if continuous {
            bits[index / 64].fetch_or(mask, Ordering::Release);
        } else {
            bits[index / 64].fetch_and(!mask, Ordering::Release);
        }
    }
    #[inline]
    fn address(&self, address: u64) -> u64 {
        (u64::from(self.pages[(address >> self.bits) as usize].load(Ordering::Relaxed)) << 12)
            + (address & self.mask)
    }
    #[inline]
    fn span(&self, address: u64) -> Span {
        let index = (address >> self.bits) as usize;
        let within = address & self.mask;
        Span {
            address: Some(self.address(address)),
            bytes: self.mask + 1 - within,
            continuous: self.continuous.as_ref().is_none_or(|bits| {
                bits[index / 64].load(Ordering::Acquire) & (1 << (index % 64)) != 0
            }),
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub(super) struct Span {
    pub address: Option<u64>,
    pub bytes: u64,
    pub continuous: bool,
}
#[derive(Debug)]
pub(super) struct PageTable {
    small: Entries,
    big: OnceLock<Entries>,
    address_bits: u8,
}
impl PageTable {
    pub fn new(machine: &Machine, address_bits: u8) -> Self {
        Self {
            small: Entries::new(machine, address_bits, 12, false),
            big: OnceLock::new(),
            address_bits,
        }
    }
    pub fn initialize(&self, machine: &Machine, big_bits: u32) {
        self.big
            .set(Entries::new(machine, self.address_bits, big_bits, true))
            .expect("GPU page table initialized twice");
    }
    /// GpuToCpuAddress consults the big table first, then the small table.
    #[inline]
    pub fn translate(&self, address: u64) -> Option<u64> {
        if address >= 1 << self.address_bits {
            return None;
        }
        let big = self.big.get()?;
        if big.kind((address >> big.bits) as usize) == MAPPED {
            return Some(big.address(address));
        }
        (self.small.kind((address >> 12) as usize) == MAPPED).then(|| self.small.address(address))
    }
    /// MemoryOperation: reserved big pages are read-zero holes; free big pages
    /// delegate to the small table. A span never crosses a hardware page edge.
    #[inline]
    pub fn span(&self, address: u64) -> Span {
        let big = self.big.get().expect("initialized GPU page table");
        match big.kind((address >> big.bits) as usize) {
            MAPPED => big.span(address),
            RESERVED => Span {
                address: None,
                bytes: big.mask + 1 - (address & big.mask),
                continuous: false,
            },
            _ => {
                let mapped = self.small.kind((address >> 12) as usize) == MAPPED;
                if mapped {
                    self.small.span(address)
                } else {
                    Span {
                        address: None,
                        bytes: 4096 - (address & 4095),
                        continuous: false,
                    }
                }
            }
        }
    }
    pub fn map(
        &self,
        address: u64,
        device: u64,
        bytes: u64,
        page_size: u32,
        memory: &DeviceMemory,
    ) {
        let table = if page_size == 4096 {
            &self.small
        } else {
            self.big.get().expect("initialized big pages")
        };
        debug_assert_eq!(u64::from(page_size), table.mask + 1);
        debug_assert_eq!((address | bytes) & table.mask, 0);
        debug_assert_eq!(device & 4095, 0);
        for offset in (0..bytes).step_by(page_size as usize) {
            let index = ((address + offset) >> table.bits) as usize;
            let at = device + offset;
            table.pages[index].store((at >> 12) as u32, Ordering::Relaxed);
            if table.continuous.is_some() {
                let first = memory.physical_address(at);
                let continuous = first.is_some_and(|first| {
                    (4096..u64::from(page_size)).step_by(4096).all(|offset| {
                        memory
                            .physical_address(at + offset)
                            .is_some_and(|next| next.get() == first.get() + offset)
                    })
                });
                table.set_continuous(index, continuous);
            }
            table.set_kind(index, MAPPED);
        }
    }
    pub fn reserve_sparse(&self, address: u64, bytes: u64) {
        let big = self.big.get().expect("initialized big pages");
        for offset in (0..bytes).step_by((big.mask + 1) as usize) {
            big.set_kind(((address + offset) >> big.bits) as usize, RESERVED);
        }
    }
    pub fn unmap(&self, address: u64, bytes: u64) {
        for table in [&self.small, self.big.get().expect("initialized big pages")] {
            for offset in (0..bytes).step_by((table.mask + 1) as usize) {
                // Leave the old value in place. Readers already seeing MAPPED
                // can consume that translation while Unmap publishes FREE.
                table.set_kind(((address + offset) >> table.bits) as usize, FREE);
            }
        }
    }
}
