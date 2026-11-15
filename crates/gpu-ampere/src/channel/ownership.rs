use crate::{BackState, FrontendHost, Registers, Result, registers::Bank};
use std::{cell::Cell, sync::Arc};
pub(super) static UNOWNED: [u64; 0xe00] = [0; 0xe00];
pub(super) struct Ownership {
    // Read directly by native code on this same thread. Interior mutability
    // keeps the retained raw view valid across Rust method callbacks.
    positions: Box<[Cell<u64>; 0xe00]>,
    shadows: Box<[u64; 0xe00]>,
    owned: [u32; 0xe00 / 32],
    bank: Arc<Bank>,
    #[cfg(feature = "performance-counters")]
    waits: (u64, std::time::Duration),
    shadow: Arc<Bank>,
}
impl Ownership {
    pub fn new(back: &BackState) -> Self {
        Self {
            positions: Box::new([const { Cell::new(0) }; 0xe00]),
            shadows: Box::new(UNOWNED),
            owned: [0; 0xe00 / 32],
            bank: back.registers.words.clone(),
            #[cfg(feature = "performance-counters")]
            waits: Default::default(),
            shadow: back.shadow.clone(),
        }
    }
    pub fn pointer(&self) -> *const u64 {
        self.positions.as_ptr().cast()
    }
    pub fn produced(&self, method: u32) -> bool {
        self.positions[method as usize / 4].get() != 0
    }
    fn wait(&mut self, position: u64, host: &mut impl FrontendHost) -> Result<()> {
        #[cfg(feature = "performance-counters")]
        let start = std::time::Instant::now();
        let result = host.wait_until(position);
        #[cfg(feature = "performance-counters")]
        {
            self.waits.0 += 1;
            self.waits.1 += start.elapsed();
        }
        result
    }
    pub fn produce(&mut self, writes: &[u32], position: u64, track: bool) {
        for &m in writes {
            let i = m as usize / 4;
            self.positions[i].set(position);
            self.owned[i / 32] |= 1 << (i % 32);
            if track {
                self.shadows[i] = position;
            }
        }
    }
    pub fn read(
        &mut self,
        method: u32,
        registers: &mut Registers,
        host: &mut impl FrontendHost,
    ) -> Result<()> {
        let i = method as usize / 4;
        let position = self.positions[i].get();
        if position != 0 {
            self.wait(position, host)?;
            registers.set(method, self.bank.get(i));
            self.positions[i].set(0);
            if self.shadows[i] == 0 {
                self.owned[i / 32] &= !(1 << (i % 32));
            }
        }
        Ok(())
    }
    pub fn take_over(
        &mut self,
        method: u32,
        mode: u32,
        shadow: &mut [u32; 0xe00],
        host: &mut impl FrontendHost,
    ) -> Result<bool> {
        let i = method as usize / 4;
        if self.owned[i / 32] & (1 << (i % 32)) == 0 {
            return Ok(false);
        }
        if mode == 3 && self.shadows[i] != 0 {
            self.wait(self.shadows[i], host)?;
            shadow[i] = self.shadow.get(i);
            self.shadows[i] = 0;
        }
        let force = self.positions[i].get() != 0 || (mode <= 1 && self.shadows[i] != 0);
        self.positions[i].set(0);
        if mode <= 1 {
            self.shadows[i] = 0;
        }
        if self.shadows[i] == 0 {
            self.owned[i / 32] &= !(1 << (i % 32));
        }
        Ok(force)
    }
}

#[cfg(feature = "performance-counters")]
impl Drop for Ownership {
    fn drop(&mut self) {
        log::info!(
            "Ampere macro ownership: read_dependencies={} wait={:?}",
            self.waits.0,
            self.waits.1
        );
    }
}
