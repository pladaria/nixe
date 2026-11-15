use super::PreparationHost;
use crate::{Registers, Result};

/// Evaluated in command order, including repeated writes of the same mode.
/// Operand memory is not reread at each draw or on an override register write.
/// https://github.com/NVIDIA/open-gpu-doc/blob/master/classes/3d/clc797.h
/// Query operand layout: https://github.com/zortos293/Lichii/blob/646b5d17135a2b2701ccc7a959afd0bbbf2f6160/src/video_core/engines/maxwell_3d.cpp#L1049
pub(super) struct Condition {
    enabled: bool,
    #[cfg(feature = "performance-counters")]
    counts: [u64; 4],
}
impl Default for Condition {
    fn default() -> Self {
        Self {
            // @SPECULATIVE: context reset is not publicly specified. Preserve
            // rendering until the guest first evaluates a condition.
            enabled: true,
            #[cfg(feature = "performance-counters")]
            counts: [0; 4],
        }
    }
}
impl Condition {
    pub fn evaluate(&mut self, regs: &Registers, host: &mut impl PreparationHost) -> Result<()> {
        let mode = match regs.get(0x1944) {
            0 => regs.get(0x1558),
            1 => 1,
            2 => 0,
            value => return Err(format!("invalid render-enable override {value:#x}").into()),
        };
        self.enabled = match mode {
            0 => false,
            1 => true,
            2..=4 => {
                if regs.get(0x1550) > 255 {
                    return Err("render condition address exceeds 40-bit VA".into());
                }
                let mut bytes = [0; 24];
                host.read_into(regs.address(0x1550, 0x1554), &mut bytes)?;
                let word =
                    |i: usize| u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
                match mode {
                    2 => word(0) != 0 && word(1) != 0,
                    3 => word(0) == word(4) && word(1) == word(5),
                    4 => word(0) != word(4) || word(1) != word(5),
                    _ => unreachable!(),
                }
            }
            value => return Err(format!("invalid render-enable mode {value:#x}").into()),
        };
        #[cfg(feature = "performance-counters")]
        {
            self.counts[0] += 1;
        }
        Ok(())
    }
    pub fn draw(&mut self, _instances: u32) -> bool {
        #[cfg(feature = "performance-counters")]
        if !self.enabled {
            self.counts[1] += 1;
            self.counts[2] += u64::from(_instances);
        }
        self.enabled
    }
    pub fn clear(&mut self) -> bool {
        #[cfg(feature = "performance-counters")]
        if !self.enabled {
            self.counts[3] += 1;
        }
        self.enabled
    }
}
#[cfg(feature = "performance-counters")]
impl Drop for Condition {
    fn drop(&mut self) {
        if self.counts != [0; 4] {
            log::info!(
                "Ampere render conditions: evaluated={} suppressed_draws={} suppressed_instances={} suppressed_clears={}",
                self.counts[0],
                self.counts[1],
                self.counts[2],
                self.counts[3]
            );
        }
    }
}
