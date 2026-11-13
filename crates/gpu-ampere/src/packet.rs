//! NVC56F channel descriptors and packets, in byte-addressed method units.
//! https://github.com/NVIDIA/open-gpu-kernel-modules/blob/main/src/common/sdk/nvidia/inc/class/clc56f.h
use crate::Result;

#[derive(Clone, Copy, Debug)]
pub struct GpfifoEntry {
    pub address: u64,
    pub words: u32,
    pub sync_wait: bool,
    /// Reporting level affects USERD GET, not method execution order.
    pub subroutine: bool,
}
pub fn decode_entries(bytes: &[u8]) -> Result<Vec<GpfifoEntry>> {
    if !bytes.len().is_multiple_of(8) {
        return Err("partial GPFIFO descriptor".into());
    }
    bytes
        .chunks_exact(8)
        .enumerate()
        .map(|(i, b)| {
            let low = u32::from_le_bytes(b[..4].try_into().unwrap());
            let high = u32::from_le_bytes(b[4..].try_into().unwrap());
            let words = (high >> 10) & 0x1f_ffff;
            // Conditional fetch requires the subdevice-mask contract. The level
            // bit is retained independently of command execution.
            if low & 3 != 0 || high & 0x100 != 0 || (words == 0 && (low != 0 || high != 0)) {
                return Err(
                    format!("unsupported GPFIFO descriptor {i}: {low:#010x} {high:#010x}").into(),
                );
            }
            let address = u64::from(low) | (u64::from(high & 255) << 32);
            if address + u64::from(words) * 4 > 1 << 40 {
                return Err("GPFIFO range exceeds 40-bit VA".into());
            }
            Ok(GpfifoEntry {
                address,
                words,
                sync_wait: high >> 31 != 0,
                subroutine: high & 0x200 != 0,
            })
        })
        .collect()
}

#[derive(Default)]
pub(crate) struct Packet {
    method: u32,
    subchannel: u8,
    remaining: u32,
    mode: u32,
}
pub(crate) enum Word {
    Header,
    End,
    Method {
        method: u32,
        subchannel: u8,
        argument: u32,
        last: bool,
    },
}
impl Packet {
    pub fn word(&mut self, word: u32) -> Result<Word> {
        if self.remaining != 0 {
            let method = self.method;
            self.remaining -= 1;
            if self.mode == 1 || self.mode == 5 {
                self.method += 4;
                if self.mode == 5 {
                    self.mode = 3;
                }
            }
            return Ok(Word::Method {
                method,
                subchannel: self.subchannel,
                argument: word,
                last: self.remaining == 0,
            });
        }
        if word == 0 {
            return Ok(Word::Header);
        }
        self.mode = word >> 29;
        if self.mode == 7 {
            if word != 7 << 29 {
                return Err("invalid END_PB_SEGMENT payload".into());
            }
            return Ok(Word::End);
        }
        // Ampere's address is 12 bits; bit 12 is not a thirteenth method bit.
        if word & 0x1000 != 0 || !matches!(self.mode, 1 | 3 | 4 | 5) {
            return Err(format!("unsupported pushbuffer header {word:#010x}").into());
        }
        self.method = (word & 0xfff) * 4;
        self.subchannel = ((word >> 13) & 7) as u8;
        let count = (word >> 16) & 0x1fff;
        if self.mode == 4 {
            Ok(Word::Method {
                method: self.method,
                subchannel: self.subchannel,
                argument: count,
                last: true,
            })
        } else {
            self.remaining = count;
            Ok(Word::Header)
        }
    }
}
