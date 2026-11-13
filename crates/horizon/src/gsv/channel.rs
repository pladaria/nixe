//! Channel ownership is independent of descriptor and CMIF-session lifetimes.
use std::{
    collections::BTreeSet,
    sync::{Arc, Mutex},
};

#[derive(Debug, Default)]
pub(crate) struct Channel {
    pub runlist: Option<u32>,
    pub space: Option<u32>,
    pub group: Option<(Arc<Mutex<super::Tsg>>, u32)>,
    pub entries: Option<u32>,
    pub inflight_jobs: u32,
    pub deterministic: bool,
    pub timeline: Option<Arc<Mutex<nixe_gpu::GuestTimeline>>>,
    pub objects: BTreeSet<u32>,
    pub client_data: u64,
    pub timeout_ms: u32,
    pub timeslice_us: u32,
    pub priority: u32,
    pub error_notifier: bool,
    pub preemption: [u32; 2],
    pub zcull: Option<(u64, u32)>,
}

pub(crate) struct GpfifoEntry {
    pub address: u64,
    pub words: u32,
}

/// AMPERE_CHANNEL_GPFIFO_A hardware descriptor format. Bit 8 is reserved,
/// unlike Maxwell's ALLOW_FLUSH field; keep generation-specific validation.
/// https://github.com/NVIDIA/open-gpu-kernel-modules/blob/main/src/common/sdk/nvidia/inc/class/clc56f.h
pub(crate) fn decode_entries(bytes: &[u8]) -> Result<Vec<GpfifoEntry>, Box<str>> {
    bytes
        .chunks_exact(8)
        .enumerate()
        .map(|(index, bytes)| {
            let low = u32::from_le_bytes(bytes[..4].try_into().unwrap());
            let high = u32::from_le_bytes(bytes[4..].try_into().unwrap());
            let words = (high >> 10) & 0x1f_ffff;
            if words == 0 || low & 3 != 0 || high & 0x100 != 0 {
                return Err(format!(
                    "unsupported Ampere GPFIFO entry: index={index} low={low:#010x} \
                high={high:#010x}"
                )
                .into_boxed_str());
            }
            let address = u64::from(low & !3) | (u64::from(high & 0xff) << 32);
            if address + u64::from(words) * 4 > 1 << 40 {
                return Err(format!(
                    "Ampere GPFIFO range exceeds 40-bit VA: index={index} \
                address={address:#x} words={words}"
                )
                .into_boxed_str());
            }
            Ok(GpfifoEntry { address, words })
        })
        .collect()
}
