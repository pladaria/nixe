//! Switch 2 batch-codec wire layout, separate from the libopus decoder.
//! @SPECULATIVE: this ABI is reconstructed from native SDK calls in the reference.
//! https://github.com/zortos293/Lichii/blob/646b5d17135a2b2701ccc7a959afd0bbbf2f6160/src/core/hle/service/audio/hardware_audio_codec.cpp

use super::*;

pub(super) fn word(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}
pub(super) fn long(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap())
}

pub(crate) fn manager_size(count: u32) -> Result<u32, HorizonIpcResult> {
    if !(1..=65536).contains(&count) {
        return Err(INVALID);
    }
    Ok((64 + count * 16).next_multiple_of(64))
}

#[derive(Debug)]
pub(crate) struct Config {
    pub rate: u32,
    pub channels: u32,
    pub streams: Option<(u32, u32)>,
    pub frames: u32,
    pub mapping: [u8; 255],
}
impl Config {
    pub fn parse(bytes: &[u8]) -> Result<Self, HorizonIpcResult> {
        if bytes.len() != 0x120 {
            return Err(INVALID);
        }
        let kind = long(bytes, 0);
        if kind == 0 {
            return Err(audio_result(513));
        }
        if kind != 1 && kind != 2 {
            return Err(INVALID);
        }
        let rate = word(bytes, 8);
        let channels = word(bytes, 12);
        if !matches!(rate, 8000 | 12000 | 16000 | 24000 | 48000) {
            return Err(HorizonIpcResult::from_raw(111 | (1001 << 9)));
        }
        if !(1..=if kind == 1 { 2 } else { 255 }).contains(&channels) {
            return Err(HorizonIpcResult::from_raw(111 | (1002 << 9)));
        }
        let duration = long(bytes, 16);
        if !matches!(duration / 1000, 2500 | 5000 | 10000 | 20000) {
            return Err(INVALID);
        }
        let mut config = Self {
            rate,
            channels,
            streams: None,
            frames: (u64::from(rate) * duration).div_ceil(1_000_000_000) as u32,
            mapping: [0; 255],
        };
        if kind == 2 {
            let streams = word(bytes, 24);
            let coupled = word(bytes, 28);
            if !(1..=255).contains(&streams) || coupled > streams || streams + coupled > channels {
                return Err(INVALID);
            }
            config.mapping.copy_from_slice(&bytes[32..287]);
            if config.mapping[..channels as usize]
                .iter()
                .any(|&channel| channel != 255 && u32::from(channel) >= streams + coupled)
            {
                return Err(INVALID);
            }
            config.streams = Some((streams, coupled));
        }
        Ok(config)
    }
    pub fn required_bytes(&self) -> Result<u32, HorizonIpcResult> {
        // Guest work sizes must not depend on optional host libopus features.
        // Config is 0x140 aligned bytes, followed by the 24-byte decoder wrapper
        // and the non-neural libopus state (8-byte aligned for multistream).
        let state = self.streams.map_or(
            if self.channels == 1 { 18468 } else { 27236 },
            |(streams, coupled)| 272 + coupled * 27240 + (streams - coupled) * 18472,
        );
        Ok(0x140 + (24_u32 + state).next_multiple_of(64))
    }
    pub fn decoder(&self) -> Result<Decoder, HorizonIpcResult> {
        Decoder::new(
            self.rate,
            self.channels,
            self.streams.map(|(streams, coupled)| {
                (streams, coupled, &self.mapping[..self.channels as usize])
            }),
        )
        .map_err(codec_error)
    }
}

#[derive(Debug)]
pub(crate) struct Layout {
    pub instance_stride: u32,
    pub instance_count: u32,
    pub input_stride: u32,
    pub output_stride: u32,
    pub io_count: u32,
    pub io_offset: u32,
    pub completion_offset: u32,
}
impl Layout {
    pub fn parse(
        data: &[u8],
        transfer: u64,
        address: u64,
        process: u64,
        properties: u64,
    ) -> Result<Self, HorizonIpcResult> {
        if data.len() != 0x60 {
            return Err(INVALID);
        }
        if word(data, 0) != 1 {
            return Err(audio_result(1537));
        }
        if data[72..].iter().any(|&byte| byte != 0) || properties & 7 == 0 {
            return Err(INVALID);
        }
        let l = Self {
            instance_stride: word(data, 52),
            instance_count: word(data, 56),
            input_stride: word(data, 60),
            output_stride: word(data, 64),
            io_count: word(data, 68),
            io_offset: word(data, 36),
            completion_offset: word(data, 44),
        };
        let manager = manager_size(l.io_count)?;
        if !(1..=65536).contains(&l.instance_count)
            || l.instance_stride == 0
            || !l.instance_stride.is_multiple_of(64)
            || l.input_stride < 0x600
            || !l.input_stride.is_multiple_of(64)
            || l.output_stride == 0
            || !l.output_stride.is_multiple_of(64)
        {
            return Err(INVALID);
        }
        let ts = u64::from(word(data, 28));
        let ps = u64::from(word(data, 48));
        if transfer == 0
            || !transfer.is_multiple_of(4096)
            || process == 0
            || !process.is_multiple_of(4096)
            || !address.is_multiple_of(4096)
            || address.checked_add(process).is_none()
            || ts > transfer
            || ps > process
            || !ts.is_multiple_of(4096)
            || !ps.is_multiple_of(4096)
        {
            return Err(ADDRESS);
        }
        let queue = (0xc0 + u64::from(l.io_count) * 64).next_multiple_of(4096);
        if word(data, 4) < manager
            || u64::from(word(data, 12))
                < u64::from(l.instance_count) * u64::from(l.instance_stride)
            || u64::from(word(data, 32))
                < u64::from(l.io_count) * (u64::from(l.input_stride) + u64::from(l.output_stride))
            || u64::from(word(data, 20)) < queue
            || u64::from(word(data, 40)) < queue
        {
            return Err(BUFFER_TOO_SMALL);
        }
        let regions: [(u64, u64); 3] = std::array::from_fn(|i| {
            (
                u64::from(word(data, 8 + i * 8)),
                u64::from(word(data, 4 + i * 8)),
            )
        });
        for (i, &(offset, size)) in regions.iter().enumerate() {
            if offset % if i == 2 { 4096 } else { 64 } != 0
                || !contains(ts, offset, size)
                || regions[..i]
                    .iter()
                    .any(|&(other, bytes)| overlaps(offset, size, other, bytes))
            {
                return Err(ADDRESS);
            }
        }
        let ios = u64::from(word(data, 32));
        let cs = u64::from(word(data, 40));
        if !l.io_offset.is_multiple_of(64)
            || !l.completion_offset.is_multiple_of(4096)
            || !contains(ps, u64::from(l.io_offset), ios)
            || !contains(ps, u64::from(l.completion_offset), cs)
            || overlaps(
                u64::from(l.io_offset),
                ios,
                u64::from(l.completion_offset),
                cs,
            )
        {
            return Err(ADDRESS);
        }
        Ok(l)
    }
}
pub(super) fn contains(total: u64, offset: u64, size: u64) -> bool {
    offset <= total && size <= total - offset
}
pub(crate) fn overlaps(a: u64, a_size: u64, b: u64, b_size: u64) -> bool {
    if a <= b {
        b - a < a_size
    } else {
        a - b < b_size
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Task {
    pub tag: u16,
    pub instance: u16,
    pub io: u16,
    pub sequence: u16,
    pub input_bytes: u32,
    pub reset: bool,
    pub notify: bool,
}
impl Task {
    pub fn parse(bytes: &[u8]) -> Result<Self, HorizonIpcResult> {
        if bytes.len() != 16 || bytes[12] > 1 || bytes[13] > 1 || bytes[14..] != [0, 0] {
            return Err(INVALID);
        }
        let short = |at| u16::from_le_bytes(bytes[at..at + 2].try_into().unwrap());
        Ok(Self {
            tag: short(0),
            instance: short(2),
            io: short(4),
            sequence: short(6),
            input_bytes: word(bytes, 8),
            reset: bytes[12] != 0,
            notify: bytes[13] != 0,
        })
    }
    pub fn completion(self, result: HorizonIpcResult, bytes: u32, duration: u64) -> Vec<u8> {
        let mut output = vec![0; 64];
        for (chunk, value) in
            output[..8]
                .chunks_exact_mut(2)
                .zip([self.tag, self.instance, self.io, self.sequence])
        {
            chunk.copy_from_slice(&value.to_le_bytes());
        }
        output[8..16].copy_from_slice(&duration.min(i64::MAX as u64).to_le_bytes());
        output[16..20].copy_from_slice(&bytes.to_le_bytes());
        output[20..24].copy_from_slice(&result.raw().to_le_bytes());
        output
    }
}
