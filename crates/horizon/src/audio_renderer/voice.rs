//! Stateful voice sources, wave-buffer ownership and status publication.
//! ABI and DSP semantics:
//! https://github.com/eden-emulator/mirror/blob/master/src/audio_core/renderer/voice/voice_info.h
//! https://github.com/eden-emulator/mirror/blob/master/src/audio_core/renderer/command/data_source/decode.cpp
use super::update::{MemoryPool, UpdateError, wide, word};
use super::worker::{RenderError, RenderMemory};
use nixe_memory::{AllocationRange, MemoryError};

use super::buffer::{Attachment, attach};
#[derive(Debug)]
struct WaveBuffer {
    data: Attachment,
    context: Option<Attachment>,
    start: u32,
    end: u32,
    looping: bool,
    end_of_stream: bool,
}
#[derive(Debug, Default)]
struct Channel {
    valid: [bool; 4],
    index: usize,
    offset: u32,
    played: u64,
    consumed: u32,
    adpcm: Adpcm,
    history: [i16; 8],
    fraction: u32,
    biquads: [super::biquad::State; 2],
}
#[derive(Debug, Default, Clone, Copy)]
struct Adpcm {
    header: u16,
    previous: i16,
    older: i16,
}
#[derive(Debug)]
pub(super) struct Voice {
    params: [u8; 0x170],
    coefficients: Option<Attachment>,
    buffers: [Option<WaveBuffer>; 4],
    channels: [Channel; 6],
    previous_volume: f32,
    stop_requested: bool,
}
impl Default for Voice {
    fn default() -> Self {
        Self {
            params: [0; 0x170],
            coefficients: None,
            buffers: std::array::from_fn(|_| None),
            channels: std::array::from_fn(|_| Channel::default()),
            previous_volume: 0.0,
            stop_requested: false,
        }
    }
}
impl Voice {
    pub(super) fn deactivate(&mut self) {
        self.params[9] = 0;
    }

    pub(super) fn uses_pool(&self, pool: usize) -> bool {
        self.params[9] != 0
            && (self
                .coefficients
                .as_ref()
                .is_some_and(|a| a.pool == Some(pool))
                || self.buffers.iter().flatten().any(|b| {
                    b.data.pool == Some(pool)
                        || b.context.as_ref().is_some_and(|a| a.pool == Some(pool))
                }))
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn update(
        &mut self,
        raw: &[u8],
        slot: usize,
        resources: usize,
        revision: u32,
        pools: &[Option<MemoryPool>],
        force: bool,
        translate: &impl Fn(u64, u64) -> Result<Vec<AllocationRange>, MemoryError>,
        output: &mut [u8],
    ) -> Result<(), UpdateError> {
        if raw[9] == 0 {
            self.params[9] = 0;
            return Ok(());
        }
        let fail = |detail: &str| {
            UpdateError::Unsupported(format!("RequestUpdate: voice slot {slot}: {detail}").into())
        };
        let channels = word(raw, 24) as usize;
        if !(1..=6).contains(&channels)
            || raw[10] > 2
            || word(raw, 0x3c) > 4
            || u16::from_le_bytes(raw[0x40..0x42].try_into().unwrap()) >= 4
        {
            return Err(UpdateError::Invalid);
        }
        if !matches!(raw[11], 2 | 5 | 6) {
            return Err(fail("sample format is not PCM16, PCM float or ADPCM"));
        }
        if raw[11] == 6 && channels != 1 {
            return Err(UpdateError::Invalid);
        }
        if word(raw, 0x5c) != u32::MAX {
            return Err(fail("splitter routing is not implemented"));
        }
        let pitch = f32::from_bits(word(raw, 0x1c));
        let volume = f32::from_bits(word(raw, 0x20));
        if !pitch.is_finite()
            || pitch < 0.0
            || !volume.is_finite()
            || word(raw, 12) == 0
            || raw[0x15e] > 2
        {
            return Err(UpdateError::Invalid);
        }
        let flags = raw[0x15c];
        if flags & !3 != 0 {
            return Err(fail("unknown voice flags"));
        }
        if raw[8] == 0
            && word(&self.params, 24) != 0
            && (self.params[11] != raw[11]
                || word(&self.params, 24) != word(raw, 24)
                || self.params[0x140..0x158] != raw[0x140..0x158])
        {
            return Err(fail(
                "changing format or channel resources without a new voice is not implemented",
            ));
        }
        for i in 0..channels {
            if word(raw, 0x140 + i * 4) as usize >= resources {
                return Err(UpdateError::Invalid);
            }
        }
        if raw[8] != 0 {
            *self = Self::default();
            self.previous_volume = volume;
            for channel in &mut self.channels {
                channel.index =
                    usize::from(u16::from_le_bytes(raw[0x40..0x42].try_into().unwrap()));
            }
        }
        self.stop_requested = raw[10] == 1 && self.params[10] != 1;
        let address = wide(raw, 0x48);
        let size = wide(raw, 0x50);
        if self.coefficients.as_ref().map(|a| (a.address, a.bytes))
            != ((address != 0 && size != 0).then_some((address, size)))
        {
            self.coefficients = attach(address, size, pools, force, translate)?;
        }
        if raw[11] == 6 && self.coefficients.as_ref().is_none_or(|a| a.bytes != 32) {
            return Err(UpdateError::Invalid);
        }
        for i in 0..4 {
            let wave = &raw[0x60 + i * 0x38..0x98 + i * 0x38];
            if wave[0x1a] != 0 {
                continue;
            }
            let start = word(wave, 16);
            let end = word(wave, 20);
            if start > i32::MAX as u32 || end > i32::MAX as u32 || start > end {
                return Err(UpdateError::Invalid);
            }
            let Some(data) = attach(wide(wave, 0), wide(wave, 8), pools, force, translate)? else {
                self.buffers[i] = None;
                for c in &mut self.channels {
                    c.valid[i] = false;
                }
                continue;
            };
            let required = match raw[11] {
                6 => {
                    u64::from(end / 14) * 8
                        + if end.is_multiple_of(14) {
                            0
                        } else {
                            1 + u64::from(end % 14).div_ceil(2)
                        }
                }
                2 => u64::from(end) * channels as u64 * 2,
                _ => u64::from(end) * channels as u64 * 4,
            };
            if required > data.bytes {
                return Err(UpdateError::Invalid);
            }
            let context = if raw[11] == 6 && revision >= 2 {
                attach(wide(wave, 0x20), wide(wave, 0x28), pools, force, translate)?
            } else {
                None
            };
            if context.as_ref().is_some_and(|a| a.bytes != 6) {
                return Err(UpdateError::Invalid);
            }
            self.buffers[i] = Some(WaveBuffer {
                data,
                context,
                start,
                end,
                looping: wave[0x18] != 0,
                end_of_stream: wave[0x19] != 0,
            });
            for c in &mut self.channels {
                c.valid[i] = true;
            }
        }
        if revision >= 5 {
            for _ in 0..raw[0x159] {
                let i = self.channels[0].index;
                for c in &mut self.channels {
                    if c.valid[i] {
                        c.valid[i] = false;
                        c.consumed = c.consumed.wrapping_add(1);
                    }
                    c.index = (i + 1) % 4;
                    c.offset = 0;
                }
            }
        }
        self.params.copy_from_slice(raw);
        let c = &self.channels[0];
        if raw[8] == 0 {
            output[..8].copy_from_slice(&c.played.to_le_bytes());
            output[8..12].copy_from_slice(&c.consumed.to_le_bytes());
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn render(
        &mut self,
        memory: &RenderMemory,
        revision: u32,
        scratch: &mut VoiceScratch,
        samples: usize,
        resources: &[[u8; 0x70]],
        previous_resources: &[[f32; 24]],
        mix: &mut [i32],
        mix_buffers: usize,
    ) -> Result<(), RenderError> {
        if self.params[9] == 0 {
            return Ok(());
        }
        let channel_count = word(&self.params, 24) as usize;
        if self.stop_requested {
            for c in &mut self.channels[..channel_count] {
                let consumed = c
                    .consumed
                    .wrapping_add(c.valid.iter().filter(|&&v| v).count() as u32);
                *c = Channel {
                    consumed,
                    index: (c.index + c.valid.iter().filter(|&&v| v).count()) % 4,
                    ..Channel::default()
                };
            }
            self.stop_requested = false;
            self.previous_volume = 0.0;
        }
        if self.params[10] != 0 || word(&self.params, 0x3c) == 0 {
            return Ok(());
        }
        let quality = self.params[0x15e];
        let history_len = super::resample::history_len(quality);
        let skip = revision >= 5 && self.params[0x15c] & 2 != 0;
        let ratio = (word(&self.params, 12) as f32 / 48000.0
            * f32::from_bits(word(&self.params, 0x1c))
            * 32768.0) as u32;
        let volume = f32::from_bits(word(&self.params, 0x20));
        let mut coefficients = [0_i16; 16];
        if self.params[11] == 6 {
            let mut data = [0; 32];
            self.coefficients
                .as_ref()
                .unwrap()
                .read(memory, 0, &mut data)?;
            for (out, input) in coefficients.iter_mut().zip(data.chunks_exact(2)) {
                *out = i16::from_le_bytes(input.try_into().unwrap());
            }
        }
        let precision = if revision >= 7 { 23 } else { 15 };
        let biquads = [
            super::biquad::Parameters::decode(&self.params[0x24..0x30]),
            super::biquad::Parameters::decode(&self.params[0x30..0x3c]),
        ];
        for channel in 0..channel_count {
            let c = &mut self.channels[channel];
            if !c.valid.iter().any(|&v| v) {
                continue;
            }
            let count = if skip {
                samples
            } else {
                ((u64::from(c.fraction) + samples as u64 * u64::from(ratio)) >> 15) as usize
            };
            if count + history_len > 0x3f00 {
                return Err("rendering: voice pitch exceeds the DSP source buffer".into());
            }
            scratch.decoded.clear();
            if !skip {
                scratch.decoded.extend_from_slice(&c.history[..history_len]);
            }
            let prefix = scratch.decoded.len();
            scratch.decoded.resize(prefix + count, 0);
            let mut written = 0;
            while written < count {
                let i = c.index;
                let Some(buffer) = self.buffers[i].as_ref().filter(|_| c.valid[i]) else {
                    break;
                };
                if c.offset == 0
                    && self.params[11] == 6
                    && let Some(context) = &buffer.context
                {
                    let mut data = [0; 6];
                    context.read(memory, 0, &mut data)?;
                    c.adpcm = Adpcm {
                        header: u16::from_le_bytes(data[..2].try_into().unwrap()),
                        previous: i16::from_le_bytes(data[2..4].try_into().unwrap()),
                        older: i16::from_le_bytes(data[4..6].try_into().unwrap()),
                    };
                }
                let available = (buffer.end - buffer.start - c.offset) as usize;
                let take = available.min(count - written);
                decode(
                    memory,
                    buffer,
                    buffer.start + c.offset,
                    &mut scratch.decoded[prefix + written..prefix + written + take],
                    self.params[11],
                    channel,
                    channel_count,
                    &coefficients,
                    &mut c.adpcm,
                    &mut scratch.bytes,
                )?;
                written += take;
                c.offset += take as u32;
                c.played = c.played.wrapping_add(take as u64);
                if c.offset < buffer.end - buffer.start {
                    continue;
                }
                c.offset = 0;
                if buffer.looping && take != 0 {
                    if revision >= 5 && self.params[0x15c] & 1 != 0 {
                        c.played = 0;
                    }
                } else {
                    c.valid[i] = false;
                    c.index = (i + 1) % 4;
                    c.consumed = c.consumed.wrapping_add(1);
                    if buffer.end_of_stream {
                        c.played = 0;
                    }
                }
                if take == 0 {
                    break;
                }
            }
            scratch.output.resize(samples, 0);
            if skip {
                for (dst, src) in scratch.output.iter_mut().zip(&scratch.decoded) {
                    *dst = i32::from(*src);
                }
            } else {
                super::resample::resample(
                    &mut scratch.output,
                    &scratch.decoded,
                    ratio,
                    &mut c.fraction,
                    quality,
                );
                c.history[..history_len]
                    .copy_from_slice(&scratch.decoded[count..count + history_len]);
            }
            for (state, params) in c.biquads.iter_mut().zip(biquads) {
                state.process(params, revision, &mut scratch.output);
            }
            let resource = word(&self.params, 0x140 + channel * 4) as usize;
            let current = &resources[resource];
            let previous = &previous_resources[resource];
            let voice_ramp = (volume - self.previous_volume) / samples as f32;
            let scale = (1_u64 << precision) as f32;
            let mut voice_gain = (self.previous_volume * scale) as i64;
            let voice_step = (voice_ramp * scale) as i64;
            for sample in &mut scratch.output {
                *sample = super::render::fixed_gain(*sample, voice_gain, precision);
                voice_gain += voice_step;
            }
            for (output, &old) in previous.iter().enumerate().take(mix_buffers) {
                let new = f32::from_bits(word(current, 4 + output * 4));
                let mut gain = (old * scale) as i64;
                let step = (((new - old) / samples as f32) * scale) as i64;
                if gain == 0 && step == 0 {
                    continue;
                }
                for (dst, &src) in mix[output * samples..(output + 1) * samples]
                    .iter_mut()
                    .zip(&scratch.output)
                {
                    *dst = dst.wrapping_add(super::render::fixed_gain(src, gain, precision));
                    gain += step;
                }
            }
        }
        self.previous_volume = volume;
        Ok(())
    }
    pub(super) fn mix_id(&self) -> usize {
        word(&self.params, 0x58) as usize
    }
}
#[derive(Debug, Default)]
pub(super) struct VoiceScratch {
    decoded: Vec<i16>,
    output: Vec<i32>,
    bytes: Vec<u8>,
}
#[allow(clippy::too_many_arguments)]
fn decode(
    memory: &RenderMemory,
    buffer: &WaveBuffer,
    start: u32,
    output: &mut [i16],
    format: u8,
    channel: usize,
    channels: usize,
    coefficients: &[i16; 16],
    context: &mut Adpcm,
    bytes: &mut Vec<u8>,
) -> Result<(), RenderError> {
    if output.is_empty() {
        return Ok(());
    }
    if format == 6 {
        let first = start / 14;
        let last = (start as usize + output.len() - 1) / 14;
        let offset = u64::from(first) * 8;
        let size = ((last - first as usize + 1) * 8).min((buffer.data.bytes - offset) as usize);
        bytes.resize(size, 0);
        buffer.data.read(memory, offset, bytes)?;
        for (i, out) in output.iter_mut().enumerate() {
            let position = start as usize + i;
            let frame = (position / 14 - first as usize) * 8;
            let sample = position % 14;
            if sample == 0 {
                context.header = u16::from(bytes[frame]);
            }
            let packed = bytes[frame + 1 + sample / 2];
            let nibble = if sample.is_multiple_of(2) {
                packed >> 4
            } else {
                packed & 15
            };
            let signed = if nibble < 8 {
                i32::from(nibble)
            } else {
                i32::from(nibble) - 16
            };
            let coefficient = ((context.header >> 4) & 7) as usize * 2;
            let prediction = i64::from(coefficients[coefficient]) * i64::from(context.previous)
                + i64::from(coefficients[coefficient + 1]) * i64::from(context.older);
            let value = (prediction
                + ((i64::from(signed) * (1_i64 << (context.header & 15))) << 11)
                + 1024)
                >> 11;
            let value = value.clamp(-32768, 32767) as i16;
            context.older = context.previous;
            context.previous = value;
            *out = value;
        }
    } else {
        let width = if format == 2 { 2 } else { 4 };
        bytes.resize(output.len() * channels * width, 0);
        buffer
            .data
            .read(memory, u64::from(start) * (channels * width) as u64, bytes)?;
        for (i, out) in output.iter_mut().enumerate() {
            let p = (i * channels + channel) * width;
            *out = if width == 2 {
                i16::from_le_bytes(bytes[p..p + 2].try_into().unwrap())
            } else {
                (f32::from_le_bytes(bytes[p..p + 4].try_into().unwrap()) * 32767.0) as i16
            };
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{audio_renderer::worker::RenderMemory, test_memory::physical as memory};
    fn buffer(bytes: &[u8], end: u32) -> WaveBuffer {
        WaveBuffer {
            data: Attachment {
                address: 0x1000,
                ranges: nixe_memory::CapturedMemory::new(memory(bytes)).unwrap(),
                bytes: bytes.len() as u64,
                pool: Some(0),
            },
            context: None,
            start: 0,
            end,
            looping: false,
            end_of_stream: false,
        }
    }
    fn put(data: &mut [u8], offset: usize, value: u32) {
        data[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
    #[test]
    fn adpcm_nibbles_prediction_saturation_and_split_blocks_preserve_context() {
        let mut bytes = vec![0x00, 0x17, 0x8f, 0x20, 0x00, 0x00, 0x00, 0x00];
        bytes.extend_from_slice(&[0x10, 0x70, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);
        let buffer = buffer(&bytes, 28);
        let mut coefficients = [0; 16];
        coefficients[2] = 2048;
        let mut whole = [0; 28];
        let mut context = Adpcm::default();
        decode(
            &RenderMemory::testing(),
            &buffer,
            0,
            &mut whole,
            6,
            0,
            1,
            &coefficients,
            &mut context,
            &mut Vec::new(),
        )
        .unwrap();
        assert_eq!(&whole[..6], &[1, 7, -8, -1, 2, 0]);
        assert_eq!(&whole[14..], &[7; 14]);
        let mut split = [0; 28];
        let mut context = Adpcm::default();
        decode(
            &RenderMemory::testing(),
            &buffer,
            0,
            &mut split[..9],
            6,
            0,
            1,
            &coefficients,
            &mut context,
            &mut Vec::new(),
        )
        .unwrap();
        decode(
            &RenderMemory::testing(),
            &buffer,
            9,
            &mut split[9..],
            6,
            0,
            1,
            &coefficients,
            &mut context,
            &mut Vec::new(),
        )
        .unwrap();
        assert_eq!(split, whole);
        let buffer = super::tests::buffer(&[0x0f, 0x78, 0, 0, 0, 0, 0, 0], 2);
        let mut clipped = [0; 2];
        decode(
            &RenderMemory::testing(),
            &buffer,
            0,
            &mut clipped,
            6,
            0,
            1,
            &coefficients,
            &mut Adpcm::default(),
            &mut Vec::new(),
        )
        .unwrap();
        assert_eq!(clipped, [32767, -32768]);
    }
    #[test]
    fn adpcm_final_frame_does_not_require_unused_nibbles() {
        let buffer = buffer(&[0x00, 0x17, 0x8f], 4);
        let mut output = [0; 4];
        decode(
            &RenderMemory::testing(),
            &buffer,
            0,
            &mut output,
            6,
            0,
            1,
            &[0; 16],
            &mut Adpcm::default(),
            &mut Vec::new(),
        )
        .unwrap();
        assert_eq!(output, [1, 7, -8, -1]);
    }
    #[test]
    fn pcm_deinterleaves_and_float_conversion_saturates() {
        let bytes: Vec<_> = [1_i16, 100, -2, -200, 3, 300]
            .into_iter()
            .flat_map(i16::to_le_bytes)
            .collect();
        let buffer = buffer(&bytes, 3);
        let mut output = [0; 2];
        decode(
            &RenderMemory::testing(),
            &buffer,
            1,
            &mut output,
            2,
            1,
            2,
            &[0; 16],
            &mut Adpcm::default(),
            &mut Vec::new(),
        )
        .unwrap();
        assert_eq!(output, [-200, 300]);
        let bytes: Vec<_> = [-2.0_f32, -0.5, 0.5, 2.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let buffer = super::tests::buffer(&bytes, 4);
        let mut output = [0; 4];
        decode(
            &RenderMemory::testing(),
            &buffer,
            0,
            &mut output,
            5,
            0,
            1,
            &[0; 16],
            &mut Adpcm::default(),
            &mut Vec::new(),
        )
        .unwrap();
        assert_eq!(output, [-32768, -16383, 16383, 32767]);
    }
    fn pcm_voice(looping: bool) -> (Voice, [u8; 0x170], Vec<Option<MemoryPool>>) {
        let mut raw = [0; 0x170];
        raw[8] = 1;
        raw[9] = 1;
        raw[11] = 2;
        for (offset, value) in [
            (12, 48000),
            (24, 1),
            (0x1c, 1_f32.to_bits()),
            (0x20, 1_f32.to_bits()),
            (0x3c, 1),
            (0x5c, u32::MAX),
            (0x74, 4),
        ] {
            put(&mut raw, offset, value);
        }
        raw[0x60..0x68].copy_from_slice(&0x1000_u64.to_le_bytes());
        raw[0x68..0x70].copy_from_slice(&8_u64.to_le_bytes());
        raw[0x78] = u8::from(looping);
        raw[0x15c] = 2;
        for i in 1..4 {
            raw[0x7a + i * 0x38] = 1;
        }
        let bytes: Vec<_> = [100_i16, 200, -300, 400]
            .into_iter()
            .flat_map(i16::to_le_bytes)
            .collect();
        let pools = vec![Some(MemoryPool {
            address: 0x1000,
            ranges: nixe_memory::CapturedMemory::new(memory(&bytes)).unwrap(),
            bytes: bytes.len() as u64,
        })];
        let mut voice = Voice::default();
        voice
            .update(&raw, 0, 1, 7, &pools, false, &|_, _| panic!(), &mut [0; 16])
            .unwrap();
        (voice, raw, pools)
    }
    fn play(voice: &mut Voice, samples: usize) -> Vec<i32> {
        let mut resource = [0; 0x70];
        put(&mut resource, 4, 1_f32.to_bits());
        let mut previous = [0_f32; 24];
        previous[0] = 1.0;
        let mut mix = vec![0; samples];
        voice
            .render(
                &RenderMemory::testing(),
                7,
                &mut VoiceScratch::default(),
                samples,
                &[resource],
                &[previous],
                &mut mix,
                1,
            )
            .unwrap();
        mix
    }
    #[test]
    fn voice_applies_both_biquads_in_order_after_source_conversion() {
        let (mut voice, mut raw, pools) = pcm_voice(true);
        raw[8] = 0;
        for offset in [0x24, 0x30] {
            raw[offset] = 1;
            raw[offset + 2..offset + 4].copy_from_slice(&8192_i16.to_le_bytes());
        }
        voice
            .update(&raw, 0, 1, 7, &pools, false, &|_, _| panic!(), &mut [0; 16])
            .unwrap();
        assert_eq!(play(&mut voice, 6), [25, 50, -75, 100, 25, 50]);
        // Disabling either stage must restore that stage's unfiltered input.
        raw[0x30] = 0;
        raw[0x7a] = 1;
        voice
            .update(&raw, 0, 1, 7, &pools, false, &|_, _| panic!(), &mut [0; 16])
            .unwrap();
        assert_eq!(play(&mut voice, 2), [-150, 200]);
    }
    #[test]
    fn real_buffer_consumption_status_pause_stop_and_looping_are_stateful() {
        let (mut voice, mut raw, pools) = pcm_voice(false);
        assert!(voice.uses_pool(0));
        assert_eq!(play(&mut voice, 3), [100, 200, -300]);
        raw[8] = 0;
        raw[10] = 2;
        raw[0x7a] = 1;
        let mut status = [0; 16];
        voice
            .update(&raw, 0, 1, 7, &pools, false, &|_, _| panic!(), &mut status)
            .unwrap();
        assert_eq!(u64::from_le_bytes(status[..8].try_into().unwrap()), 3);
        assert_eq!(play(&mut voice, 3), [0; 3]);
        raw[10] = 0;
        voice
            .update(&raw, 0, 1, 7, &pools, false, &|_, _| panic!(), &mut status)
            .unwrap();
        assert_eq!(play(&mut voice, 3), [400, 0, 0]);
        voice
            .update(&raw, 0, 1, 7, &pools, false, &|_, _| panic!(), &mut status)
            .unwrap();
        assert_eq!(word(&status, 8), 1);
        assert_eq!(u64::from_le_bytes(status[..8].try_into().unwrap()), 4);
        let (mut looping, mut raw, pools) = pcm_voice(true);
        assert_eq!(play(&mut looping, 6), [100, 200, -300, 400, 100, 200]);
        assert_eq!(looping.channels[0].consumed, 0);
        assert_eq!(looping.channels[0].offset, 2);
        raw[8] = 0;
        raw[10] = 1;
        raw[0x7a] = 1;
        looping
            .update(&raw, 0, 1, 7, &pools, false, &|_, _| panic!(), &mut status)
            .unwrap();
        assert_eq!(play(&mut looping, 2), [0; 2]);
        assert_eq!(looping.channels[0].consumed, 1);
        assert_eq!(looping.channels[0].played, 0);
    }
    #[test]
    fn forced_mapping_and_pool_bounds_are_real_memory_attachments() {
        let (_, _, pools) = pcm_voice(false);
        assert!(attach(0x1000, 9, &pools, false, &|_, _| panic!()).is_err());
        let mapped = attach(0x2000, 4, &pools, true, &|address, size| {
            assert_eq!((address, size), (0x2000, 4));
            Ok(memory(&[1, 2, 3, 4]))
        })
        .unwrap()
        .unwrap();
        let mut bytes = [0; 4];
        mapped
            .read(&RenderMemory::testing(), 0, &mut bytes)
            .unwrap();
        assert_eq!(bytes, [1, 2, 3, 4]);
        assert!((mapped.pool).is_none());
    }
}
