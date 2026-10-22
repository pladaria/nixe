//! Voice decoding, scalar mix commands and stereo device output.
//!
//! Integer mix buffers, Q15/Q23 gains and device saturation:
//! https://github.com/eden-emulator/mirror/blob/master/src/audio_core/renderer/command/mix/mix.cpp
//! https://github.com/eden-emulator/mirror/blob/master/src/audio_core/renderer/command/mix/volume.cpp
//! https://github.com/eden-emulator/mirror/blob/master/src/audio_core/renderer/command/sink/device.cpp

use super::{
    RendererParameters,
    update::{RendererUpdateState, word},
    worker::{RenderError, RenderMemory},
};

#[derive(Debug)]
pub(super) struct RenderBuffers {
    samples: Vec<i32>,
    offsets: Vec<usize>,
    order: Vec<(usize, usize)>,
    voice_scratch: super::voice::VoiceScratch,
}

fn gain(sample: i32, volume: f32, precision: u32) -> Result<i32, Box<str>> {
    let scaled = volume * (1_u64 << precision) as f32;
    if !scaled.is_finite() || scaled.abs() >= (1_u64 << 63) as f32 {
        return Err("rendering: mix gain cannot be represented in fixed point".into());
    }
    Ok(fixed_gain(sample, scaled as i64, precision))
}

pub(super) fn fixed_gain(sample: i32, gain: i64, precision: u32) -> i32 {
    let raw = i128::from(sample) * i128::from(gain);
    // Match Common::FixedPoint::to_int, including negative fractional samples.
    // https://github.com/eden-emulator/mirror/blob/master/src/common/fixed_point.h
    let rounded = raw + ((raw & ((1_i128 << precision) - 1)) >> 1);
    (rounded >> precision) as i32
}

// Host adaptation after the guest device command's s16 saturation. Channel
// order is FL, FR, C, LFE, BL, BR. These are host playback coefficients, not the
// separate guest DownMix6chTo2ch command's configurable coefficients.
// Adapted from Eden (GPL-3.0-or-later), Copyright 2026 Eden Emulator Project;
// Yuzu (GPL-2.0-or-later), Copyright 2018 yuzu Emulator Project:
// https://github.com/eden-emulator/mirror/blob/master/src/audio_core/sink/sink_stream.cpp
fn surround_to_stereo(channels: [i16; 6], volume: f32) -> [i16; 2] {
    let [left, right, center, lfe, back_left, back_right] = channels.map(f32::from);
    let left = left + center * 0.596 + lfe * 0.354 + back_left * 0.707;
    let right = right + center * 0.596 + lfe * 0.354 + back_right * 0.707;
    [left, right]
        .map(|sample| ((sample * volume) as i32).clamp(i16::MIN as i32, i16::MAX as i32) as i16)
}

impl RenderBuffers {
    pub(super) fn new(params: &RendererParameters) -> Self {
        Self {
            samples: vec![0; params.mixes as usize * params.samples as usize],
            offsets: vec![0; params.submixes as usize + 1],
            order: Vec::with_capacity(params.submixes as usize + 1),
            voice_scratch: super::voice::VoiceScratch::default(),
        }
    }

    pub(super) fn render(
        &mut self,
        memory: &RenderMemory,
        state: &mut RendererUpdateState,
        params: &RendererParameters,
        pcm: &mut [i16],
        volume: f32,
    ) -> Result<bool, RenderError> {
        self.samples.fill(0);
        self.prepare_mixes(state, params)?;
        let samples = params.samples as usize;
        for voice in &mut state.voices {
            let id = voice.mix_id();
            let Some(mix) = state.mixes.get(id).and_then(Option::as_ref) else {
                continue;
            };
            let buffers = word(mix.as_slice(), 8) as usize;
            voice.render(
                memory,
                params.revision_number().unwrap(),
                &mut self.voice_scratch,
                samples,
                &state.resources,
                &state.previous_resources,
                &mut self.samples
                    [self.offsets[id] * samples..(self.offsets[id] + buffers) * samples],
                buffers,
            )?;
        }
        for (previous, current) in state.previous_resources.iter_mut().zip(&state.resources) {
            for (i, value) in previous.iter_mut().enumerate() {
                *value = f32::from_bits(word(current, 4 + i * 4));
            }
        }
        self.mix_and_output(state, params, pcm, volume)
            .map_err(Into::into)
    }

    fn prepare_mixes(
        &mut self,
        state: &RendererUpdateState,
        params: &RendererParameters,
    ) -> Result<(), Box<str>> {
        let mut offset = 0;
        self.order.clear();
        for (id, mix) in state.mixes.iter().enumerate() {
            let Some(mix) = mix else { continue };
            if word(mix.as_slice(), 4) != params.sample_rate {
                return Err(format!(
                    "rendering: mix {id} requires sample-rate conversion ({} Hz)",
                    word(mix.as_slice(), 4)
                )
                .into());
            }
            self.offsets[id] = offset;
            offset += word(mix.as_slice(), 8) as usize;
            let mut depth = 0;
            let mut current = id;
            while current != 0 {
                let Some(mix) = state.mixes.get(current).and_then(Option::as_ref) else {
                    break;
                };
                let next = word(mix.as_slice(), 0x924);
                if next == i32::MAX as u32 {
                    break;
                }
                current = next as usize;
                depth += 1;
                if depth > state.mixes.len() {
                    return Err("rendering: cyclic mix routing".into());
                }
            }
            self.order.push((id, depth));
        }
        // No allocation and no comparison closure holding mutable graph state.
        self.order
            .sort_unstable_by_key(|&(_, depth)| std::cmp::Reverse(depth));
        Ok(())
    }

    fn mix_and_output(
        &mut self,
        state: &mut RendererUpdateState,
        params: &RendererParameters,
        pcm: &mut [i16],
        volume: f32,
    ) -> Result<bool, Box<str>> {
        let samples = params.samples as usize;
        let precision = if params.revision_number().unwrap() >= 7 {
            23
        } else {
            15
        };
        for &(id, _) in &self.order {
            let mix = state.mixes[id].as_ref().unwrap();
            let count = word(mix.as_slice(), 8) as usize;
            for &slot in &state.effect_order[id] {
                if slot == usize::MAX {
                    break;
                }
                state.effects[slot].process(
                    &mut self.samples
                        [self.offsets[id] * samples..(self.offsets[id] + count) * samples],
                    samples,
                    count,
                )?;
            }
            let volume = f32::from_bits(word(mix.as_slice(), 0));
            if id == 0 {
                if volume != 1.0 {
                    for sample in &mut self.samples
                        [self.offsets[id] * samples..(self.offsets[id] + count) * samples]
                    {
                        *sample = gain(*sample, volume, precision)?;
                    }
                }
                continue;
            }
            let dest_id = word(mix.as_slice(), 0x924) as usize;
            let Some(dest) = state.mixes.get(dest_id).and_then(Option::as_ref) else {
                continue;
            };
            let dest_count = word(dest.as_slice(), 8) as usize;
            for input in 0..count {
                for output in 0..dest_count {
                    let volume = volume
                        * f32::from_bits(word(mix.as_slice(), 0x24 + (input * 24 + output) * 4));
                    if volume == 0.0 {
                        continue;
                    }
                    let src = (self.offsets[id] + input) * samples;
                    let dst = (self.offsets[dest_id] + output) * samples;
                    for sample in 0..samples {
                        self.samples[dst + sample] = self.samples[dst + sample].wrapping_add(gain(
                            self.samples[src + sample],
                            volume,
                            precision,
                        )?);
                    }
                }
            }
        }
        let mut device_sink = None;
        for (id, sink) in state.sinks.iter().enumerate() {
            if sink[0] == 1 && sink[1] != 0 {
                if device_sink.is_some() {
                    return Err(
                        "rendering: multiple active device sinks are not implemented".into(),
                    );
                }
                if !matches!(word(sink, 0x120), 1 | 2 | 6) || sink[0x12b] != 0 {
                    return Err(format!("rendering: sink {id} requires {} channels / downmix={}; mono, stereo and 5.1 device output without guest downmix are currently supported", word(sink, 0x120), sink[0x12b] != 0).into());
                }
                device_sink = Some(sink);
            }
        }
        let Some(sink) = device_sink else {
            return Ok(false);
        };
        let count = state
            .mixes
            .first()
            .and_then(Option::as_ref)
            .map_or(0, |mix| word(mix.as_slice(), 8));
        let channels = word(sink, 0x120) as usize;
        let mut offsets = [0; 6];
        for (channel, offset) in offsets.iter_mut().enumerate().take(channels) {
            let input = usize::from(sink[0x124 + channel]);
            if input >= count as usize {
                return Err(format!(
                    "rendering: device input {input} is outside the final mix's {count} buffers"
                )
                .into());
            }
            *offset = (self.offsets[0] + input) * samples;
        }
        for sample in 0..samples {
            let mut frame = [0; 6];
            for channel in 0..channels {
                frame[channel] = self.samples[offsets[channel] + sample]
                    .clamp(i16::MIN as i32, i16::MAX as i32)
                    as i16;
            }
            let stereo = if channels == 6 {
                surround_to_stereo(frame, volume)
            } else {
                let stereo = [frame[0], frame[1]];
                if volume == 1.0 {
                    stereo
                } else {
                    stereo.map(|sample| {
                        ((f32::from(sample) * volume) as i32)
                            .clamp(i16::MIN as i32, i16::MAX as i32) as i16
                    })
                }
            };
            pcm[sample * 2..sample * 2 + 2].copy_from_slice(&stereo);
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn surround_host_conversion_preserves_channel_order_and_clips_the_sum() {
        for (channel, expected) in [
            (0, [10000, 0]),
            (1, [0, 10000]),
            (2, [5960, 5960]),
            (3, [3540, 3540]),
            (4, [7070, 0]),
            (5, [0, 7070]),
        ] {
            let mut frame = [0; 6];
            frame[channel] = 10000;
            assert_eq!(surround_to_stereo(frame, 1.0), expected);
        }
        assert_eq!(surround_to_stereo([i16::MAX; 6], 1.0), [i16::MAX; 2]);
        assert_eq!(surround_to_stereo([i16::MIN; 6], 1.0), [i16::MIN; 2]);
        // Device gain must act before the host's downmix sum is saturated.
        assert_eq!(surround_to_stereo([20000; 6], 0.5), [26570; 2]);
        assert_eq!(surround_to_stereo([20000; 6], 0.0), [0; 2]);
    }
    #[test]
    fn submix_routes_integer_samples_and_device_sink_interleaves_and_saturates() {
        let mut data = [0; super::super::PARAMETER_SIZE];
        for (offset, value) in [
            (0, 48000_u32),
            (4, 3),
            (8, 4),
            (12, 1),
            (20, 1),
            (48, u32::from_le_bytes(*b"REV7")),
        ] {
            data[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
        let params = RendererParameters::decode(&data);
        let mut state = RendererUpdateState::default();
        for id in 0..2 {
            let mut mix = Box::new([0; 0x930]);
            mix[..4].copy_from_slice(&1_f32.to_le_bytes());
            mix[4..8].copy_from_slice(&48000_u32.to_le_bytes());
            mix[8..12].copy_from_slice(&2_u32.to_le_bytes());
            if id == 1 {
                mix[0x24..0x28].copy_from_slice(&0.5_f32.to_le_bytes());
                mix[0x24 + 25 * 4..0x28 + 25 * 4].copy_from_slice(&0.5_f32.to_le_bytes());
            }
            state.mixes.push(Some(mix));
            state.effect_order.push(Vec::new());
        }
        let mut sink = [0; 0x140];
        sink[0] = 1;
        sink[1] = 1;
        sink[0x120..0x124].copy_from_slice(&2_u32.to_le_bytes());
        sink[0x124] = 1; // Reverse the final mix channels.
        state.sinks = vec![sink];
        let mut buffers = RenderBuffers::new(&params);
        buffers.prepare_mixes(&state, &params).unwrap();
        buffers.samples[6..].copy_from_slice(&[40000, -40000, 3, 6, -6, 65536]);
        let mut pcm = [0; 6];
        assert!(
            buffers
                .mix_and_output(&mut state, &params, &mut pcm, 1.0)
                .unwrap()
        );
        assert_eq!(pcm, [3, 20000, -3, -20000, 32767, 1]);
        buffers.samples.fill(0);
        buffers.samples[6..].copy_from_slice(&[40000, -40000, 3, 6, -6, 65536]);
        assert!(
            buffers
                .mix_and_output(&mut state, &params, &mut pcm, 0.5)
                .unwrap()
        );
        assert_eq!(pcm, [1, 10000, -1, -10000, 16383, 0]);
        buffers.samples.fill(0);
        buffers.samples[6..].copy_from_slice(&[40000, -40000, 3, 6, -6, 65536]);
        assert!(
            buffers
                .mix_and_output(&mut state, &params, &mut pcm, 1.85)
                .unwrap()
        );
        assert_eq!(pcm, [5, 32767, -5, -32768, 32767, 1]);
        // A new block clears stale samples instead of repeating the last block.
        assert!(
            buffers
                .render(&RenderMemory::testing(), &mut state, &params, &mut pcm, 1.0)
                .unwrap()
        );
        assert_eq!(pcm, [0; 6]);
    }
    #[test]
    fn effect_runs_before_submix_routing_and_volume() {
        use crate::test_memory::physical;
        let mut raw_params = [0; super::super::PARAMETER_SIZE];
        for (offset, value) in [
            (0, 48000_u32),
            (4, 3),
            (8, 6),
            (12, 1),
            (20, 1),
            (24, 1),
            (48, u32::from_le_bytes(*b"REV7")),
        ] {
            raw_params[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
        let params = RendererParameters::decode(&raw_params);
        let mut state = RendererUpdateState::default();
        for id in 0..2 {
            let mut mix = Box::new([0; 0x930]);
            mix[..4].copy_from_slice(&(if id == 0 { 1_f32 } else { 0.5 }).to_le_bytes());
            mix[4..8].copy_from_slice(&48000_u32.to_le_bytes());
            mix[8..12].copy_from_slice(&(if id == 0 { 2_u32 } else { 4 }).to_le_bytes());
            if id == 1 {
                mix[0x24 + 48 * 4..0x28 + 48 * 4].copy_from_slice(&1_f32.to_le_bytes());
                mix[0x24 + 73 * 4..0x28 + 73 * 4].copy_from_slice(&1_f32.to_le_bytes());
            }
            state.mixes.push(Some(mix));
        }
        let mut sink = [0; 0x140];
        sink[0] = 1;
        sink[1] = 1;
        sink[0x120..0x124].copy_from_slice(&2_u32.to_le_bytes());
        sink[0x125] = 1;
        state.sinks = vec![sink];
        let ranges = physical(&vec![0; 4096]);
        let mut raw = [0; 0xc0];
        raw[0] = 4;
        raw[1] = 1;
        raw[4..8].copy_from_slice(&1_u32.to_le_bytes());
        raw[8..16].copy_from_slice(&0x1000_u64.to_le_bytes());
        raw[16..24].copy_from_slice(&4096_u64.to_le_bytes());
        raw[0x21] = 1;
        raw[0x26] = 2;
        raw[0x27] = 3;
        raw[0x2c..0x2e].copy_from_slice(&2_u16.to_le_bytes());
        raw[0x2e..0x30].copy_from_slice(&2_u16.to_le_bytes());
        let mut effect = super::super::effect::Effect::default();
        effect
            .update(
                &raw,
                0,
                7,
                48000,
                &[],
                true,
                &|_, _| Ok(ranges.clone()),
                true,
            )
            .unwrap();
        state.effects = vec![effect];
        state.effect_order = vec![Vec::new(), vec![0]];
        let mut buffers = RenderBuffers::new(&params);
        buffers.prepare_mixes(&state, &params).unwrap();
        buffers.samples[6..12].copy_from_slice(&[1000, -1000, 3, 6, -6, 100]);
        let mut pcm = [0; 6];
        buffers
            .mix_and_output(&mut state, &params, &mut pcm, 1.0)
            .unwrap();
        assert_eq!(pcm, [500, 3, -500, -3, 1, 50]);
    }

    #[test]
    fn fixed_point_gain_matches_reference_fractional_rounding_and_signs() {
        for precision in [15, 23] {
            assert_eq!(gain(1000, 0.5, precision).unwrap(), 500);
            assert_eq!(gain(-1000, 0.5, precision).unwrap(), -500);
            assert_eq!(gain(1, 0.75, precision).unwrap(), 1);
            assert_eq!(gain(-1, 0.75, precision).unwrap(), -1);
            assert_eq!(gain(1, 0.5, precision).unwrap(), 0);
            assert_eq!(gain(-1, 0.5, precision).unwrap(), -1);
        }
        assert!(gain(1, f32::NAN, 23).is_err());
    }
}
