// SPDX-FileCopyrightText: Copyright 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
//! Q14 AXFX2-style reverb, adapted from Yuzu; no allocations in steady-state DSP.
//! https://github.com/JamesDoesGaming/yuzu/blob/master/src/audio_core/renderer/command/effect/reverb.cpp
//! Stereo early-tap attenuation is also confirmed by Ryujinx's ReverbState.
#![allow(clippy::excessive_precision)] // Preserve the reference Q14 coefficient tables.
use super::super::update::{UpdateError, word};
use super::fixed::Q14 as Q;
const FDN_MAX: [f32; 4] = [
    53.9532470703125,
    79.19256591796875,
    116.23876953125,
    170.61529541015625,
];
const DECAY_MAX: [f32; 4] = [7.0, 9.0, 13.0, 17.0];
const EARLY_TIMES: [[f32; 11]; 5] = [
    [
        0.000000, 3.500000, 2.799988, 3.899963, 2.699951, 13.399963, 7.899963, 8.399963, 9.899963,
        12.000000, 12.500000,
    ],
    [
        0.000000, 11.799988, 5.500000, 11.199951, 10.399963, 38.099976, 22.199951, 29.599976,
        21.199951, 24.799988, 40.000000,
    ],
    [
        0.000000, 41.500000, 20.500000, 41.299988, 0.000000, 29.500000, 33.799988, 45.199951,
        46.799988, 0.000000, 50.000000,
    ],
    [
        33.099976, 43.299988, 22.799988, 37.899963, 14.899963, 35.299988, 17.899963, 34.199951,
        0.000000, 43.299988, 50.000000,
    ],
    [
        0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
        0.000000, 0.000000,
    ],
];
const EARLY_GAINS: [[f32; 10]; 5] = [
    [
        0.699951, 0.679993, 0.699951, 0.679993, 0.699951, 0.679993, 0.699951, 0.679993, 0.679993,
        0.679993,
    ],
    [
        0.699951, 0.679993, 0.699951, 0.679993, 0.699951, 0.679993, 0.679993, 0.679993, 0.679993,
        0.679993,
    ],
    [
        0.500000, 0.699951, 0.699951, 0.679993, 0.500000, 0.679993, 0.679993, 0.699951, 0.679993,
        0.000000,
    ],
    [
        0.929993, 0.919983, 0.869995, 0.859985, 0.939941, 0.809998, 0.799988, 0.769958, 0.759949,
        0.649963,
    ],
    [
        0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
        0.000000,
    ],
];
const FDN_TIMES: [[f32; 4]; 5] = [
    [53.953247, 79.192566, 116.238770, 130.615295],
    [53.953247, 79.192566, 116.238770, 170.615295],
    [5.000000, 10.000000, 5.000000, 10.000000],
    [47.029968, 71.000000, 103.000000, 170.000000],
    [53.953247, 79.192566, 116.238770, 170.615295],
];
const DECAY_TIMES: [[f32; 4]; 5] = [
    [7.000000, 9.000000, 13.000000, 17.000000],
    [7.000000, 9.000000, 13.000000, 17.000000],
    [1.000000, 1.000000, 1.000000, 1.000000],
    [7.000000, 7.000000, 13.000000, 9.000000],
    [7.000000, 9.000000, 13.000000, 17.000000],
];
#[derive(Clone, Debug)]
pub(super) struct Parameters {
    inputs: [usize; 6],
    outputs: [usize; 6],
    channels: usize,
    rate: Q,
    early_mode: usize,
    early_gain: Q,
    pre_delay: Q,
    late_mode: usize,
    late_gain: Q,
    decay_time: Q,
    hf_ratio: Q,
    colouration: Q,
    base: Q,
    wet: Q,
    dry: Q,
    long_pre: bool,
}
impl Parameters {
    pub(super) fn decode(
        raw: &[u8],
        revision: u32,
        sample_rate: u32,
        enabled: bool,
    ) -> Result<Self, UpdateError> {
        let valid_channels = |n| matches!(n, 1 | 2 | 4 | 6);
        let max = u16::from_le_bytes(raw[12..14].try_into().unwrap()) as usize;
        let requested = u16::from_le_bytes(raw[14..16].try_into().unwrap()) as usize;
        if !valid_channels(max) || raw[0x40] > 2 {
            return Err(UpdateError::Invalid);
        }
        let channels = if valid_channels(requested) {
            requested
        } else {
            max
        };
        if channels > max {
            return Err(UpdateError::Invalid);
        }
        let q = |offset| Q::raw(i64::from(word(raw, offset) as i32));
        let mut params = Self {
            inputs: std::array::from_fn(|i| raw[i] as usize),
            outputs: std::array::from_fn(|i| raw[6 + i] as usize),
            channels,
            rate: q(16),
            early_mode: word(raw, 20) as usize,
            early_gain: q(24),
            pre_delay: q(28),
            late_mode: word(raw, 32) as usize,
            late_gain: q(36),
            decay_time: q(40),
            hf_ratio: q(44),
            colouration: q(48),
            base: q(52),
            wet: q(56),
            dry: q(60),
            long_pre: revision >= 3,
        };
        if enabled {
            if params.rate != Q::float(sample_rate as f32 / 1000.0) {
                return Err(UpdateError::Unsupported(
                    "RequestUpdate: Reverb sample rate must match its renderer (Q14 kHz)".into(),
                ));
            }
            if params.early_mode >= 5
                || params.late_mode >= 5
                || params.pre_delay.0 < 0
                || params.decay_time.0 <= 0
                || params.hf_ratio.0 < 0
                || !(Q::raw(0)..=Q::ONE).contains(&params.colouration)
            {
                return Err(UpdateError::Invalid);
            }
        }
        // REV1–REV7 expect FL, FR, BL, BR, C, LFE on the wire. The DSP operates
        // in FL, FR, C, LFE, BL, BR order. Revision 11 removed this permutation.
        if channels == 6 && revision < 11 {
            params.inputs.swap(2, 4);
            params.inputs.swap(3, 5);
            params.outputs.swap(2, 4);
            params.outputs.swap(3, 5);
        }
        Ok(params)
    }
    pub(super) fn validate_routing(&self, count: usize) -> Result<(), Box<str>> {
        if self.inputs[..self.channels]
            .iter()
            .chain(&self.outputs[..self.channels])
            .any(|i| *i >= count)
        {
            return Err(format!(
                "rendering: Reverb input/output exceeds its mix's {count} buffers"
            )
            .into());
        }
        Ok(())
    }
    pub(super) fn bypass(&self, buffers: &mut [i32], samples: usize) {
        for c in 0..self.channels {
            let src = self.inputs[c] * samples;
            let dst = self.outputs[c] * samples;
            if src != dst {
                buffers.copy_within(src..src + samples, dst);
            }
        }
    }
}
#[derive(Debug)]
struct Delay {
    buffer: Vec<Q>,
    input: usize,
    output: usize,
    count: usize,
    maximum: usize,
    decay: Q,
}
impl Delay {
    fn new(count: usize) -> Self {
        Self {
            buffer: vec![Q::raw(0); count + 1],
            input: 0,
            output: 0,
            count,
            maximum: count,
            decay: Q::raw(0),
        }
    }
    fn read(&self) -> Q {
        self.buffer[self.output]
    }
    fn write(&mut self, sample: Q) {
        self.buffer[self.input] = sample;
        self.input += 1;
        if self.input >= self.count {
            self.input = 0;
        }
    }
    fn tick(&mut self, sample: Q) -> Q {
        let value = self.read();
        self.output += 1;
        if self.output >= self.count {
            self.output = 0;
        }
        self.write(sample);
        value
    }
    fn tap(&self, index: usize) -> Q {
        self.buffer[(self.input + self.count - (index + 1) % self.count) % self.count]
    }
    fn set(&mut self, count: usize) {
        self.count = count.min(self.maximum).saturating_sub(1).max(1);
    }
}
#[derive(Debug)]
pub(super) struct State {
    pre: Delay,
    center: Delay,
    early_times: [usize; 10],
    early_gains: [Q; 10],
    pre_time: usize,
    decay: [Delay; 4],
    fdn: [Delay; 4],
    hf_gain: [Q; 4],
    hf_previous: [Q; 4],
    feedback: [Q; 4],
}
impl State {
    pub(super) fn new(p: &Parameters) -> Self {
        let count = |ms| ((Q::float(ms) * p.rate).0 >> 14) as usize;
        let mut state = Self {
            pre: Delay::new(count(if p.long_pre { 350.0 } else { 150.0 })),
            center: Delay::new(count(5.0)),
            early_times: [0; 10],
            early_gains: [Q::raw(0); 10],
            pre_time: 0,
            decay: std::array::from_fn(|i| Delay::new(count(DECAY_MAX[i]))),
            fdn: std::array::from_fn(|i| Delay::new(count(FDN_MAX[i]))),
            hf_gain: [Q::raw(0); 4],
            hf_previous: [Q::raw(0); 4],
            feedback: [Q::raw(0); 4],
        };
        state.update(p);
        state
    }
    pub(super) fn update(&mut self, p: &Parameters) {
        let pow10 = |value: f32| {
            if value >= 0.0 {
                1.0
            } else if value <= -5.3 {
                0.0
            } else {
                10_f32.powf(value)
            }
        };
        for i in 0..10 {
            self.early_times[i] = ((p.pre_delay + Q::float(EARLY_TIMES[p.early_mode][i])) * p.rate)
                .int()
                .max(0) as usize;
            self.early_times[i] = self.early_times[i].min(self.pre.maximum) + 1;
            self.early_gains[i] = p.early_gain * Q::float(EARLY_GAINS[p.early_mode][i]);
        }
        if p.channels == 2 {
            self.early_gains[4] = self.early_gains[4] * Q::float(0.5);
            self.early_gains[5] = self.early_gains[5] * Q::float(0.5);
        }
        self.pre_time = (((p.pre_delay + Q::float(EARLY_TIMES[p.early_mode][10])) * p.rate)
            .int()
            .max(0) as usize)
            .min(self.pre.maximum);
        // Calculate per effect/sample rate, not a process-global cached cosine.
        let cosine = Q::float((Q::float(1280.0) / p.rate).value().to_radians().cos());
        for i in 0..4 {
            self.fdn[i].set((Q::float(FDN_TIMES[p.late_mode][i]) * p.rate).int() as usize);
            self.decay[i].set((Q::float(DECAY_TIMES[p.late_mode][i]) * p.rate).int() as usize);
            self.decay[i].decay = Q::float(0.5999755859375) * (Q::ONE - p.colouration);
            let a = Q::raw(-3 * ((self.fdn[i].maximum + self.decay[i].maximum) as i64) * 16384);
            let b = a / (p.decay_time * p.rate);
            let (c, d) = if p.hf_ratio > Q::float(0.99493408203125) {
                (Q::raw(0), Q::ONE)
            } else if p.hf_ratio.0 == 0 {
                (Q::ONE, Q::raw(0))
            } else {
                let e = pow10(
                    ((Q::ONE / p.hf_ratio - Q::ONE) * Q::float(2.0) / Q::float(100.0)
                        * (b / Q::float(10.0)))
                    .value(),
                );
                let f = 1.0 - e;
                if f == 0.0 {
                    (Q::raw(0), Q::ONE)
                } else {
                    let g = Q::float(2.0) - cosine * Q::float(e) * Q::float(2.0);
                    let h = (g.value().powi(2) - f.powi(2) * 4.0).max(0.0).sqrt();
                    let c = (g - Q::float(h)) / Q::float(f * 2.0);
                    (c, Q::ONE - c)
                }
            };
            self.hf_previous[i] = c;
            self.hf_gain[i] =
                Q::float(pow10((b / Q::float(1000.0)).value())) * d * Q::float(0.70709228515625);
            self.feedback[i] = Q::raw(0);
        }
    }
    pub(super) fn process(&mut self, p: &Parameters, buffers: &mut [i32], samples: usize) {
        let taps = match p.channels {
            1 => [0; 10],
            2 => [0, 0, 1, 1, 0, 1, 0, 0, 1, 1],
            4 => [0, 0, 1, 1, 0, 1, 2, 2, 3, 3],
            6 => [0, 0, 1, 1, 2, 2, 4, 4, 5, 5],
            _ => unreachable!(),
        };
        for sample in 0..samples {
            let input: [Q; 6] = std::array::from_fn(|c| {
                if c < p.channels {
                    Q::raw(i64::from(buffers[p.inputs[c] * samples + sample]) << 14)
                } else {
                    Q::raw(0)
                }
            });
            let mut output = [Q::raw(0); 6];
            for (i, &channel) in taps.iter().enumerate() {
                let early = self.pre.tap(self.early_times[i]) * self.early_gains[i];
                output[channel] = output[channel] + early;
                if p.channels == 6 {
                    output[3] = output[3] + early;
                }
            }
            if p.channels == 6 {
                output[3] = output[3] * Q::float(0.2);
            }
            let sum = input[..p.channels]
                .iter()
                .copied()
                .fold(Q::raw(0), |a, b| a + b);
            self.pre.write(Q::raw(sum.0.wrapping_mul(64)) * p.base);
            for i in 0..4 {
                self.feedback[i] =
                    self.feedback[i] * self.hf_previous[i] + self.fdn[i].read() * self.hf_gain[i];
            }
            let late = self.pre.tap(self.pre_time) * p.late_gain;
            let matrix = [
                self.feedback[2] + self.feedback[1] + late,
                -self.feedback[0] - self.feedback[3] + late,
                self.feedback[0] - self.feedback[3] + late,
                self.feedback[1] - self.feedback[2] + late,
            ];
            let mut all = [Q::raw(0); 4];
            for i in 0..4 {
                let mixed = matrix[i] - self.decay[i].read() * self.decay[i].decay;
                all[i] = self.decay[i].tick(mixed) + mixed * self.decay[i].decay;
                self.fdn[i].tick(all[i]);
            }
            for c in 0..p.channels {
                let allpass = if p.channels == 6 {
                    match c {
                        0 => all[0],
                        1 => all[1],
                        2 => self.center.tick((all[2] - all[3]) * Q::float(0.5)),
                        3 | 5 => all[3],
                        4 => all[2],
                        _ => unreachable!(),
                    }
                } else {
                    all[c]
                };
                let wet = (output[c] + allpass) * p.wet;
                let value = input[c] * p.dry + Q::raw(wet.0 / 64);
                buffers[p.outputs[c] * samples + sample] = value.int();
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    fn raw(channels: u16) -> [u8; 0xa0] {
        let mut bytes = [0; 0xa0];
        for i in 0..6 {
            bytes[i] = i as u8;
            bytes[6 + i] = i as u8;
        }
        bytes[12..14].copy_from_slice(&channels.to_le_bytes());
        bytes[14..16].copy_from_slice(&channels.to_le_bytes());
        for (offset, value) in [
            (16, 48 << 14),
            (20, 1),
            (24, 16384),
            (28, 10 << 14),
            (32, 1),
            (36, 16384),
            (40, 1000 << 14),
            (44, 8192),
            (48, 6553),
            (52, 8192),
            (56, 8192),
            (60, 12288),
        ] {
            bytes[offset..offset + 4].copy_from_slice(&(value as u32).to_le_bytes());
        }
        bytes
    }
    #[test]
    fn impulse_response_matches_independent_cpp_reference_for_each_channel_count() {
        // Generated by the original Yuzu C++ DSP and its FixedPoint implementation,
        // with the stereo-tap *= and late pre-delay TapOut fixes confirmed by
        // Ryujinx/Eden. These hashes
        // cover 300 ms of early/late response across 60 separate worker blocks.
        for (channels, expected) in [
            (
                1,
                "d5e95a48135ab3dd7d3f1b32759efe581431b02312216c71d32af35209d646ab",
            ),
            (
                2,
                "c74096968bed58db2e6e7b9a4adf4de1a4ee65125fe72c5f19b49a95b913a128",
            ),
            (
                4,
                "d3e775dcd22701df8fc55be4bb347c856dec61b1740eece10ec7679ab619067a",
            ),
            (
                6,
                "040b7729ec0a4f38023dd35ebb22bdfff9ccdda52e22d25cb8a414e66220a5df",
            ),
        ] {
            // Use standard DSP channel order here; the legacy wire permutation
            // is verified separately below, independent of impulse generation.
            let p = Parameters::decode(&raw(channels), 11, 48000, true).unwrap();
            let mut state = State::new(&p);
            let mut hash = Sha256::new();
            for frame in 0..60 {
                let mut samples = vec![0; usize::from(channels) * 240];
                if frame == 0 {
                    for c in 0..usize::from(channels) {
                        samples[c * 240] = 10000 - c as i32 * 1500;
                    }
                }
                state.process(&p, &mut samples, 240);
                for value in samples {
                    hash.update(value.to_le_bytes());
                }
            }
            assert_eq!(
                format!("{:x}", hash.finalize()),
                expected,
                "{channels} channels"
            );
        }
    }
    #[test]
    fn wire_mapping_and_disabled_bypass_preserve_samples_without_gains() {
        let p = Parameters::decode(&raw(6), 7, 48000, true).unwrap();
        assert_eq!(p.inputs, [0, 1, 4, 5, 2, 3]);
        assert_eq!(p.outputs, [0, 1, 4, 5, 2, 3]);
        let mut bytes = raw(2);
        bytes[6] = 2;
        bytes[7] = 3;
        let p = Parameters::decode(&bytes, 7, 48000, false).unwrap();
        let mut data = vec![1000, -1000, 23, -23, 0, 0, 0, 0];
        p.bypass(&mut data, 2);
        assert_eq!(data, [1000, -1000, 23, -23, 1000, -1000, 23, -23]);
        assert!(p.validate_routing(3).is_err());
        let mut wrong_rate = raw(2);
        wrong_rate[16..20].copy_from_slice(&(32_u32 << 14).to_le_bytes());
        assert!(Parameters::decode(&wrong_rate, 7, 48000, true).is_err());
    }
    #[test]
    fn every_early_and_late_mode_is_stable_and_updates_retain_allocated_delay_memory() {
        for early in 0..5 {
            for late in 0..5 {
                let mut bytes = raw(2);
                bytes[20..24].copy_from_slice(&(early as u32).to_le_bytes());
                bytes[32..36].copy_from_slice(&(late as u32).to_le_bytes());
                let p = Parameters::decode(&bytes, 7, 48000, true).unwrap();
                let mut state = State::new(&p);
                let pre = state.pre.buffer.as_ptr();
                let fdn = state.fdn[0].buffer.as_ptr();
                let mut data = vec![0; 480];
                data[0] = 10000;
                state.process(&p, &mut data, 240);
                state.update(&p);
                assert_eq!(state.pre.buffer.as_ptr(), pre);
                assert_eq!(state.fdn[0].buffer.as_ptr(), fdn);
                for _ in 0..100 {
                    data.fill(0);
                    state.process(&p, &mut data, 240);
                    assert!(data.iter().all(|n| n.unsigned_abs() < 1_000_000));
                }
            }
        }
    }
}
