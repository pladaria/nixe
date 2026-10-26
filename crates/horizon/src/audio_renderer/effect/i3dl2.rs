//! I3DL2 room reflections and four-branch feedback delay network, in Q14.
//! https://github.com/eden-emulator/mirror/blob/master/src/audio_core/renderer/command/effect/i3dl2_reverb.cpp
//! https://github.com/eden-emulator/mirror/blob/master/src/audio_core/renderer/effect/i3dl2.h
#![allow(clippy::excessive_precision)]

use super::super::update::{UpdateError, word};
use super::fixed::Q14 as Q;

const FDN_MIN: [f32; 4] = [5.0, 6.0, 13.0, 14.0];
const FDN_MAX: [f32; 4] = [45.7042007446, 82.7817001343, 149.938293457, 271.575805664];
const DECAY_A: [f32; 4] = [17.0, 13.0, 9.0, 7.0];
const DECAY_B: [f32; 4] = [19.0, 11.0, 10.0, 6.0];
const TAP_TIMES: [f32; 20] = [
    0.0171360000968,
    0.0591540001333,
    0.161733001471,
    0.390186011791,
    0.425262004137,
    0.455410987139,
    0.689737021923,
    0.74590998888,
    0.833844006062,
    0.859502017498,
    0.0,
    0.0750240013003,
    0.168788000941,
    0.299901008606,
    0.337442994118,
    0.371903002262,
    0.599011003971,
    0.716741025448,
    0.817858994007,
    0.85166400671,
];
const TAP_GAINS: [f32; 20] = [
    0.67096, 0.61027, 1.0, 0.3568, 0.68361, 0.65978, 0.51939, 0.24712, 0.45945, 0.45021, 0.64196,
    0.54879, 0.92925, 0.3827, 0.72867, 0.69794, 0.5464, 0.24563, 0.45214, 0.44042,
];
const TAP_CHANNELS: [[usize; 20]; 4] = [
    [0; 20],
    [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 0, 0, 0, 0, 3, 3, 3],
    [2, 0, 0, 1, 1, 1, 1, 4, 4, 4, 1, 1, 1, 0, 0, 0, 0, 5, 5, 5],
];

#[derive(Debug)]
pub(super) struct Parameters {
    inputs: [usize; 6],
    outputs: [usize; 6],
    channels: usize,
    sample_rate: u32,
    room_hf: f32,
    reference_hf: f32,
    decay: f32,
    hf_ratio: f32,
    room: f32,
    reflections: f32,
    reverb: f32,
    diffusion: f32,
    reflection_delay: f32,
    late_delay: f32,
    density: f32,
    dry: f32,
}

impl Parameters {
    pub(super) fn decode(
        raw: &[u8],
        revision: u32,
        rate: u32,
        enabled: bool,
    ) -> Result<Self, UpdateError> {
        let valid_channels = |n| matches!(n, 1 | 2 | 4 | 6);
        let maximum = u16::from_le_bytes(raw[12..14].try_into().unwrap()) as usize;
        let count = u16::from_le_bytes(raw[14..16].try_into().unwrap()) as usize;
        if !valid_channels(maximum) || raw[0x48] > 2 {
            return Err(UpdateError::Invalid);
        }
        let f = |offset| f32::from_bits(word(raw, offset));
        let mut p = Self {
            inputs: std::array::from_fn(|i| raw[i] as usize),
            outputs: std::array::from_fn(|i| raw[6 + i] as usize),
            channels: if valid_channels(count) {
                count
            } else {
                maximum
            },
            sample_rate: word(raw, 0x14),
            room_hf: f(0x18),
            reference_hf: f(0x1c),
            decay: f(0x20),
            hf_ratio: f(0x24),
            room: f(0x28),
            reflections: f(0x2c),
            reverb: f(0x30),
            diffusion: f(0x34),
            reflection_delay: f(0x38),
            late_delay: f(0x3c),
            density: f(0x40),
            dry: f(0x44),
        };
        if p.channels > maximum {
            return Err(UpdateError::Invalid);
        }
        if enabled {
            if p.sample_rate != rate {
                return Err(UpdateError::Unsupported(
                    "RequestUpdate: I3DL2 sample rate must match its renderer (Hz)".into(),
                ));
            }
            if [
                p.room_hf,
                p.reference_hf,
                p.decay,
                p.hf_ratio,
                p.room,
                p.reflections,
                p.reverb,
                p.diffusion,
                p.reflection_delay,
                p.late_delay,
                p.density,
                p.dry,
            ]
            .iter()
            .any(|v| !v.is_finite())
                || p.decay <= 0.0
                || p.hf_ratio <= 0.0
                || p.reference_hf <= 0.0
                || p.reference_hf >= rate as f32 * 0.5
                || p.reflection_delay < 0.0
                || p.late_delay < 0.0
                || p.reflection_delay + p.late_delay > 0.4
                || !(0.0..=100.0).contains(&p.density)
                || !(0.0..=100.0).contains(&p.diffusion)
            {
                return Err(UpdateError::Invalid);
            }
        }
        if p.channels == 6 && revision < 11 {
            p.inputs.swap(2, 4);
            p.inputs.swap(3, 5);
            p.outputs.swap(2, 4);
            p.outputs.swap(3, 5);
        }
        Ok(p)
    }

    pub(super) fn validate_routing(&self, count: usize) -> Result<(), Box<str>> {
        if self.inputs[..self.channels]
            .iter()
            .chain(&self.outputs[..self.channels])
            .any(|i| *i >= count)
        {
            return Err(
                format!("rendering: I3DL2 input/output exceeds its mix's {count} buffers").into(),
            );
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
    values: Vec<Q>,
    input: usize,
    output: usize,
    maximum: usize,
    delay: usize,
}
impl Delay {
    fn new(maximum: usize) -> Self {
        Self {
            values: vec![Q::raw(0); maximum + 1],
            input: maximum,
            output: 0,
            maximum,
            delay: maximum,
        }
    }
    fn set_delay(&mut self, delay: usize) {
        if delay <= self.maximum {
            self.delay = delay;
            self.input = (self.output + delay) % (self.maximum + 1);
        }
    }
    fn read(&self) -> Q {
        self.values[self.output]
    }
    fn tick(&mut self, value: Q) -> Q {
        self.values[self.input] = value;
        self.input += 1;
        if self.input >= self.maximum {
            self.input = 0;
        }
        let out = self.read();
        self.output += 1;
        if self.output >= self.maximum {
            self.output = 0;
        }
        out
    }
    fn tap(&self, delay: usize) -> Q {
        let count = self.maximum + 1;
        self.values[(self.input + count - (delay + 1)) % count]
    }
}

#[derive(Debug)]
pub(super) struct State {
    early: Delay,
    center: Delay,
    feedback: [Delay; 4],
    decay_a: [Delay; 4],
    decay_b: [Delay; 4],
    taps: [usize; 20],
    late_tap: usize,
    input_filter: f32,
    input_coeff: [f32; 2],
    shelf: [f32; 4],
    coefficients: [[f32; 3]; 4],
    early_gain: f32,
    late_gain: f32,
    diffusion: f32,
    dry: f32,
}

fn gain(exponent: f32) -> f32 {
    if exponent >= 0.0 {
        1.0
    } else if exponent <= -5.3 {
        0.0
    } else {
        10_f32.powf(exponent)
    }
}
fn scaled_time(ms: f32, rate: Q) -> Q {
    Q::float(ms) * rate
}

impl State {
    pub(super) fn new(p: &Parameters) -> Self {
        let rate = Q::float(p.sample_rate as f32 / 1000.0);
        let count = |ms| (scaled_time(ms, rate).0 >> 14) as usize;
        let mut s = Self {
            early: Delay::new(count(400.0)),
            center: Delay::new(count(5.0)),
            feedback: std::array::from_fn(|i| Delay::new(count(FDN_MAX[i]))),
            decay_a: std::array::from_fn(|i| Delay::new(count(DECAY_A[i]))),
            decay_b: std::array::from_fn(|i| Delay::new(count(DECAY_B[i]))),
            taps: [0; 20],
            late_tap: 0,
            input_filter: 0.0,
            input_coeff: [0.0; 2],
            shelf: [0.0; 4],
            coefficients: [[0.0; 3]; 4],
            early_gain: 0.0,
            late_gain: 0.0,
            diffusion: 0.0,
            dry: 0.0,
        };
        s.update(p);
        s
    }

    pub(super) fn update(&mut self, p: &Parameters) {
        let rate = Q::float(p.sample_rate as f32 / 1000.0);
        self.dry = p.dry;
        self.early_gain = gain(Q::float((p.room + p.reflections).min(5000.0) / 2000.0).value());
        self.late_gain = gain(Q::float((p.room + p.reverb).min(5000.0) / 2000.0).value());
        let hf = Q::float(gain(p.room_hf / 2000.0));
        let lowpass = if hf >= Q::ONE {
            0.0
        } else {
            let angle =
                (p.reference_hf * 256.0 / p.sample_rate as f32) * std::f32::consts::PI / 180.0;
            let a = Q::float(1.0 - hf.value());
            let b = Q::float(2.0) + Q::float(-angle.cos()) * (hf * Q::float(2.0));
            let c = Q::float((b.value().powf(2.0) + a.value().powf(2.0) * -4.0).sqrt());
            ((b - c) / (a * Q::float(2.0))).value().min(0.99723)
        };
        self.input_coeff = [lowpass, 1.0 - lowpass];
        self.late_tap =
            scaled_time((p.reflection_delay + p.late_delay) * 1000.0, rate).int() as usize;
        self.diffusion = p.diffusion * 0.6 * 0.01;
        for i in 0..4 {
            let delay = scaled_time(
                FDN_MIN[i] + p.density / 100.0 * (FDN_MAX[i] - FDN_MIN[i]),
                rate,
            )
            .int() as usize;
            self.feedback[i].set_delay(delay);
            let total =
                (self.feedback[i].delay + self.decay_a[i].delay + self.decay_b[i].delay) as f32;
            let a = total * -60.0 / (p.decay * p.sample_rate as f32);
            let b = a / p.hf_ratio;
            let angle = ((p.reference_hf * 0.5) * 128.0 / p.sample_rate as f32)
                * std::f32::consts::PI
                / 180.0;
            let c = angle.cos() / angle.sin();
            let d = gain((b - a) / 40.0);
            // The DSP coefficient is rounded to four decimal places.
            #[allow(clippy::approx_constant)]
            let e = gain((b + a) / 40.0) * 0.7071;
            self.coefficients[i] = [
                ((c * d + 1.0) * e) / (c + d),
                ((1.0 - c * d) * e) / (c + d),
                (c - d) / (c + d),
            ];
        }
        let spread = (p.late_delay * 0.9998 + 0.02) * 1000.0;
        for (i, tap) in self.taps.iter_mut().enumerate() {
            *tap = (scaled_time(p.reflection_delay * 1000.0 + spread * TAP_TIMES[i], rate).int()
                as usize)
                .min(self.early.maximum);
        }
    }

    pub(super) fn process(&mut self, p: &Parameters, buffers: &mut [i32], samples: usize) {
        let table = match p.channels {
            1 => 0,
            2 => 1,
            4 => 2,
            6 => 3,
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
            let late = self.early.tap(self.late_tap);
            let mut output = [Q::raw(0); 6];
            for i in 0..20 {
                let value = self.early.tap(self.taps[i]) * Q::float(TAP_GAINS[i]);
                let c = TAP_CHANNELS[table][i];
                output[c] = output[c] + value;
                if p.channels == 6 {
                    output[3] = output[3] + value;
                }
            }
            let sum = input[..p.channels]
                .iter()
                .copied()
                .fold(Q::raw(0), |a, b| a + b);
            self.input_filter = (sum * Q::float(self.input_coeff[1])
                + Q::float(self.input_filter * self.input_coeff[0]))
            .value();
            self.early.tick(Q::float(self.input_filter));
            let filtered: [Q; 4] = std::array::from_fn(|i| {
                let v = self.feedback[i].read() * Q::float(self.coefficients[i][0])
                    + Q::float(self.shelf[i]);
                self.shelf[i] = (v * Q::float(self.coefficients[i][2])
                    + self.feedback[i].read() * Q::float(self.coefficients[i][1]))
                .value();
                v
            });
            let late = late * Q::float(self.late_gain);
            let mixed = [
                filtered[1] + filtered[2] + late,
                -filtered[0] - filtered[3] + late,
                filtered[0] - filtered[3] + late,
                filtered[1] - filtered[2] + late,
            ];
            let all: [Q; 4] = std::array::from_fn(|i| {
                let mut v = mixed[i];
                for (line, gain) in [
                    (&mut self.decay_a[i], self.diffusion),
                    (&mut self.decay_b[i], self.diffusion * -0.9),
                ] {
                    let g = Q::float(gain);
                    let mixed = v - line.read() * g;
                    v = line.tick(mixed) + mixed * g;
                }
                self.feedback[i].tick(v);
                v
            });
            for c in 0..p.channels {
                let wet = if p.channels == 6 {
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
                let dry = Q::float(self.dry * input[c].value());
                let value = output[c] * Q::float(self.early_gain) + wet + dry;
                buffers[p.outputs[c] * samples + sample] =
                    value.value().clamp(-8388600.0, 8388600.0) as i32;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    fn packet(channels: u16, rate: u32) -> [u8; 0xa0] {
        let mut raw = [0; 0xa0];
        for i in 0..6 {
            raw[i] = i as u8;
            raw[6 + i] = i as u8;
        }
        raw[12..14].copy_from_slice(&channels.to_le_bytes());
        raw[14..16].copy_from_slice(&channels.to_le_bytes());
        raw[0x14..0x18].copy_from_slice(&rate.to_le_bytes());
        for (offset, value) in [
            (0x18, -1000_f32),
            (0x1c, 5000.0),
            (0x20, 1.49),
            (0x24, 0.83),
            (0x28, -1000.0),
            (0x2c, -2602.0),
            (0x30, 200.0),
            (0x34, 100.0),
            (0x38, 0.007),
            (0x3c, 0.011),
            (0x40, 100.0),
            (0x44, 0.7),
        ] {
            raw[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
        raw
    }

    #[test]
    fn impulse_and_parameter_updates_match_independent_reference_at_both_renderer_rates() {
        // Hashes from the reference C++ DSP/FixedPoint, 120 consecutive 240-sample
        // blocks, with a density/decay/HF update at block 40 that retains the tail.
        for (rate, channels, expected) in [
            (
                32000,
                1,
                "ffb7e42c909a8b09354f6ff55cc9f0307446cc691fcfb76cb827a788b4739469",
            ),
            (
                32000,
                2,
                "8d680a87339c4d1ed933c649afd5f042fe34d9989c949e2045f6091fa56aa52e",
            ),
            (
                32000,
                4,
                "2b9211631aa9341374fe4b3263146784d1193c8f855f7be563537d708d66c38d",
            ),
            (
                32000,
                6,
                "0bfb537b7c2d3f9d4042e8bc30804df7c65eb3b7c615774e523d9f4428fb8b95",
            ),
            (
                48000,
                1,
                "aba89be6cc8f91dddd1ad12d85d0062400da8a36b7585f4a2c0f00ad97076747",
            ),
            (
                48000,
                2,
                "f4d1ac6b201a49b84deccd8a5b15eef89f57562ac5437d9af74ec646f9332ac4",
            ),
            (
                48000,
                4,
                "7629c6cc1fd6a06890d7695bd67917af481394832c7a34d988c36a330a82d0cb",
            ),
            (
                48000,
                6,
                "546dfe175182838dd821fa233fb6414d6b3bd1f8e5248bb152501bcb44c2b89d",
            ),
        ] {
            let raw = packet(channels, rate);
            let mut p = Parameters::decode(&raw, 11, rate, true).unwrap();
            let mut state = State::new(&p);
            let mut buffers = vec![0; usize::from(channels) * 240];
            let mut hash = Sha256::new();
            for block in 0..120 {
                buffers.fill(0);
                if block == 0 {
                    for ch in 0..usize::from(channels) {
                        buffers[ch * 240] = 10000 * (ch as i32 + 1);
                    }
                }
                if block == 40 {
                    p.density = 35.0;
                    p.decay = 0.9;
                    p.room_hf = -200.0;
                    state.update(&p);
                }
                state.process(&p, &mut buffers, 240);
                for v in &buffers {
                    hash.update(v.to_le_bytes());
                }
            }
            assert_eq!(
                format!("{:x}", hash.finalize()),
                expected,
                "rate={rate}, channels={channels}"
            );
        }
    }

    #[test]
    fn ipc_updates_preserve_pending_initialization_tail_and_workbuffer() {
        use super::super::{Effect, MemoryPool};
        let ranges = crate::test_memory::physical(&vec![0; 4096]);
        let pools = [Some(MemoryPool {
            address: 0x1000,
            ranges: nixe_memory::CapturedMemory::new(ranges).unwrap(),
            bytes: 4096,
        })];
        let mut raw = [0; 0xc0];
        raw[0] = 5;
        raw[1] = 1;
        raw[2] = 1;
        raw[8..16].copy_from_slice(&0x1000_u64.to_le_bytes());
        raw[16..24].copy_from_slice(&0x1000_u64.to_le_bytes());
        raw[0x20..].copy_from_slice(&packet(2, 48000));
        let mut effect = Effect::default();
        assert_eq!(
            effect
                .update(&raw, 0, 11, 48000, &pools, false, &|_, _| panic!(), false)
                .unwrap(),
            3
        );
        assert!(effect.uses_pool(0));
        raw[1] = 0;
        raw[0x68] = 2;
        effect
            .update(&raw, 0, 11, 48000, &pools, false, &|_, _| panic!(), true)
            .unwrap();
        let Effect::I3dl2(e) = &effect else { panic!() };
        assert_eq!(e.pending, 0);
        let mut buffers = vec![0; 480];
        buffers[0] = 100000;
        effect
            .process(
                &crate::audio_renderer::worker::RenderMemory::testing(),
                &mut buffers,
                240,
                2,
            )
            .unwrap();
        let Effect::I3dl2(e) = &effect else { panic!() };
        let state = std::ptr::from_ref(e.state.as_ref().unwrap());
        assert_eq!(e.pending, 2);
        let mut tail = false;
        for _ in 0..15 {
            effect
                .update(&raw, 0, 11, 48000, &pools, false, &|_, _| panic!(), true)
                .unwrap();
            let Effect::I3dl2(e) = &effect else { panic!() };
            assert_eq!(std::ptr::from_ref(e.state.as_ref().unwrap()), state);
            buffers.fill(0);
            effect
                .process(
                    &crate::audio_renderer::worker::RenderMemory::testing(),
                    &mut buffers,
                    240,
                    2,
                )
                .unwrap();
            tail |= buffers.iter().any(|v| *v != 0);
        }
        assert!(tail, "IPC updates must retain delay-line history");
        raw[2] = 0;
        assert_eq!(
            effect
                .update(&raw, 0, 11, 48000, &pools, false, &|_, _| panic!(), true)
                .unwrap(),
            3
        );
        buffers.fill(123);
        effect
            .process(
                &crate::audio_renderer::worker::RenderMemory::testing(),
                &mut buffers,
                240,
                2,
            )
            .unwrap();
        assert!(buffers.iter().all(|v| *v == 123));
        assert!(!effect.uses_pool(0));
        assert_eq!(
            effect
                .update(&raw, 0, 11, 48000, &pools, false, &|_, _| panic!(), true)
                .unwrap(),
            4
        );
    }

    #[test]
    fn disabled_effects_bypass_dry_gain_and_validate_routing_and_parameters() {
        let mut raw = packet(6, 48000);
        for i in 0..6 {
            raw[6 + i] = 6 + i as u8;
        }
        let p = Parameters::decode(&raw, 10, 48000, false).unwrap();
        assert_eq!(p.outputs, [6, 7, 10, 11, 8, 9]);
        assert!(p.validate_routing(12).is_ok());
        assert!(p.validate_routing(11).is_err());
        let mut buffers: Vec<i32> = (0..36).collect();
        p.bypass(&mut buffers, 3);
        assert_eq!(&buffers[..18], &buffers[18..]);
        raw[0x20..0x24].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(Parameters::decode(&raw, 11, 48000, true).is_err());
        assert!(Parameters::decode(&raw, 11, 48000, false).is_ok());
        let raw = packet(2, 48000);
        assert!(Parameters::decode(&raw, 11, 32000, true).is_err());
    }
}
