//! Stateful look-ahead limiter and revision 9+ per-channel statistics.
//! https://github.com/eden-emulator/mirror/blob/master/src/audio_core/renderer/effect/light_limiter.h
//! https://github.com/eden-emulator/mirror/blob/master/src/audio_core/renderer/command/effect/light_limiter.cpp
use super::super::{
    buffer::Attachment,
    update::{UpdateError, word},
};
use super::fixed::Q15 as Q;

#[derive(Debug)]
pub(in crate::audio_renderer) struct Limiter {
    pub(super) mix: usize,
    pub(super) order: usize,
    work: Attachment,
    revision: u32,
    enabled: bool,
    new: bool,
    was_enabled: Option<bool>,
    inputs: [usize; 6],
    outputs: [usize; 6],
    channels: usize,
    pending: u8,
    attack: f32,
    release: f32,
    threshold: f32,
    input_gain: f32,
    output_gain: f32,
    lookahead: usize,
    maximum: usize,
    mode: u32,
    statistics: bool,
    reset_statistics: bool,
    maximum_sample: [f32; 6],
    minimum_gain: [f32; 6],
    state: Option<State>,
}
#[derive(Debug)]
struct State {
    average: [Q; 6],
    gain: [Q; 6],
    offset: [usize; 6],
    delay: [Vec<Q>; 6],
}
impl Limiter {
    pub(super) fn new(work: Attachment, revision: u32) -> Self {
        Self {
            mix: 0,
            order: 0,
            work,
            revision,
            enabled: false,
            new: true,
            was_enabled: None,
            inputs: [0; 6],
            outputs: [0; 6],
            channels: 0,
            pending: 0,
            attack: 0.0,
            release: 0.0,
            threshold: 0.0,
            input_gain: 0.0,
            output_gain: 0.0,
            lookahead: 0,
            maximum: 0,
            mode: 0,
            statistics: false,
            reset_statistics: false,
            maximum_sample: [0.0; 6],
            minimum_gain: [1.0; 6],
            state: None,
        }
    }
    pub(super) fn uses_pool(&self, pool: usize) -> bool {
        self.enabled && self.work.pool == Some(pool)
    }
    pub(super) fn update(
        &mut self,
        raw: &[u8],
        rate: u32,
        started: bool,
    ) -> Result<u8, UpdateError> {
        let p = &raw[0x20..];
        let valid = |v| matches!(v, 1 | 2 | 4 | 6);
        let maximum = u16::from_le_bytes(p[12..14].try_into().unwrap()) as usize;
        let channels = u16::from_le_bytes(p[14..16].try_into().unwrap()) as usize;
        if !valid(maximum) || p[0x40] > 2 {
            return Err(UpdateError::Invalid);
        }
        let channels = if valid(channels) { channels } else { maximum };
        if channels > maximum {
            return Err(UpdateError::Invalid);
        }
        let f = |i| f32::from_bits(word(p, i));
        let attack = f(0x24);
        let release = f(0x28);
        let threshold = f(0x2c);
        let input_gain = f(0x30);
        let output_gain = f(0x34);
        let minimum = word(p, 0x38) as i32;
        let maximum = word(p, 0x3c) as i32;
        let mode = word(p, 0x44);
        if raw[2] != 0
            && (word(p, 0x10) != rate
                || minimum <= 0
                || maximum < minimum
                || ![attack, release, threshold, input_gain, output_gain]
                    .iter()
                    .all(|f| f.is_finite())
                || !(0.0..=1.0).contains(&attack)
                || !(0.0..=1.0).contains(&release)
                || threshold < 0.0
                || input_gain < 0.0
                || output_gain < 0.0
                || mode > 1)
        {
            return Err(UpdateError::Invalid);
        }
        self.mix = word(raw, 4) as usize;
        self.order = word(raw, 24) as usize;
        self.enabled = raw[2] != 0;
        self.inputs = std::array::from_fn(|i| p[i] as usize);
        self.outputs = std::array::from_fn(|i| p[i + 6] as usize);
        self.channels = channels;
        self.pending = if self.state.is_none() { 0 } else { p[0x40] };
        self.attack = attack;
        self.release = release;
        self.threshold = threshold;
        self.input_gain = input_gain;
        self.output_gain = output_gain;
        self.lookahead = minimum.max(0) as usize;
        self.maximum = maximum.max(0) as usize;
        self.mode = mode;
        self.statistics = self.revision >= 9 && p[0x41] != 0;
        self.reset_statistics |= p[0x42] != 0;
        Ok(
            if self.new || (started && self.was_enabled != Some(false)) {
                3
            } else {
                4
            },
        )
    }
    pub(super) fn write_result(&self, output: &mut [u8]) {
        for (i, &value) in self
            .maximum_sample
            .iter()
            .chain(&self.minimum_gain)
            .enumerate()
        {
            output[i * 4..i * 4 + 4].copy_from_slice(&value.to_le_bytes());
        }
    }
    pub(super) fn process(
        &mut self,
        buffers: &mut [i32],
        samples: usize,
        count: usize,
    ) -> Result<(), Box<str>> {
        if self.inputs[..self.channels]
            .iter()
            .chain(&self.outputs[..self.channels])
            .any(|i| *i >= count)
        {
            return Err(format!(
                "rendering: Limiter input/output exceeds its mix's {count} buffers"
            )
            .into());
        }
        if self.enabled {
            if self.pending == 0 || self.state.is_none() {
                self.state = Some(State {
                    average: [Q::raw(0); 6],
                    gain: [Q::ONE; 6],
                    offset: [0; 6],
                    delay: std::array::from_fn(|c| {
                        if c < self.channels {
                            vec![Q::raw(0); self.maximum]
                        } else {
                            Vec::new()
                        }
                    }),
                });
            }
            let s = self.state.as_mut().unwrap();
            if s.delay[..self.channels]
                .iter()
                .enumerate()
                .any(|(c, d)| d.len() < self.lookahead || s.offset[c] >= d.len())
            {
                return Err(
                    "rendering: Limiter look-ahead grew without initializing its state".into(),
                );
            }
            if self.statistics && self.reset_statistics {
                self.maximum_sample[..self.channels].fill(0.0);
                self.minimum_gain[..self.channels].fill(1.0);
            }
            for sample in 0..samples {
                for c in 0..self.channels {
                    // Normalization by 32768 converts the integer sample
                    // directly to the Q15 storage representation.
                    let input = Q::raw(i64::from(buffers[self.inputs[c] * samples + sample]))
                        * Q::float(self.input_gain);
                    let absolute = if input < Q::raw(0) { -input } else { input };
                    let coeff = if absolute > s.average[c] {
                        self.attack
                    } else {
                        self.release
                    };
                    s.average[c] = s.average[c]
                        + Q::float(((absolute - s.average[c]) * Q::float(coeff)).value());
                    let a = s.average[c].0 as f64 / 32768.0;
                    let q = (a * 512.0) as i32;
                    let reciprocal_estimate = 1.0 / ((f64::from(q) + 0.5) / 512.0);
                    let estimate = ((256.0 * reciprocal_estimate + 0.5) as i32) as f32 / 256.0;
                    let mut reciprocal = Q::float(estimate);
                    if self.mode != 1 {
                        let temp = Q::float(2.0) - s.average[c] * reciprocal;
                        reciprocal = Q::float(2.0) - s.average[c] * temp;
                    }
                    let attenuation = if s.average[c] > Q::float(self.threshold) {
                        Q::float(self.threshold) * reciprocal
                    } else {
                        Q::ONE
                    };
                    let coeff = if attenuation < s.gain[c] {
                        self.attack
                    } else {
                        self.release
                    };
                    s.gain[c] = s.gain[c] + (attenuation - s.gain[c]) * Q::float(coeff);
                    let delayed = s.delay[c][s.offset[c]];
                    s.delay[c][s.offset[c]] = input;
                    s.offset[c] = (s.offset[c] + 1) % self.lookahead;
                    let output =
                        delayed * s.gain[c] * Q::float(self.output_gain) * Q::raw(32768_i64 << 15);
                    buffers[self.outputs[c] * samples + sample] = output
                        .long()
                        .clamp(i64::from(i32::MIN), i64::from(i32::MAX))
                        as i32;
                    if self.statistics {
                        self.maximum_sample[c] = self.maximum_sample[c].max(absolute.value());
                        self.minimum_gain[c] = self.minimum_gain[c].min(s.gain[c].value());
                    }
                }
            }
        } else {
            for c in 0..self.channels {
                let src = self.inputs[c] * samples;
                let dst = self.outputs[c] * samples;
                if src != dst {
                    buffers.copy_within(src..src + samples, dst);
                }
            }
        }
        self.was_enabled = Some(self.enabled);
        self.new = false;
        self.pending = 2;
        self.reset_statistics = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::{Effect, MemoryPool};
    use crate::audio_renderer::worker::RenderMemory;
    use sha2::{Digest, Sha256};

    fn packet(channels: u16, mode: u32) -> [u8; 0xc0] {
        let mut raw = [0; 0xc0];
        raw[0] = 7;
        raw[1] = 1;
        raw[2] = 1;
        raw[8..16].copy_from_slice(&0x1000_u64.to_le_bytes());
        raw[16..24].copy_from_slice(&4096_u64.to_le_bytes());
        let p = &mut raw[0x20..];
        for i in 0..6 {
            p[i] = i as u8;
            p[6 + i] = i as u8;
        }
        p[12..14].copy_from_slice(&channels.to_le_bytes());
        p[14..16].copy_from_slice(&channels.to_le_bytes());
        p[0x10..0x14].copy_from_slice(&48000_u32.to_le_bytes());
        p[0x38..0x3c].copy_from_slice(&120_u32.to_le_bytes());
        p[0x3c..0x40].copy_from_slice(&240_u32.to_le_bytes());
        p[0x41] = 1;
        p[0x44..0x48].copy_from_slice(&mode.to_le_bytes());
        for (i, f) in [
            (0x24, 0.2_f32),
            (0x28, 0.03),
            (0x2c, 0.65),
            (0x30, 1.3),
            (0x34, 0.8),
        ] {
            p[i..i + 4].copy_from_slice(&f.to_le_bytes());
        }
        raw
    }
    fn pools() -> [Option<MemoryPool>; 1] {
        [Some(MemoryPool {
            address: 0x1000,
            bytes: 4096,
            ranges: nixe_memory::CapturedMemory::new(crate::test_memory::physical(&vec![0; 4096]))
                .unwrap(),
        })]
    }
    #[test]
    fn lookahead_compression_and_statistics_match_reference_in_both_modes() {
        for (channels, mode, expected) in [
            (
                1,
                0,
                "c8768f0373bc01cfdec68d6352217c0dc17370640519abf953790544c173d1cf",
            ),
            (
                1,
                1,
                "bd13ec5737a50bb9b053d789354884f3533046871cc4fe30cff976f5a9948a18",
            ),
            (
                2,
                0,
                "8132e2b9da602074422be848fa4ac29136a5c035224f7d9334e9acdfbb5cbddc",
            ),
            (
                2,
                1,
                "78ac7201dfe4d55258f5a52cc0ad63d89f609ec3b8020b41c22cfeebda07b5fe",
            ),
            (
                4,
                0,
                "c1c28ed33155c7d058e52fbb6465ffbf331e2b5b3ed11ef402da0588494098e8",
            ),
            (
                4,
                1,
                "720d7124e0c351e91894713447319180410eaf8e2149e9f010e5ff410235c515",
            ),
            (
                6,
                0,
                "3237834c9d8abdb7a51428acedaea5a5bf6e04db5cd59ab2fec8412100b2df1f",
            ),
            (
                6,
                1,
                "6fed3d9d2d209605b5954d94bdb075e503195b77a9ff9d7885108ce6d9d5b26c",
            ),
        ] {
            // Independent C++ FixedPoint<49,15> output for 40 blocks, including
            // live attack/threshold changes and a statistics reset at block 20.
            let mut raw = packet(channels, mode);
            let pools = pools();
            let mut effect = Effect::default();
            let mut hash = Sha256::new();
            let memory = RenderMemory::testing();
            let mut buffers = vec![0_i32; usize::from(channels) * 240];
            for block in 0..40 {
                if block == 20 {
                    raw[0x44..0x48].copy_from_slice(&0.35_f32.to_le_bytes());
                    raw[0x4c..0x50].copy_from_slice(&0.4_f32.to_le_bytes());
                    raw[0x62] = 1;
                    raw[0x60] = 1;
                }
                effect
                    .update(&raw, 0, 11, 48000, &pools, false, &|_, _| panic!(), true)
                    .unwrap();
                for c in 0..usize::from(channels) {
                    for i in 0..240 {
                        buffers[c * 240 + i] =
                            (((block * 240 + i) * 97 + c * 7919) % 100001) as i32 - 50000;
                    }
                }
                effect
                    .process(&memory, &mut buffers, 240, usize::from(channels))
                    .unwrap();
                for sample in &buffers {
                    hash.update(sample.to_le_bytes());
                }
                let mut stats = [0; 0x80];
                effect.write_result(&mut stats);
                hash.update(&stats[..0x30]);
                raw[1] = 0;
                raw[0x60] = 2;
                raw[0x62] = 0;
            }
            assert_eq!(
                format!("{:x}", hash.finalize()),
                expected,
                "channels={channels}, mode={mode}"
            );
        }
    }
    #[test]
    fn disabled_limiter_bypasses_gains_and_release_state_tracks_worker_consumption() {
        let mut raw = packet(2, 1);
        let pools = pools();
        let memory = RenderMemory::testing();
        let mut effect = Effect::default();
        assert_eq!(
            effect
                .update(&raw, 0, 11, 48000, &pools, false, &|_, _| panic!(), true)
                .unwrap(),
            3
        );
        assert!(effect.uses_pool(0));
        effect.process(&memory, &mut [0; 480], 240, 2).unwrap();
        raw[1] = 0;
        raw[2] = 0;
        raw[0x60] = 2;
        assert_eq!(
            effect
                .update(&raw, 0, 11, 48000, &pools, false, &|_, _| panic!(), true)
                .unwrap(),
            3
        );
        let mut data = [456; 480];
        effect.process(&memory, &mut data, 240, 2).unwrap();
        assert!(data.iter().all(|s| *s == 456));
        assert!(!effect.uses_pool(0));
        assert_eq!(
            effect
                .update(&raw, 0, 11, 48000, &pools, false, &|_, _| panic!(), true)
                .unwrap(),
            4
        );
        raw[2] = 1;
        raw[0x58..0x5c].copy_from_slice(&0_u32.to_le_bytes());
        assert!(
            effect
                .update(&raw, 0, 11, 48000, &pools, false, &|_, _| panic!(), true)
                .is_err()
        );
    }
}
