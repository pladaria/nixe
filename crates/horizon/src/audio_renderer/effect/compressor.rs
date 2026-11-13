//! Linked-channel compressor with persistent envelope and gain state.
//! https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/audio_core/renderer/command/effect/compressor.cpp
use super::super::update::word;

#[derive(Debug)]
pub(in crate::audio_renderer) struct Compressor {
    pub mix: usize,
    pub order: usize,
    parameters: [u8; 0x38],
    state: State,
    enabled: bool,
    was_enabled: Option<bool>,
}
impl Default for Compressor {
    fn default() -> Self {
        Self {
            mix: 0,
            order: 0,
            parameters: [0; 0x38],
            state: State::default(),
            enabled: false,
            was_enabled: None,
        }
    }
}
#[derive(Debug, Default)]
struct State {
    energy: f32,
    previous_gain: f32,
    gain: f32,
    slope: f32,
    low: f32,
    high: f32,
    output_gain: f32,
}

// These decimal coefficients and the fractional exponent are part of the
// reference DSP algorithm, rather than substitutions for mathematical constants.
#[allow(clippy::approx_constant)]
fn gain(value: f32) -> f32 {
    let exponent = value / 20.0 * 3.3219;
    ((exponent - exponent.trunc()) * 0.69315).exp2()
}

impl Compressor {
    pub fn update(&mut self, raw: &[u8], started: bool) -> u8 {
        self.mix = word(raw, 4) as usize;
        self.order = word(raw, 24) as usize;
        self.enabled = raw[2] != 0;
        self.parameters.copy_from_slice(&raw[0x20..0x58]);
        if self.was_enabled.is_none() || (started && self.was_enabled != Some(false)) {
            3
        } else {
            4
        }
    }
    pub fn process(
        &mut self,
        buffers: &mut [i32],
        samples: usize,
        count: usize,
    ) -> Result<(), Box<str>> {
        let p = &self.parameters;
        let channels = i16::from_le_bytes(p[14..16].try_into().unwrap());
        if !matches!(channels, 1 | 2 | 4 | 6) {
            return Err(
                format!("rendering: unsupported Compressor channel count {channels}").into(),
            );
        }
        let channels = channels as usize;
        if p[..channels]
            .iter()
            .chain(&p[6..6 + channels])
            .any(|&i| usize::from(i) >= count)
        {
            return Err("rendering: Compressor input/output exceeds its mix's buffers".into());
        }
        if self.enabled {
            let threshold = f32::from_bits(word(p, 0x14));
            let ratio = f32::from_bits(word(p, 0x18));
            let energy_rate = f32::from_bits(word(p, 0x24));
            let attack = f32::from_bits(word(p, 0x28));
            let release = f32::from_bits(word(p, 0x2c));
            let output = f32::from_bits(word(p, 0x30));
            if p[0x34] > 2
                || ratio <= 0.0
                || ![threshold, ratio, energy_rate, attack, release, output]
                    .into_iter()
                    .all(f32::is_finite)
            {
                return Err("rendering: invalid Compressor coefficients/state".into());
            }
            if p[0x34] == 0 {
                self.state = State {
                    previous_gain: 1.0,
                    gain: 1.0,
                    ..Default::default()
                };
            }
            if p[0x34] <= 1 {
                let inverse = 1.0 / ratio;
                let makeup = if p[0x35] != 0 {
                    threshold * 0.5 * (inverse - 1.0) - 3.0
                } else {
                    0.0
                };
                self.state.slope = (1.0 - inverse) / 6.0;
                self.state.low = threshold - 1.5;
                self.state.high = threshold + 1.5;
                self.state.output_gain = gain(output + makeup);
            }
            let state = &mut self.state;
            let prior_energy = state.energy;
            for sample in 0..samples {
                let mut energy = 0.0;
                for &input in &p[..channels] {
                    let value = i64::from(buffers[usize::from(input) * samples + sample]);
                    // Q49.15 multiplication retains the DSP's wrapping precision.
                    energy += value.wrapping_mul(value).wrapping_shl(15) as f32 / 32768.0;
                }
                state.energy += energy_rate * (energy / channels as f32 - prior_energy);
                let (level, mut target) = if state.energy >= 1.0e-10 {
                    (state.energy.log10() * 10.0, 1.0)
                } else {
                    (-100.0, 0.0)
                };
                if level >= state.low {
                    let reduction = if level >= state.high {
                        (1.0 / ratio - 1.0) * (level - threshold)
                    } else {
                        (level - state.low).powi(2) * -state.slope
                    };
                    target = gain(reduction);
                }
                let difference = state.previous_gain - target;
                let rate = if difference <= 0.08 { release } else { attack };
                if !((-0.08..=0.08).contains(&difference) && (state.gain - target).abs() >= 0.001) {
                    state.previous_gain = target;
                }
                state.gain += (target - state.gain) * rate;
                // Read each route when consumed: output/input aliases preserve
                // channel order without copying a block of samples.
                for channel in 0..channels {
                    let input = usize::from(p[channel]) * samples + sample;
                    let output = usize::from(p[6 + channel]) * samples + sample;
                    buffers[output] =
                        (buffers[input] as f32 * state.gain * state.output_gain) as i32;
                }
            }
        } else {
            for channel in 0..channels {
                let input = usize::from(p[channel]) * samples;
                let output = usize::from(p[6 + channel]) * samples;
                if input != output {
                    buffers.copy_within(input..input + samples, output);
                }
            }
        }
        self.parameters[0x34] = 2;
        self.was_enabled = Some(self.enabled);
        Ok(())
    }
}
