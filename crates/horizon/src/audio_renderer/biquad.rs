//! Voice filters use signed Q14 coefficients, positive feedback coefficients,
//! and revision-dependent DSP state/rounding. They run after SRC, before gain.
//! https://switchbrew.github.io/libnx/audren_8h_source.html
//! https://github.com/eden-emulator/mirror/blob/master/src/audio_core/renderer/command/effect/biquad_filter.cpp
//! https://github.com/eden-emulator/mirror/blob/master/src/audio_core/renderer/command/command_generator.cpp

#[derive(Debug, Default, Clone, Copy)]
pub(super) struct Parameters {
    enabled: bool,
    feed: [i16; 3],
    feedback: [i16; 2],
    float: Option<([f32; 3], [f32; 2])>,
}
impl Parameters {
    pub(super) fn q14(feed: [i16; 3], feedback: [i16; 2]) -> Self {
        Self {
            enabled: true,
            feed,
            feedback,
            float: None,
        }
    }
    pub(super) fn decode(raw: &[u8]) -> Self {
        let coefficient = |i| i16::from_le_bytes(raw[i..i + 2].try_into().unwrap());
        Self {
            enabled: raw[0] != 0,
            feed: [coefficient(2), coefficient(4), coefficient(6)],
            feedback: [coefficient(8), coefficient(10)],
            float: None,
        }
    }
    pub(super) fn decode_float(raw: &[u8]) -> Self {
        let coefficient = |i| f32::from_le_bytes(raw[i..i + 4].try_into().unwrap());
        Self {
            enabled: raw[0] != 0,
            float: Some((
                [coefficient(4), coefficient(8), coefficient(12)],
                [coefficient(16), coefficient(20)],
            )),
            ..Self::default()
        }
    }
}

#[derive(Debug, Default)]
pub(super) enum State {
    #[default]
    Inactive,
    Integer([i64; 2]),
    Float([f64; 4]),
}
impl State {
    pub(super) fn process(&mut self, params: Parameters, revision: u32, samples: &mut [i32]) {
        if !params.enabled {
            *self = Self::Inactive;
            return;
        }
        if revision >= 7 {
            if !matches!(self, Self::Float(_)) {
                *self = Self::Float([0.0; 4]);
            }
            let Self::Float(history) = self else {
                unreachable!()
            };
            let (feed, feedback) = if let Some((feed, feedback)) = params.float {
                (feed.map(f64::from), feedback.map(f64::from))
            } else {
                (
                    params.feed.map(|v| f64::from(v) / 16384.0),
                    params.feedback.map(|v| f64::from(v) / 16384.0),
                )
            };
            for sample in samples {
                let input = f64::from(*sample);
                let output = input * feed[0]
                    + history[0] * feed[1]
                    + history[1] * feed[2]
                    + history[2] * feedback[0]
                    + history[3] * feedback[1];
                // Integer output saturates/truncates, while feedback retains
                // the unquantized result across samples and renderer frames.
                *sample = output.clamp(f64::from(i32::MIN), f64::from(i32::MAX)) as i32;
                *history = [input, history[0], output, history[2]];
            }
        } else {
            if !matches!(self, Self::Integer(_)) {
                *self = Self::Integer([0; 2]);
            }
            let Self::Integer(delay) = self else {
                unreachable!()
            };
            let feed = params.feed.map(i64::from);
            let feedback = params.feedback.map(i64::from);
            for sample in samples {
                let input = i64::from(*sample);
                let output = ((input * feed[0] + delay[0] + 8192) >> 14)
                    .clamp(i64::from(i32::MIN), i64::from(i32::MAX));
                *delay = [
                    delay[1] + feed[1] * input + feedback[0] * output,
                    feed[2] * input + feedback[1] * output,
                ];
                *sample = output as i32;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn filter() -> Parameters {
        Parameters {
            enabled: true,
            feed: [8192, 0, 0],
            feedback: [8192, 0],
            float: None,
        }
    }

    #[test]
    fn revision_controls_quantization_and_feedback_survives_frame_boundaries() {
        for revision in [6, 7, 10] {
            let mut whole = [1024, 0, 0, 0, 0, 0, 0, 0];
            State::default().process(filter(), revision, &mut whole);
            assert_eq!(whole, [512, 256, 128, 64, 32, 16, 8, 4]);
            let mut split = [1024, 0, 0, 0, 0, 0, 0, 0];
            let mut state = State::default();
            state.process(filter(), revision, &mut split[..3]);
            state.process(filter(), revision, &mut split[3..]);
            assert_eq!(split, whole);
        }
        let mut integer = [1, 0, 0];
        let mut float = integer;
        State::default().process(filter(), 6, &mut integer);
        State::default().process(filter(), 7, &mut float);
        assert_eq!(integer, [1, 1, 1]);
        assert_eq!(float, [0, 0, 0]);
    }

    #[test]
    fn disabling_resets_state_but_coefficient_updates_retain_it() {
        for revision in [6, 7] {
            let mut state = State::default();
            state.process(filter(), revision, &mut [1024]);
            let mut silence = [0];
            state.process(
                Parameters {
                    feed: [0; 3],
                    ..filter()
                },
                revision,
                &mut silence,
            );
            assert_eq!(silence, [256]);
            state.process(
                Parameters {
                    enabled: false,
                    ..filter()
                },
                revision,
                &mut silence,
            );
            assert_eq!(silence, [256]);
            silence[0] = 0;
            state.process(filter(), revision, &mut silence);
            assert_eq!(silence, [0]);
        }
    }

    #[test]
    fn signed_coefficients_and_signed_output_saturation_are_preserved() {
        let mut bytes = [0; 12];
        bytes[0] = 1;
        for (out, c) in bytes[2..].chunks_exact_mut(2).zip([-16384_i16, 0, 0, 0, 0]) {
            out.copy_from_slice(&c.to_le_bytes());
        }
        for revision in [6, 7] {
            let mut samples = [-100, 100];
            State::default().process(Parameters::decode(&bytes), revision, &mut samples);
            assert_eq!(samples, [100, -100]);
            let mut samples = [i32::MAX, i32::MIN];
            State::default().process(
                Parameters {
                    feed: [32767, 0, 0],
                    feedback: [0, 0],
                    ..filter()
                },
                revision,
                &mut samples,
            );
            assert_eq!(samples, [i32::MAX, i32::MIN]);
        }
    }

    #[test]
    fn all_feedforward_and_feedback_taps_contribute_in_order() {
        let params = Parameters {
            feed: [8192, 4096, 2048],
            feedback: [8192, -4096],
            ..filter()
        };
        for revision in [6, 7] {
            let mut samples = [1024, 0, 0, 0, 0, 0];
            State::default().process(params, revision, &mut samples);
            assert_eq!(samples, [512, 512, 256, 0, -64, -32]);
        }
    }
}
