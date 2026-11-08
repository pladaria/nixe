// SPDX-FileCopyrightText: Copyright 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
//! Q14 delay lines, feedback matrix and low-pass state from Eden's DelayCommand.
//! https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/audio_core/renderer/command/effect/delay.cpp
//! https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/audio_core/renderer/effect/delay.cpp
use super::{Attachment, UpdateError, fixed::Q14 as Q, word};

#[derive(Debug)]
pub(in crate::audio_renderer) struct Delay {
    pub(super) mix: usize,
    pub(super) order: usize,
    work: Attachment,
    revision: u32,
    params: Parameters,
    state: Option<State>,
    pending: u8,
    enabled: bool,
    new: bool,
    was_enabled: Option<bool>,
}
#[derive(Debug, Default)]
struct Parameters {
    inputs: [usize; 6],
    outputs: [usize; 6],
    channels: usize,
    maximum_time: u32,
    time: u32,
    rate: Q,
    input: Q,
    feedback: Q,
    wet: Q,
    dry: Q,
    spread: Q,
    lowpass: Q,
}
#[derive(Debug)]
struct Line {
    samples: Vec<Q>,
    cursor: usize,
}
#[derive(Debug)]
struct State {
    lines: [Line; 6],
    matrix: [[Q; 6]; 6],
    lowpass_gain: Q,
    lowpass_feedback: Q,
    lowpass_z: [Q; 6],
}
impl Delay {
    pub(super) fn new(work: Attachment, revision: u32) -> Self {
        Self {
            mix: 0,
            order: 0,
            work,
            revision,
            params: Parameters::default(),
            state: None,
            pending: 0,
            enabled: false,
            new: true,
            was_enabled: None,
        }
    }
    pub(super) fn uses_pool(&self, pool: usize) -> bool {
        self.enabled && self.work.pool == Some(pool)
    }
    pub(super) fn update(&mut self, raw: &[u8], started: bool) -> Result<u8, UpdateError> {
        let p = &raw[0x20..];
        let valid = |n| matches!(n, 1 | 2 | 4 | 6);
        let maximum = u16::from_le_bytes(p[12..14].try_into().unwrap()) as usize;
        let requested = u16::from_le_bytes(p[14..16].try_into().unwrap()) as usize;
        if !valid(maximum) || p[0x34] > 2 {
            return Err(UpdateError::Invalid);
        }
        let q = |offset| Q::raw(i64::from(word(p, offset) as i32));
        let mut params = Parameters {
            inputs: std::array::from_fn(|i| usize::from(p[i])),
            outputs: std::array::from_fn(|i| usize::from(p[i + 6])),
            channels: if valid(requested) { requested } else { maximum },
            maximum_time: word(p, 16),
            time: word(p, 20),
            rate: q(24),
            input: q(28),
            feedback: q(32),
            wet: q(36),
            dry: q(40),
            spread: q(44),
            lowpass: q(48),
        };
        if params.channels == 6 && self.revision < 11 {
            params.inputs.swap(2, 4);
            params.inputs.swap(3, 5);
            params.outputs.swap(2, 4);
            params.outputs.swap(3, 5);
        }
        self.mix = word(raw, 4) as usize;
        self.order = word(raw, 24) as usize;
        self.enabled = raw[2] != 0;
        if self.pending == 2 && valid(requested) {
            self.pending = p[0x34];
        }
        self.params = params;
        Ok(
            if self.new || (started && self.was_enabled != Some(false)) {
                3
            } else {
                4
            },
        )
    }
    pub(super) fn process(
        &mut self,
        buffers: &mut [i32],
        samples: usize,
        count: usize,
    ) -> Result<(), Box<str>> {
        let p = &self.params;
        if p.inputs[..p.channels]
            .iter()
            .chain(&p.outputs[..p.channels])
            .any(|&i| i >= count)
        {
            return Err(
                format!("rendering: Delay input/output exceeds its mix's {count} buffers").into(),
            );
        }
        if self.enabled {
            match self.state.as_mut() {
                Some(state) if self.pending != 0 => {
                    if self.pending == 1 {
                        state.update(p);
                    }
                }
                _ => self.state = Some(State::new(p)),
            }
            self.state.as_mut().unwrap().process(p, buffers, samples)?;
        } else {
            for c in 0..p.channels {
                let src = p.inputs[c] * samples;
                let dst = p.outputs[c] * samples;
                if src != dst {
                    buffers.copy_within(src..src + samples, dst);
                }
            }
        }
        self.was_enabled = Some(self.enabled);
        self.new = false;
        self.pending = 2;
        Ok(())
    }
}
impl State {
    fn new(p: &Parameters) -> Self {
        // The reference uses Q32 for the maximum, Q14 for delay time, and
        // rounds both down. A zero-length delay still owns a one-sample ring.
        let maximum = (super::fixed::Fixed::<32>::float(0.064).0 as i128
            * (p.rate.0 >> 14) as i128
            * i128::from(p.maximum_time))
            >> 32;
        let delay = Q::raw(i64::from(p.time) << 14) * (p.rate / Q::float(1000.0));
        let length = ((delay.0 >> 14) as i128).min(maximum).max(1) as usize;
        let mut state = Self {
            lines: std::array::from_fn(|i| Line {
                samples: if i < p.channels {
                    vec![Q::raw(0); length]
                } else {
                    Vec::new()
                },
                cursor: 0,
            }),
            matrix: [[Q::raw(0); 6]; 6],
            lowpass_gain: Q::raw(0),
            lowpass_feedback: Q::raw(0),
            lowpass_z: [Q::raw(0); 6],
        };
        state.update(p);
        state
    }
    fn update(&mut self, p: &Parameters) {
        let feedback = p.feedback * Q::raw(16056); // 0.97998046875 in Q14.
        let own = feedback * (Q::ONE - p.spread);
        let spread = if p.channels >= 4 {
            Q::raw(p.spread.0 >> 1)
        } else {
            p.spread
        };
        let cross = spread * feedback;
        self.matrix = [[Q::raw(0); 6]; 6];
        for i in 0..p.channels {
            self.matrix[i][i] = own;
        }
        let links: &[(usize, usize)] = match p.channels {
            1 => {
                self.matrix[0][0] = feedback;
                &[]
            }
            2 => &[(0, 1)],
            4 => &[(0, 1), (0, 2), (1, 3), (2, 3)],
            6 => {
                self.matrix[3][3] = p.feedback;
                &[(0, 2), (0, 4), (1, 2), (1, 5), (4, 5)]
            }
            _ => unreachable!("validated delay channel count"),
        };
        for &(a, b) in links {
            self.matrix[a][b] = cross;
            self.matrix[b][a] = cross;
        }
        self.lowpass_feedback = p.lowpass * Q::raw(15564); // 0.949951171875 in Q14.
        self.lowpass_gain = Q::ONE - self.lowpass_feedback;
    }
    fn process(
        &mut self,
        p: &Parameters,
        buffers: &mut [i32],
        samples: usize,
    ) -> Result<(), Box<str>> {
        if self.lines[..p.channels]
            .iter()
            .any(|line| line.samples.is_empty())
        {
            return Err("rendering: Delay channel count grew without state initialization".into());
        }
        for sample in 0..samples {
            let input: [Q; 6] = std::array::from_fn(|c| {
                if c < p.channels {
                    Q::raw((i64::from(buffers[p.inputs[c] * samples + sample]) * 64) << 14)
                } else {
                    Q::raw(0)
                }
            });
            let delayed: [Q; 6] = std::array::from_fn(|c| {
                if c < p.channels {
                    self.lines[c].samples[self.lines[c].cursor]
                } else {
                    Q::raw(0)
                }
            });
            for (c, &input_sample) in input.iter().enumerate().take(p.channels) {
                let mut feedback = Q::raw(0);
                for (j, &value) in delayed[..p.channels].iter().enumerate() {
                    feedback = feedback + value * self.matrix[j][c];
                }
                let gained = input_sample * p.input + feedback;
                self.lowpass_z[c] =
                    gained * self.lowpass_gain + self.lowpass_z[c] * self.lowpass_feedback;
                let line = &mut self.lines[c];
                line.samples[line.cursor] = self.lowpass_z[c];
                line.cursor = (line.cursor + 1) % line.samples.len();
            }
            for c in 0..p.channels {
                let output = input[c] * p.dry + delayed[c] * p.wet;
                buffers[p.outputs[c] * samples + sample] = ((output.0 >> 14) as i32) / 64;
            }
        }
        Ok(())
    }
}
