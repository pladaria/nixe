//! Mix-buffer biquad effects use the same DSP primitive as voice filters.
//! https://github.com/eden-emulator/mirror/blob/master/src/audio_core/renderer/effect/biquad_filter.h
//! https://github.com/eden-emulator/mirror/blob/master/src/audio_core/renderer/command/command_buffer.cpp
use super::super::{
    biquad,
    update::{UpdateError, word},
};

#[derive(Debug)]
pub(in crate::audio_renderer) struct Biquad {
    pub(super) mix: usize,
    pub(super) order: usize,
    inputs: [usize; 6],
    outputs: [usize; 6],
    channels: usize,
    parameters: biquad::Parameters,
    states: [biquad::State; 6],
    revision: u32,
    pending: u8,
    enabled: bool,
    new: bool,
    was_enabled: Option<bool>,
}
impl Biquad {
    pub(super) fn new(revision: u32) -> Self {
        Self {
            mix: 0,
            order: 0,
            inputs: [0; 6],
            outputs: [0; 6],
            channels: 0,
            parameters: Default::default(),
            states: std::array::from_fn(|_| Default::default()),
            revision,
            pending: 0,
            enabled: false,
            new: true,
            was_enabled: None,
        }
    }
    pub(super) fn update(&mut self, raw: &[u8], started: bool) -> Result<u8, UpdateError> {
        let p = &raw[0x20..];
        let modern = self.revision >= 9;
        let channels = p[if modern { 0x24 } else { 0x16 }] as usize;
        let pending = p[if modern { 0x25 } else { 0x17 }];
        if channels > 6 || pending > 2 {
            return Err(UpdateError::Invalid);
        }
        let mut coeff = [0_i16; 5];
        for (i, coeff) in coeff.iter_mut().enumerate() {
            *coeff = if modern {
                let f = f32::from_bits(word(p, 0x10 + i * 4));
                if !f.is_finite() {
                    return Err(UpdateError::Invalid);
                }
                // Effect ABI v2 supplies floats, but command generation still
                // quantizes and clamps to Q14 before the shared filter runs.
                (f * 16384.0).clamp(i16::MIN as f32, i16::MAX as f32) as i16
            } else {
                i16::from_le_bytes(p[12 + i * 2..14 + i * 2].try_into().unwrap())
            };
        }
        self.mix = word(raw, 4) as usize;
        self.order = word(raw, 24) as usize;
        self.inputs = std::array::from_fn(|i| p[i] as usize);
        self.outputs = std::array::from_fn(|i| p[6 + i] as usize);
        self.channels = channels;
        self.pending = pending;
        self.enabled = raw[2] != 0;
        self.parameters = biquad::Parameters::q14(
            coeff[..3].try_into().unwrap(),
            coeff[3..].try_into().unwrap(),
        );
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
        if self.inputs[..self.channels]
            .iter()
            .chain(&self.outputs[..self.channels])
            .any(|i| *i >= count)
        {
            return Err(format!(
                "rendering: Biquad input/output exceeds its mix's {count} buffers"
            )
            .into());
        }
        for c in 0..self.channels {
            let src = self.inputs[c] * samples;
            let dst = self.outputs[c] * samples;
            if src != dst {
                buffers.copy_within(src..src + samples, dst);
            }
            if self.enabled {
                if self.pending == 0 || (self.pending == 1 && self.revision < 6) {
                    self.states[c] = Default::default();
                }
                self.states[c].process(
                    self.parameters,
                    self.revision,
                    &mut buffers[dst..dst + samples],
                );
            }
        }
        self.was_enabled = Some(self.enabled);
        self.new = false;
        self.pending = 2;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::Effect;
    use super::*;

    fn packet(revision: u32) -> [u8; 0xc0] {
        let mut raw = [0; 0xc0];
        raw[0] = 6;
        raw[1] = 1;
        raw[2] = 1;
        let p = &mut raw[0x20..];
        p[6] = 1;
        if revision >= 9 {
            p[0x10..0x14].copy_from_slice(&0.50005_f32.to_le_bytes());
            p[0x1c..0x20].copy_from_slice(&0.5_f32.to_le_bytes());
            p[0x24] = 1;
        } else {
            p[0x0c..0x0e].copy_from_slice(&8192_i16.to_le_bytes());
            p[0x12..0x14].copy_from_slice(&8192_i16.to_le_bytes());
            p[0x16] = 1;
        }
        raw
    }

    #[test]
    fn both_effect_abis_use_the_shared_filter_and_preserve_history_across_updates() {
        for revision in [5, 6, 7, 9, 11, 15] {
            let mut raw = packet(revision);
            let state_offset = if revision >= 9 { 0x45 } else { 0x37 };
            let mut effect = Effect::default();
            assert_eq!(
                effect
                    .update(
                        &raw,
                        0,
                        revision,
                        48000,
                        &[],
                        false,
                        &|_, _| panic!("biquad has no work buffer"),
                        true
                    )
                    .unwrap(),
                3
            );
            assert!(!effect.uses_pool(0));
            let mut data = [32768, 0, 0, 0];
            effect
                .process(
                    &crate::audio_renderer::worker::RenderMemory::testing(),
                    &mut data,
                    2,
                    2,
                )
                .unwrap();
            assert_eq!(&data[2..], &[16384, 8192]);
            raw[1] = 0;
            raw[state_offset] = 2;
            effect
                .update(&raw, 0, revision, 48000, &[], false, &|_, _| panic!(), true)
                .unwrap();
            data.fill(0);
            effect
                .process(
                    &crate::audio_renderer::worker::RenderMemory::testing(),
                    &mut data,
                    2,
                    2,
                )
                .unwrap();
            assert_eq!(&data[2..], &[4096, 2048]);
            raw[state_offset] = 1;
            effect
                .update(&raw, 0, revision, 48000, &[], false, &|_, _| panic!(), true)
                .unwrap();
            data.fill(0);
            effect
                .process(
                    &crate::audio_renderer::worker::RenderMemory::testing(),
                    &mut data,
                    2,
                    2,
                )
                .unwrap();
            assert_eq!(
                &data[2..],
                if revision < 6 { &[0, 0] } else { &[1024, 512] }
            );
            raw[2] = 0;
            raw[state_offset] = 2;
            effect
                .update(&raw, 0, revision, 48000, &[], false, &|_, _| panic!(), true)
                .unwrap();
            data = [123, 456, 0, 0];
            effect
                .process(
                    &crate::audio_renderer::worker::RenderMemory::testing(),
                    &mut data,
                    2,
                    2,
                )
                .unwrap();
            assert_eq!(&data[2..], &[123, 456]);
            assert_eq!(
                effect
                    .update(&raw, 0, revision, 48000, &[], false, &|_, _| panic!(), true)
                    .unwrap(),
                4
            );
        }
    }

    #[test]
    fn coefficient_clamping_reset_and_bad_routes_are_handled_before_filtering() {
        let mut raw = packet(11);
        raw[0x30..0x34].copy_from_slice(&10_f32.to_le_bytes());
        raw[0x3c..0x40].copy_from_slice(&0_f32.to_le_bytes());
        let mut b = Biquad::new(11);
        b.update(&raw, true).unwrap();
        let mut data = [16384, 0];
        assert!(b.process(&mut data, 1, 1).is_err());
        b.process(&mut data, 1, 2).unwrap();
        assert_eq!(data[1], 32767);
        raw[0x30..0x34].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(b.update(&raw, true).is_err());
        let mut raw = packet(11);
        raw[0x44] = 7;
        assert!(b.update(&raw, true).is_err());
    }
}
