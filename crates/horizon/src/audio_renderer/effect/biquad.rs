//! Mix-buffer biquad effects retain their wire parameters until command generation.
//! https://github.com/eden-emulator/mirror/blob/master/src/audio_core/renderer/effect/biquad_filter.h
//! https://github.com/eden-emulator/mirror/blob/master/src/audio_core/renderer/command/command_buffer.cpp
//! https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/audio_core/renderer/command/command_generator.cpp
use super::super::{biquad, update::word};

#[derive(Debug)]
pub(in crate::audio_renderer) struct Biquad {
    pub(super) mix: usize,
    pub(super) order: usize,
    parameters: [u8; 0x28],
    states: [biquad::State; 6],
    revision: u32,
    enabled: bool,
    new: bool,
    was_enabled: Option<bool>,
}
impl Biquad {
    pub(super) fn new(revision: u32) -> Self {
        Self {
            mix: 0,
            order: 0,
            parameters: [0; 0x28],
            states: std::array::from_fn(|_| Default::default()),
            revision,
            enabled: false,
            new: true,
            was_enabled: None,
        }
    }
    pub(super) fn update(&mut self, raw: &[u8], started: bool) -> u8 {
        self.mix = word(raw, 4) as usize;
        self.order = word(raw, 24) as usize;
        self.enabled = raw[2] != 0;
        let size = if self.revision >= 9 { 0x28 } else { 0x18 };
        self.parameters[..size].copy_from_slice(&raw[0x20..0x20 + size]);
        if self.new || (started && self.was_enabled != Some(false)) {
            3
        } else {
            4
        }
    }
    pub(super) fn process(
        &mut self,
        buffers: &mut [i32],
        samples: usize,
        count: usize,
    ) -> Result<(), Box<str>> {
        let p = &self.parameters;
        let modern = self.revision >= 9;
        let channels = p[if modern { 0x24 } else { 0x16 }] as i8;
        let pending = p[if modern { 0x25 } else { 0x17 }];
        // Match GenerateBiquadFilterEffectCommand: parameters without an active
        // command do not reject the renderer's entire RequestUpdate. In
        // particular, channel_count is signed and the state gates generation.
        if channels > 0 && pending <= 2 {
            let channels = channels as usize;
            if channels > self.states.len() {
                return Err("rendering: Biquad channel count exceeds six channels".into());
            }
            if p[..channels]
                .iter()
                .chain(&p[6..6 + channels])
                .any(|&i| usize::from(i) >= count)
            {
                return Err(format!(
                    "rendering: Biquad input/output exceeds its mix's {count} buffers"
                )
                .into());
            }
            let mut coeff = [0_i16; 5];
            if self.enabled {
                for (i, coeff) in coeff.iter_mut().enumerate() {
                    *coeff = if modern {
                        let f = f32::from_bits(word(p, 0x10 + i * 4));
                        if !f.is_finite() {
                            return Err("rendering: Biquad coefficient is not finite".into());
                        }
                        // Eden quantizes effect v2 coefficients to Q14 when
                        // generating its command, even for float processing.
                        (f * 16384.0).clamp(i16::MIN as f32, i16::MAX as f32) as i16
                    } else {
                        i16::from_le_bytes(p[12 + i * 2..14 + i * 2].try_into().unwrap())
                    };
                }
            }
            let parameters = biquad::Parameters::q14(
                coeff[..3].try_into().unwrap(),
                coeff[3..].try_into().unwrap(),
            );
            for c in 0..channels {
                let src = usize::from(p[c]) * samples;
                let dst = usize::from(p[6 + c]) * samples;
                if src != dst {
                    buffers.copy_within(src..src + samples, dst);
                }
                if self.enabled {
                    if pending == 0 || (pending == 1 && self.revision < 6) {
                        self.states[c] = Default::default();
                    }
                    self.states[c].process(
                        parameters,
                        self.revision,
                        &mut buffers[dst..dst + samples],
                    );
                }
            }
        }
        self.was_enabled = Some(self.enabled);
        self.new = false;
        // BiquadFilterInfo::UpdateForCommandGeneration updates the first valid
        // state field; when neither layout has one, it updates both fields.
        // https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/audio_core/renderer/effect/biquad_filter.cpp
        if self.parameters[0x17] <= 2 {
            self.parameters[0x17] = 2;
        } else if self.parameters[0x25] <= 2 {
            self.parameters[0x25] = 2;
        } else {
            self.parameters[0x17] = 2;
            self.parameters[0x25] = 2;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::Effect;

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
}
