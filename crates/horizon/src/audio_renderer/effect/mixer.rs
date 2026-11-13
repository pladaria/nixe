//! Buffer-mixer effects add ordered, gain-scaled input channels to destinations.
//! https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/audio_core/renderer/effect/buffer_mixer.h
//! https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/audio_core/renderer/command/command_generator.cpp
use super::super::{
    render::mix_add,
    update::{UpdateError, word},
};
#[derive(Debug)]
pub(in crate::audio_renderer) struct Mixer {
    pub mix: usize,
    pub order: usize,
    inputs: [u8; 24],
    outputs: [u8; 24],
    volumes: [f32; 24],
    channels: usize,
    precision: u32,
    enabled: bool,
    new: bool,
    was_enabled: Option<bool>,
}
impl Mixer {
    pub fn new(revision: u32) -> Self {
        Self {
            mix: 0,
            order: 0,
            inputs: [0; 24],
            outputs: [0; 24],
            volumes: [0.; 24],
            channels: 0,
            precision: if revision >= 7 { 23 } else { 15 },
            enabled: false,
            new: true,
            was_enabled: None,
        }
    }
    pub fn update(&mut self, raw: &[u8], started: bool) -> Result<u8, UpdateError> {
        self.mix = word(raw, 4) as usize;
        self.order = word(raw, 24) as usize;
        self.enabled = raw[2] != 0;
        let p = &raw[0x20..];
        self.inputs.copy_from_slice(&p[..24]);
        self.outputs.copy_from_slice(&p[24..48]);
        self.volumes = std::array::from_fn(|i| f32::from_bits(word(p, 48 + i * 4)));
        self.channels = word(p, 0x90) as usize;
        Ok(
            if self.new || (started && self.was_enabled != Some(false)) {
                3
            } else {
                4
            },
        )
    }
    pub fn process(
        &mut self,
        buffers: &mut [i32],
        samples: usize,
        count: usize,
    ) -> Result<(), Box<str>> {
        if self.enabled {
            if self.channels > 24 {
                return Err("rendering: buffer mixer exceeds 24 routes".into());
            }
            for route in 0..self.channels {
                let volume = self.volumes[route];
                if volume == 0. {
                    continue;
                }
                let input = usize::from(self.inputs[route]);
                let output = usize::from(self.outputs[route]);
                if input >= count || output >= count {
                    return Err("rendering: buffer mixer route exceeds its mix's buffers".into());
                }
                // The input may alias the destination or a preceding route's
                // output. Consume in order without a scratch buffer or copy.
                mix_add(buffers, input, output, samples, volume, self.precision)?;
            }
        }
        self.new = false;
        self.was_enabled = Some(self.enabled);
        Ok(())
    }
}
