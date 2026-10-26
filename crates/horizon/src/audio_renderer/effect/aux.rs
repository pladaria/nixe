//! Guest CPU/DSP auxiliary sample exchange. Rings retain their physical pools.
//! https://github.com/eden-emulator/mirror/blob/master/src/audio_core/renderer/effect/aux_.h
//! https://github.com/eden-emulator/mirror/blob/master/src/audio_core/renderer/command/effect/aux_.cpp
use super::super::{
    buffer::{Attachment, attach},
    update::{MemoryPool, UpdateError, wide, word},
    worker::{RenderError, RenderMemory},
};
use nixe_memory::{AllocationRange, MemoryError};

#[derive(Debug)]
pub(in crate::audio_renderer) struct Aux {
    pub(super) mix: usize,
    pub(super) order: usize,
    inputs: [usize; 24],
    outputs: [usize; 24],
    channels: usize,
    capacity: usize,
    send: Attachment,
    receive: Attachment,
    enabled: bool,
    new: bool,
    was_enabled: Option<bool>,
}
impl Aux {
    pub(super) fn new(
        raw: &[u8],
        pools: &[Option<MemoryPool>],
        force: bool,
        translate: &impl Fn(u64, u64) -> Result<Vec<AllocationRange>, MemoryError>,
    ) -> Result<Self, UpdateError> {
        let p = &raw[0x20..];
        let size = 0x80 + u64::from(word(p, 0x38)) * 4;
        let send =
            attach(wide(p, 0x40), size, pools, force, translate)?.ok_or(UpdateError::Invalid)?;
        let receive =
            attach(wide(p, 0x50), size, pools, force, translate)?.ok_or(UpdateError::Invalid)?;
        Ok(Self {
            mix: 0,
            order: 0,
            inputs: [0; 24],
            outputs: [0; 24],
            channels: 0,
            capacity: 0,
            send,
            receive,
            enabled: false,
            new: true,
            was_enabled: None,
        })
    }
    pub(super) fn uses_pool(&self, pool: usize) -> bool {
        self.enabled && (self.send.pool == Some(pool) || self.receive.pool == Some(pool))
    }
    pub(super) fn update(&mut self, raw: &[u8], started: bool) -> Result<u8, UpdateError> {
        let p = &raw[0x20..];
        let channels = word(p, 0x30) as usize;
        let capacity = word(p, 0x38) as usize;
        if channels > 24
            || 0x80 + capacity as u64 * 4 > self.send.bytes
            || 0x80 + capacity as u64 * 4 > self.receive.bytes
        {
            return Err(UpdateError::Invalid);
        }
        self.mix = word(raw, 4) as usize;
        self.order = word(raw, 24) as usize;
        self.inputs = std::array::from_fn(|i| p[i] as usize);
        self.outputs = std::array::from_fn(|i| p[24 + i] as usize);
        self.channels = channels;
        self.capacity = capacity;
        self.enabled = raw[2] != 0;
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
        memory: &RenderMemory,
        buffers: &mut [i32],
        samples: usize,
        count: usize,
    ) -> Result<(), RenderError> {
        if self.inputs[..self.channels]
            .iter()
            .chain(&self.outputs[..self.channels])
            .any(|i| *i >= count)
        {
            return Err(
                format!("rendering: Aux input/output exceeds its mix's {count} buffers")
                    .into_boxed_str()
                    .into(),
            );
        }
        if self.enabled && self.channels != 0 {
            if self.capacity < samples {
                return Err("rendering: Aux sample block exceeds its ring capacity".into());
            }
            let mut bytes = [0; 4];
            self.send.read(memory, 0x44, &mut bytes)?;
            let write = u32::from_le_bytes(bytes) as usize;
            self.receive.read(memory, 0x40, &mut bytes)?;
            let read = u32::from_le_bytes(bytes) as usize;
            for c in 0..self.channels {
                let src = self.inputs[c] * samples;
                let dst = self.outputs[c] * samples;
                let write_start = write + c * samples;
                let read_start = read + c * samples;
                if write_start > self.capacity || read_start > self.capacity {
                    return Err("rendering: Aux ring cursor exceeds its sample capacity".into());
                }
                let write_start = write_start % self.capacity;
                let read_start = read_start % self.capacity;
                let first = samples.min(self.capacity - write_start);
                self.send.write(
                    memory,
                    0x80 + write_start as u64 * 4,
                    bytemuck::cast_slice(&buffers[src..src + first]),
                )?;
                self.send.write(
                    memory,
                    0x80,
                    bytemuck::cast_slice(&buffers[src + first..src + samples]),
                )?;
                if c + 1 == self.channels {
                    self.send.write(
                        memory,
                        0x44,
                        &(((write + self.channels * samples) % self.capacity) as u32).to_le_bytes(),
                    )?;
                }
                let first = samples.min(self.capacity - read_start);
                self.receive.read(
                    memory,
                    0x80 + read_start as u64 * 4,
                    bytemuck::cast_slice_mut(&mut buffers[dst..dst + first]),
                )?;
                self.receive.read(
                    memory,
                    0x80,
                    bytemuck::cast_slice_mut(&mut buffers[dst + first..dst + samples]),
                )?;
                // Offsets commit only after the final channel, after data writes.
                if c + 1 == self.channels {
                    self.receive.write(
                        memory,
                        0x40,
                        &(((read + self.channels * samples) % self.capacity) as u32).to_le_bytes(),
                    )?;
                }
            }
        } else if self.channels != 0 {
            for attachment in [&self.send, &self.receive] {
                let mut header = [0; 16];
                attachment.read(memory, 0x40, &mut header)?;
                header[..8].fill(0);
                header[12..].fill(0);
                attachment.write(memory, 0x40, &header)?;
            }
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
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::Effect;
    use super::*;

    #[test]
    fn physical_rings_exchange_each_channel_wrap_and_commit_cursors_after_the_data() {
        for (channels, capacity, cursor, samples) in [(2, 12, 9, 3), (1, 10, 8, 4)] {
            let ranges = crate::test_memory::physical(&vec![0; 4096]);
            let pools = [Some(MemoryPool {
                address: 0x1000,
                ranges: nixe_memory::CapturedMemory::new(ranges).unwrap(),
                bytes: 4096,
            })];
            let mut raw = [0; 0xc0];
            raw[0] = 2;
            raw[1] = 1;
            raw[2] = 1;
            let p = &mut raw[0x20..];
            for c in 0..channels {
                p[c] = c as u8;
                p[24 + c] = (channels + c) as u8;
            }
            p[0x30..0x34].copy_from_slice(&(channels as u32).to_le_bytes());
            p[0x38..0x3c].copy_from_slice(&(capacity as u32).to_le_bytes());
            p[0x40..0x48].copy_from_slice(&0x1000_u64.to_le_bytes());
            p[0x50..0x58].copy_from_slice(&0x1800_u64.to_le_bytes());
            let memory = RenderMemory::testing();
            let mut effect = Effect::default();
            assert_eq!(
                effect
                    .update(&raw, 0, 11, 48000, &pools, false, &|_, _| panic!(), true)
                    .unwrap(),
                3
            );
            assert!(effect.uses_pool(0));
            let Effect::Aux(aux) = &effect else { panic!() };
            let send_info = [7, cursor as u32, 123, 99];
            let receive_info = [cursor as u32, 5, 222, 888];
            aux.send
                .write(&memory, 0x40, bytemuck::cast_slice(&send_info))
                .unwrap();
            aux.receive
                .write(&memory, 0x40, bytemuck::cast_slice(&receive_info))
                .unwrap();
            let returns: Vec<i32> = (0..capacity).map(|i| 100 + i as i32).collect();
            aux.receive
                .write(&memory, 0x80, bytemuck::cast_slice(&returns))
                .unwrap();
            let input: Vec<i32> = (0..channels * samples)
                .map(|i| 11 * (i as i32 + 1))
                .collect();
            let mut buffers = input.clone();
            buffers.resize(channels * samples * 2, 0);
            effect
                .process(&memory, &mut buffers, samples, channels * 2)
                .unwrap();
            assert_eq!(&buffers[..input.len()], input);
            for i in 0..input.len() {
                assert_eq!(buffers[input.len() + i], returns[(cursor + i) % capacity]);
            }
            let Effect::Aux(aux) = &effect else { panic!() };
            let mut sent = vec![0_i32; capacity];
            aux.send
                .read(&memory, 0x80, bytemuck::cast_slice_mut(&mut sent))
                .unwrap();
            for i in 0..input.len() {
                assert_eq!(sent[(cursor + i) % capacity], input[i]);
            }
            let mut info = [0_u32; 4];
            aux.send
                .read(&memory, 0x40, bytemuck::cast_slice_mut(&mut info))
                .unwrap();
            let next = ((cursor + input.len()) % capacity) as u32;
            assert_eq!(info, [7, next, 123, 99]);
            aux.receive
                .read(&memory, 0x40, bytemuck::cast_slice_mut(&mut info))
                .unwrap();
            assert_eq!(info, [next, 5, 222, 888]);
            raw[1] = 0;
            raw[2] = 0;
            assert_eq!(
                effect
                    .update(&raw, 0, 11, 48000, &pools, false, &|_, _| panic!(), true)
                    .unwrap(),
                3
            );
            buffers.fill(1234);
            effect
                .process(&memory, &mut buffers, samples, channels * 2)
                .unwrap();
            assert!(buffers.iter().all(|v| *v == 1234));
            let Effect::Aux(aux) = &effect else { panic!() };
            aux.send
                .read(&memory, 0x40, bytemuck::cast_slice_mut(&mut info))
                .unwrap();
            assert_eq!(info, [0, 0, 123, 0]);
            aux.receive
                .read(&memory, 0x40, bytemuck::cast_slice_mut(&mut info))
                .unwrap();
            assert_eq!(info, [0, 0, 222, 0]);
            assert!(!effect.uses_pool(0));
            assert_eq!(
                effect
                    .update(&raw, 0, 11, 48000, &pools, false, &|_, _| panic!(), true)
                    .unwrap(),
                4
            );
        }
    }
}
