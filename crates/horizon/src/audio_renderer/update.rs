//! REV1–REV7 renderer updates. Rendering is a separate operation.
//!
//! Wire layouts, section order and status transitions:
//! https://github.com/eden-emulator/mirror/blob/master/src/audio_core/renderer/behavior/info_updater.cpp
//! https://github.com/eden-emulator/mirror/blob/master/src/audio_core/renderer/memory/pool_mapper.cpp

use super::RendererParameters;
use nixe_memory::{
    CanonicalBackingRange, CanonicalRangeTranslationError, CanonicalRangeTranslationErrorReason,
};
use std::ops::Range;

pub(crate) const HEADER_SIZE: usize = 0x40;
const MIX_SIZE: usize = 0x930;
const UNUSED_MIX: u32 = i32::MAX as u32;

pub(super) fn word(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap())
}

pub(super) fn wide(data: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap())
}

fn put_word(data: &mut [u8], offset: usize, value: u32) {
    data[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

#[derive(Debug)]
pub(crate) enum UpdateError {
    Invalid,
    Unsupported(Box<str>),
    Mapping(CanonicalRangeTranslationError),
}

#[derive(Debug)]
pub(crate) struct UpdateLayout {
    behavior: Range<usize>,
    pools: Range<usize>,
    resources: Range<usize>,
    voices: Range<usize>,
    effects: Range<usize>,
    splitters: Range<usize>,
    mixes: Range<usize>,
    sinks: Range<usize>,
    performance: Range<usize>,
    pub(crate) output_size: usize,
}

impl UpdateLayout {
    pub(crate) fn parse(
        params: &RendererParameters,
        header: &[u8; HEADER_SIZE],
        input_size: usize,
    ) -> Result<Self, UpdateError> {
        let voices = params.voices as usize;
        let effects = params.effects as usize;
        let pool_count = voices * 4 + effects;
        for (offset, expected) in [
            (4, 0x10),
            (8, pool_count * 0x20),
            (12, voices * 0x170),
            (16, voices * 0x70),
            (20, effects * 0xc0),
            (28, params.sinks as usize * 0x140),
            (32, 0x10),
        ] {
            if word(header, offset) as usize != expected {
                return Err(UpdateError::Invalid);
            }
        }
        let mix_size = word(header, 24) as usize;
        let revision = params.revision_number().unwrap();
        if params.splitters != 0 || params.splitter_destinations != 0 || word(header, 36) > 0x20 {
            return Err(UpdateError::Unsupported(
                "RequestUpdate: splitter updates are not implemented".into(),
            ));
        }
        if !matches!(word(header, 36), 0 | 0x20) || (revision < 2 && word(header, 36) != 0) {
            return Err(UpdateError::Invalid);
        }
        let mix_prefix = if revision >= 7 { 0x20 } else { 0 };
        let mix_limit = (params.submixes as usize + 1) * MIX_SIZE + mix_prefix;
        if mix_size < mix_prefix
            || mix_size > mix_limit
            || !(mix_size - mix_prefix).is_multiple_of(MIX_SIZE)
            || (revision < 7 && mix_size != mix_limit)
        {
            return Err(UpdateError::Invalid);
        }
        let mut cursor = HEADER_SIZE;
        let mut section = |offset| -> Result<Range<usize>, UpdateError> {
            let size = word(header, offset) as usize;
            let end = cursor
                .checked_add(size)
                .filter(|&end| end <= input_size)
                .ok_or(UpdateError::Invalid)?;
            let range = cursor..end;
            cursor = end;
            Ok(range)
        };
        // Voice resources precede voices on the wire, despite header field order.
        let behavior = section(4)?;
        let pools = section(8)?;
        let resources = section(16)?;
        let voices = section(12)?;
        let effects = section(20)?;
        let splitters = section(36)?;
        let mixes = section(24)?;
        let sinks = section(28)?;
        let performance = section(32)?;
        if cursor != input_size {
            return Err(UpdateError::Invalid);
        }
        let output_size = HEADER_SIZE
            + pool_count * 0x10
            + params.voices as usize * 0x10
            + params.effects as usize * 0x10
            + params.sinks as usize * 0x20
            + 0x10
            + 0xb0
            + if revision >= 5 { 0x10 } else { 0 };
        Ok(Self {
            behavior,
            pools,
            resources,
            voices,
            effects,
            splitters,
            mixes,
            sinks,
            performance,
            output_size,
        })
    }
}

#[derive(Debug)]
pub(super) struct MemoryPool {
    pub(super) address: u64,
    pub(super) backing: CanonicalBackingRange,
}

#[derive(Debug, Default)]
pub(crate) struct RendererUpdateState {
    pools: Vec<Option<MemoryPool>>,
    pub(super) resources: Vec<[u8; 0x70]>,
    pub(super) previous_resources: Vec<[f32; 24]>,
    pub(super) voices: Vec<super::voice::Voice>,
    pub(super) mixes: Vec<Option<Box<[u8; MIX_SIZE]>>>,
    pub(super) sinks: Vec<[u8; 0x140]>,
    pub(super) effects: Vec<super::effect::Effect>,
    pub(super) effect_order: Vec<Vec<usize>>,
    pub(crate) started: bool,
    pub(super) frames: u64,
    pub(crate) failure: Option<Box<str>>,
    performance_target: u32,
    force_mapping: bool,
    updates: u64,
}

impl RendererUpdateState {
    pub(crate) fn update(
        &mut self,
        params: &RendererParameters,
        layout: &UpdateLayout,
        input: &[u8],
        translate: impl Fn(u64, u64) -> Result<CanonicalBackingRange, CanonicalRangeTranslationError>,
    ) -> Result<Vec<u8>, UpdateError> {
        let revision = params.revision_number().unwrap();
        let behavior = &input[layout.behavior.clone()];
        if word(behavior, 0) != params.revision {
            return Err(UpdateError::Invalid);
        }
        // This policy is consumed by voice/effect buffer attachment, separately
        // from the pool attachment requests processed below.
        self.force_mapping = wide(behavior, 8) & 1 != 0;
        if !layout.splitters.is_empty() {
            let splitters = &input[layout.splitters.clone()];
            if word(splitters, 4) != 0 || word(splitters, 8) != 0 {
                return Err(UpdateError::Invalid);
            }
        }
        let mut output = vec![0; layout.output_size];
        // Advertise the server's supported ABI revision, not a newer firmware ABI.
        put_word(&mut output, 0, u32::from_le_bytes(*b"REV7"));
        put_word(&mut output, 60, layout.output_size as u32);
        let pool_count = params.voices as usize * 4 + params.effects as usize;
        self.pools.resize_with(pool_count, || None);
        let mut cursor = HEADER_SIZE;
        for (i, data) in input[layout.pools.clone()].chunks_exact(0x20).enumerate() {
            let address = wide(data, 0);
            let size = wide(data, 8);
            let requested = word(data, 16);
            if matches!(requested, 2 | 4)
                && address != 0
                && size != 0
                && address.is_multiple_of(4096)
                && size.is_multiple_of(4096)
            {
                let status = match requested {
                    4 => match translate(address, size) {
                        Ok(backing) => {
                            self.pools[i] = Some(MemoryPool { address, backing });
                            5 // Attached: actual retained canonical RAM, no identity-map stub.
                        }
                        Err(error)
                            if matches!(
                                error.reason,
                                CanonicalRangeTranslationErrorReason::Unmapped
                                    | CanonicalRangeTranslationErrorReason::AddressOverflow
                                    | CanonicalRangeTranslationErrorReason::PermissionDenied
                                    | CanonicalRangeTranslationErrorReason::DeviceMemory
                            ) =>
                        {
                            self.pools[i] = None;
                            0 // Failed attachment leaves the output state invalid.
                        }
                        Err(error) => return Err(UpdateError::Mapping(error)),
                    },
                    2 if !self.voices.iter().any(|voice| voice.uses_pool(i))
                        && !self.effects.iter().any(|effect| effect.uses_pool(i))
                        && self.pools[i].as_ref().is_some_and(|pool| {
                            pool.address == address && pool.backing.size() == size
                        }) =>
                    {
                        self.pools[i] = None;
                        3 // Detached.
                    }
                    _ => 0,
                };
                put_word(&mut output, cursor, status);
            }
            cursor += 0x10;
        }
        put_word(&mut output, 8, (pool_count * 0x10) as u32);

        self.resources.resize(params.voices as usize, [0; 0x70]);
        self.previous_resources
            .resize(params.voices as usize, [0.0; 24]);
        self.voices
            .resize_with(params.voices as usize, super::voice::Voice::default);
        for (stored, data) in self
            .resources
            .iter_mut()
            .zip(input[layout.resources.clone()].chunks_exact(0x70))
        {
            stored[0x64] = data[0x64];
            if data[0x64] != 0 {
                stored[4..0x64].copy_from_slice(&data[4..0x64]);
            }
        }
        for voice in &mut self.voices {
            voice.deactivate();
        }
        for voice in input[layout.voices.clone()].chunks_exact(0x170) {
            if voice[9] != 0 {
                let id = word(voice, 0) as usize;
                let Some(state) = self.voices.get_mut(id) else {
                    return Err(UpdateError::Invalid);
                };
                state.update(
                    voice,
                    id,
                    self.resources.len(),
                    revision,
                    &self.pools,
                    self.force_mapping,
                    &translate,
                    &mut output[cursor..cursor + 0x10],
                )?;
                if voice[8] != 0 {
                    for channel in 0..word(voice, 24) as usize {
                        let resource = word(voice, 0x140 + channel * 4) as usize;
                        for (i, value) in self.previous_resources[resource].iter_mut().enumerate() {
                            *value = f32::from_bits(word(&self.resources[resource], 4 + i * 4));
                        }
                    }
                }
            }
            cursor += 0x10;
        }
        put_word(&mut output, 12, params.voices * 0x10);
        self.effects
            .resize_with(params.effects as usize, super::effect::Effect::default);
        for (i, raw) in input[layout.effects.clone()].chunks_exact(0xc0).enumerate() {
            output[cursor] = self.effects[i].update(
                raw,
                i,
                revision,
                params.sample_rate,
                &self.pools,
                self.force_mapping,
                &translate,
                self.started,
            )?;
            cursor += 0x10;
        }
        put_word(&mut output, 20, params.effects * 0x10);

        if params.mixes == 0 {
            return Err(UpdateError::Invalid);
        }
        self.mixes
            .resize_with(params.submixes as usize + 1, || None);
        let mixes = &input[layout.mixes.clone()];
        let mix_data = if revision >= 7 {
            if word(mixes, 4) as usize != (mixes.len() - 0x20) / MIX_SIZE {
                return Err(UpdateError::Invalid);
            }
            &mixes[0x20..]
        } else {
            mixes
        };
        for (index, data) in mix_data.chunks_exact(MIX_SIZE).enumerate() {
            let id = if revision >= 7 {
                word(data, 16) as usize
            } else {
                index
            };
            let Some(stored) = self.mixes.get_mut(id) else {
                return Err(UpdateError::Invalid);
            };
            if data[12] == 0 {
                *stored = None;
            } else {
                if word(data, 8) > 24 || word(data, 20) > params.effects {
                    return Err(UpdateError::Invalid);
                }
                if word(data, 0x928) != u32::MAX {
                    return Err(UpdateError::Unsupported(
                        "RequestUpdate: mix routing through splitters is not implemented".into(),
                    ));
                }
                // Keep dirty mix updates in the same slots across calls.
                *stored = Some(Box::new(data.try_into().unwrap()));
            }
        }
        let buffers: u64 = self
            .mixes
            .iter()
            .flatten()
            .map(|mix| u64::from(word(mix.as_slice(), 8)))
            .sum();
        if buffers > u64::from(params.mixes) {
            return Err(UpdateError::Invalid);
        }
        for (id, mix) in self
            .mixes
            .iter()
            .enumerate()
            .filter_map(|(id, mix)| mix.as_ref().map(|mix| (id, mix)))
        {
            if id == 0 {
                continue;
            }
            let mut next = word(mix.as_slice(), 0x924);
            let mut hops = 0;
            while next != UNUSED_MIX && next != 0 {
                let Some(target) = self.mixes.get(next as usize) else {
                    return Err(UpdateError::Invalid);
                };
                let Some(target) = target else {
                    break;
                };
                hops += 1;
                if hops >= self.mixes.len() {
                    return Err(UpdateError::Invalid);
                }
                next = word(target.as_slice(), 0x924);
            }
        }

        // Effect process_order is an index, not a sortable priority: command
        // generation ends at the first empty slot in each mix's order table.
        // https://github.com/eden-emulator/mirror/blob/master/src/audio_core/renderer/mix/mix_info.cpp
        self.effect_order.resize_with(self.mixes.len(), Vec::new);
        for order in &mut self.effect_order {
            order.clear();
        }
        for (slot, effect) in self.effects.iter().enumerate() {
            let Some((mix, index)) = effect.routing() else {
                continue;
            };
            if mix == UNUSED_MIX as usize {
                continue;
            }
            let Some(order) = self.effect_order.get_mut(mix) else {
                return Err(UpdateError::Invalid);
            };
            if index >= self.effects.len() {
                return Err(UpdateError::Invalid);
            }
            if order.len() <= index {
                order.resize(index + 1, usize::MAX);
            }
            if order[index] != usize::MAX {
                return Err(UpdateError::Invalid);
            }
            order[index] = slot;
        }

        self.sinks.resize(params.sinks as usize, [0; 0x140]);
        for (i, data) in input[layout.sinks.clone()].chunks_exact(0x140).enumerate() {
            match data[0] {
                0 => self.sinks[i].fill(0),
                1 => {
                    if data[1] != 0 {
                        let count = word(data, 0x120) as usize;
                        if count > 6 || data[0x124..0x124 + count].iter().any(|&channel| u32::from(channel) >= params.mixes) {
                            return Err(UpdateError::Invalid);
                        }
                    }
                    // Device input routing updates on an in-use transition;
                    // downmix parameters can change without that transition.
                    // https://github.com/eden-emulator/mirror/blob/master/src/audio_core/renderer/sink/device_sink_info.cpp
                    let stored = &mut self.sinks[i];
                    if stored[0] != 1 || stored[1] != data[1] { stored.copy_from_slice(data); }
                    else { stored[0x12b..0x13c].copy_from_slice(&data[0x12b..0x13c]); }
                }
                kind => return Err(UpdateError::Unsupported(format!("RequestUpdate: sink slot {i} uses type {kind}; circular-buffer sink updates are not implemented").into())),
            }
            cursor += 0x20; // Device sinks do not report a circular-buffer write position.
        }
        put_word(&mut output, 28, params.sinks * 0x20);
        self.performance_target = word(&input[layout.performance.clone()], 0);
        put_word(&mut output, 32, 0x10);
        if self.started && params.performance_frames != 0 {
            return Err(UpdateError::Unsupported(
                "RequestUpdate: renderer performance histories are not implemented".into(),
            ));
        }
        cursor += 0x10; // No performance history storage was requested.
        put_word(&mut output, 4, 0xb0);
        cursor += 0xb0; // Attachments were resolved to retained canonical memory.
        if revision >= 5 {
            put_word(&mut output, 40, 0x10);
            output[cursor..cursor + 8].copy_from_slice(&self.frames.to_le_bytes());
            cursor += 0x10;
        }
        debug_assert_eq!(cursor, output.len());
        self.updates += 1;
        if self.updates == 1 {
            log::debug!(
                "audio renderer update accepted: pools={} resources={} dirty-mix-bytes={} sinks={} performance-target={:#x} force-mapping={}",
                pool_count,
                params.voices,
                mix_data.len(),
                params.sinks,
                self.performance_target,
                self.force_mapping
            );
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nixe_memory::{CanonicalAllocation, MemoryPermissions};

    fn params() -> RendererParameters {
        let mut data = [0; super::super::PARAMETER_SIZE];
        for (offset, value) in [
            (0, 48000),
            (4, 240),
            (8, 4),
            (12, 1),
            (16, 1),
            (20, 1),
            (24, 1),
            (48, u32::from_le_bytes(*b"REV7")),
        ] {
            put_word(&mut data, offset, value);
        }
        RendererParameters::decode(&data)
    }

    // Independent packet construction from the reference ABI's named sections;
    // deliberately puts resources before voices and includes dirty mix framing.
    fn packet(dirty_mix: bool) -> Vec<u8> {
        let mix_size = 0x20 + if dirty_mix { MIX_SIZE } else { 0 };
        let size = 0x40 + 0x10 + 5 * 0x20 + 0x70 + 0x170 + 0xc0 + mix_size + 0x140 + 0x10;
        let mut data = vec![0; size];
        for (offset, value) in [
            (0, u32::from_le_bytes(*b"REV7")),
            (4, 0x10),
            (8, 5 * 0x20),
            (12, 0x170),
            (16, 0x70),
            (20, 0xc0),
            (24, mix_size as u32),
            (28, 0x140),
            (32, 0x10),
            (60, size as u32),
            (0x40, u32::from_le_bytes(*b"REV7")),
        ] {
            put_word(&mut data, offset, value);
        }
        let mix = 0x40 + 0x10 + 5 * 0x20 + 0x70 + 0x170 + 0xc0;
        put_word(&mut data, mix + 4, u32::from(dirty_mix));
        if dirty_mix {
            let start = mix + 0x20;
            put_word(&mut data, start, 1_f32.to_bits());
            put_word(&mut data, start + 4, 48000);
            put_word(&mut data, start + 8, 2);
            data[start + 12] = 1;
            put_word(&mut data, start + 0x924, UNUSED_MIX);
            put_word(&mut data, start + 0x928, u32::MAX);
        }
        let sink = mix + mix_size;
        data[sink] = 1;
        data[sink + 1] = 1;
        data[sink + 0x20..sink + 0x2c].copy_from_slice(b"MainAudioOut");
        put_word(&mut data, sink + 0x120, 2);
        data[sink + 0x125] = 1;
        data
    }

    fn layout(data: &[u8]) -> Result<UpdateLayout, UpdateError> {
        UpdateLayout::parse(
            &params(),
            data[..HEADER_SIZE].try_into().unwrap(),
            data.len(),
        )
    }

    #[test]
    fn stopped_update_reports_real_inactive_status_and_preserves_dirty_mixes() {
        let mut state = RendererUpdateState::default();
        let first = packet(true);
        let output = state
            .update(&params(), &layout(&first).unwrap(), &first, |_, _| {
                panic!("no pool attachment requested")
            })
            .unwrap();
        assert_eq!(output.len(), 0x1a0);
        for (offset, expected) in [
            (0, u32::from_le_bytes(*b"REV7")),
            (4, 0xb0),
            (8, 0x50),
            (12, 0x10),
            (20, 0x10),
            (28, 0x20),
            (32, 0x10),
            (40, 0x10),
            (60, 0x1a0),
        ] {
            assert_eq!(word(&output, offset), expected);
        }
        assert!(output[0x90..0xa0].iter().all(|&byte| byte == 0)); // Unused voice.
        assert_eq!(output[0xa0], 4); // Invalid effect reports Removed, not Used.
        assert_eq!(wide(&output, 0x190), 0); // Actual rendered frames.
        let mut second = packet(false);
        second[0x48] = 1;
        state
            .update(&params(), &layout(&second).unwrap(), &second, |_, _| {
                panic!("no attachment")
            })
            .unwrap();
        assert_eq!(word(state.mixes[0].as_ref().unwrap().as_slice(), 8), 2);
        assert_eq!(state.sinks[0][0x125], 1);
        assert_eq!(state.updates, 2);
        assert!(state.force_mapping);
    }

    #[test]
    fn active_voice_update_renders_attached_pcm_and_reports_consumption_on_the_wire() {
        let params = params();
        let mut data = packet(true);
        let layout = layout(&data).unwrap();
        let memory = CanonicalAllocation::zeroed(4096, 4096).unwrap();
        memory
            .write(
                0,
                &[100_i16, -200, 300, -400]
                    .into_iter()
                    .flat_map(i16::to_le_bytes)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        data[0x50..0x58].copy_from_slice(&0x1000_u64.to_le_bytes());
        data[0x58..0x60].copy_from_slice(&4096_u64.to_le_bytes());
        put_word(&mut data, 0x60, 4);
        let resource = layout.resources.start;
        put_word(&mut data, resource + 4, 1_f32.to_bits());
        put_word(&mut data, resource + 8, 1_f32.to_bits());
        data[resource + 0x64] = 1;
        let voice = &mut data[layout.voices.clone()];
        voice[8] = 1;
        voice[9] = 1;
        voice[11] = 2;
        for (offset, value) in [
            (12, 48000),
            (24, 1),
            (0x1c, 1_f32.to_bits()),
            (0x20, 1_f32.to_bits()),
            (0x3c, 1),
            (0x5c, u32::MAX),
            (0x74, 4),
        ] {
            put_word(voice, offset, value);
        }
        voice[0x60..0x68].copy_from_slice(&0x1000_u64.to_le_bytes());
        voice[0x68..0x70].copy_from_slice(&8_u64.to_le_bytes());
        voice[0x15c] = 2;
        for i in 1..4 {
            voice[0x7a + i * 0x38] = 1;
        }
        let mut state = RendererUpdateState::default();
        state
            .update(&params, &layout, &data, |address, size| {
                assert_eq!((address, size), (0x1000, 4096));
                Ok(memory.backing_range(MemoryPermissions::READ_WRITE).unwrap())
            })
            .unwrap();
        let mut pcm = vec![0; 480];
        let mut renderer = super::super::render::RenderBuffers::new(&params);
        assert!(renderer.render(&mut state, &params, &mut pcm, 1.0).unwrap());
        assert_eq!(&pcm[..8], &[100, 100, -200, -200, 300, 300, -400, -400]);
        assert!(pcm[8..].iter().all(|&sample| sample == 0));
        put_word(&mut data, 0x60, 0);
        data[layout.voices.start + 8] = 0;
        data[layout.voices.start + 0x7a] = 1;
        let output = state
            .update(&params, &layout, &data, |_, _| {
                panic!("attached buffers must not be remapped")
            })
            .unwrap();
        assert_eq!(wide(&output, 0x90), 4);
        assert_eq!(word(&output, 0x98), 1);
        renderer.render(&mut state, &params, &mut pcm, 1.0).unwrap();
        assert!(pcm.iter().all(|&sample| sample == 0));
    }

    #[test]
    fn pool_attachment_retains_real_bytes_and_detach_checks_identity() {
        let memory = CanonicalAllocation::zeroed(4096, 4096).unwrap();
        memory.write(0, &[0xab]).unwrap();
        let backing = memory.backing_range(MemoryPermissions::READ_WRITE).unwrap();
        let mut state = RendererUpdateState::default();
        let mut data = packet(true);
        data[0x50..0x58].copy_from_slice(&0x1000_u64.to_le_bytes());
        data[0x58..0x60].copy_from_slice(&4096_u64.to_le_bytes());
        put_word(&mut data, 0x60, 4);
        let output = state
            .update(
                &params(),
                &layout(&data).unwrap(),
                &data,
                |address, size| {
                    assert_eq!((address, size), (0x1000, 4096));
                    Ok(backing.clone())
                },
            )
            .unwrap();
        assert_eq!(word(&output, 0x40), 5);
        drop(backing);
        drop(memory);
        let mut byte = [0];
        state.pools[0]
            .as_ref()
            .unwrap()
            .backing
            .read(0, &mut byte)
            .unwrap();
        assert_eq!(byte, [0xab]);
        put_word(&mut data, 0x60, 2);
        data[0x50..0x58].copy_from_slice(&0x2000_u64.to_le_bytes());
        let output = state
            .update(&params(), &layout(&data).unwrap(), &data, |_, _| {
                panic!("detach must not remap")
            })
            .unwrap();
        assert_eq!(word(&output, 0x40), 0);
        assert!(state.pools[0].is_some());
        data[0x50..0x58].copy_from_slice(&0x1000_u64.to_le_bytes());
        let output = state
            .update(&params(), &layout(&data).unwrap(), &data, |_, _| {
                panic!("detach must not remap")
            })
            .unwrap();
        assert_eq!(word(&output, 0x40), 3);
        assert!(state.pools[0].is_none());
    }

    #[test]
    fn malformed_sections_and_active_dsp_features_remain_explicit() {
        let mut data = packet(true);
        put_word(&mut data, 16, 0x60);
        assert!(matches!(layout(&data), Err(UpdateError::Invalid)));
        let mut data = packet(true);
        data.pop();
        assert!(matches!(layout(&data), Err(UpdateError::Invalid)));
        let mut data = packet(true);
        let voice = 0x40 + 0x10 + 5 * 0x20 + 0x70;
        data[voice + 0x170] = 3;
        assert!(
            matches!(RendererUpdateState::default().update(&params(), &layout(&data).unwrap(), &data, |_, _| panic!()), Err(UpdateError::Unsupported(detail)) if detail.contains("Delay"))
        );
    }
}
