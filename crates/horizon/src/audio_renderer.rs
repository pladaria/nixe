//! Horizon audio-renderer configuration and session lifetime.
//!
//! https://switchbrew.org/wiki/Audio_services#IAudioRenderer
//! Session limits and ownership:
//! https://github.com/eden-emulator/mirror/blob/master/src/audio_core/audio_render_manager.cpp

use crate::audio_device::{AudioDeviceSession, AudioDeviceState};
use nixe_memory::{CanonicalWriteBatch, CanonicalWriteBatchError};
use nixe_runtime::{
    EventObject, ExternalEventSource, ReadableEventObject, TransferMemoryObject,
    WritableEventObject,
};
use std::sync::{Arc, Mutex, Weak};
mod biquad;
mod buffer;
mod effect;
mod render;
mod resample;
pub(crate) mod update;
mod voice;
mod worker;

pub(crate) const PARAMETER_SIZE: usize = 0x34;

pub(crate) fn revision_number(revision: u32) -> Option<u32> {
    let revision = revision.checked_sub(u32::from_le_bytes(*b"REV0"))?;
    (revision & 0x00ff_ffff == 0 && (1..=15).contains(&(revision >> 24))).then_some(revision >> 24)
}

#[derive(Clone, Debug)]
pub(crate) struct RendererParameters {
    pub(crate) sample_rate: u32,
    pub(crate) samples: u32,
    pub(crate) mixes: u32,
    pub(crate) submixes: u32,
    pub(crate) voices: u32,
    pub(crate) sinks: u32,
    pub(crate) effects: u32,
    pub(crate) performance_frames: u32,
    pub(crate) splitters: u32,
    pub(crate) splitter_destinations: u32,
    pub(crate) external_context_size: u32,
    pub(crate) revision: u32,
    pub(crate) rendering_device: u8,
    pub(crate) execution_mode: u8,
}

impl RendererParameters {
    pub(crate) fn decode(data: &[u8; PARAMETER_SIZE]) -> Self {
        let word = |offset: usize| u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
        Self {
            sample_rate: word(0),
            samples: word(4),
            mixes: word(8),
            submixes: word(12),
            voices: word(16),
            sinks: word(20),
            effects: word(24),
            performance_frames: word(28),
            splitters: word(36),
            splitter_destinations: word(40),
            external_context_size: word(44),
            revision: word(48),
            rendering_device: data[34],
            execution_mode: data[35],
        }
    }

    pub(crate) fn revision_number(&self) -> Option<u32> {
        revision_number(self.revision)
    }

    // REV1–REV7 negotiated work-memory layout, including revision-dependent
    // routing, performance records and command storage. Keep calculations wide:
    // untrusted counts must never wrap into a small guest allocation.
    // Adapted from Yuzu (GPL-2.0-or-later), Copyright 2018 yuzu emulator team:
    // https://github.com/eden-emulator/mirror/blob/d27abf5546bc392d498dedf87daed57e96268b0f/src/core/hle/service/audio/audren_u.cpp#L350
    pub(crate) fn work_buffer_size(&self, revision: u32) -> Option<u64> {
        let align = |value: u128, alignment: u128| value.div_ceil(alignment) * alignment;
        let samples = u128::from(self.samples);
        let mixes = u128::from(self.mixes);
        let submixes = u128::from(self.submixes);
        let voices = u128::from(self.voices);
        let sinks = u128::from(self.sinks);
        let effects = u128::from(self.effects);
        let splitters = u128::from(self.splitters);
        let destinations = u128::from(self.splitter_destinations);
        let nodes = submixes + 1;
        let entries = effects + submixes + sinks + voices + 1;

        let mut size = align((mixes + 6) * 4 * (samples + 240) + submixes + sinks, 64);
        size += align(u128::from(self.external_context_size), 64) + align(4 * mixes, 64);
        size += align(8 * nodes, 16) + align(0x940 * nodes, 16) + align(1024 * submixes, 16);
        if revision >= 2 {
            let node_states = 4 * nodes * nodes + 12 * nodes + 2 * (align(nodes, 64) / 8);
            size += align(node_states + 4 * nodes * nodes, 16);
        }
        size += align(8 * voices, 16) + align(0x220 * voices, 16);
        size += align(0xd0 * voices, 16) + align(0x100 * voices, 16);
        size += 0x48 + align(4 * (effects + voices) * 0x20, 16);
        if revision >= 2 {
            size += destinations + align(0x20 * splitters, 16) + align(0xe0 * destinations, 16);
        }
        size = align(size, 64);
        size += align(0x280 * (submixes + sinks), 64);
        size += align(0x2b0 * effects, 16) + align(0x170 * sinks, 16);
        size += align(0x100 * voices + 63, 16);
        let (performance_header, performance_entry) = if revision >= 5 {
            (0x30, 0x18)
        } else {
            (0x18, 0x10)
        };
        let performance_frame = performance_header + performance_entry * (entries + 100);
        size += align(
            performance_frame * u128::from(self.performance_frames) + 1,
            64,
        ) + 64
            + 128;
        size += if revision >= 5 {
            effects * 0x540
                + 0x24
                + 0x1c * 24
                + 0x28 * (entries + 100)
                + sinks * 0x130
                + destinations * 24 * 0x24
                + submixes * (0x24 + 0x1c * 24 * 24)
                + voices * (splitters * 0x50 + 0x9c + 2 * 0x2c + 0x20 + 0x13c)
                + 126
        } else {
            0x18000 + 126
        };
        u64::try_from(align(size, 4096)).ok()
    }
}

/// Renderer slots belong to the service, not to a particular manager connection.
/// Weak reservations release a slot only after the last session clone closes.
#[derive(Clone, Debug, Default)]
pub struct AudioRendererManagerSession {
    slots: Arc<Mutex<[Weak<AudioRendererState>; 2]>>,
    pub(crate) devices: Arc<AudioDeviceState>,
}

#[derive(Clone, Debug)]
pub struct AudioRendererSession(pub(crate) Arc<AudioRendererState>);

#[derive(Debug)]
pub(crate) struct AudioRendererState {
    pub(crate) parameters: RendererParameters,
    pub(crate) event: ReadableEventObject,
    signal: WritableEventObject,
    pub(crate) rendering_time_limit: Mutex<u32>,
    pub(crate) update: Arc<Mutex<update::RendererUpdateState>>,
    worker: Mutex<Option<worker::RendererWorker>>,
    devices: Arc<AudioDeviceState>,
    // Closing the client's copied handle must not free the renderer's work memory.
    _work_memory: TransferMemoryObject,
    _applet_resource_user_id: u64,
    _process_id: u64,
}

impl AudioRendererManagerSession {
    pub(crate) fn set_host_paused(&self, paused: bool) {
        for slot in self.slots.lock().unwrap().iter() {
            if let Some(state) = slot.upgrade() {
                state.set_host_paused(paused);
            }
        }
    }
    pub(crate) fn device(
        &self,
        revision: u32,
        applet_resource_user_id: u64,
        backend: Arc<dyn nixe_audio::AudioBackend>,
    ) -> Result<AudioDeviceSession, nixe_audio::AudioError> {
        AudioDeviceSession::new(
            self.devices.clone(),
            revision,
            applet_resource_user_id,
            backend,
        )
    }

    pub(crate) fn open(
        &self,
        parameters: RendererParameters,
        work_memory: TransferMemoryObject,
        work_size: u64,
        applet_resource_user_id: u64,
        process_id: u64,
    ) -> Result<Option<AudioRendererSession>, CanonicalWriteBatchError> {
        let mut slots = self.slots.lock().unwrap();
        let Some(slot) = slots.iter_mut().find(|slot| slot.strong_count() == 0) else {
            return Ok(None);
        };
        // Initialize the transferred backing itself, independent of the owner's
        // virtual alias permissions or later closure of its transfer handle.
        // https://github.com/eden-emulator/mirror/blob/master/src/audio_core/renderer/system.cpp
        let zeroes = [0; 4096];
        let mut writes = CanonicalWriteBatch::new();
        let mut offset = 0;
        while offset < work_size {
            let count = (work_size - offset).min(zeroes.len() as u64) as usize;
            writes.stage(work_memory.backing(), offset, &zeroes[..count])?;
            offset += count as u64;
        }
        writes.commit()?;
        let (signal, event) = EventObject::create_pair_with_source(ExternalEventSource::Device);
        let state = Arc::new(AudioRendererState {
            parameters,
            event,
            signal,
            rendering_time_limit: Mutex::new(100),
            update: Arc::new(Mutex::new(update::RendererUpdateState::default())),
            worker: Mutex::new(None),
            devices: self.devices.clone(),
            _work_memory: work_memory,
            _applet_resource_user_id: applet_resource_user_id,
            _process_id: process_id,
        });
        *slot = Arc::downgrade(&state);
        Ok(Some(AudioRendererSession(state)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nixe_memory::{CanonicalAllocation, GuestVirtualAddress, MemoryPermissions};

    fn parameters(revision: u8) -> RendererParameters {
        let mut data = [0; PARAMETER_SIZE];
        for (offset, value) in [
            (0, 48000_u32),
            (4, 240),
            (8, 30),
            (12, 1),
            (16, 96),
            (20, 1),
            (24, 14),
        ] {
            data[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
        data[48..].copy_from_slice(&[b'R', b'E', b'V', b'0' + revision]);
        RendererParameters::decode(&data)
    }

    #[test]
    fn negotiated_sizes_match_independently_executed_yuzu_reference() {
        // Golden results from compiling the unmodified sizing lambdas at the
        // pinned reference above, including a splitter/performance configuration.
        for (revision, expected) in [
            (1, 0x4f000),
            (2, 0x4f000),
            (4, 0x4f000),
            (5, 0x50000),
            (7, 0x50000),
        ] {
            let params = parameters(revision);
            assert_eq!(params.revision_number(), Some(u32::from(revision)));
            assert_eq!(params.work_buffer_size(u32::from(revision)), Some(expected));
        }
        let mut params = parameters(7);
        params.samples = 160;
        params.mixes = 24;
        params.submixes = 3;
        params.voices = 12;
        params.sinks = 2;
        params.effects = 8;
        params.external_context_size = 0x123;
        params.splitters = 4;
        params.splitter_destinations = 12;
        params.performance_frames = 3;
        assert_eq!(params.work_buffer_size(7), Some(0x2e000));
    }

    #[test]
    fn malformed_revisions_and_unrepresentable_sizes_are_rejected() {
        let mut params = parameters(7);
        for revision in [
            0,
            u32::from_le_bytes(*b"REV0"),
            u32::from_le_bytes(*b"rev7"),
            u32::from_le_bytes(*b"REV7") + 1,
        ] {
            params.revision = revision;
            assert_eq!(params.revision_number(), None);
        }
        params.voices = u32::MAX;
        params.splitters = u32::MAX;
        assert_eq!(params.work_buffer_size(7), None);
    }

    #[test]
    fn device_gains_survive_connections_and_reach_existing_and_future_renderers() {
        #[derive(Debug)]
        struct Backend;
        impl nixe_audio::AudioBackend for Backend {
            fn output_layout(&self) -> Result<nixe_audio::ChannelLayout, nixe_audio::AudioError> {
                Ok(nixe_audio::ChannelLayout::Stereo)
            }
            fn open(
                &self,
                _: nixe_audio::AudioFeed,
            ) -> Result<Box<dyn nixe_audio::AudioDevice>, nixe_audio::AudioError> {
                panic!("metadata does not start playback")
            }
        }
        let manager = AudioRendererManagerSession::default();
        let connection = manager.clone();
        let device = connection.device(7, 1, Arc::new(Backend)).unwrap();
        let memory = CanonicalAllocation::zeroed(0x50000, 4096).unwrap();
        let transfer = TransferMemoryObject::new(
            GuestVirtualAddress::new(0x1000),
            0x50000,
            MemoryPermissions::NONE,
            memory.backing_range(MemoryPermissions::READ_WRITE).unwrap(),
        );
        device
            .state
            .set_volume(crate::audio_device::ACTIVE_DEVICE, 0.5);
        let first = manager
            .open(parameters(7), transfer.clone(), 0x50000, 1, 1)
            .unwrap()
            .unwrap();
        assert_eq!(
            first.0.devices.volume(crate::audio_device::ACTIVE_DEVICE),
            0.5
        );
        device
            .state
            .set_volume(crate::audio_device::ACTIVE_DEVICE, 0.25);
        assert_eq!(
            first.0.devices.volume(crate::audio_device::ACTIVE_DEVICE),
            0.25
        );
        drop(device);
        drop(connection);
        let second = manager
            .open(parameters(7), transfer, 0x50000, 1, 1)
            .unwrap()
            .unwrap();
        assert_eq!(
            second.0.devices.volume(crate::audio_device::ACTIVE_DEVICE),
            0.25
        );
        drop(manager);
        assert_eq!(
            first.0.devices.volume(crate::audio_device::ACTIVE_DEVICE),
            0.25
        );
    }

    #[test]
    fn renderer_initializes_retained_memory_and_releases_slots_after_the_last_clone() {
        let memory = CanonicalAllocation::zeroed(0x51000, 4096).unwrap();
        memory.write(0, &vec![0xcc; 0x51000]).unwrap();
        let transfer = TransferMemoryObject::new(
            GuestVirtualAddress::new(0x1000),
            0x51000,
            MemoryPermissions::NONE,
            memory.backing_range(MemoryPermissions::READ_WRITE).unwrap(),
        );
        let manager = AudioRendererManagerSession::default();
        let another_manager_connection = manager.clone();
        let open = |manager: &AudioRendererManagerSession| {
            manager
                .open(parameters(7), transfer.clone(), 0x50000, 1, 1)
                .unwrap()
        };
        let first = open(&manager).unwrap();
        let second = open(&another_manager_connection).unwrap();
        assert!(open(&manager).is_none());
        let mut bytes = vec![0; 0x51000];
        memory.read(0, &mut bytes).unwrap();
        assert!(bytes[..0x50000].iter().all(|&byte| byte == 0));
        assert!(bytes[0x50000..].iter().all(|&byte| byte == 0xcc));
        assert!(!first.0.event.is_signalled());
        let clone = first.clone();
        drop(first);
        drop(manager);
        assert!(open(&another_manager_connection).is_none());
        drop(clone);
        assert!(open(&another_manager_connection).is_some());
        drop(second);
    }
}
