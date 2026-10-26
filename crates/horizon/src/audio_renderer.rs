//! Horizon audio-renderer configuration and session lifetime.
//!
//! https://switchbrew.org/wiki/Audio_services#IAudioRenderer
//! Session limits and ownership:
//! https://github.com/eden-emulator/mirror/blob/master/src/audio_core/audio_render_manager.cpp

use crate::audio_device::{AudioDeviceSession, AudioDeviceState};
use nixe_runtime::{
    EventObject, ExternalEventSource, ReadableEventObject, TransferMemoryObject,
    WritableEventObject,
};
use std::sync::{Arc, Mutex, OnceLock, Weak};
mod biquad;
mod buffer;
mod effect;
mod render;
mod resample;
mod splitter;
mod state;
pub(crate) mod update;
mod voice;
mod worker;
pub(crate) use worker::RenderError;

pub(crate) const PARAMETER_SIZE: usize = 0x34;
pub(crate) const CURRENT_REVISION: u32 = 15;

pub(crate) fn revision_number(revision: u32) -> Option<u32> {
    let revision = revision.checked_sub(u32::from_le_bytes(*b"REV0"))?;
    (revision & 0x00ff_ffff == 0 && (1..=CURRENT_REVISION).contains(&(revision >> 24)))
        .then_some(revision >> 24)
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
            revision: word(48),
            rendering_device: data[34],
            execution_mode: data[35],
        }
    }

    pub(crate) fn revision_number(&self) -> Option<u32> {
        revision_number(self.revision)
    }

    // Work storage follows the reference renderer's concrete metadata and
    // command sizes, without depending on Rust struct layout. Wide arithmetic
    // prevents guest counts from wrapping into a smaller allocation.
    // https://github.com/eden-emulator/mirror/blob/master/src/audio_core/renderer/system.cpp
    // https://switchbrew.org/wiki/Audio_services#BehaviorInfo_Revision
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

        let mut size = align(4 * mixes, 64) + submixes * 256 * 4 + nodes * 2368;
        size += voices * (560 + 200 + 296);
        size += align(nodes * 8, 16) + align(voices * 8, 16);
        size += align(((sinks + submixes) * 240 + samples) * 4 * (mixes + 6), 64);
        if revision >= 2 {
            let node_states = 4 * nodes * nodes + 48 * nodes + 2 * (align(nodes, 64) / 8);
            let edges = align(nodes * nodes, 64) / 8;
            size += align(node_states + edges, 16);
            size += 264 * destinations + 32 * splitters;
            if revision >= 5 {
                size += align(4 * destinations, 16);
            }
        }
        size += (effects + 4 * voices) * 32;
        if revision >= 9 {
            size += effects * 128;
        }
        size = align(size + 0x50, 64);
        size += (sinks + submixes) * 616 + effects * 1592;
        size += align(voices * 296, 64) + sinks * 352;
        if revision >= 9 {
            size += effects * 128;
        }
        let (performance_header, performance_entry) = if revision >= 5 {
            (0x30, 0x18)
        } else {
            (0x18, 0x10)
        };
        let performance_frame = performance_header + performance_entry * (entries + 100);
        if self.performance_frames != 0 {
            size += align(
                performance_frame * (u128::from(self.performance_frames) + 1) + 0xc0,
                256,
            );
        }
        size += if revision >= 5 {
            effects * 1592
                + 48
                + 40 * 24 * 24
                + 48
                + 40 * 24
                + 64 * (entries + 100)
                + sinks * 320
                + destinations * 24 * 48
                + voices
                    * (splitters * 96 + 2 * 80 + 40 + 328 + if revision >= 8 { 296 } else { 288 })
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
    pub(crate) update: Arc<state::UpdateSlot>,
    pub(crate) failure: Arc<OnceLock<RenderError>>,
    worker: Mutex<Option<worker::RendererWorker>>,
    devices: Arc<AudioDeviceState>,
    // Closing the client's copied handle must not free the renderer's work memory.
    _work_memory: Arc<TransferMemoryObject>,
    _applet_resource_user_id: u64,
    _process_id: u64,
}

impl AudioRendererManagerSession {
    pub(crate) fn set_host_paused(&self, paused: bool) {
        let states: Vec<_> = self
            .slots
            .lock()
            .unwrap()
            .iter()
            .filter_map(Weak::upgrade)
            .collect();
        for state in states {
            state.set_host_paused(paused);
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

    pub(crate) async fn open(
        &self,
        parameters: RendererParameters,
        work_memory: Arc<TransferMemoryObject>,
        work_size: u64,
        applet_resource_user_id: u64,
        process_id: u64,
    ) -> Result<Option<AudioRendererSession>, Arc<nixe_memory::MemoryError>> {
        let (signal, event) = EventObject::create_pair_with_source(ExternalEventSource::Device);
        let state = Arc::new(AudioRendererState {
            parameters,
            event,
            signal,
            rendering_time_limit: Mutex::new(100),
            update: Arc::default(),
            failure: Arc::default(),
            worker: Mutex::new(None),
            devices: self.devices.clone(),
            _work_memory: work_memory,
            _applet_resource_user_id: applet_resource_user_id,
            _process_id: process_id,
        });
        {
            let mut slots = self.slots.lock().unwrap();
            let Some(slot) = slots.iter_mut().find(|slot| slot.strong_count() == 0) else {
                return Ok(None);
            };
            *slot = Arc::downgrade(&state);
        }
        // The pending open owns its slot and transfer object, so the service
        // lock is not held while the device resolves an exact work-memory write.
        let zeroes = [0; 4096];
        let mut offset = 0;
        while offset < work_size {
            let count = (work_size - offset).min(zeroes.len() as u64) as usize;
            crate::system_memory::write_transfer(&state._work_memory, offset, &zeroes[..count])
                .await?;
            offset += count as u64;
        }
        Ok(Some(AudioRendererSession(state)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_memory::{run, transfer};

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
    fn negotiated_sizes_match_independently_executed_reference() {
        // Golden results from compiling System::GetWorkBufferSize and its
        // helpers in the reference linked above, including native struct sizes.
        for (revision, expected) in [
            (1, 0x5c000),
            (2, 0x5c000),
            (4, 0x5c000),
            (5, 0x66000),
            (7, 0x66000),
            (8, 0x66000),
            (9, 0x67000),
            (11, 0x67000),
            (15, 0x67000),
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
        params.splitters = 4;
        params.splitter_destinations = 12;
        params.performance_frames = 3;
        for revision in [7, 8, 9, 15] {
            assert_eq!(params.work_buffer_size(revision), Some(0x4a000));
        }
    }

    #[test]
    fn malformed_revisions_and_unrepresentable_sizes_are_rejected() {
        let mut params = parameters(7);
        for revision in [
            0,
            u32::from_le_bytes(*b"REV0"),
            u32::from_le_bytes(*b"REV@"),
            u32::from_le_bytes(*b"rev7"),
            u32::from_le_bytes(*b"REV7") + 1,
        ] {
            params.revision = revision;
            assert!((params.revision_number()).is_none());
        }
        params.voices = u32::MAX;
        params.splitters = u32::MAX;
        assert!((params.work_buffer_size(7)).is_none());
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
        let transfer = transfer(&vec![0; 0x50000]);
        device
            .state
            .set_volume(crate::audio_device::ACTIVE_DEVICE, 0.5);
        let first = run(manager.open(parameters(7), transfer.clone(), 0x50000, 1, 1))
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
        let second = run(manager.open(parameters(7), transfer, 0x50000, 1, 1))
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
        let transfer = transfer(&vec![0xcc; 0x51000]);
        let ranges = transfer.ranges().clone();
        let manager = AudioRendererManagerSession::default();
        let another_manager_connection = manager.clone();
        let open = |manager: &AudioRendererManagerSession| {
            run(manager.open(parameters(7), transfer.clone(), 0x50000, 1, 1)).unwrap()
        };
        let first = open(&manager).unwrap();
        let second = open(&another_manager_connection).unwrap();
        assert!(open(&manager).is_none());
        let mut bytes = vec![0; 0x51000];
        ranges[0]
            .allocation()
            .machine()
            .read_captured_into(&ranges, 0, &mut bytes)
            .unwrap()
            .into_ready()
            .unwrap();
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
