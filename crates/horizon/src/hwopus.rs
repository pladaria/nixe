//! Hardware Opus service state and guest work-memory negotiation.
//! https://switchbrew.org/wiki/Audio_services#hwopus
//! https://github.com/eden-emulator/mirror/blob/master/src/audio_core/opus/decoder_manager.cpp
//! https://github.com/eden-emulator/mirror/blob/master/src/audio_core/adsp/apps/opus/opus_decode_object.h
use crate::HorizonIpcResult;
use nixe_audio::opus::Decoder;
use nixe_runtime::TransferMemoryObject;
use std::sync::{Arc, Mutex, Weak};
use std::time::Instant;

fn result(description: u32) -> HorizonIpcResult {
    HorizonIpcResult::from_raw(111 | (description << 9))
}

pub(crate) fn codec_error(error: nixe_audio::opus::Error) -> HorizonIpcResult {
    result(match error.0 {
        -1 => 2,
        -2 => 3,
        -3 => 4,
        -4 => 17,
        -5 => 5,
        -6 => 6,
        -7 => 7,
        _ => 259,
    })
}
pub(crate) const BUFFER_TOO_SMALL: HorizonIpcResult = HorizonIpcResult::from_raw(111 | (3 << 9));
pub(crate) const INPUT_TOO_SMALL: HorizonIpcResult = HorizonIpcResult::from_raw(111 | (8 << 9));
pub(crate) const BAD_ARGUMENT: HorizonIpcResult = HorizonIpcResult::from_raw(111 | (2 << 9));

#[derive(Debug, Clone)]
pub(crate) struct Parameters {
    pub(crate) sample_rate: u32,
    pub(crate) channels: u32,
    pub(crate) streams: Option<(u32, u32)>,
    pub(crate) large_frames: bool,
    mapping: [u8; 256],
}
impl Parameters {
    pub(crate) fn decode(raw: &[u8], multi: bool, extended: bool) -> Self {
        let word = |offset| u32::from_le_bytes(raw[offset..offset + 4].try_into().unwrap());
        let mut params = Self {
            sample_rate: word(0),
            channels: word(4),
            streams: multi.then(|| (word(8), word(12))),
            large_frames: extended && raw[if multi { 16 } else { 8 }] != 0,
            mapping: [0; 256],
        };
        if multi {
            let offset = if extended { 0x18 } else { 0x10 };
            params.mapping.copy_from_slice(&raw[offset..offset + 256]);
        }
        params
    }

    pub(crate) fn work_size(&self) -> Result<u32, HorizonIpcResult> {
        if !(1..=if self.streams.is_some() { 255 } else { 2 }).contains(&self.channels) {
            return Err(result(1002));
        }
        if !matches!(self.sample_rate, 8000 | 12000 | 16000 | 24000 | 48000) {
            return Err(result(1001));
        }
        if let Some((streams, coupled)) = self.streams
            && (streams == 0
                || streams > self.channels
                || coupled > streams
                || streams + coupled > self.channels)
        {
            return Err(result(1001));
        }
        let state = Decoder::state_size(self.channels, self.streams).map_err(codec_error)? as u32;
        let input = self
            .streams
            .map_or(0x600, |(streams, _)| (1500 * streams).next_multiple_of(64));
        let output = ((if self.large_frames { 5760 } else { 1920 }) * self.channels
            / (48000 / self.sample_rate))
            .next_multiple_of(64);
        // The reference decoder's 64-bit metadata occupies 32 bytes, followed
        // by libopus state, compressed input and aligned PCM staging capacity.
        Ok(32 + state + input + output)
    }

    pub(crate) fn input_capacity(&self) -> usize {
        self.streams.map_or(0x600, |(streams, _)| {
            (1500 * streams).next_multiple_of(64) as usize
        })
    }
}

#[derive(Clone, Debug)]
pub struct OpusManagerSession {
    slots: Arc<Mutex<[Weak<OpusState>; 24]>>,
    pub(crate) domain: crate::object::DomainSession<OpusDecoderSession>,
}

impl Default for OpusManagerSession {
    fn default() -> Self {
        Self {
            slots: Arc::default(),
            domain: crate::object::DomainSession::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct OpusDecoderSession {
    pub(crate) state: Arc<OpusState>,
    pub(crate) domain: crate::object::DomainSession<()>,
}

#[derive(Debug)]
pub(crate) struct OpusState {
    pub(crate) params: Parameters,
    decoder: Mutex<Decoder>,
    _work_memory: Arc<TransferMemoryObject>,
}

impl OpusManagerSession {
    pub(crate) fn open_session(&self) -> Self {
        Self {
            slots: self.slots.clone(),
            domain: crate::object::DomainSession::new(),
        }
    }
    pub(crate) fn open(
        &self,
        params: Parameters,
        work: Arc<TransferMemoryObject>,
        bytes: u32,
    ) -> Result<OpusDecoderSession, HorizonIpcResult> {
        if bytes < params.work_size()? || u64::from(bytes) > work.size() {
            return Err(BUFFER_TOO_SMALL);
        }
        let mut slots = self.slots.lock().unwrap();
        let slot = slots
            .iter_mut()
            .find(|slot| slot.strong_count() == 0)
            .ok_or_else(|| result(385))?;
        let streams = params.streams.map(|(streams, coupled)| {
            (
                streams,
                coupled,
                &params.mapping[..params.channels as usize],
            )
        });
        let decoder =
            Decoder::new(params.sample_rate, params.channels, streams).map_err(codec_error)?;
        let state = Arc::new(OpusState {
            params,
            decoder: Mutex::new(decoder),
            _work_memory: work,
        });
        *slot = Arc::downgrade(&state);
        Ok(OpusDecoderSession {
            state,
            domain: crate::object::DomainSession::new(),
        })
    }
}

impl OpusState {
    pub(crate) fn decode(
        &self,
        packet: &[u8],
        capacity: usize,
        reset: bool,
    ) -> Result<(Vec<i16>, u32, u64), HorizonIpcResult> {
        let channels = self.params.channels as usize;
        let max_frames = self.params.sample_rate as usize / 1000
            * if self.params.large_frames { 120 } else { 40 };
        let frames = max_frames.min(capacity / (channels * 2));
        if frames == 0 {
            return Err(BUFFER_TOO_SMALL);
        }
        let mut pcm = vec![0_i16; frames * channels];
        let mut decoder = self.decoder.lock().unwrap();
        let start = Instant::now();
        let frames = decoder
            .decode(packet, &mut pcm, reset)
            .map_err(codec_error)?;
        let elapsed = start.elapsed().as_micros() as u64;
        pcm.truncate(frames * channels);
        // Guest PCM is little endian; this is a no-op on supported Linux hosts.
        for sample in &mut pcm {
            *sample = sample.to_le();
        }
        Ok((pcm, frames as u32, elapsed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> Parameters {
        Parameters {
            sample_rate: 48000,
            channels: 1,
            streams: None,
            large_frames: false,
            mapping: [0; 256],
        }
    }

    #[test]
    fn work_negotiation_validates_counts_and_tracks_native_state_and_frame_limits() {
        let mut parameters = params();
        let state = Decoder::state_size(1, None).unwrap() as u32;
        assert_eq!(parameters.work_size(), Ok(state + 3488));
        parameters.large_frames = true;
        assert_eq!(parameters.work_size(), Ok(state + 7328));
        parameters.streams = Some((1, 0));
        let multi = Decoder::state_size(1, Some((1, 0))).unwrap() as u32;
        assert_eq!(parameters.work_size(), Ok(multi + 7328));
        for channels in [0, 256, u32::MAX] {
            parameters.channels = channels;
            assert_eq!(parameters.work_size(), Err(result(1002)));
        }
        parameters.channels = 2;
        for streams in [(0, 0), (1, 2), (2, 1), (u32::MAX, u32::MAX)] {
            parameters.streams = Some(streams);
            assert_eq!(parameters.work_size(), Err(result(1001)));
        }
        parameters.streams = None;
        parameters.sample_rate = 44100;
        assert_eq!(parameters.work_size(), Err(result(1001)));
    }

    #[test]
    fn decoder_slots_and_work_memory_outlive_client_handles_until_the_last_clone() {
        let manager = OpusManagerSession::default();
        let connection = manager.clone();
        let work = crate::test_memory::transfer(&vec![0; 65536]);
        let retained = Arc::downgrade(&work);
        let mut decoders: Vec<_> = (0..24)
            .map(|_| manager.open(params(), work.clone(), 65536).unwrap())
            .collect();
        assert_eq!(
            connection.open(params(), work.clone(), 65536).unwrap_err(),
            result(385)
        );
        let first = decoders.pop().unwrap();
        let clone = first.clone();
        drop(first);
        assert_eq!(
            connection.open(params(), work.clone(), 65536).unwrap_err(),
            result(385)
        );
        drop(clone);
        decoders.push(connection.open(params(), work.clone(), 65536).unwrap());
        drop(work);
        assert!(retained.upgrade().is_some());
        drop(decoders);
        assert!(retained.upgrade().is_none());
    }

    #[test]
    fn domain_conversion_is_local_to_the_connection_and_clones_share_children() {
        let registry = OpusManagerSession::default();
        let first = registry.open_session();
        let second = registry.open_session();
        let clone = first.clone();
        assert_eq!(first.domain.convert(), 1);
        assert!(clone.domain.is_domain());
        assert!(!second.domain.is_domain());
        assert!(!registry.domain.is_domain());
        let work = crate::test_memory::transfer(&vec![0; 65536]);
        let decoder = first.open(params(), work, 65536).unwrap();
        let id = first.domain.insert_object(decoder).unwrap();
        assert!(clone.domain.object(id).is_some());
        assert!(second.domain.object(id).is_none());
        assert!(clone.domain.close_object(id));
        assert!(first.domain.object(id).is_none());
    }

    #[test]
    fn large_frames_decode_120_ms_and_default_capacity_rejects_them() {
        // Six CBR CELT silence frames, 20 ms each (RFC 6716 packet code 3).
        let packet = [
            0xfb, 6, 0xff, 0xfe, 0xff, 0xfe, 0xff, 0xfe, 0xff, 0xfe, 0xff, 0xfe, 0xff, 0xfe,
        ];
        let manager = OpusManagerSession::default();
        let work = crate::test_memory::transfer(&vec![0; 65536]);
        let normal = manager.open(params(), work.clone(), 65536).unwrap();
        assert_eq!(
            normal.state.decode(&packet, 11520, false).unwrap_err(),
            BUFFER_TOO_SMALL
        );
        let mut parameters = params();
        parameters.large_frames = true;
        let large = manager.open(parameters, work, 65536).unwrap();
        let (pcm, samples, _) = large.state.decode(&packet, 11520, true).unwrap();
        assert_eq!(samples, 5760);
        assert_eq!(pcm.len(), 5760);
        assert!(pcm.iter().all(|&sample| sample == 0));
        assert_eq!(
            large.state.decode(&[0xff], 11520, false).unwrap_err(),
            result(17)
        );
    }
}
