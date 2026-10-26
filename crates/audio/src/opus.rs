//! Stateful libopus decoding. PCM capacity is measured in samples per channel.
//! https://opus-codec.org/docs/opus_api-1.3.1/group__opus__decoder.html
//! https://opus-codec.org/docs/opus_api-1.3.1/group__opus__multistream.html
use std::{ffi::c_void, ptr::NonNull};

unsafe extern "C" {
    fn opus_decoder_get_size(channels: i32) -> i32;
    fn opus_decoder_create(rate: i32, channels: i32, error: *mut i32) -> *mut c_void;
    fn opus_decoder_destroy(state: *mut c_void);
    fn opus_decode(
        state: *mut c_void,
        packet: *const u8,
        bytes: i32,
        pcm: *mut i16,
        frames: i32,
        fec: i32,
    ) -> i32;
    fn opus_decoder_ctl(state: *mut c_void, request: i32, ...) -> i32;
    fn opus_multistream_decoder_get_size(streams: i32, coupled: i32) -> i32;
    fn opus_multistream_decoder_create(
        rate: i32,
        channels: i32,
        streams: i32,
        coupled: i32,
        mapping: *const u8,
        error: *mut i32,
    ) -> *mut c_void;
    fn opus_multistream_decoder_destroy(state: *mut c_void);
    fn opus_multistream_decode(
        state: *mut c_void,
        packet: *const u8,
        bytes: i32,
        pcm: *mut i16,
        frames: i32,
        fec: i32,
    ) -> i32;
    fn opus_multistream_decoder_ctl(state: *mut c_void, request: i32, ...) -> i32;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Error(pub i32);

fn checked(result: i32) -> Result<usize, Error> {
    if result < 0 {
        Err(Error(result))
    } else {
        Ok(result as usize)
    }
}

#[derive(Debug)]
pub struct Decoder {
    state: NonNull<c_void>,
    channels: usize,
    multistream: bool,
}

// libopus has no thread affinity. All mutation requires exclusive access.
unsafe impl Send for Decoder {}

impl Decoder {
    pub fn state_size(channels: u32, streams: Option<(u32, u32)>) -> Result<usize, Error> {
        let size = unsafe {
            if let Some((streams, coupled)) = streams {
                if streams == 0 || streams > 255 || coupled > streams {
                    return Err(Error(-1));
                }
                opus_multistream_decoder_get_size(streams as i32, coupled as i32)
            } else {
                if !(1..=2).contains(&channels) {
                    return Err(Error(-1));
                }
                opus_decoder_get_size(channels as i32)
            }
        };
        if size <= 0 {
            Err(Error(-1))
        } else {
            Ok(size as usize)
        }
    }

    pub fn new(
        rate: u32,
        channels: u32,
        streams: Option<(u32, u32, &[u8])>,
    ) -> Result<Self, Error> {
        if !(1..=255).contains(&channels) || rate > i32::MAX as u32 {
            return Err(Error(-1));
        }
        let mut error = 0;
        let state = unsafe {
            if let Some((streams, coupled, mapping)) = streams {
                if streams == 0
                    || streams > 255
                    || coupled > streams
                    || mapping.len() < channels as usize
                {
                    return Err(Error(-1));
                }
                opus_multistream_decoder_create(
                    rate as i32,
                    channels as i32,
                    streams as i32,
                    coupled as i32,
                    mapping.as_ptr(),
                    &mut error,
                )
            } else {
                opus_decoder_create(rate as i32, channels as i32, &mut error)
            }
        };
        checked(error)?;
        let state = NonNull::new(state).ok_or(Error(-7))?;
        Ok(Self {
            state,
            channels: channels as usize,
            multistream: streams.is_some(),
        })
    }

    pub fn decode(&mut self, packet: &[u8], pcm: &mut [i16], reset: bool) -> Result<usize, Error> {
        let bytes = i32::try_from(packet.len()).map_err(|_| Error(-1))?;
        let frames = i32::try_from(pcm.len() / self.channels).map_err(|_| Error(-1))?;
        unsafe {
            if reset {
                checked(if self.multistream {
                    opus_multistream_decoder_ctl(self.state.as_ptr(), 4028)
                } else {
                    opus_decoder_ctl(self.state.as_ptr(), 4028)
                })?;
            }
            checked(if self.multistream {
                opus_multistream_decode(
                    self.state.as_ptr(),
                    packet.as_ptr(),
                    bytes,
                    pcm.as_mut_ptr(),
                    frames,
                    0,
                )
            } else {
                opus_decode(
                    self.state.as_ptr(),
                    packet.as_ptr(),
                    bytes,
                    pcm.as_mut_ptr(),
                    frames,
                    0,
                )
            })
        }
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        unsafe {
            if self.multistream {
                opus_multistream_decoder_destroy(self.state.as_ptr());
            } else {
                opus_decoder_destroy(self.state.as_ptr());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_real_packets_at_all_rates_and_maps_multistream_channels() {
        for rate in [8000, 12000, 16000, 24000, 48000] {
            let mut single = Decoder::new(rate, 1, None).unwrap();
            let mut multi = Decoder::new(rate, 2, Some((1, 0, &[0, 0]))).unwrap();
            let frames = rate as usize / 50;
            let mut mono = vec![123; frames];
            let mut stereo = vec![123; frames * 2];
            assert_eq!(
                single.decode(&[0xf8, 0xff, 0xfe], &mut mono, false),
                Ok(frames)
            );
            assert_eq!(
                multi.decode(&[0xf8, 0xff, 0xfe], &mut stereo, false),
                Ok(frames)
            );
            for (frame, &sample) in stereo.chunks_exact(2).zip(&mono) {
                assert_eq!(frame, [sample, sample]);
            }
            assert!(mono.iter().all(|&v| v == 0));
            mono.fill(123);
            assert_eq!(
                single.decode(&[0xf8, 0xff, 0xfe], &mut mono, true),
                Ok(frames)
            );
            assert!(mono.iter().all(|&v| v == 0));
            assert_eq!(
                single.decode(&[0xf8, 0xff, 0xfe], &mut [0; 1], false),
                Err(Error(-2))
            );
        }
    }
}
