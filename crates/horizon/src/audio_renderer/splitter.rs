//! Splitter connections and per-destination gain history.
//! https://github.com/eden-emulator/mirror/blob/master/src/audio_core/renderer/splitter/splitter_context.cpp
//! https://github.com/eden-emulator/mirror/blob/master/src/audio_core/renderer/splitter/splitter_info.cpp
//! https://github.com/eden-emulator/mirror/blob/master/src/audio_core/renderer/splitter/splitter_destinations_data.h
use super::update::{UpdateError, word};

pub(super) const UNUSED_MIX: usize = i32::MAX as usize;
pub(super) const UNUSED_SPLITTER: usize = u32::MAX as usize;

#[derive(Debug, Default)]
pub(super) struct Splitters {
    pub(super) infos: Vec<Splitter>,
    pub(super) destinations: Vec<Destination>,
}

#[derive(Debug, Default)]
pub(super) struct Splitter {
    pub(super) destinations: Vec<usize>,
}

#[derive(Debug)]
pub(super) struct Destination {
    pub(super) mix: usize,
    pub(super) in_use: bool,
    pub(super) volumes: [f32; 24],
    pub(super) previous: [f32; 24],
    pub(super) used: bool,
}

impl Default for Destination {
    fn default() -> Self {
        Self {
            mix: UNUSED_MIX,
            in_use: false,
            volumes: [0.0; 24],
            previous: [0.0; 24],
            used: false,
        }
    }
}

impl Splitters {
    pub(super) fn update(
        &mut self,
        raw: &[u8],
        revision: u32,
        infos: usize,
        destinations: usize,
        mixes: usize,
    ) -> Result<bool, UpdateError> {
        self.infos.resize_with(infos, Splitter::default);
        self.destinations
            .resize_with(destinations, Destination::default);
        if infos == 0 || destinations == 0 {
            return if raw.is_empty()
                || (raw.len() == 0x20 && word(raw, 4) == 0 && word(raw, 8) == 0)
            {
                Ok(false)
            } else {
                Err(UpdateError::Invalid)
            };
        }
        if raw.is_empty() {
            return Err(UpdateError::Invalid);
        }
        if raw.len() < 0x20 || word(raw, 0) != u32::from_le_bytes(*b"SNDH") {
            return Err(UpdateError::Invalid);
        }
        let mut cursor = 0x20;
        let mut changed = false;
        for _ in 0..word(raw, 4) {
            let header = raw.get(cursor..cursor + 0x10).ok_or(UpdateError::Invalid)?;
            if word(header, 0) != u32::from_le_bytes(*b"SNDI") {
                return Err(UpdateError::Invalid);
            }
            let id = word(header, 4) as usize;
            let count = word(header, 12) as usize;
            let info = self.infos.get_mut(id).ok_or(UpdateError::Invalid)?;
            // The destination array immediately follows the header; each
            // record also owns twelve trailing reserved bytes.
            let end = cursor
                .checked_add(0x1c + count * 4)
                .ok_or(UpdateError::Invalid)?;
            let data = raw.get(cursor + 0x10..end).ok_or(UpdateError::Invalid)?;
            let count = if revision < 5 {
                count.min(destinations / infos)
            } else {
                count
            };
            let ids = &data[..count * 4];
            changed |= info.destinations.len() != count
                || info
                    .destinations
                    .iter()
                    .zip(ids.chunks_exact(4))
                    .any(|(&a, b)| a != word(b, 0) as usize);
            info.destinations.clear();
            for data in ids.chunks_exact(4) {
                let id = word(data, 0) as usize;
                if id >= destinations {
                    return Err(UpdateError::Invalid);
                }
                info.destinations.push(id);
            }
            cursor = end;
        }
        let (stride, flags) = if revision >= 15 {
            (0xa8, 0x9c)
        } else if revision >= 12 {
            (0x90, 0x84)
        } else {
            (0x70, 0x6c)
        };
        for _ in 0..word(raw, 8) {
            let data = raw
                .get(cursor..cursor + stride)
                .ok_or(UpdateError::Invalid)?;
            if word(data, 0) != u32::from_le_bytes(*b"SNDD") {
                return Err(UpdateError::Invalid);
            }
            let destination = self
                .destinations
                .get_mut(word(data, 4) as usize)
                .ok_or(UpdateError::Invalid)?;
            let mix = word(data, 0x68) as usize;
            if mix != UNUSED_MIX && mix >= mixes {
                return Err(UpdateError::Invalid);
            }
            let in_use = data[flags] != 0;
            if in_use && revision >= 12 {
                let filter_size = if revision >= 15 { 0x18 } else { 0xc };
                if data[0x6c] != 0 || data[0x6c + filter_size] != 0 {
                    return Err(UpdateError::Unsupported(
                        "RequestUpdate: splitter biquad processing is not implemented".into(),
                    ));
                }
            }
            if in_use && revision >= 13 && data[flags + 1] != 0 {
                return Err(UpdateError::Unsupported(
                    "RequestUpdate: explicit splitter gain-history reset is not implemented".into(),
                ));
            }
            changed |= destination.mix != mix || destination.in_use != in_use;
            destination.mix = mix;
            for (i, volume) in destination.volumes.iter_mut().enumerate() {
                *volume = f32::from_bits(word(data, 8 + i * 4));
                if !volume.is_finite() {
                    return Err(UpdateError::Invalid);
                }
            }
            if !destination.in_use && in_use {
                destination.previous = destination.volumes;
                destination.used = false;
            }
            destination.in_use = in_use;
            cursor += stride;
        }
        // The header owns the section's trailing alignment padding.
        if raw.len() - cursor >= 16 {
            return Err(UpdateError::Invalid);
        }
        Ok(changed)
    }

    pub(super) fn finish_voices(&mut self) {
        for info in &self.infos {
            for &id in &info.destinations {
                let destination = &mut self.destinations[id];
                if destination.in_use && destination.used {
                    destination.previous = destination.volumes;
                }
                destination.used = false;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn put(raw: &mut [u8], offset: usize, value: u32) {
        raw[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn packet(revision: u32) -> (Vec<u8>, usize) {
        let (size, flags) = if revision >= 15 {
            (0xa8_usize, 0x9c)
        } else if revision >= 12 {
            (0x90, 0x84)
        } else {
            (0x70, 0x6c)
        };
        let mut raw = vec![0; (0x40 + size).next_multiple_of(16)];
        raw[..4].copy_from_slice(b"SNDH");
        put(&mut raw, 4, 1);
        put(&mut raw, 8, 1);
        raw[0x20..0x24].copy_from_slice(b"SNDI");
        put(&mut raw, 0x28, 48000);
        put(&mut raw, 0x2c, 1);
        raw[0x40..0x44].copy_from_slice(b"SNDD");
        put(&mut raw, 0x48, 0.5_f32.to_bits());
        raw[0x40 + flags] = 1;
        (raw, flags)
    }

    #[test]
    fn gains_preserve_history_until_consumed_and_reactivation_starts_at_current_gain() {
        for revision in [5, 9, 12, 13, 15] {
            let (mut raw, flags) = packet(revision);
            let mut splitters = Splitters::default();
            assert!(splitters.update(&raw, revision, 1, 1, 1).unwrap());
            assert_eq!(splitters.infos[0].destinations, [0]);
            assert_eq!(splitters.destinations[0].previous[0], 0.5);
            put(&mut raw, 0x48, 1_f32.to_bits());
            assert!(!splitters.update(&raw, revision, 1, 1, 1).unwrap());
            assert_eq!(splitters.destinations[0].volumes[0], 1.0);
            splitters.finish_voices();
            assert_eq!(splitters.destinations[0].previous[0], 0.5);
            splitters.destinations[0].used = true;
            splitters.finish_voices();
            assert_eq!(splitters.destinations[0].previous[0], 1.0);
            raw[0x40 + flags] = 0;
            assert!(splitters.update(&raw, revision, 1, 1, 1).unwrap());
            raw[0x40 + flags] = 1;
            put(&mut raw, 0x48, 0.25_f32.to_bits());
            assert!(splitters.update(&raw, revision, 1, 1, 1).unwrap());
            assert_eq!(splitters.destinations[0].previous[0], 0.25);
        }
    }

    #[test]
    fn malformed_destinations_and_unsupported_active_processing_fail_explicitly() {
        let (raw, _) = packet(9);
        for offset in [0, 0x20, 0x40, 0x24, 0x30, 0x44, 0xa8] {
            let mut bad = raw.clone();
            put(&mut bad, offset, u32::MAX);
            assert!(matches!(
                Splitters::default().update(&bad, 9, 1, 1, 1),
                Err(UpdateError::Invalid)
            ));
        }
        let mut short = raw;
        short.pop();
        assert!(matches!(
            Splitters::default().update(&short, 9, 1, 1, 1),
            Err(UpdateError::Invalid)
        ));
        for revision in [12, 15] {
            let (mut raw, _) = packet(revision);
            raw[0xac] = 1;
            assert!(
                matches!(Splitters::default().update(&raw, revision, 1, 1, 1), Err(UpdateError::Unsupported(detail)) if detail.contains("biquad"))
            );
        }
    }
}
