//! Stateful renderer effects. Parameter updates are separate from DSP execution.
mod reverb;
use super::buffer::{Attachment, attach};
use super::update::{MemoryPool, UpdateError, wide, word};
use nixe_memory::CanonicalRangeTranslationError;

#[derive(Debug, Default)]
pub(super) enum Effect {
    #[default]
    Invalid,
    Reverb(Box<Reverb>),
}
#[derive(Debug)]
pub(super) struct Reverb {
    pub(super) mix: usize,
    pub(super) order: usize,
    enabled: bool,
    new: bool,
    was_enabled: Option<bool>,
    // The guest work buffer is an opaque DSP allocation. Like Yuzu, we keep
    // actual delay state on the host while retaining its canonical attachment.
    work: Attachment,
    params: reverb::Parameters,
    state: Option<reverb::State>,
    pending: u8,
}
impl Effect {
    pub(super) fn uses_pool(&self, pool: usize) -> bool {
        matches!(self, Self::Reverb(effect) if effect.enabled && effect.work.pool == Some(pool))
    }
    pub(super) fn routing(&self) -> Option<(usize, usize)> {
        match self {
            Self::Invalid => None,
            Self::Reverb(e) => Some((e.mix, e.order)),
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn update(
        &mut self,
        raw: &[u8],
        slot: usize,
        revision: u32,
        sample_rate: u32,
        pools: &[Option<MemoryPool>],
        force: bool,
        translate: &impl Fn(
            u64,
            u64,
        ) -> Result<
            nixe_memory::CanonicalBackingRange,
            CanonicalRangeTranslationError,
        >,
        started: bool,
    ) -> Result<u8, UpdateError> {
        if raw[0] == 0 {
            *self = Self::Invalid;
            return Ok(4);
        }
        if raw[0] != 4 {
            const NAMES: [&str; 10] = [
                "Invalid",
                "Mix",
                "Aux",
                "Delay",
                "Reverb",
                "I3DL2 Reverb",
                "Biquad Filter",
                "Light Limiter",
                "Capture",
                "Compressor",
            ];
            let kind = NAMES.get(raw[0] as usize).copied().unwrap_or("unknown");
            return Err(UpdateError::Unsupported(format!("RequestUpdate: effect slot {slot} uses {kind} (type={}, enabled={}); this effect's updates are not implemented", raw[0], raw[2] != 0).into()));
        }
        let params = reverb::Parameters::decode(&raw[0x20..], revision, sample_rate, raw[2] != 0)?;
        let new = raw[1] != 0 || matches!(self, Self::Invalid);
        if new {
            let work = attach(wide(raw, 8), wide(raw, 16), pools, force, translate)?
                .ok_or(UpdateError::Invalid)?;
            *self = Self::Reverb(Box::new(Reverb {
                mix: word(raw, 4) as usize,
                order: word(raw, 24) as usize,
                enabled: raw[2] != 0,
                new: true,
                was_enabled: None,
                work,
                params,
                state: None,
                pending: 0,
            }));
        } else if let Self::Reverb(effect) = self {
            effect.mix = word(raw, 4) as usize;
            effect.order = word(raw, 24) as usize;
            effect.enabled = raw[2] != 0;
            // Initialized/Updating must survive further IPC updates until a
            // worker frame consumes them. Only Updated accepts a new transition.
            if effect.pending == 2 {
                effect.pending = raw[0x60];
            }
            effect.params = params;
        }
        let Self::Reverb(effect) = self else {
            unreachable!()
        };
        // Reference status is published from previous command generation usage,
        // rather than claiming a freshly-disabled effect has already completed.
        Ok(
            if effect.new || (started && effect.was_enabled != Some(false)) {
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
        let Self::Reverb(effect) = self else {
            return Ok(());
        };
        effect.params.validate_routing(count)?;
        if effect.enabled {
            match effect.state.as_mut() {
                Some(state) if effect.pending != 0 => {
                    if effect.pending == 1 {
                        state.update(&effect.params);
                    }
                }
                _ => effect.state = Some(reverb::State::new(&effect.params)),
            }
            effect
                .state
                .as_mut()
                .unwrap()
                .process(&effect.params, buffers, samples);
        } else {
            effect.params.bypass(buffers, samples);
        }
        effect.was_enabled = Some(effect.enabled);
        effect.new = false;
        effect.pending = 2;
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn packet() -> [u8; 0xc0] {
        let mut p = [0; 0xc0];
        p[0] = 4;
        p[1] = 1;
        p[2] = 1;
        p[8..16].copy_from_slice(&0x1000_u64.to_le_bytes());
        p[16..24].copy_from_slice(&0x1000_u64.to_le_bytes());
        p[0x20] = 0;
        p[0x21] = 1;
        p[0x26] = 0;
        p[0x27] = 1;
        p[0x2c..0x2e].copy_from_slice(&2_u16.to_le_bytes());
        p[0x2e..0x30].copy_from_slice(&2_u16.to_le_bytes());
        for (offset, n) in [
            (0x30, 48_u32 << 14),
            (0x38, 1 << 14),
            (0x44, 1 << 14),
            (0x48, 1000 << 14),
            (0x4c, 1 << 14),
            (0x54, 1 << 14),
            (0x58, 1 << 14),
            (0x5c, 1 << 14),
        ] {
            p[offset..offset + 4].copy_from_slice(&n.to_le_bytes());
        }
        p
    }
    #[test]
    fn updates_wait_for_worker_consumption_and_preserve_tail_and_workbuffer_ownership() {
        use nixe_memory::{CanonicalAllocation, MemoryPermissions};
        let memory = CanonicalAllocation::zeroed(4096, 4096).unwrap();
        let backing = memory.backing_range(MemoryPermissions::READ_WRITE).unwrap();
        let pools = vec![Some(MemoryPool {
            address: 0x1000,
            backing,
        })];
        let mut effect = Effect::default();
        let mut raw = packet();
        assert_eq!(
            effect
                .update(&raw, 0, 7, 48000, &pools, false, &|_, _| panic!(), false)
                .unwrap(),
            3
        );
        assert!(effect.uses_pool(0));
        assert_eq!(effect.routing(), Some((0, 0)));
        raw[1] = 0;
        raw[0x60] = 2;
        effect
            .update(&raw, 0, 7, 48000, &pools, false, &|_, _| panic!(), true)
            .unwrap();
        let Effect::Reverb(e) = &effect else { panic!() };
        assert_eq!(e.pending, 0);
        assert!(e.state.is_none());
        let mut samples = vec![0; 480];
        samples[0] = 10000;
        effect.process(&mut samples, 240, 2).unwrap();
        let Effect::Reverb(e) = &effect else { panic!() };
        assert_eq!(e.pending, 2);
        let state = std::ptr::from_ref(e.state.as_ref().unwrap());
        effect
            .update(&raw, 0, 7, 48000, &pools, false, &|_, _| panic!(), true)
            .unwrap();
        let Effect::Reverb(e) = &effect else { panic!() };
        assert_eq!(std::ptr::from_ref(e.state.as_ref().unwrap()), state);
        samples.fill(0);
        effect.process(&mut samples, 240, 2).unwrap();
        assert!(
            samples.iter().any(|x| *x != 0),
            "IPC updates must preserve the pending reverb tail"
        );
        raw[2] = 0;
        assert_eq!(
            effect
                .update(&raw, 0, 7, 48000, &pools, false, &|_, _| panic!(), true)
                .unwrap(),
            3
        );
        samples.fill(500);
        effect.process(&mut samples, 240, 2).unwrap();
        assert!(samples.iter().all(|x| *x == 500));
        assert_eq!(
            effect
                .update(&raw, 0, 7, 48000, &pools, false, &|_, _| panic!(), true)
                .unwrap(),
            4
        );
        assert!(!effect.uses_pool(0));
        raw[0] = 0;
        effect
            .update(&raw, 0, 7, 48000, &pools, false, &|_, _| panic!(), true)
            .unwrap();
        assert!(matches!(effect, Effect::Invalid));
    }
}
