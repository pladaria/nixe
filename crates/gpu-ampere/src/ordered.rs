//! Resolved effects and bounded ordinary-state blocks on the shared ordered ring.
//! https://github.com/zortos293/Lichii/blob/646b5d17135a2b2701ccc7a959afd0bbbf2f6160/src/video_core/gpu_split.h
use crate::{DirtyFlags, Result, Source};
use nixe_gpu::ordered::{self, ENGINE, LOCATION, MACRO_END, METHOD, ORIGIN};
// Word tags are interpreted by the ISA-specific consumer. Tag 3 is available
// here; Maxwell uses it for its own multi-method representation.
const STATE: u64 = 3;
const RECONCILE: u64 = 4;
const MACRO: u64 = 7;
const MAX_STATE_WRITES: usize = 128;
// Tag: 3 bits; count: 8 bits; dirty domains: 46 bits. A single header
// keeps even one-write blocks compact without a separate scalar path.
const STATE_DIRTY_SHIFT: u32 = 11;
const _: () = {
    assert!(MAX_STATE_WRITES <= 0xff);
    assert!(STATE_DIRTY_SHIFT + DirtyFlags::BITS <= 64);
};
pub enum Operation<'a> {
    Method(Source),
    Reconcile(Source),
    Macro(Source, u32, &'a [u8]),
    Constants(Source, &'a [ConstantWord]),
    State(Source, DirtyFlags, &'a [u64]),
    Flush,
}
#[derive(Clone, Copy, Default)]
pub struct ConstantWord {
    pub method: u32,
    pub value: u32,
    pub offset: u32,
}
pub struct Writer<T> {
    ring: ordered::Sender<T>,
    previous: Option<Source>,
    state_source: Source,
    state: [u64; MAX_STATE_WRITES + 1],
    state_count: usize,
    #[cfg(feature = "performance-counters")]
    statistics: Statistics,
}
#[cfg(feature = "performance-counters")]
#[derive(Default, Debug)]
struct Statistics {
    state_writes: u64,
    state_blocks: u64,
    state_words: u64,
    scalar_effects: u64,
    block_sizes: [u64; 8],
}
impl<T> Writer<T> {
    pub fn new(ring: ordered::Sender<T>) -> Self {
        Self {
            ring,
            previous: None,
            state_source: Source::default(),
            state: [0; MAX_STATE_WRITES + 1],
            state_count: 0,
            #[cfg(feature = "performance-counters")]
            statistics: Statistics::default(),
        }
    }
    pub fn state(&mut self, source: Source, dirty: DirtyFlags) -> Result<()> {
        if self.state_count == MAX_STATE_WRITES
            || (self.state_count != 0
                && (source.channel != self.state_source.channel
                    || source.submission != self.state_source.submission
                    || source.entry != self.state_source.entry
                    || source.pushbuffer != self.state_source.pushbuffer
                    || source.class != self.state_source.class
                    || source.subchannel != self.state_source.subchannel))
        {
            self.flush_state()?;
        }
        self.state_source = source;
        self.state[self.state_count + 1] =
            u64::from(source.method) | (u64::from(source.argument) << 32);
        self.state[0] |= dirty.0 << STATE_DIRTY_SHIFT;
        self.state_count += 1;
        Ok(())
    }
    pub fn flush_state(&mut self) -> Result<()> {
        if self.state_count == 0 {
            return Ok(());
        }
        self.select(self.state_source)?;
        self.state[0] |= STATE | ((self.state_count as u64) << 3);
        self.ring
            .write(&self.state[..self.state_count + 1], &[])
            .map_err(|_| "preparation owner stopped")?;
        #[cfg(feature = "performance-counters")]
        {
            self.statistics.state_writes += self.state_count as u64;
            self.statistics.state_blocks += 1;
            self.statistics.state_words += self.state_count as u64 + 1;
            self.statistics.block_sizes[self.state_count.ilog2() as usize] += 1;
        }
        self.state_count = 0;
        self.state[0] = 0;
        Ok(())
    }
    fn select(&mut self, source: Source) -> Result<()> {
        let old = self.previous;
        if old.is_none_or(|s| {
            s.entry != source.entry
                || s.pushbuffer != source.pushbuffer
                || s.submission != source.submission
        }) {
            self.ring
                .write(
                    &[ORIGIN | (u64::from(source.entry) << 32), source.pushbuffer],
                    &[],
                )
                .map_err(|_| "preparation owner stopped")?;
        }
        if old.is_none_or(|s| s.offset != source.offset || s.entry != source.entry) {
            self.ring
                .write(&[LOCATION | (u64::from(source.offset) << 32)], &[])
                .map_err(|_| "preparation owner stopped")?;
        }
        if old.is_none_or(|s| s.class != source.class || s.subchannel != source.subchannel) {
            self.ring
                .write(
                    &[ENGINE
                        | (u64::from(source.subchannel) << 3)
                        | (u64::from(source.class) << 32)],
                    &[],
                )
                .map_err(|_| "preparation owner stopped")?;
        }
        self.previous = Some(source);
        Ok(())
    }
    pub fn method(&mut self, source: Source) -> Result<()> {
        self.flush_state()?;
        self.select(source)?;
        self.ring
            .write(
                &[METHOD
                    | (u64::from(source.subchannel) << 3)
                    | (u64::from(source.method) << 8)
                    | (u64::from(source.argument) << 32)],
                &[],
            )
            .map_err(|_| "preparation owner stopped")?;
        #[cfg(feature = "performance-counters")]
        {
            self.statistics.scalar_effects += 1;
        }
        Ok(())
    }
    pub fn reconcile(&mut self, source: Source) -> Result<()> {
        self.flush_state()?;
        self.select(source)?;
        self.ring
            .write(
                &[
                    RECONCILE
                        | (u64::from(source.method) << 8)
                        | (u64::from(source.argument) << 32),
                ],
                &[],
            )
            .map_err(|_| "preparation owner stopped".into())
    }
    pub fn run_macro(&mut self, source: Source, start: u32, parameters: &[u8]) -> Result<u64> {
        self.flush_state()?;
        self.select(source)?;
        assert!(parameters.len() <= 0x4000 && parameters.len().is_multiple_of(4) && start < 0x1000);
        // The ring owns this copy until preparation finishes or is cancelled.
        self.ring
            .write(
                &[MACRO
                    | (u64::from(start) << 8)
                    | (u64::from(source.method) << 24)
                    | (((parameters.len() / 4) as u64) << 40)],
                parameters,
            )
            .map_err(|_| "preparation owner stopped")?;
        Ok(self.ring.position())
    }
    pub fn draw_boundary(&mut self) -> Result<()> {
        self.flush_state()?;
        self.ring
            .write(&[MACRO_END], &[])
            .map_err(|_| "preparation owner stopped".into())
    }
    pub fn send(&mut self, control: T) -> Result<()> {
        self.flush_state()?;
        self.ring
            .send(control)
            .map_err(|_| "preparation owner stopped".into())
    }
    pub fn wait_until(&mut self, position: u64) -> Result<()> {
        self.flush_state()?;
        self.ring
            .wait_until(position)
            .map_err(|_| "preparation owner stopped".into())
    }
    /// Positions describe admitted ring entries; buffered state has no memory
    /// effects of its own. Flush it before forming a dependency on pending draws.
    pub fn position(&self) -> u64 {
        self.ring.position()
    }
    pub fn flush(&mut self) -> Result<()> {
        self.flush_state()?;
        self.ring.flush();
        Ok(())
    }
    pub fn finish(&mut self) -> Result<()> {
        self.flush()?;
        #[cfg(feature = "performance-counters")]
        log::info!(
            "Ampere ordered admission: {:?}; ring_wait={:?} ring_peak_words={} ring_publications={} ring_batches={}",
            self.statistics,
            self.ring.wait,
            self.ring.peak,
            self.ring.publications,
            self.ring.batches
        );
        Ok(())
    }
    pub fn reset(&mut self) {
        debug_assert_eq!(self.state_count, 0, "control boundary must flush state");
        self.previous = None;
    }
}
pub fn consume<T>(
    words: &mut ordered::Words<'_, T>,
    source: &mut Source,
    mut method: impl FnMut(Operation<'_>) -> Result<()>,
) -> Result<()> {
    while !words.remaining().is_empty() {
        let batch = words.remaining();
        let value = batch[0];
        let mut consumed = 1;
        match value & 7 {
            ORIGIN => {
                source.entry = (value >> 32) as u32;
                source.pushbuffer = batch[1];
                consumed = 2;
            }
            LOCATION => source.offset = (value >> 32) as u32,
            ENGINE => {
                source.class = (value >> 32) as u32;
                source.subchannel = ((value >> 3) & 7) as u8;
            }
            STATE => {
                let count = ((value >> 3) & 0xff) as usize;
                let writes = &batch[1..1 + count];
                let last = writes[count - 1];
                source.method = last as u32;
                source.argument = (last >> 32) as u32;
                method(Operation::State(
                    *source,
                    DirtyFlags(value >> STATE_DIRTY_SHIFT),
                    writes,
                ))?;
                consumed = 1 + count;
            }
            RECONCILE => {
                source.method = ((value >> 8) & 0xffff) as u32;
                source.argument = (value >> 32) as u32;
                method(Operation::Reconcile(*source))?;
            }
            MACRO => {
                let count = (value >> 40) as usize;
                let start = ((value >> 8) & 0xffff) as u32;
                source.method = ((value >> 24) & 0xffff) as u32;
                let storage = &batch[1..1 + count.div_ceil(2)];
                // SAFETY: exactly count initialized little-endian words follow
                // the header. The ring retains them until the callback returns.
                let parameters =
                    unsafe { std::slice::from_raw_parts(storage.as_ptr().cast::<u8>(), count * 4) };
                source.argument = parameters
                    .get(..4)
                    .map_or(0, |word| u32::from_le_bytes(word.try_into().unwrap()));
                method(Operation::Macro(*source, start, parameters))?;
                consumed = 1 + count.div_ceil(2);
            }
            METHOD => {
                source.method = ((value >> 8) & 0xffff) as u32;
                source.argument = (value >> 32) as u32;
                source.subchannel = ((value >> 3) & 7) as u8;
                if source.class == crate::Class::ThreeD as u32
                    && (0x2390..=0x23cc).contains(&source.method)
                {
                    let mut data = [ConstantWord::default(); 64];
                    let mut count = 1;
                    data[0] = ConstantWord {
                        method: source.method,
                        value: source.argument,
                        offset: source.offset,
                    };
                    let mut offset = source.offset;
                    while count < data.len() && consumed < batch.len() {
                        let mut next = consumed;
                        if batch[next] & 7 == LOCATION {
                            offset = (batch[next] >> 32) as u32;
                            next += 1;
                        }
                        let Some(&word) = batch.get(next) else {
                            break;
                        };
                        let m = ((word >> 8) & 0xffff) as u32;
                        if word & 7 != METHOD
                            || ((word >> 3) & 7) as u8 != source.subchannel
                            || !(0x2390..=0x23cc).contains(&m)
                        {
                            break;
                        }
                        data[count] = ConstantWord {
                            method: m,
                            value: (word >> 32) as u32,
                            offset,
                        };
                        count += 1;
                        consumed = next + 1;
                    }
                    method(Operation::Constants(*source, &data[..count]))
                        .map_err(|e| source.error(e))?;
                    let last = data[count - 1];
                    source.offset = last.offset;
                    source.method = last.method;
                    source.argument = last.value;
                } else {
                    method(Operation::Method(*source)).map_err(|e| source.error(e))?;
                }
            }
            MACRO_END => method(Operation::Flush)?,
            _ => unreachable!("admitted ordered method record"),
        }
        words.advance(consumed);
    }
    Ok(())
}
