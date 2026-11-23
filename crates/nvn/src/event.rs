//! Physical event words are shared by CPU release operations and GPU completion.
use super::*;
use nixe_memory::{AllocationRange, ArenaOffset, device_memory::DeviceMemory};
use std::{
    collections::HashMap,
    sync::{
        Weak,
        atomic::{AtomicBool, AtomicU32, Ordering},
    },
};

#[derive(Clone, Copy)]
pub enum Signal {
    Write,
    Add,
}
impl Signal {
    pub fn decode(mode: u32) -> Result<Self> {
        match mode {
            0 => Ok(Self::Write),
            1 => Ok(Self::Add),
            _ => Err("invalid NVN event signal mode".into()),
        }
    }
}

pub struct EventWord {
    physical: ArenaOffset,
    // A completion may outlive the original SMMU mapping.
    range: AllocationRange,
}
impl EventWord {
    pub fn capture(memory: &DeviceMemory, address: u64) -> Result<Self> {
        let mapping = memory
            .buffer_mapping(address, 4)
            .map_err(|e| e.to_string())?;
        let mut ranges = mapping.pin(0, 4).map_err(|e| e.to_string())?;
        let range = ranges
            .pop()
            .filter(|_| ranges.is_empty())
            .ok_or("fragmented NVN event word")?;
        let physical = range
            .physical_extents()
            .next()
            .filter(|e| e.bytes == 4 && e.start.get().is_multiple_of(4))
            .ok_or("unaligned NVN event word")?
            .start;
        Ok(Self { physical, range })
    }

    fn atomic(&self) -> &AtomicU32 {
        let address = self
            .range
            .allocation()
            .machine()
            .arena()
            .host_address(self.physical, 4)
            .expect("retained physical event word");
        // SAFETY: this word is naturally aligned, remains mapped, and CPU
        // event accesses use atomic operations of the same width.
        unsafe { AtomicU32::from_ptr(address.as_ptr().cast()) }
    }
    fn load(&self) -> u32 {
        u32::from_le(self.atomic().load(Ordering::Acquire))
    }
    fn publish(&self, signal: Signal, value: u32) -> u32 {
        let word = self.atomic();
        match signal {
            Signal::Write => {
                word.store(value.to_le(), Ordering::Release);
                value
            }
            Signal::Add => {
                #[cfg(target_endian = "little")]
                let previous = word.fetch_add(value, Ordering::AcqRel);
                #[cfg(target_endian = "big")]
                let previous = word
                    .fetch_update(Ordering::AcqRel, Ordering::Acquire, |value_before| {
                        Some(u32::from_le(value_before).wrapping_add(value).to_le())
                    })
                    .unwrap();
                u32::from_le(previous).wrapping_add(value)
            }
        }
    }
}

#[derive(Clone, Copy)]
enum Predicate {
    Equal(u32),
    AtLeast(u32),
}
impl Predicate {
    fn decode(mode: u32, value: u32) -> Result<Self> {
        match mode {
            0 => Ok(Self::Equal(value)),
            1 => Ok(Self::AtLeast(value)),
            _ => Err("invalid NVN event wait mode".into()),
        }
    }
    fn matches(self, current: u32) -> bool {
        match self {
            Self::Equal(value) => current == value,
            Self::AtLeast(value) => (current.wrapping_sub(value) as i32) >= 0,
        }
    }
}

struct Latch {
    predicate: Predicate,
    predecessor: u64,
    ready: AtomicBool,
}

pub struct EventWait {
    word: EventWord,
    latch: Arc<Latch>,
}

#[derive(Default)]
struct WordState {
    completed: u64,
    pending: Option<(u64, Option<u32>)>,
    waiting: Vec<Weak<Latch>>,
}
impl WordState {
    fn ready(&self, word: &EventWord, predicate: Predicate) -> bool {
        if let Some((_, queued)) = self.pending {
            // The consumer's Vulkan work follows the encoded producer on the
            // same physical queue, even when its completion is still pending.
            queued.is_some_and(|value| predicate.matches(value))
        } else {
            predicate.matches(word.load())
        }
    }
}

#[derive(Default)]
pub struct EventSignals(Mutex<HashMap<u64, WordState>>);

impl EventSignals {
    pub fn wait(&self, word: EventWord, mode: u32, expected: u32) -> Result<Option<EventWait>> {
        let predicate = Predicate::decode(mode, expected)?;
        let mut words = self.0.lock().unwrap();
        let state = words.entry(word.physical.get()).or_default();
        if state.ready(&word, predicate) {
            if state.pending.is_none() && state.waiting.is_empty() {
                words.remove(&word.physical.get());
            }
            return Ok(None);
        }
        let latch = Arc::new(Latch {
            predicate,
            predecessor: state.pending.map_or(state.completed, |(serial, _)| serial),
            ready: AtomicBool::new(false),
        });
        state.waiting.push(Arc::downgrade(&latch));
        Ok(Some(EventWait { word, latch }))
    }

    pub fn ready(&self, wait: &EventWait) -> bool {
        if wait.latch.ready.load(Ordering::Acquire) {
            return true;
        }
        let mut words = self.0.lock().unwrap();
        let key = wait.word.physical.get();
        let Some(state) = words.get_mut(&key) else {
            return wait.latch.ready.load(Ordering::Acquire);
        };
        if !state.ready(&wait.word, wait.latch.predicate) {
            return false;
        }
        state.waiting.retain(|latch| {
            !std::ptr::eq(latch.as_ptr(), Arc::as_ptr(&wait.latch)) && latch.strong_count() != 0
        });
        if state.pending.is_none() && state.waiting.is_empty() {
            words.remove(&key);
        }
        true
    }

    pub fn record(&self, word: &EventWord, signal: Signal, value: u32) -> u64 {
        let mut words = self.0.lock().unwrap();
        let state = words.entry(word.physical.get()).or_default();
        let serial = state.pending.map_or(state.completed, |(serial, _)| serial) + 1;
        state.pending = Some((serial, matches!(signal, Signal::Write).then_some(value)));
        serial
    }

    pub fn publish(&self, word: EventWord, signal: Signal, payload: u32, serial: Option<u64>) {
        // The cache owners can be locked by a queue arming a wait. Notify
        // before taking the event lock, as for other physical RAM writes.
        word.range
            .allocation()
            .machine()
            .notify_resolved_write(&word.range);
        let mut words = self.0.lock().unwrap();
        let value = word.publish(signal, payload);
        let key = word.physical.get();
        let Some(state) = words.get_mut(&key) else {
            return;
        };
        if let Some(serial) = serial {
            state.completed = serial;
            if state.pending.is_some_and(|(last, _)| last == serial) {
                state.pending = None;
            }
        }
        // Latch the equality at publication. Another producer can change the
        // word again before the recording owner resumes the waiting queue.
        // https://github.com/seedds/nxnative/blob/a366365/dm/gfx/nvn/nvn_queue.inc
        state.waiting.retain(|latch| {
            let Some(latch) = latch.upgrade() else {
                return false;
            };
            if serial.map_or(state.pending.is_none(), |serial| {
                serial >= latch.predecessor
            }) && latch.predicate.matches(value)
            {
                latch.ready.store(true, Ordering::Release);
                false
            } else {
                true
            }
        });
        if state.pending.is_none() && state.waiting.is_empty() {
            words.remove(&key);
        }
    }
}

pub fn signal(c: &Call) -> Result<u64> {
    let event: Event = c.read(c.args[0])?;
    if event.cpu == 0 {
        return Err("NVN event pool is not CPU accessible".into());
    }
    let signal = Signal::decode(c.args[1] as u32)?;
    let word = EventWord::capture(c.session.graphics()?.heap.device(), event.gpu)?;
    c.session
        .events
        .publish(word, signal, c.args[2] as u32, None);
    Ok(0)
}
