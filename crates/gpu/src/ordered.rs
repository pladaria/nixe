//! Ordered SPSC word ring. Method payloads are written in place and consumed
//! before their storage can be reused. Control values retain their Rust ownership.
//! https://github.com/zortos293/Lichii/blob/b4b830c8faf1fb5d7b2ad1b892117f4cfd708546/src/video_core/gpu_split.h
use std::cell::UnsafeCell;
use std::mem::{MaybeUninit, size_of};
use std::sync::{
    Arc, OnceLock,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use std::thread::{self, Thread};
#[cfg(feature = "performance-counters")]
use std::time::Duration;
#[cfg(feature = "performance-counters")]
use std::time::Instant;

const WRAP: u64 = 0;
const WORDS: u64 = 1;
const CONTROL: u64 = 2;
const PUBLISH_WORDS: u64 = 256;
const EARLY_PUBLISH_WORDS: u64 = 8;
#[repr(align(128))]
struct Cursor(AtomicU64);
// These flags are written by opposite threads during sleep handshakes. Keep
// their cache lines independent from each other and from the quiet transition.
#[repr(align(128))]
struct Flag(AtomicBool);
struct Shared<T> {
    words: Box<[UnsafeCell<MaybeUninit<u64>>]>,
    written: Cursor,
    read: Cursor,
    idle_at: Cursor,
    wait_target: Cursor,
    producer_alive: AtomicBool,
    consumer_alive: AtomicBool,
    producer_waiting: Flag,
    consumer_waiting: Flag,
    consumer_quiet: Flag,
    producer: OnceLock<Thread>,
    consumer: OnceLock<Thread>,
    marker: std::marker::PhantomData<T>,
}
// Only one producer writes unpublished words. A consumer lease prevents their
// reuse until execution returns. Control values move exactly once out of the ring.
unsafe impl<T: Send> Sync for Shared<T> {}
impl<T> Shared<T> {
    fn pointer(&self, position: u64) -> *mut u64 {
        self.words[position as usize & (self.words.len() - 1)]
            .get()
            .cast()
    }
    fn consumed(&self, position: u64) {
        self.read.0.store(position, Ordering::SeqCst);
        if self.producer_waiting.0.load(Ordering::SeqCst)
            && let Some(thread) = self.producer.get()
        {
            thread.unpark();
        }
    }
}
impl<T> Drop for Shared<T> {
    fn drop(&mut self) {
        let mut position = *self.read.0.get_mut();
        let end = *self.written.0.get_mut();
        while position < end {
            // All headers below written are initialized, including wrap records.
            let header = unsafe { self.pointer(position).read() };
            if header & 3 == CONTROL {
                unsafe { self.pointer(position + 1).cast::<T>().read_unaligned() };
            }
            position += header >> 2;
        }
    }
}
pub struct Sender<T> {
    shared: Arc<Shared<T>>,
    position: u64,
    read: u64,
    published: u64,
    publish_every: u64,
    batch: Option<u64>,
    #[cfg(feature = "performance-counters")]
    pub wait: Duration,
    #[cfg(feature = "performance-counters")]
    pub peak: u64,
    #[cfg(feature = "performance-counters")]
    pub occupancy: u64,
    #[cfg(feature = "performance-counters")]
    pub publications: u64,
    #[cfg(feature = "performance-counters")]
    pub batches: u64,
}
pub struct Receiver<T> {
    shared: Arc<Shared<T>>,
    position: u64,
    record_end: u64,
    published: u64,
    released: u64,
}
pub enum Read<'a, T> {
    Control(T),
    Words(Words<'a, T>),
}
/// A lexical lease prevents nested consumption and reuse of unread words.
/// Advancing the lease releases only methods whose execution has returned.
pub struct Words<'a, T> {
    receiver: &'a mut Receiver<T>,
}
impl<T> Words<'_, T> {
    pub fn remaining(&self) -> &[u64] {
        unsafe {
            std::slice::from_raw_parts(
                self.receiver.shared.pointer(self.receiver.position),
                (self.receiver.record_end - self.receiver.position) as usize,
            )
        }
    }
    pub fn advance(&mut self, words: usize) {
        assert!(words as u64 <= self.receiver.record_end - self.receiver.position);
        self.receiver.position += words as u64;
        self.receiver.release(false);
    }
}
impl<T> Drop for Words<'_, T> {
    fn drop(&mut self) {
        // On failure remaining methods are discarded, never interpreted as
        // outer headers. No owned control values occur inside a word payload.
        self.receiver.position = self.receiver.record_end;
        self.receiver.release(false);
    }
}
pub fn channel<T: Send>(capacity: usize) -> (Sender<T>, Receiver<T>) {
    assert!(capacity.is_power_of_two() && capacity > 1 + size_of::<T>().div_ceil(8));
    let shared = Arc::new(Shared {
        words: (0..capacity)
            .map(|_| UnsafeCell::new(MaybeUninit::uninit()))
            .collect(),
        written: Cursor(AtomicU64::new(0)),
        read: Cursor(AtomicU64::new(0)),
        idle_at: Cursor(AtomicU64::new(u64::MAX)),
        wait_target: Cursor(AtomicU64::new(u64::MAX)),
        producer_alive: AtomicBool::new(true),
        consumer_alive: AtomicBool::new(true),
        producer_waiting: Flag(AtomicBool::new(false)),
        consumer_waiting: Flag(AtomicBool::new(false)),
        consumer_quiet: Flag(AtomicBool::new(true)),
        producer: OnceLock::new(),
        consumer: OnceLock::new(),
        marker: std::marker::PhantomData,
    });
    (
        Sender {
            shared: Arc::clone(&shared),
            position: 0,
            read: 0,
            published: 0,
            publish_every: PUBLISH_WORDS,
            batch: None,
            #[cfg(feature = "performance-counters")]
            wait: Duration::ZERO,
            #[cfg(feature = "performance-counters")]
            peak: 0,
            #[cfg(feature = "performance-counters")]
            occupancy: 0,
            #[cfg(feature = "performance-counters")]
            publications: 0,
            #[cfg(feature = "performance-counters")]
            batches: 0,
        },
        Receiver {
            shared,
            position: 0,
            record_end: 0,
            published: 0,
            released: 0,
        },
    )
}
impl<T> Sender<T> {
    fn seal(&mut self) {
        if let Some(start) = self.batch.take() {
            debug_assert!(
                (start as usize & (self.shared.words.len() - 1)) + (self.position - start) as usize
                    <= self.shared.words.len(),
                "word batch crosses the physical ring boundary"
            );
            #[cfg(feature = "performance-counters")]
            {
                self.batches += 1;
            }
            unsafe {
                self.shared
                    .pointer(start)
                    .write(((self.position - start) << 2) | WORDS);
            }
        }
    }
    pub fn flush(&mut self) {
        self.seal();
        if self.position == self.published {
            return;
        }
        self.published = self.position;
        self.publish_every = (self.publish_every * 2).min(PUBLISH_WORDS);
        self.shared.written.0.store(self.position, Ordering::SeqCst);
        #[cfg(feature = "performance-counters")]
        {
            let depth = self.position - self.shared.read.0.load(Ordering::Relaxed);
            self.peak = self.peak.max(depth);
            self.occupancy += depth;
            self.publications += 1;
        }
        if self.shared.consumer_waiting.0.load(Ordering::SeqCst)
            && let Some(thread) = self.shared.consumer.get()
        {
            thread.unpark();
        }
    }
    fn space(&mut self, words: u64) -> Result<(), Disconnected> {
        if self.position + words - self.read <= self.shared.words.len() as u64 {
            return Ok(());
        }
        self.flush();
        #[cfg(feature = "performance-counters")]
        let started = Instant::now();
        let result = self.wait_until(self.position + words - self.shared.words.len() as u64);
        #[cfg(feature = "performance-counters")]
        {
            self.wait += started.elapsed();
        }
        result
    }
    pub fn consumed(&mut self, position: u64) -> bool {
        if self.read < position {
            self.read = self.shared.read.0.load(Ordering::Acquire);
        }
        self.read >= position
    }
    /// Ask an empty consumer to settle once, without waking it again on each
    /// poll of a dependency. Explicit progress requests remain independent.
    pub fn needs_idle_hook(&mut self) -> bool {
        !self.shared.consumer_quiet.0.load(Ordering::Acquire) && self.consumed(self.position)
    }
    pub fn wait_until(&mut self, position: u64) -> Result<(), Disconnected> {
        if self.read >= position {
            return Ok(());
        }
        self.read = self.shared.read.0.load(Ordering::Acquire);
        if self.read >= position {
            return Ok(());
        }
        self.shared.wait_target.0.store(position, Ordering::SeqCst);
        self.flush();
        if self.shared.consumer_waiting.0.load(Ordering::SeqCst)
            && let Some(thread) = self.shared.consumer.get()
        {
            thread.unpark();
        }
        let result = self.wait_for_consumed(position);
        self.shared.wait_target.0.store(u64::MAX, Ordering::Relaxed);
        if result.is_ok() && self.read == self.position {
            self.publish_every = EARLY_PUBLISH_WORDS;
        }
        result
    }
    fn wait_for_consumed(&mut self, position: u64) -> Result<(), Disconnected> {
        for _ in 0..1024 {
            self.read = self.shared.read.0.load(Ordering::Acquire);
            if self.read >= position {
                return Ok(());
            }
            std::hint::spin_loop();
        }
        let s = &self.shared;
        s.producer.get_or_init(thread::current);
        s.producer_waiting.0.store(true, Ordering::SeqCst);
        let result = loop {
            self.read = s.read.0.load(Ordering::SeqCst);
            if self.read >= position {
                break Ok(());
            }
            if !s.consumer_alive.load(Ordering::SeqCst) {
                break Err(Disconnected);
            }
            thread::park();
        };
        s.producer_waiting.0.store(false, Ordering::Relaxed);
        result
    }
    fn reserve(&mut self, words: u64) -> Result<(), Disconnected> {
        assert!(words < self.shared.words.len() as u64);
        let index = self.position as usize & (self.shared.words.len() - 1);
        let rest = (self.shared.words.len() - index) as u64;
        if rest < words {
            self.seal();
            self.space(rest)?;
            unsafe {
                self.shared.pointer(self.position).write((rest << 2) | WRAP);
            }
            self.position += rest;
            self.flush();
        }
        self.space(words)
    }
    pub fn send(&mut self, value: T) -> Result<(), T> {
        if !self.shared.consumer_alive.load(Ordering::Acquire) {
            return Err(value);
        }
        self.seal();
        let words = 1 + size_of::<T>().div_ceil(8) as u64;
        if self.reserve(words).is_err() {
            return Err(value);
        }
        unsafe {
            self.shared
                .pointer(self.position)
                .write((words << 2) | CONTROL);
            self.shared
                .pointer(self.position + 1)
                .cast::<T>()
                .write_unaligned(value);
        }
        self.position += words;
        self.flush();
        Ok(())
    }
    /// Append a word header and an optional literal byte block directly into
    /// the ring. Only the final partial word needs padding.
    pub fn write(&mut self, prefix: &[u64], bytes: &[u8]) -> Result<(), Disconnected> {
        let count = prefix.len() as u64 + bytes.len().div_ceil(8) as u64;
        if count == 0 {
            return Ok(());
        }
        if !self.shared.consumer_alive.load(Ordering::Acquire) {
            return Err(Disconnected);
        }
        let index = self.position as usize & (self.shared.words.len() - 1);
        let rest = self.shared.words.len() - index;
        // A preceding append can fill the ring exactly without publishing.
        // Its batch must end there even though this append fits at index zero.
        if self.batch.is_some() && (index == 0 || rest < count as usize) {
            self.seal();
        }
        if self.batch.is_none() {
            self.reserve(count + 1)?;
            self.batch = Some(self.position);
            self.position += 1;
        } else {
            // Reserving space may publish the open batch before waiting.
            self.space(count + 1)?;
            if self.batch.is_none() {
                self.reserve(count + 1)?;
                self.batch = Some(self.position);
                self.position += 1;
            }
        }
        unsafe {
            let destination = self.shared.pointer(self.position);
            std::ptr::copy_nonoverlapping(prefix.as_ptr(), destination, prefix.len());
            if !bytes.is_empty() {
                if !bytes.len().is_multiple_of(8) {
                    destination.add(count as usize - 1).write(0);
                }
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    destination.add(prefix.len()).cast(),
                    bytes.len(),
                );
            }
        }
        self.position += count;
        let unpublished = self.position - self.published;
        if unpublished >= self.publish_every
            || (unpublished >= EARLY_PUBLISH_WORDS
                && self.shared.idle_at.0.load(Ordering::Relaxed) == self.published)
        {
            self.flush();
        }
        Ok(())
    }
    pub fn position(&self) -> u64 {
        self.position
    }
}
#[derive(Debug)]
pub struct Disconnected;
impl<T> Receiver<T> {
    fn release(&mut self, force: bool) {
        if self.position != self.released
            && (force
                || self.position == self.published
                || self.position - self.released >= 4096
                || self.position >= self.shared.wait_target.0.load(Ordering::Relaxed))
        {
            self.shared.consumed(self.position);
            self.released = self.position;
        }
    }
    pub fn ready(&mut self) -> bool {
        // Only callable without a live payload borrow. Also discard a forgotten
        // lease so its remaining words cannot be mistaken for outer headers.
        self.position = self.record_end;
        self.release(false);
        loop {
            if self.position == self.published {
                self.published = self.shared.written.0.load(Ordering::Acquire);
                if self.position == self.published {
                    return false;
                }
            }
            let header = unsafe { self.shared.pointer(self.position).read() };
            if header & 3 != WRAP {
                return true;
            }
            // Padding may be published before the following record. Consuming
            // it must not report an available message and then return None.
            self.position += header >> 2;
            self.record_end = self.position;
            self.release(false);
        }
    }
    pub fn try_recv(&mut self) -> Option<Read<'_, T>> {
        if !self.ready() {
            return None;
        }
        // The consumer alone changes quiet state. Publish the transition before
        // its first record; subsequent records need no shared atomic write.
        if self.shared.consumer_quiet.0.load(Ordering::Relaxed) {
            self.shared.consumer_quiet.0.store(false, Ordering::SeqCst);
        }
        let start = self.position;
        let header = unsafe { self.shared.pointer(start).read() };
        let end = start + (header >> 2);
        self.record_end = end;
        match header & 3 {
            CONTROL => {
                self.position = end;
                Some(Read::Control(unsafe {
                    self.shared.pointer(start + 1).cast::<T>().read_unaligned()
                }))
            }
            WORDS => {
                self.position = start + 1;
                Some(Read::Words(Words { receiver: self }))
            }
            _ => unreachable!("ready skips ring padding"),
        }
    }
    /// Announce an empty queue before spinning so the producer publishes early.
    /// Run the idle hook only after the transient gap, or on an explicit request.
    pub fn wait<E>(
        &mut self,
        stopped: impl Fn() -> bool,
        requested: impl Fn() -> bool,
        mut idle: impl FnMut() -> Result<(), E>,
    ) -> Result<bool, E> {
        self.shared.consumer.get_or_init(thread::current);
        self.shared
            .idle_at
            .0
            .store(self.position, Ordering::Relaxed);
        for _ in 0..1024 {
            if self.ready() {
                return Ok(true);
            }
            if stopped() {
                return Ok(false);
            }
            if requested() {
                break;
            }
            std::hint::spin_loop();
        }
        loop {
            if self.ready() {
                return Ok(true);
            }
            self.shared
                .idle_at
                .0
                .store(self.position, Ordering::Relaxed);
            if !self.shared.consumer_quiet.0.load(Ordering::Relaxed) || requested() {
                idle()?;
                self.shared.consumer_quiet.0.store(true, Ordering::Release);
            }
            self.shared.consumer_waiting.0.store(true, Ordering::SeqCst);
            self.published = self.shared.written.0.load(Ordering::SeqCst);
            // A wait target may have been installed since the last release.
            self.release(true);
            if self.position != self.published {
                self.shared
                    .consumer_waiting
                    .0
                    .store(false, Ordering::Relaxed);
                if self.ready() {
                    return Ok(true);
                }
                continue;
            }
            if stopped() || !self.shared.producer_alive.load(Ordering::SeqCst) {
                self.shared
                    .consumer_waiting
                    .0
                    .store(false, Ordering::Relaxed);
                return Ok(false);
            }
            if !requested() {
                thread::park();
            }
            self.shared
                .consumer_waiting
                .0
                .store(false, Ordering::Relaxed);
        }
    }
}
impl<T> Drop for Sender<T> {
    fn drop(&mut self) {
        self.flush();
        self.shared.producer_alive.store(false, Ordering::SeqCst);
        if let Some(thread) = self.shared.consumer.get() {
            thread.unpark();
        }
    }
}
impl<T> Drop for Receiver<T> {
    fn drop(&mut self) {
        self.position = self.record_end;
        self.release(true);
        self.shared.consumer_alive.store(false, Ordering::SeqCst);
        if let Some(thread) = self.shared.producer.get() {
            thread.unpark();
        }
    }
}
pub const METHOD: u64 = 0;
pub const LOCATION: u64 = 1;
pub const MACRO_END: u64 = 2;
pub const MULTI: u64 = 3;
pub const RAW: u64 = 4;
pub const ORIGIN: u64 = 5;
pub const ENGINE: u64 = 6;
pub const HOST: u64 = 7;
