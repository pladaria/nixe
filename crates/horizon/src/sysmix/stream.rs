//! Native output rings retain physical RAM; the guest owns write, the service owns read.
//! https://switchbrew.org/wiki/Audio_services#(S2)_audsmx:u
// @SPECULATIVE: ring offsets, stream formats and command semantics follow SDK
// observations in the reference. Public documentation does not describe them.
// https://github.com/zortos293/Lichii/blob/646b5d17135a2b2701ccc7a959afd0bbbf2f6160/src/core/hle/service/audio/nx2_sysmix.cpp
use super::{User, system::Clock};
use crate::{HorizonIpcResult as R, audio_device::AudioDeviceState};
use nixe_audio::{AudioBackend, AudioOutput, ChannelLayout, PcmFormat};
use nixe_memory::{
    Access, AccessResult, AllocationRange, CapturedMemory, GuestVirtualAddress, MemoryPermissions,
};
use nixe_runtime::EventObject;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering, fence},
};

const READ: u64 = 0x44;
const WRITE: u64 = 0x1004;
const PAYLOAD: u64 = 0x1040;
const FRAMES: usize = 240;
pub(crate) fn supported_format(kind: u32, channels: u32, rate: u32) -> bool {
    matches!(kind, 2 | 5) && matches!(channels, 2 | 6) && rate == 48_000
}
fn memory_error(error: impl std::fmt::Display) -> R {
    log::error!("native audio ring access failed: {error}");
    R::AUDIO_INVALID_ADDRESS_INFO
}
fn audio_error(error: nixe_audio::AudioError) -> R {
    log::error!("native audio output failed: {error}");
    R::AUDIO_OPERATION_FAILED
}
struct Open {
    output: AudioOutput,
    kind: u32,
    channels: usize,
    slots: u32,
    consumer: u32,
    engaged: bool,
    faulted: bool,
    mode: u32,
    threshold: u32,
    gain: f32,
    free: Vec<Vec<i16>>,
    allocated: usize,
    input: [u8; FRAMES * 6 * 4],
    accepted_frames: u64,
}
impl Open {
    fn notify(&self, filled: u32, event: &EventObject) {
        if self.mode == 2 && self.slots - 1 - filled >= self.threshold {
            event.signal();
        }
    }
}
pub(crate) struct Stream {
    pub(crate) id: u32,
    pub(crate) user: Arc<User>,
    backend: Arc<dyn AudioBackend>,
    devices: Arc<AudioDeviceState>,
    ring: CapturedMemory,
    read_word: Vec<AllocationRange>,
    size: u64,
    state: Mutex<Option<Open>>,
    name: Mutex<Box<str>>,
    cancelled: AtomicBool,
    clock: Arc<Clock>,
    pub(crate) state_event: EventObject,
    pub(crate) data_event: EventObject,
}
impl std::fmt::Debug for Stream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeAudioStream")
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}
#[derive(Debug)]
pub(crate) struct StreamSession {
    pub(crate) stream: Arc<Stream>,
}
impl Drop for StreamSession {
    fn drop(&mut self) {
        self.stream.close();
    }
}
impl Stream {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        id: u32,
        user: Arc<User>,
        backend: Arc<dyn AudioBackend>,
        devices: Arc<AudioDeviceState>,
        kernel: &Arc<nixe_scheduler::SchedulerState>,
        address: u64,
        size: u64,
        flags: u32,
        clock: Arc<Clock>,
    ) -> Result<Arc<Self>, R> {
        if flags != 0x24 {
            return Err(R::AUDIO_NOT_SUPPORTED);
        }
        if address == 0
            || !address.is_multiple_of(4096)
            || !size.is_multiple_of(4096)
            || size < 0x2000
            || address.checked_add(size).is_none()
        {
            return Err(R::AUDIO_INVALID_ADDRESS_INFO);
        }
        let ring = CapturedMemory::new(
            user.memory
                .space()
                .capture(
                    GuestVirtualAddress::new(address),
                    size,
                    MemoryPermissions::READ_WRITE,
                )
                .map_err(memory_error)?,
        )
        .map_err(memory_error)?;
        let read_word = ring.capture(READ, 4).map_err(memory_error)?;
        Ok(Arc::new(Self {
            id,
            user,
            backend,
            devices,
            ring,
            read_word,
            size,
            state: Mutex::new(None),
            name: Mutex::new("Native audio".into()),
            cancelled: AtomicBool::new(false),
            clock,
            state_event: EventObject::new(kernel),
            data_event: EventObject::new(kernel),
        }))
    }
    fn machine(&self) -> &nixe_memory::Machine {
        self.user.memory.account().machine()
    }
    async fn read_word(&self, offset: u64) -> Result<u32, R> {
        let mut bytes = [0; 4];
        if let Access::Pending(pending) = self
            .machine()
            .read_captured_into(&self.ring, offset, &mut bytes)
            .map_err(memory_error)?
        {
            match pending.await.map_err(memory_error)? {
                AccessResult::Read(data) if data.len() == 4 => bytes.copy_from_slice(&data),
                _ => return Err(R::AUDIO_INVALID_ADDRESS_INFO),
            }
        }
        // Captured aligned words use one native RAM load; acquire pairs it
        // with the guest producer's release publication of its payload.
        fence(Ordering::Acquire);
        Ok(u32::from_le_bytes(bytes))
    }
    pub(crate) async fn open(
        &self,
        kind: u32,
        channels: u32,
        rate: u32,
        frames: u32,
        prefill: u32,
        flags: u32,
    ) -> Result<(), R> {
        if !supported_format(kind, channels, rate) || prefill != 0 || flags != 0 {
            return Err(R::AUDIO_NOT_SUPPORTED);
        }
        if frames == 0 || frames >= i32::MAX as u32 {
            return Err(R::AUDIO_INVALID_ARGUMENT);
        }
        let slots = frames + 1;
        let frame_bytes = u64::from(channels) * if kind == 2 { 2 } else { 4 };
        let minimum = (0x1000
            + (0x40 + u64::from(slots.min(16)) * 8).next_multiple_of(64)
            + (u64::from(slots) * frame_bytes).next_multiple_of(64))
        .next_multiple_of(4096);
        if self.size < minimum || PAYLOAD + u64::from(slots) * frame_bytes > self.size {
            return Err(R::AUDIO_INSUFFICIENT_BUFFER);
        }
        let read = self.read_word(READ).await?;
        let write = self.read_word(WRITE).await?;
        if read >= slots || write >= slots {
            return Err(R::AUDIO_INVALID_ADDRESS_INFO);
        }
        let mut state = self.state.lock().unwrap();
        if state.is_some() {
            return Err(R::AUDIO_OPERATION_FAILED);
        }
        let output = AudioOutput::open(
            self.backend.as_ref(),
            PcmFormat {
                sample_rate: std::num::NonZeroU32::new(rate).unwrap(),
                layout: if channels == 6 {
                    ChannelLayout::Surround51
                } else {
                    ChannelLayout::Stereo
                },
            },
            // The 5 ms mixer tick recycles completed host blocks. Completion
            // does not correspond to a guest event or require another wakeup.
            Arc::new(|_| {}),
        )
        .map_err(audio_error)?;
        self.cancelled.store(false, Ordering::Release);
        self.data_event.clear();
        *state = Some(Open {
            output,
            kind,
            channels: channels as usize,
            slots,
            consumer: read,
            engaged: false,
            faulted: false,
            mode: 0,
            threshold: 0,
            gain: 1.0,
            free: Vec::new(),
            allocated: 0,
            input: [0; FRAMES * 6 * 4],
            accepted_frames: 0,
        });
        self.state_event.signal();
        log::info!(
            "native audio stream {} opened: name={:?}, sample-type={kind}, channels={channels}, sample-rate={rate}, ring-frames={slots}",
            self.id,
            self.name.lock().unwrap()
        );
        Ok(())
    }
    pub(crate) fn close(&self) {
        // Cancellation precedes the state lock, waking a worker that holds it
        // across a coherent GPU dependency. No guest thread or service teardown
        // has to wait for that dependency merely to close the stream.
        self.clock.cancel(&self.cancelled);
        if let Some(state) = self.state.lock().unwrap().take() {
            log::debug!(
                "native audio stream {} closed: accepted-frames={}",
                self.id,
                state.accepted_frames
            );
            drop(state);
        }
        self.state_event.signal();
    }
    pub(crate) fn engage(&self, engaged: bool) -> Result<(), R> {
        let paused = self.clock.paused();
        let mut state = self.state.lock().unwrap();
        let state = state
            .as_mut()
            .filter(|s| !s.faulted)
            .ok_or(R::AUDIO_OPERATION_FAILED)?;
        if engaged {
            state.output.start().map_err(audio_error)?;
        }
        state
            .output
            .set_host_paused(paused || !engaged)
            .map_err(audio_error)?;
        state.engaged = engaged;
        self.state_event.signal();
        self.clock.wake();
        Ok(())
    }
    pub(super) fn pause(&self, paused: bool) {
        if let Some(state) = self.state.lock().unwrap().as_mut()
            && let Err(error) = state.output.set_host_paused(paused || !state.engaged)
        {
            Self::fault(state, &self.state_event, audio_error(error));
        }
    }
    pub(crate) fn status(&self) -> u32 {
        self.state
            .lock()
            .unwrap()
            .as_ref()
            .filter(|s| !s.faulted)
            .map_or(0, |s| if s.engaged { 2 } else { 1 })
    }
    pub(crate) fn gain(&self) -> f32 {
        self.state.lock().unwrap().as_ref().map_or(1.0, |s| s.gain)
    }
    pub(crate) fn set_gain(&self, gain: f32, ramp: i64) -> Result<(), R> {
        if ramp != 0 {
            return Err(R::AUDIO_NOT_SUPPORTED);
        }
        if !gain.is_finite() || !(0.0..=128.0).contains(&gain) {
            return Err(R::AUDIO_INVALID_ARGUMENT);
        }
        self.state
            .lock()
            .unwrap()
            .as_mut()
            .ok_or(R::AUDIO_OPERATION_FAILED)?
            .gain = gain;
        Ok(())
    }
    pub(crate) fn rename(&self, name: Box<str>) {
        *self.name.lock().unwrap() = name;
    }
    pub(crate) async fn notify(&self, mode: u32, threshold: i32) -> Result<(), R> {
        if !matches!(mode, 0 | 2) {
            return Err(R::AUDIO_NOT_SUPPORTED);
        }
        loop {
            let (consumer, accepted) = self
                .state
                .lock()
                .unwrap()
                .as_ref()
                .filter(|s| !s.faulted)
                .map(|s| (s.consumer, s.accepted_frames))
                .ok_or(R::AUDIO_OPERATION_FAILED)?;
            let read = self.read_word(READ).await?;
            let write = self.read_word(WRITE).await?;
            let mut state = self.state.lock().unwrap();
            let state = state
                .as_mut()
                .filter(|s| !s.faulted)
                .ok_or(R::AUDIO_OPERATION_FAILED)?;
            // Coherent reads may yield while the mixer consumes a block.
            // Retry that snapshot instead of treating our own cursor advance
            // as an invalid guest write to the service-owned cursor.
            if state.accepted_frames != accepted {
                continue;
            }
            if threshold < 0 || threshold as u32 >= state.slots {
                return Err(R::AUDIO_INVALID_ARGUMENT);
            }
            if read != consumer || read >= state.slots || write >= state.slots {
                return Err(R::AUDIO_INVALID_ADDRESS_INFO);
            }
            state.mode = mode;
            state.threshold = threshold as u32;
            state.notify((write + state.slots - read) % state.slots, &self.data_event);
            return Ok(());
        }
    }
    fn read(&self, offset: u64, bytes: &mut [u8]) -> Result<(), R> {
        if let Access::Pending(pending) = self
            .machine()
            .read_captured_into(&self.ring, offset, bytes)
            .map_err(memory_error)?
        {
            match self.clock.wait(pending, &self.cancelled)? {
                AccessResult::Read(data) if data.len() == bytes.len() => {
                    bytes.copy_from_slice(&data)
                }
                _ => return Err(R::AUDIO_INVALID_ADDRESS_INFO),
            }
        }
        Ok(())
    }
    fn cursors(&self, state: &Open) -> Result<(u32, u32), R> {
        let mut read = [0; 4];
        let mut write = [0; 4];
        self.read(READ, &mut read)?;
        self.read(WRITE, &mut write)?;
        fence(Ordering::Acquire);
        let read = u32::from_le_bytes(read);
        let write = u32::from_le_bytes(write);
        if read != state.consumer || read >= state.slots || write >= state.slots {
            return Err(R::AUDIO_INVALID_ADDRESS_INFO);
        }
        Ok((read, (write + state.slots - read) % state.slots))
    }
    fn publish_read(&self, previous: u32, replacement: u32) -> Result<(), R> {
        let result = self
            .machine()
            .access_captured(
                &self.read_word,
                nixe_memory::AccessOperation::Atomic(nixe_memory::AtomicAccess {
                    bytes: 4,
                    ordering: nixe_memory::AccessOrdering::Release,
                    operation: nixe_memory::AtomicOperation::CompareExchange {
                        expected: u128::from(previous),
                        replacement: u128::from(replacement),
                    },
                }),
            )
            .map_err(memory_error)?;
        let result = match result {
            Access::Ready(r) => r,
            Access::Pending(p) => self.clock.wait(p, &self.cancelled)?,
        };
        match result {
            AccessResult::Atomic { written: true, .. } => Ok(()),
            _ => Err(R::AUDIO_INVALID_ADDRESS_INFO),
        }
    }
    fn fault(state: &mut Open, event: &EventObject, error: R) {
        state.faulted = true;
        state.engaged = false;
        let _ = state.output.set_host_paused(true);
        event.signal();
        log::error!("native audio stream fault: {error:?}");
    }
    pub(super) fn pump(&self) {
        let mut guard = match self.state.try_lock() {
            Ok(guard) => guard,
            Err(std::sync::TryLockError::WouldBlock) => return,
            Err(error) => panic!("native audio stream state poisoned: {error}"),
        };
        let Some(state) = guard.as_mut().filter(|s| !s.faulted) else {
            return;
        };
        if self.cancelled.load(Ordering::Acquire) {
            return;
        }
        if let Err(error) = self.pump_open(state)
            && !self.cancelled.load(Ordering::Acquire)
        {
            Self::fault(state, &self.state_event, error);
        }
    }
    fn pump_open(&self, state: &mut Open) -> Result<(), R> {
        let (mut read, mut filled) = self.cursors(state)?;
        state.notify(filled, &self.data_event);
        if !state.engaged {
            return Ok(());
        }
        state
            .output
            .set_volume(state.gain * self.devices.volume(crate::audio_device::ACTIVE_DEVICE))
            .map_err(audio_error)?;
        while let Some(samples) = state.output.recycle_buffer().map_err(audio_error)? {
            state.free.push(samples);
        }
        let limit = state
            .output
            .streaming_queue_blocks(FRAMES)
            .map_err(audio_error)?;
        while state.allocated < limit {
            state.free.push(Vec::with_capacity(FRAMES * state.channels));
            state.allocated += 1;
        }
        let queued = state.output.buffer_count().map_err(audio_error)?;
        for _ in queued..limit {
            if filled == 0 {
                break;
            }
            let Some(mut samples) = state.free.pop() else {
                break;
            };
            let taken = filled.min(FRAMES as u32);
            let first = taken.min(state.slots - read);
            let frame_bytes = state.channels * if state.kind == 2 { 2 } else { 4 };
            let split = first as usize * frame_bytes;
            self.read(
                PAYLOAD + u64::from(read) * frame_bytes as u64,
                &mut state.input[..split],
            )?;
            if first < taken {
                self.read(
                    PAYLOAD,
                    &mut state.input[split..taken as usize * frame_bytes],
                )?;
            }
            samples.resize(taken as usize * state.channels, 0);
            for (index, sample) in samples.iter_mut().enumerate() {
                *sample = if state.kind == 2 {
                    i16::from_le_bytes(state.input[index * 2..index * 2 + 2].try_into().unwrap())
                } else {
                    let value = f32::from_le_bytes(
                        state.input[index * 4..index * 4 + 4].try_into().unwrap(),
                    );
                    if value.is_finite() {
                        (value.clamp(-1.0, 1.0) * 32768.0).clamp(-32768.0, 32767.0) as i16
                    } else {
                        0
                    }
                };
            }
            // The shared host feed accepts short blocks. Keep actual samples;
            // only a real callback underrun supplies silence, without padding
            // a short guest tail between otherwise continuous blocks.
            state
                .output
                .append(state.accepted_frames, samples)
                .map_err(audio_error)?;
            let next = (read + taken) % state.slots;
            self.publish_read(read, next)?;
            state.consumer = next;
            state.accepted_frames += u64::from(taken);
            (read, filled) = self.cursors(state)?;
            state.notify(filled, &self.data_event);
        }
        Ok(())
    }
}
