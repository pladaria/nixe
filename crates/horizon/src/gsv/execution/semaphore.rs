//! Channel acquires retain physical backing and latch matching GPU writes.
//! https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/video_core/nx2.h
use super::*;
use nixe_gpu::physical::Footprint;
use nixe_memory::ArenaOffset;

pub(super) struct Wait {
    pub acquire: ampere::SemaphoreAcquire,
    capture: Arc<GpuCapture>,
    physical: u64,
    latched: AtomicBool,
    thread: thread::Thread,
}
impl Wait {
    pub fn ready(&self, memory: &Memory<'_>) -> Result<bool> {
        if self.latched.load(Ordering::Acquire) {
            return Ok(true);
        }
        let bytes = memory.captured_read(&self.capture, vec![0; 4])?;
        Ok(self
            .acquire
            .matches(u32::from_le_bytes(bytes.try_into().unwrap()))
            || self.latched.load(Ordering::Acquire))
    }
}

#[derive(Default)]
pub(super) struct Waiters {
    active: AtomicBool,
    slots: Mutex<BTreeMap<u64, Arc<Wait>>>,
}
impl Waiters {
    pub fn active(&self) -> bool {
        self.active.load(Ordering::Acquire)
    }
    pub fn register(
        &self,
        acquire: ampere::SemaphoreAcquire,
        memory: &Memory<'_>,
    ) -> Result<Arc<Wait>> {
        let capture = Arc::new(
            GpuMemory::new(memory.space)
                .capture(acquire.address, 4)
                .map_err(|e| acquire.source.error(e))?,
        );
        let physical = capture.ranges[0]
            .physical_extents()
            .next()
            .filter(|extent| extent.bytes == 4)
            .ok_or("semaphore word is not physically contiguous")?
            .start
            .get();
        let wait = Arc::new(Wait {
            acquire,
            capture,
            physical,
            latched: AtomicBool::new(false),
            thread: thread::current(),
        });
        self.slots
            .lock()
            .unwrap()
            .insert(acquire.source.channel, wait.clone());
        self.active.store(true, Ordering::Release);
        Ok(wait)
    }
    pub fn remove(&self, channel: u64) {
        let mut slots = self.slots.lock().unwrap();
        slots.remove(&channel);
        self.active.store(!slots.is_empty(), Ordering::Release);
    }
    pub fn written(
        &self,
        capture: &nixe_memory::CapturedMemory,
        offset: u64,
        bytes: &[u8],
        mut read: impl FnMut(&GpuCapture) -> Result<Vec<u8>>,
    ) -> Result<()> {
        if !self.active.load(Ordering::Acquire) {
            return Ok(());
        }
        let waiters: Vec<_> = self.slots.lock().unwrap().values().cloned().collect();
        let ranges = capture
            .capture(offset, bytes.len() as u64)
            .map_err(|e| e.to_string().into_boxed_str())?;
        let mut at = 0;
        for extent in ranges.iter().flat_map(|range| range.physical_extents()) {
            let start = extent.start.get();
            let end = start + extent.bytes;
            for wait in &waiters {
                if wait.latched.load(Ordering::Relaxed)
                    || wait.physical >= end
                    || wait.physical + 4 <= start
                {
                    continue;
                }
                let value = if wait.physical >= start && wait.physical + 4 <= end {
                    let index = at + (wait.physical - start) as usize;
                    u32::from_le_bytes(bytes[index..index + 4].try_into().unwrap())
                } else {
                    // A partial DMA write still changes the complete semaphore word.
                    let bytes = read(&wait.capture)?;
                    u32::from_le_bytes(bytes.try_into().unwrap())
                };
                if wait.acquire.matches(value) {
                    wait.latched.store(true, Ordering::Release);
                    wait.thread.unpark();
                }
            }
            at += extent.bytes as usize;
        }
        Ok(())
    }

    /// Native copies keep pixels on the GPU. Only an overlapping semaphore
    /// needs a coherent read now, before a subsequent producer can overwrite it.
    pub fn written_gpu(
        &self,
        destination: &Footprint,
        mut read: impl FnMut(&GpuCapture) -> Result<Vec<u8>>,
    ) -> Result<()> {
        if !self.active() {
            return Ok(());
        }
        let mut affected = smallvec::SmallVec::<[_; 4]>::new();
        {
            let slots = self.slots.lock().unwrap();
            for wait in slots.values() {
                if wait.latched.load(Ordering::Relaxed) {
                    continue;
                }
                let mut intersects = false;
                for (index, span) in destination.mapping().spans().iter().enumerate() {
                    if span.start.get() >= wait.physical + 4
                        || span.start.get() + span.bytes <= wait.physical
                    {
                        continue;
                    }
                    destination
                        .visit(index, ArenaOffset::new(wait.physical), 4, |_| {
                            intersects = true;
                            false
                        })
                        .map_err(|e| e.to_string().into_boxed_str())?;
                    if intersects {
                        break;
                    }
                }
                if intersects {
                    affected.push(wait.clone());
                }
            }
        }
        for wait in affected {
            let bytes = read(&wait.capture)?;
            if wait
                .acquire
                .matches(u32::from_le_bytes(bytes.try_into().unwrap()))
            {
                wait.latched.store(true, Ordering::Release);
                wait.thread.unpark();
            }
        }
        Ok(())
    }
}
