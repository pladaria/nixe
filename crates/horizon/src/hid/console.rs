//! Console motion resources, with the same owners and wire layout as Eden.
//! https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/hid_core/resources/six_axis/seven_six_axis.cpp
//! https://switchbrew.org/wiki/HID_services#InitializeSevenSixAxisSensor
use super::*;

mod motion;
pub(super) use motion::ConsoleMotion;

const CAPACITY: usize = 0x21;
const ENTRY_SIZE: usize = 0x50;

#[derive(Debug)]
pub(super) struct SevenSixAxis {
    pub active: bool,
    pub memory: Option<Arc<nixe_runtime::TransferMemoryObject>>,
    pub saved_timestamp: u64,
    pub global_timestamp: u64,
    pending: Option<(Arc<nixe_runtime::TransferMemoryObject>, PendingAccess)>,
    lifo: Vec<u8>,
    tail: usize,
    count: usize,
    sampling_number: u64,
}

impl Default for SevenSixAxis {
    fn default() -> Self {
        let mut lifo = vec![0; 0x20 + CAPACITY * ENTRY_SIZE];
        put_u64(&mut lifo, 8, CAPACITY as u64);
        Self {
            active: false,
            memory: None,
            saved_timestamp: 0,
            global_timestamp: 0,
            pending: None,
            lifo,
            tail: 0,
            count: 0,
            sampling_number: 0,
        }
    }
}

impl SevenSixAxis {
    pub fn poll(&mut self) -> Result<bool, HidInputError> {
        let Some((_, access)) = self.pending.as_mut() else {
            return Ok(true);
        };
        match access.poll() {
            None => Ok(false),
            Some(Err(error)) => Err(HidInputError::Memory(error)),
            Some(Ok(AccessResult::Written(bytes))) => {
                self.lifo = bytes;
                self.pending = None;
                Ok(true)
            }
            Some(Ok(_)) => Err(HidInputError::Memory(Arc::new(MemoryError::Execution(
                "invalid console motion write completion".into(),
            )))),
        }
    }

    pub fn publish(
        &mut self,
        state: &HidState,
        motion: &ConsoleMotion,
    ) -> Result<(), HidInputError> {
        if !self.active || self.memory.is_none() {
            self.count = 0;
            self.tail = 0;
            return Ok(());
        }
        let memory = self.memory.as_ref().unwrap();
        self.global_timestamp = state.kernel.clock().scheduler_time_ns();
        self.sampling_number = self.sampling_number.wrapping_add(1);
        self.tail = (self.tail + 1) % CAPACITY;
        self.count = (self.count + 1).min(CAPACITY - 1);
        let entry = &mut self.lifo[0x20 + self.tail * ENTRY_SIZE..][..ENTRY_SIZE];
        put_u64(entry, 0, self.sampling_number.wrapping_shl(1));
        put_u64(
            entry,
            0x10,
            self.global_timestamp.wrapping_sub(self.saved_timestamp),
        );
        put_u64(entry, 0x18, self.sampling_number);
        put_u64(entry, 0x20, 1);
        for (i, value) in motion.acceleration.into_iter().enumerate() {
            put_f32(entry, 0x28 + 4 * i, value);
        }
        for (i, value) in motion.gyroscope.into_iter().enumerate() {
            put_f32(entry, 0x34 + 4 * i, value);
        }
        let [x, y, z, w] = motion.quaternion;
        for (i, value) in [y, x, -w, -z].into_iter().enumerate() {
            put_f32(entry, 0x40 + 4 * i, value);
        }
        put_u64(&mut self.lifo, 0x10, self.tail as u64);
        put_u64(&mut self.lifo, 0x18, self.count as u64);
        let bytes = std::mem::take(&mut self.lifo);
        match memory.ranges()[0]
            .allocation()
            .machine()
            .write_captured_owned(memory.ranges(), 0, bytes)
            .map_err(|error| HidInputError::Memory(Arc::new(error)))?
        {
            Access::Ready(bytes) => self.lifo = bytes,
            Access::Pending(mut access) => {
                state.watch_publication(&mut access);
                self.pending = Some((memory.clone(), access));
            }
        }
        Ok(())
    }
}
