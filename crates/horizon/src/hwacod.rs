//! Retained memory and decoder ownership for native batch audio.
//! https://switchbrew.org/wiki/Audio_services#Cmd100
//! @SPECULATIVE: manager/ring semantics follow the SDK-derived reference protocol.
//! https://github.com/zortos293/Lichii/blob/646b5d17135a2b2701ccc7a959afd0bbbf2f6160/src/core/hle/service/audio/hardware_audio_codec.cpp

mod protocol;
use crate::{
    HorizonIpcResult,
    hwopus::{BUFFER_TOO_SMALL, codec_error},
};
use nixe_audio::opus::Decoder;
use nixe_memory::{
    Access, AccessOperation, AccessOrdering, AccessResult, AtomicAccess, AtomicOperation,
    AtomicTransform, Machine, MemoryError,
};
use nixe_runtime::{EventObject, TransferMemoryObject};
pub(crate) use protocol::{Config, Layout, Task, manager_size, overlaps};
use std::sync::{Arc, Mutex};

pub(crate) const fn audio_result(description: u32) -> HorizonIpcResult {
    HorizonIpcResult::from_raw(153 | (description << 9))
}
pub(crate) const INVALID: HorizonIpcResult = audio_result(900);
pub(crate) const ADDRESS: HorizonIpcResult = audio_result(42);
pub(crate) const FAILED: HorizonIpcResult = audio_result(2);
const FULL: HorizonIpcResult = audio_result(8);

#[derive(Debug, Default)]
pub(crate) struct Tags(Mutex<std::collections::BTreeSet<u16>>);
#[derive(Debug)]
pub(crate) struct Tag {
    owner: Arc<Tags>,
    pub value: u16,
}
impl Tags {
    pub fn acquire(self: &Arc<Self>) -> Result<Tag, HorizonIpcResult> {
        let mut live = self.0.lock().unwrap();
        let value = (1..=u16::MAX)
            .find(|value| !live.contains(value))
            .ok_or(audio_result(5))?;
        live.insert(value);
        Ok(Tag {
            owner: self.clone(),
            value,
        })
    }
}
impl Drop for Tag {
    fn drop(&mut self) {
        self.owner.0.lock().unwrap().remove(&self.value);
    }
}

#[derive(Debug, Clone)]
pub struct AudioCodecSession {
    pub(crate) root: crate::hwopus::OpusObject,
    pub(crate) domain: crate::object::DomainSession<crate::hwopus::OpusObject>,
}
impl AudioCodecSession {
    pub(crate) fn new(root: crate::hwopus::OpusObject) -> Self {
        Self {
            root,
            domain: crate::object::DomainSession::new(),
        }
    }
}

#[derive(Debug)]
struct Instance {
    config: Config,
    decoder: Decoder,
}
#[derive(Debug, Clone, Copy)]
struct Lease {
    slot: u32,
    instance: u16,
}
#[derive(Debug)]
pub(crate) struct State {
    instances: Vec<Option<Instance>>,
    leases: Vec<Option<Lease>>,
    pub producer: u32,
    consumer: u32,
    outstanding: u32,
    faulted: bool,
}
#[derive(Debug)]
pub(crate) struct Manager {
    pub owner: u64,
    pub tag: Tag,
    pub layout: Layout,
    pub event: EventObject,
    pub publication: Arc<crate::ipc_wire::operation::Publication>,
    pub state: Mutex<State>,
    machine: Machine,
    io: Arc<TransferMemoryObject>,
    _work: Arc<TransferMemoryObject>,
}
impl Manager {
    pub fn new(
        owner: u64,
        tag: Tag,
        layout: Layout,
        event: EventObject,
        machine: Machine,
        io: Arc<TransferMemoryObject>,
        work: Arc<TransferMemoryObject>,
    ) -> Self {
        let state = State {
            instances: (0..layout.instance_count).map(|_| None).collect(),
            leases: vec![None; layout.io_count as usize],
            producer: 0,
            consumer: 0,
            outstanding: 0,
            faulted: false,
        };
        Self {
            owner,
            tag,
            layout,
            event,
            publication: Arc::default(),
            state: Mutex::new(state),
            machine,
            io,
            _work: work,
        }
    }
    pub async fn initialize(self: &Arc<Self>) -> Result<(), Arc<MemoryError>> {
        let ring = u64::from(self.layout.completion_offset);
        if self.atomic(ring + 0x44, None).await?.0 != 0 {
            return Err(Arc::new(MemoryError::Invalid(
                "audio completion consumer is not initially zero",
            )));
        }
        self.write(ring, vec![0; 64]).await?;
        self.write(
            ring + 0x80,
            vec![0; (self.layout.io_count as usize + 1) * 64],
        )
        .await
    }
    pub fn initialize_instance(&self, id: u32, config: Config) -> Result<(), HorizonIpcResult> {
        let mut s = self.state.lock().unwrap();
        if s.faulted {
            return Err(FAILED);
        }
        let instance = s.instances.get_mut(id as usize).ok_or(INVALID)?;
        if instance.is_some() {
            return Err(INVALID);
        }
        if config.required_bytes()? > self.layout.instance_stride
            || (config.frames * config.channels * 2).next_multiple_of(64)
                > self.layout.output_stride
        {
            return Err(BUFFER_TOO_SMALL);
        }
        let decoder = config.decoder()?;
        *instance = Some(Instance { config, decoder });
        Ok(())
    }
    pub fn finalize_instance(&self, id: u32) -> Result<(), HorizonIpcResult> {
        let mut s = self.state.lock().unwrap();
        if s.leases
            .iter()
            .flatten()
            .any(|lease| u32::from(lease.instance) == id)
        {
            return Err(FULL);
        }
        s.instances
            .get_mut(id as usize)
            .and_then(Option::take)
            .ok_or(INVALID)?;
        Ok(())
    }
    pub async fn reclaim(
        self: &Arc<Self>,
    ) -> Result<Result<(), HorizonIpcResult>, Arc<MemoryError>> {
        let ring = u64::from(self.layout.completion_offset);
        let consumer = self.atomic(ring + 0x44, None).await?.0;
        let producer = self.atomic(ring + 4, None).await?.0;
        let mut s = self.state.lock().unwrap();
        if s.faulted {
            return Ok(Err(FAILED));
        }
        let capacity = self.layout.io_count + 1;
        if consumer >= capacity || producer != s.producer {
            return Ok(Err(INVALID));
        }
        let consumed = (consumer + capacity - s.consumer) % capacity;
        if consumed > s.outstanding {
            return Ok(Err(INVALID));
        }
        let old = s.consumer;
        let reclaimed = |lease: &Lease| (lease.slot + capacity - old) % capacity < consumed;
        if s.leases
            .iter()
            .flatten()
            .filter(|lease| reclaimed(lease))
            .count()
            != consumed as usize
        {
            return Ok(Err(INVALID));
        }
        for lease in &mut s.leases {
            if lease.as_ref().is_some_and(reclaimed) {
                *lease = None;
            }
        }
        s.consumer = consumer;
        s.outstanding -= consumed;
        Ok(Ok(()))
    }
    pub fn validate(&self, tasks: &[Task]) -> Result<(), HorizonIpcResult> {
        let s = self.state.lock().unwrap();
        if s.faulted {
            return Err(FAILED);
        }
        if tasks.len() > (self.layout.io_count - s.outstanding) as usize {
            return Err(FULL);
        }
        let mut seen = vec![false; self.layout.io_count as usize];
        for task in tasks {
            if task.tag != self.tag.value
                || !s
                    .instances
                    .get(task.instance as usize)
                    .is_some_and(Option::is_some)
                || u32::from(task.io) >= self.layout.io_count
                || task.input_bytes > self.layout.input_stride
            {
                return Err(INVALID);
            }
            if s.leases[task.io as usize].is_some()
                || std::mem::replace(&mut seen[task.io as usize], true)
            {
                return Err(FULL);
            }
        }
        Ok(())
    }
    pub fn decode(
        &self,
        task: Task,
        packet: &[u8],
        pcm: &mut [i16],
    ) -> Result<usize, HorizonIpcResult> {
        let mut s = self.state.lock().unwrap();
        let instance = s.instances[task.instance as usize]
            .as_mut()
            .ok_or(INVALID)?;
        if packet.len() < 8 {
            return Err(crate::hwopus::INPUT_TOO_SMALL);
        }
        let bytes = u32::from_be_bytes(packet[..4].try_into().unwrap()) as usize;
        if bytes == 0 {
            return Err(HorizonIpcResult::from_raw(111 | (17 << 9)));
        }
        if bytes > packet.len() - 8 {
            return Err(BUFFER_TOO_SMALL);
        }
        let config = &instance.config;
        let samples = config.frames as usize * config.channels as usize;
        let capacity = samples.min(pcm.len());
        let frames = instance
            .decoder
            .decode(&packet[8..8 + bytes], &mut pcm[..capacity], task.reset)
            .map_err(codec_error)?;
        Ok(frames * config.channels as usize)
    }
    pub fn io_offset(&self, task: Task) -> u64 {
        u64::from(self.layout.io_offset)
            + u64::from(task.io)
                * (u64::from(self.layout.input_stride) + u64::from(self.layout.output_stride))
    }
    pub async fn commit(
        self: &Arc<Self>,
        task: Task,
        completion: Vec<u8>,
    ) -> Result<(), Arc<MemoryError>> {
        let current = self.state.lock().unwrap().producer;
        let next = (current + 1) % (self.layout.io_count + 1);
        let ring = u64::from(self.layout.completion_offset);
        self.write(ring + 0x80 + u64::from(current) * 64, completion)
            .await?;
        if !self.atomic(ring + 4, Some((current, next))).await?.1 {
            return Err(Arc::new(MemoryError::Invalid(
                "audio completion producer changed",
            )));
        }
        let mut s = self.state.lock().unwrap();
        s.leases[task.io as usize] = Some(Lease {
            slot: current,
            instance: task.instance,
        });
        s.producer = next;
        s.outstanding += 1;
        drop(s);
        if task.notify {
            self.event.signal();
        }
        Ok(())
    }
    pub fn fault(&self) {
        self.state.lock().unwrap().faulted = true;
        self.event.signal();
    }
    pub async fn read(
        self: &Arc<Self>,
        offset: u64,
        size: usize,
    ) -> Result<Vec<u8>, Arc<MemoryError>> {
        match self
            .machine
            .read_captured_owned(self.io.ranges(), offset, vec![0; size])
            .map_err(Arc::new)?
        {
            Access::Ready(bytes) => Ok(bytes),
            Access::Pending(wait) => {
                wait.retain_owner(self.clone());
                match wait.await? {
                    AccessResult::Read(bytes) => Ok(bytes),
                    _ => Err(Arc::new(MemoryError::Invalid("audio read completion"))),
                }
            }
        }
    }
    pub async fn write(
        self: &Arc<Self>,
        offset: u64,
        bytes: Vec<u8>,
    ) -> Result<(), Arc<MemoryError>> {
        match self
            .machine
            .write_captured_owned(self.io.ranges(), offset, bytes)
            .map_err(Arc::new)?
        {
            Access::Ready(_) => Ok(()),
            Access::Pending(wait) => {
                wait.retain_owner(self.clone());
                match wait.await? {
                    AccessResult::Written(_) => Ok(()),
                    _ => Err(Arc::new(MemoryError::Invalid("audio write completion"))),
                }
            }
        }
    }
    async fn atomic(
        self: &Arc<Self>,
        offset: u64,
        exchange: Option<(u32, u32)>,
    ) -> Result<(u32, bool), Arc<MemoryError>> {
        let ranges = self.io.ranges().capture(offset, 4).map_err(Arc::new)?;
        let operation = AtomicAccess {
            bytes: 4,
            ordering: if exchange.is_some() {
                AccessOrdering::Release
            } else {
                AccessOrdering::Acquire
            },
            operation: exchange.map_or(
                AtomicOperation::ReadModifyWrite {
                    transform: AtomicTransform::Add,
                    operand: 0,
                },
                |(expected, replacement)| AtomicOperation::CompareExchange {
                    expected: u128::from(expected),
                    replacement: u128::from(replacement),
                },
            ),
        };
        let result = match self
            .machine
            .access_captured(&ranges, AccessOperation::Atomic(operation))
            .map_err(Arc::new)?
        {
            Access::Ready(result) => result,
            Access::Pending(wait) => {
                wait.retain_owner(self.clone());
                wait.await?
            }
        };
        match result {
            AccessResult::Atomic { previous, written } => Ok((previous as u32, written)),
            _ => Err(Arc::new(MemoryError::Invalid("audio atomic completion"))),
        }
    }
}
