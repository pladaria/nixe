use super::command::Observer;
use nixe_gpu::native::NativeWork;
use nixe_gpu::{FrontendSubmissionId, GuestSyncpointId};
use nixe_gpu_maxwell::*;
use nixe_memory::{Allocation, MemoryPermissions, PAGE_SIZE};
use nixe_memory::{AllocationRange, Machine};

pub struct Fixture {
    pub memory: Machine,
    pub allocation: Allocation,
    pub space: MaxwellGpuAddressSpace,
    pub channel: MaxwellGpuChannel,
    pub dispatch: MaxwellFrontendDispatch,
    pub frontend: std::cell::RefCell<MaxwellNativeFrontend>,
    pub scratch: MaxwellCommandSourceScratch,
}
impl Fixture {
    pub fn new(build: impl FnOnce(u64) -> Vec<Vec<u32>>) -> Self {
        Self::with_completion(4, None, build)
    }
    pub fn with_completion(
        flags: u32,
        completion: Option<nixe_gpu::ReservedTimelinePoint>,
        build: impl FnOnce(u64) -> Vec<Vec<u32>>,
    ) -> Self {
        let memory = Machine::new(64 * PAGE_SIZE).unwrap();
        let allocation = memory
            .account(64 * PAGE_SIZE)
            .unwrap()
            .allocate(64 * PAGE_SIZE)
            .unwrap();
        let input = AllocationRange::new(allocation.pin(), 0, 64 * PAGE_SIZE).unwrap();
        let mut space = MaxwellGpuAddressSpace::new(
            memory.clone(),
            MaxwellAddressSpaceId::new(1),
            SWITCH_1_GM20B_PROFILE,
        );
        space
            .initialize(MaxwellAddressSpaceInitialization::default())
            .unwrap();
        let mapping = space
            .map(MaxwellMapRequest {
                allocation: MaxwellAllocationId::new(1),
                backing: vec![input.clone()],
                backing_offset: 0,
                size: 64 * PAGE_SIZE,
                allocation_alignment: PAGE_SIZE as u32,
                page_size: 0,
                kind: 0,
                cacheable: false,
                permissions: MemoryPermissions::READ_WRITE,
                fixed_offset: None,
            })
            .unwrap();
        let base = mapping.offset().get();
        let entries = build(base);
        let mut descriptors = Vec::new();
        let mut offset = PAGE_SIZE;
        for words in &entries {
            let bytes: Vec<_> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
            memory
                .write_resolved(&[input.slice(offset, bytes.len() as u64).unwrap()], &bytes)
                .unwrap();
            let pointer = base + offset;
            descriptors.extend_from_slice(&(pointer as u32).to_le_bytes());
            descriptors.extend_from_slice(
                &(((pointer >> 32) as u32) | ((words.len() as u32) << 10)).to_le_bytes(),
            );
            offset += bytes.len() as u64;
        }
        let mut channel = MaxwellGpuChannel::new(
            MaxwellChannelId::new(1),
            MaxwellChannelOwner::new(1),
            SWITCH_1_GM20B_PROFILE,
        );
        channel.bind_address_space(space.memory().id()).unwrap();
        channel
            .bind_memory_manager(MaxwellMemoryManagerId::new(1))
            .unwrap();
        channel
            .allocate_gpfifo(128, false, GuestSyncpointId::new(1))
            .unwrap();
        let decoded = decode_gpfifo_submission(
            SWITCH_1_GM20B_PROFILE,
            128,
            MaxwellGpfifoSubmitRequest {
                entry_count: entries.len() as u32,
                flags,
                fence_id: 0,
                fence_value: completion
                    .as_ref()
                    .map_or(0, nixe_gpu::ReservedTimelinePoint::increments),
            },
            &descriptors,
        )
        .unwrap();
        let submission = validate_gpfifo_submission(
            &channel.configuration(),
            FrontendSubmissionId::new(1),
            decoded,
            space.memory(),
        )
        .unwrap();
        let mut scheduler = MaxwellScheduler::default();
        scheduler
            .enqueue(&channel.configuration(), submission, None, completion)
            .unwrap();
        let dispatch = scheduler
            .dispatch_next(true, space.memory())
            .unwrap()
            .unwrap();
        Self {
            memory,
            allocation,
            space,
            channel,
            dispatch,
            frontend: std::cell::RefCell::new(MaxwellNativeFrontend::new(Default::default())),
            scratch: Default::default(),
        }
    }
    pub fn run(
        &mut self,
        consume: &mut dyn FnMut(NativeWork),
    ) -> Result<usize, MaxwellFrontendDispatchError> {
        let mut host = Observer {
            memory: &self.memory,
            consume,
            controls: 0,
        };
        super::command::ready(stream_maxwell_frontend(
            &self.dispatch,
            &mut self.channel,
            self.space.memory(),
            &self.frontend,
            &mut self.scratch,
            &mut host,
        ))?;
        Ok(host.controls)
    }
    pub fn read(&self, offset: u64, size: usize) -> Vec<u8> {
        let mut bytes = vec![0; size];
        self.memory
            .capture_stream(
                &[AllocationRange::new(self.allocation.pin(), offset, size as u64).unwrap()],
                &mut bytes,
            )
            .unwrap();
        bytes
    }
}
pub fn header(mode: u32, method: u32, count: u32) -> u32 {
    (mode << 29) | (count << 16) | (method / 4)
}
pub fn packet(words: &mut Vec<u32>, method: u32, args: &[u32]) {
    words.push(header(1, method, args.len() as u32));
    words.extend_from_slice(args);
}
