//! Bounded domain code cache and executor-local lookup acceleration.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque, hash_map::DefaultHasher};
use std::hash::{Hash, Hasher};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError, Weak};

use nixe_cpu::error::{FrontendError, FrontendInternalError};
use nixe_cpu::ir::region::IrRegion;
use nixe_cpu::location::{ExecutionState, LocationDescriptor};
use nixe_cpu::memory::{CodePageDependency, CpuMemory};
use nixe_cpu::translate::RegionTranslationConfig;
use nixe_memory::{
    AddressSpaceId, GuestPhysicalPageId, GuestVirtualAddress, MappingGeneration,
    MemoryInvalidation, MemoryInvalidationCursor, MemoryInvalidationKind,
};

use crate::abi::{EXECUTION_STATE_A32, EXECUTION_STATE_A64, EXECUTION_STATE_T32, ExecutionFrame};
use crate::compiler::{CompiledRegion, CompilerError};
use crate::configuration::JitConfiguration;
use crate::links::{INDIRECT_LINK_WAYS, LinkKind, LinkTable, NativeLinkTarget};
use crate::performance::CachePerformanceSnapshot;

const DEFAULT_MAX_LIVE_IR_OPERATIONS: u64 = 32 * 1024 * 1024;
const LOCAL_LOOKUP_SLOTS: usize = 64;
const QUIESCENT_EPOCH: u64 = 0;
const COMPILATION_ACTIVE: u8 = 0;
const COMPILATION_STALE: u8 = 1;
const COMPILATION_STOPPED: u8 = 2;

type RegionId = u64;

// Ryujinx likewise gives LCQ code a native entry counter and requests an HCQ
// replacement after the counter reaches its threshold:
// https://git.axenov.dev/Museum/ryujinx/src/commit/b5cf8b8af9a8316a32322c8a7dc6d1798490e241/ARMeilleure/Translation/Translator.cs
pub(crate) const HOT_PROMOTION_ENTRIES: u32 = 100;
pub(crate) const PROMOTION_COUNTING: u32 = 0;
const PROMOTION_QUEUED: u32 = 1;
const PROMOTION_COMPILING: u32 = 2;
const PROMOTION_OPTIMIZED: u32 = 3;
const PROMOTION_STALE: u32 = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CodeTier {
    Light,
    Optimized,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LinkOutcome {
    Published,
    AlreadyPresent,
    /// The PIC was full and a bounded round-robin replacement was published.
    PicFull,
}

/// Stable state addressed directly by light-tier native code.
#[repr(C, align(8))]
pub(crate) struct PromotionCell {
    pub(crate) entries: AtomicU32,
    pub(crate) state: AtomicU32,
    region_id: AtomicU64,
    cancellation: AtomicU8,
}

impl PromotionCell {
    pub(crate) fn new() -> Self {
        Self {
            entries: AtomicU32::new(0),
            state: AtomicU32::new(PROMOTION_COUNTING),
            region_id: AtomicU64::new(0),
            cancellation: AtomicU8::new(COMPILATION_ACTIVE),
        }
    }

    pub(crate) fn install_region(&self, region_id: RegionId) {
        self.region_id.store(region_id, Ordering::Release);
    }

    fn queue(&self) -> bool {
        self.state
            .compare_exchange(
                PROMOTION_COUNTING,
                PROMOTION_QUEUED,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }

    pub(crate) fn begin_compilation(&self) -> bool {
        self.state
            .compare_exchange(
                PROMOTION_QUEUED,
                PROMOTION_COMPILING,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }

    pub(crate) fn rearm(&self) {
        if self
            .state
            .compare_exchange(
                PROMOTION_QUEUED,
                PROMOTION_COUNTING,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
        {
            self.entries
                .store(HOT_PROMOTION_ENTRIES - 1, Ordering::Release);
        }
    }

    fn mark_optimized(&self) {
        self.state.store(PROMOTION_OPTIMIZED, Ordering::Release);
    }

    pub(crate) fn mark_stale(&self) {
        self.cancellation
            .store(COMPILATION_STALE, Ordering::Release);
        self.state.store(PROMOTION_STALE, Ordering::Release);
    }

    pub(crate) fn cancellation(&self) -> CompilationCancellation<'_> {
        CompilationCancellation {
            state: &self.cancellation,
        }
    }
}

pub(crate) const PROMOTION_ENTRIES_OFFSET: usize = core::mem::offset_of!(PromotionCell, entries);
pub(crate) const PROMOTION_STATE_OFFSET: usize = core::mem::offset_of!(PromotionCell, state);

/// Frontend policy which affects generated semantics and therefore cache identity.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum TranslationMode {
    Baseline,
}

impl TranslationMode {
    pub(crate) fn config(self) -> RegionTranslationConfig {
        match self {
            Self::Baseline => RegionTranslationConfig::default(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct RegionKey {
    address_space: AddressSpaceId,
    location: LocationDescriptor,
    translation_mode: TranslationMode,
    root_code_mapping: CodePageDependency,
}

impl RegionKey {
    pub(crate) const fn new(
        address_space: AddressSpaceId,
        location: LocationDescriptor,
        translation_mode: TranslationMode,
        root_code_mapping: CodePageDependency,
    ) -> Self {
        Self {
            address_space,
            location,
            translation_mode,
            root_code_mapping,
        }
    }
}

pub(crate) fn root_code_mapping(
    memory: &dyn CpuMemory,
    address_space: AddressSpaceId,
    location: LocationDescriptor,
) -> Result<CodePageDependency, FrontendError> {
    let dependencies = match location.execution_state {
        ExecutionState::A64 | ExecutionState::A32 => {
            memory
                .fetch32(address_space, location.pc)
                .map_err(FrontendError::InstructionFetch)?
                .dependencies
        }
        ExecutionState::T32 => {
            memory
                .fetch16(address_space, location.pc)
                .map_err(FrontendError::InstructionFetch)?
                .dependencies
        }
    };
    dependencies.iter().next().ok_or_else(|| {
        FrontendError::Internal(FrontendInternalError::new(
            None,
            "instruction fetch returned no root code dependency",
        ))
    })
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct MappingDependency {
    location: LocationDescriptor,
    dependency: CodePageDependency,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct MappingIndexKey {
    address_space: AddressSpaceId,
    location: GuestVirtualAddress,
    generation: MappingGeneration,
}

#[derive(Debug)]
struct RegionIdentity {
    keys: Box<[RegionKey]>,
    code_dependencies: Box<[CodePageDependency]>,
    mapping_dependencies: Box<[MappingDependency]>,
}

pub(crate) struct PendingRegion {
    compiled: CompiledRegion,
    identity: RegionIdentity,
    ir_operations: u64,
    tier: CodeTier,
    source_ir: Option<Arc<IrRegion>>,
    promotion: Option<Arc<PromotionCell>>,
}

impl PendingRegion {
    pub(crate) fn new(
        address_space: AddressSpaceId,
        translation_mode: TranslationMode,
        region: Arc<IrRegion>,
        compiled: CompiledRegion,
        tier: CodeTier,
        promotion: Option<Arc<PromotionCell>>,
    ) -> Result<Self, CacheError> {
        let mut keys = Vec::with_capacity(region.metadata.entries.len());
        for entry in &region.metadata.entries {
            let block = region.block(entry.block).ok_or_else(|| {
                CacheError::Internal("region entry references a missing block".into())
            })?;
            let root_code_mapping = block
                .metadata
                .sources
                .first()
                .and_then(|source| source.dependencies.iter().next())
                .ok_or_else(|| {
                    CacheError::Internal("region entry has no root code dependency".into())
                })?;
            keys.push(RegionKey::new(
                address_space,
                entry.location,
                translation_mode,
                root_code_mapping,
            ));
        }

        let mut mapping_dependencies = Vec::new();
        for block in &region.blocks {
            for source in &block.metadata.sources {
                for dependency in source.dependencies.iter() {
                    let dependency = MappingDependency {
                        location: source.location,
                        dependency,
                    };
                    if !mapping_dependencies.contains(&dependency) {
                        mapping_dependencies.push(dependency);
                    }
                }
            }
        }

        Ok(Self {
            compiled,
            identity: RegionIdentity {
                keys: keys.into_boxed_slice(),
                code_dependencies: region.metadata.code_dependencies.clone(),
                mapping_dependencies: mapping_dependencies.into_boxed_slice(),
            },
            ir_operations: u64::from(region.metadata.ir_operation_count),
            tier,
            source_ir: (tier == CodeTier::Light).then_some(region),
            promotion,
        })
    }
}

pub(crate) struct CachedRegion {
    id: RegionId,
    live: AtomicBool,
    compiled: CompiledRegion,
    identity: RegionIdentity,
    ir_operations: u64,
    links: LinkTable,
    tier: CodeTier,
    source_ir: Option<Arc<IrRegion>>,
    promotion: Option<Arc<PromotionCell>>,
}

impl CachedRegion {
    pub(crate) const fn id(&self) -> RegionId {
        self.id
    }

    pub(crate) fn compiled(&self) -> &CompiledRegion {
        &self.compiled
    }

    fn is_live(&self) -> bool {
        self.live.load(Ordering::Acquire)
    }

    pub(crate) fn install_dispatch(&self, frame: &mut ExecutionFrame) {
        frame.dispatch.link_table = self.links.base_address();
        frame.dispatch.metadata = std::ptr::from_ref(&self.compiled.metadata).addr();
        frame.dispatch.region_id = self.id;
        frame.dispatch.retired = 0;
    }
}

pub(crate) struct PromotionRequest {
    pub(crate) key: RegionKey,
    pub(crate) baseline_id: RegionId,
    pub(crate) address_space: AddressSpaceId,
    pub(crate) mode: TranslationMode,
    pub(crate) region: Arc<IrRegion>,
    pub(crate) promotion: Arc<PromotionCell>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum CacheError {
    Frontend(FrontendError),
    Compiler(CompilerError),
    Capacity(Box<str>),
    Internal(Box<str>),
    Stale,
    Cancelled,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct InvalidationEffect {
    pub(crate) retired_regions: u64,
}

impl From<FrontendError> for CacheError {
    fn from(error: FrontendError) -> Self {
        Self::Frontend(error)
    }
}

impl From<CompilerError> for CacheError {
    fn from(error: CompilerError) -> Self {
        match error.cancellation_reason() {
            Some(CompilationCancellationReason::Stale) => Self::Stale,
            Some(CompilationCancellationReason::Stopped) => Self::Cancelled,
            None => Self::Compiler(error),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct CacheLimits {
    max_live_segments: usize,
    max_live_mapped_bytes: usize,
    max_live_ir_operations: u64,
    max_concurrent_compilations: usize,
}

impl CacheLimits {
    const fn from_configuration(configuration: &JitConfiguration) -> Self {
        Self {
            max_live_segments: configuration.max_cached_regions(),
            max_live_mapped_bytes: configuration.max_cache_bytes(),
            max_live_ir_operations: DEFAULT_MAX_LIVE_IR_OPERATIONS,
            max_concurrent_compilations: configuration.max_concurrent_compilations(),
        }
    }
}

enum CacheSlot {
    Ready(RegionId),
    Compiling(Arc<CompilationFlight>),
}

struct CompilationFlight {
    invalidation_revision: u64,
    cancellation: AtomicU8,
    result: Mutex<Option<Result<Arc<CachedRegion>, CacheError>>>,
    ready: Condvar,
}

impl CompilationFlight {
    fn new(invalidation_revision: u64) -> Self {
        Self {
            invalidation_revision,
            cancellation: AtomicU8::new(COMPILATION_ACTIVE),
            result: Mutex::new(None),
            ready: Condvar::new(),
        }
    }

    fn complete(&self, result: Result<Arc<CachedRegion>, CacheError>) {
        let mut state = lock(&self.result);
        if state.is_none() {
            *state = Some(result);
            self.ready.notify_all();
        }
    }

    fn cancel(&self, error: CacheError) {
        let reason = match error {
            CacheError::Stale => COMPILATION_STALE,
            CacheError::Cancelled => COMPILATION_STOPPED,
            _ => unreachable!("only terminal cache changes cancel compilation"),
        };
        let _ = self.cancellation.compare_exchange(
            COMPILATION_ACTIVE,
            reason,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
        self.complete(Err(error));
    }

    fn cancellation(&self) -> CompilationCancellation<'_> {
        CompilationCancellation {
            state: &self.cancellation,
        }
    }

    fn wait(&self) -> Result<Arc<CachedRegion>, CacheError> {
        let mut state = lock(&self.result);
        while state.is_none() {
            state = self
                .ready
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner);
        }
        state.clone().expect("single-flight result became ready")
    }
}

/// Cooperative cancellation observed only by the JIT frontend/compiler work
/// owned by one cache miss. It never enters the engine-neutral contract.
#[derive(Clone, Copy)]
pub(crate) struct CompilationCancellation<'a> {
    state: &'a AtomicU8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CompilationCancellationReason {
    Stale,
    Stopped,
}

impl CompilationCancellation<'_> {
    #[must_use]
    pub(crate) fn reason(self) -> Option<CompilationCancellationReason> {
        match self.state.load(Ordering::Acquire) {
            COMPILATION_ACTIVE => None,
            COMPILATION_STALE => Some(CompilationCancellationReason::Stale),
            COMPILATION_STOPPED => Some(CompilationCancellationReason::Stopped),
            _ => unreachable!("compilation cancellation state is private and validated"),
        }
    }

    pub(crate) fn check(self) -> Result<(), CacheError> {
        match self.state.load(Ordering::Acquire) {
            COMPILATION_ACTIVE => Ok(()),
            COMPILATION_STALE => Err(CacheError::Stale),
            COMPILATION_STOPPED => Err(CacheError::Cancelled),
            _ => unreachable!("compilation cancellation state is private and validated"),
        }
    }
}

struct CacheState {
    accepting_compilation: bool,
    slots: HashMap<RegionKey, CacheSlot>,
    regions: HashMap<RegionId, Arc<CachedRegion>>,
    retirement_order: VecDeque<RegionId>,
    physical_index: HashMap<GuestPhysicalPageId, BTreeSet<RegionId>>,
    mapping_index: BTreeMap<MappingIndexKey, BTreeSet<RegionId>>,
    next_region_id: RegionId,
    live_mapped_bytes: usize,
    live_ir_operations: u64,
    invalidation_revision: u64,
    invalidation_cursor: MemoryInvalidationCursor,
    active_compilations: usize,
    compiling_slots: usize,
    active_links: HashMap<LinkRef, ActiveLink>,
    incoming_links: HashMap<RegionId, BTreeSet<LinkRef>>,
    link_targets: HashMap<usize, Arc<NativeLinkTarget>>,
    retirement_epoch: u64,
    executors: Vec<Weak<AtomicU64>>,
    retired: VecDeque<RetiredBatch>,
    background_failure: Option<CacheError>,
    performance: Option<CachePerformanceState>,
}

struct CachePerformanceState {
    snapshot: CachePerformanceSnapshot,
    builds_by_key: HashMap<RegionKey, u64>,
}

#[derive(Clone, Copy, Default)]
struct InvalidationRecordCounts {
    executable_content: u64,
    mapping: u64,
    instruction_cache: u64,
}

impl CachePerformanceState {
    fn new() -> Self {
        Self {
            snapshot: CachePerformanceSnapshot::default(),
            builds_by_key: HashMap::new(),
        }
    }

    fn record_build(&mut self, key: RegionKey) {
        self.snapshot.global_builds = self.snapshot.global_builds.saturating_add(1);
        let builds = self.builds_by_key.entry(key).or_default();
        if *builds == 0 {
            self.snapshot.unique_region_keys = self.snapshot.unique_region_keys.saturating_add(1);
        } else {
            self.snapshot.repeat_builds = self.snapshot.repeat_builds.saturating_add(1);
            if *builds == 1 {
                self.snapshot.recompiled_region_keys =
                    self.snapshot.recompiled_region_keys.saturating_add(1);
            }
        }
        *builds = builds.saturating_add(1);
    }

    fn record_invalidation(
        &mut self,
        records: InvalidationRecordCounts,
        history_lost: bool,
        retired_regions: u64,
        links_unlinked: u64,
        fast_irrelevant: bool,
    ) {
        let snapshot = &mut self.snapshot;
        snapshot.invalidation_batches = snapshot.invalidation_batches.saturating_add(1);
        snapshot.invalidation_executable_content = snapshot
            .invalidation_executable_content
            .saturating_add(records.executable_content);
        snapshot.invalidation_mapping = snapshot
            .invalidation_mapping
            .saturating_add(records.mapping);
        snapshot.invalidation_instruction_cache = snapshot
            .invalidation_instruction_cache
            .saturating_add(records.instruction_cache);
        snapshot.invalidation_history_lost = snapshot
            .invalidation_history_lost
            .saturating_add(u64::from(history_lost));
        snapshot.regions_retired_invalidation = snapshot
            .regions_retired_invalidation
            .saturating_add(retired_regions);
        snapshot.links_unlinked_invalidation = snapshot
            .links_unlinked_invalidation
            .saturating_add(links_unlinked);
        snapshot.fast_irrelevant_invalidations = snapshot
            .fast_irrelevant_invalidations
            .saturating_add(u64::from(fast_irrelevant));
    }
}

struct RetiredBatch {
    epoch: u64,
    regions: Vec<Arc<CachedRegion>>,
    link_targets: Vec<Arc<NativeLinkTarget>>,
}

impl RetiredBatch {
    fn new(epoch: u64) -> Self {
        Self {
            epoch,
            regions: Vec::new(),
            link_targets: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct LinkRef {
    source: RegionId,
    site: u32,
    way: u8,
}

#[derive(Clone, Copy, Debug)]
struct ActiveLink {
    target: RegionId,
    target_address: usize,
    location: LocationDescriptor,
}

impl CacheState {
    fn new(performance_enabled: bool) -> Self {
        Self {
            accepting_compilation: true,
            slots: HashMap::new(),
            regions: HashMap::new(),
            retirement_order: VecDeque::new(),
            physical_index: HashMap::new(),
            mapping_index: BTreeMap::new(),
            next_region_id: 1,
            live_mapped_bytes: 0,
            live_ir_operations: 0,
            invalidation_revision: 0,
            invalidation_cursor: MemoryInvalidationCursor::INITIAL,
            active_compilations: 0,
            compiling_slots: 0,
            active_links: HashMap::new(),
            incoming_links: HashMap::new(),
            link_targets: HashMap::new(),
            retirement_epoch: 1,
            executors: Vec::new(),
            retired: VecDeque::new(),
            background_failure: None,
            performance: performance_enabled.then(CachePerformanceState::new),
        }
    }

    fn next_retirement_epoch(&mut self) -> Result<u64, CacheError> {
        self.retirement_epoch = self.retirement_epoch.checked_add(1).ok_or_else(|| {
            CacheError::Capacity("domain cache retirement epoch is exhausted".into())
        })?;
        Ok(self.retirement_epoch)
    }

    fn retired_batch(&mut self, epoch: u64) -> &mut RetiredBatch {
        if self.retired.back().is_none_or(|batch| batch.epoch != epoch) {
            self.retired.push_back(RetiredBatch::new(epoch));
        }
        self.retired
            .back_mut()
            .expect("retirement batch was inserted")
    }

    fn retire(&mut self, id: RegionId, epoch: u64) {
        let Some(region) = self.regions.get(&id).cloned() else {
            return;
        };
        if let Some(promotion) = &region.promotion {
            promotion.mark_stale();
        }
        self.unlink_region(id, epoch);
        region.live.store(false, Ordering::Release);
        self.regions.remove(&id);
        self.slots
            .retain(|_, slot| !matches!(slot, CacheSlot::Ready(region_id) if *region_id == id));
        for dependency in &region.identity.code_dependencies {
            remove_index(&mut self.physical_index, dependency.page, id);
        }
        for dependency in &region.identity.mapping_dependencies {
            remove_ordered_index(
                &mut self.mapping_index,
                MappingIndexKey {
                    address_space: region.identity.keys[0].address_space,
                    location: dependency.location.pc,
                    generation: dependency.dependency.mapping_generation,
                },
                id,
            );
        }
        self.live_mapped_bytes = self
            .live_mapped_bytes
            .checked_sub(region.compiled.mapped_len())
            .expect("live native-byte accounting includes every region");
        self.live_ir_operations = self
            .live_ir_operations
            .checked_sub(region.ir_operations)
            .expect("live compilation-work accounting includes every region");
        self.retired_batch(epoch).regions.push(region);
    }

    fn unlink_region(&mut self, id: RegionId, epoch: u64) {
        let mut links: BTreeSet<_> = self.incoming_links.remove(&id).unwrap_or_default();
        links.extend(
            self.active_links
                .keys()
                .filter(|link| link.source == id)
                .copied(),
        );
        for link in links {
            self.unlink(link, epoch);
        }
    }

    fn unlink(&mut self, link: LinkRef, epoch: u64) {
        let Some(active) = self.active_links.remove(&link) else {
            return;
        };
        if let Some(incoming) = self.incoming_links.get_mut(&active.target) {
            incoming.remove(&link);
            if incoming.is_empty() {
                self.incoming_links.remove(&active.target);
            }
        }
        let Some(source) = self.regions.get(&link.source) else {
            return;
        };
        let Some(cell) = source
            .links
            .site(link.site as usize)
            .and_then(|site| site.cell(usize::from(link.way)))
        else {
            return;
        };
        cell.clear_if(std::ptr::with_exposed_provenance_mut(active.target_address));
        if let Some(payload) = self.link_targets.remove(&active.target_address) {
            self.retired_batch(epoch).link_targets.push(payload);
        }
    }

    fn begin_shutdown(&mut self) -> Result<Vec<Arc<CompilationFlight>>, CacheError> {
        if !self.accepting_compilation {
            return Ok(Vec::new());
        }
        self.accepting_compilation = false;
        let next_revision = self.invalidation_revision.checked_add(1).ok_or_else(|| {
            CacheError::Capacity("domain cache invalidation identities are exhausted".into())
        })?;
        let flights = self
            .slots
            .values()
            .filter_map(|slot| match slot {
                CacheSlot::Compiling(flight) => Some(Arc::clone(flight)),
                CacheSlot::Ready(_) => None,
            })
            .collect();
        let region_ids: Vec<_> = self.regions.keys().copied().collect();
        let retirement_epoch = self.next_retirement_epoch()?;
        for id in region_ids {
            self.retire(id, retirement_epoch);
        }
        self.slots.clear();
        self.compiling_slots = 0;
        self.retirement_order.clear();
        self.physical_index.clear();
        self.mapping_index.clear();
        self.live_mapped_bytes = 0;
        self.live_ir_operations = 0;
        self.invalidation_revision = next_revision;
        Ok(flights)
    }
}

/// Sole owner of compiled-region identity and metadata for one engine domain.
pub(crate) struct ProcessCodeCache {
    limits: CacheLimits,
    state: Mutex<CacheState>,
    compilation_ready: Condvar,
}

struct CompilationPermit<'a> {
    cache: &'a ProcessCodeCache,
}

impl Drop for CompilationPermit<'_> {
    fn drop(&mut self) {
        let mut state = lock(&self.cache.state);
        state.active_compilations = state
            .active_compilations
            .checked_sub(1)
            .expect("each active compilation owns one permit");
        self.cache.compilation_ready.notify_one();
    }
}

pub(crate) struct ThreadEpoch {
    cache: Arc<ProcessCodeCache>,
    active: Arc<AtomicU64>,
}

impl Drop for ThreadEpoch {
    fn drop(&mut self) {
        self.active.store(QUIESCENT_EPOCH, Ordering::Release);
        self.cache.reclaim_retired();
    }
}

pub(crate) struct NativeEpochGuard {
    cache: Arc<ProcessCodeCache>,
    active: Arc<AtomicU64>,
}

impl Drop for NativeEpochGuard {
    fn drop(&mut self) {
        self.active.store(QUIESCENT_EPOCH, Ordering::Release);
        self.cache.reclaim_retired();
    }
}

impl ProcessCodeCache {
    pub(crate) fn new(configuration: JitConfiguration) -> Self {
        let performance_enabled = configuration.performance_report_directory().is_some();
        Self {
            limits: CacheLimits::from_configuration(&configuration),
            state: Mutex::new(CacheState::new(performance_enabled)),
            compilation_ready: Condvar::new(),
        }
    }

    fn acquire_compilation(
        &self,
        flight: &CompilationFlight,
    ) -> Result<CompilationPermit<'_>, CacheError> {
        let mut state = lock(&self.state);
        loop {
            flight.cancellation().check()?;
            if !state.accepting_compilation {
                return Err(CacheError::Cancelled);
            }
            if state.active_compilations < self.limits.max_concurrent_compilations {
                state.active_compilations += 1;
                return Ok(CompilationPermit { cache: self });
            }
            state = self
                .compilation_ready
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }

    pub(crate) fn register_thread(self: &Arc<Self>) -> ThreadEpoch {
        let active = Arc::new(AtomicU64::new(QUIESCENT_EPOCH));
        let mut state = lock(&self.state);
        state
            .executors
            .retain(|executor| executor.strong_count() != 0);
        state.executors.push(Arc::downgrade(&active));
        ThreadEpoch {
            cache: Arc::clone(self),
            active,
        }
    }

    pub(crate) fn begin_native(
        self: &Arc<Self>,
        region: &Arc<CachedRegion>,
        executor: &ThreadEpoch,
    ) -> Result<NativeEpochGuard, CacheError> {
        let state = lock(&self.state);
        if !region.is_live() || !state.regions.contains_key(&region.id) {
            return Err(CacheError::Stale);
        }
        executor
            .active
            .store(state.retirement_epoch, Ordering::Release);
        Ok(NativeEpochGuard {
            cache: Arc::clone(self),
            active: Arc::clone(&executor.active),
        })
    }

    fn reclaim_retired(&self) {
        let reclaimed = {
            let mut state = lock(&self.state);
            state
                .executors
                .retain(|executor| executor.strong_count() != 0);
            let oldest_active = state
                .executors
                .iter()
                .filter_map(Weak::upgrade)
                .map(|executor| executor.load(Ordering::Acquire))
                .filter(|epoch| *epoch != QUIESCENT_EPOCH)
                .min();
            let reclaim_count = state
                .retired
                .iter()
                .take_while(|batch| oldest_active.is_none_or(|active| active >= batch.epoch))
                .count();
            state.retired.drain(..reclaim_count).collect::<Vec<_>>()
        };
        // Published allocations may take the executable-arena lock on drop;
        // never nest that lock under the code-cache lock.
        drop(reclaimed);
    }

    pub(crate) fn apply_invalidations(
        &self,
        records: &[MemoryInvalidation],
        through: MemoryInvalidationCursor,
        history_lost: bool,
    ) -> Result<InvalidationEffect, CacheError> {
        let (flights, effect) = {
            let mut state = lock(&self.state);
            if state.invalidation_cursor >= through {
                return Ok(InvalidationEffect::default());
            }
            let mut record_counts = InvalidationRecordCounts::default();
            for record in records
                .iter()
                .filter(|record| record.cursor > state.invalidation_cursor)
            {
                match record.kind {
                    MemoryInvalidationKind::ExecutableContent { .. } => {
                        record_counts.executable_content =
                            record_counts.executable_content.saturating_add(1);
                    }
                    MemoryInvalidationKind::Mapping { .. } => {
                        record_counts.mapping = record_counts.mapping.saturating_add(1);
                    }
                    MemoryInvalidationKind::InstructionCache { .. } => {
                        record_counts.instruction_cache =
                            record_counts.instruction_cache.saturating_add(1);
                    }
                }
            }
            let mut affected = BTreeSet::new();
            if history_lost {
                affected.extend(state.regions.keys().copied());
            } else {
                for record in records
                    .iter()
                    .filter(|record| record.cursor > state.invalidation_cursor)
                {
                    match record.kind {
                        MemoryInvalidationKind::ExecutableContent { first, second } => {
                            if let Some(regions) = state.physical_index.get(&first) {
                                affected.extend(regions.iter().copied());
                            }
                            if let Some(second) = second
                                && let Some(regions) = state.physical_index.get(&second)
                            {
                                affected.extend(regions.iter().copied());
                            }
                        }
                        MemoryInvalidationKind::Mapping {
                            address_space,
                            start,
                            size,
                        } => {
                            let end = start.get().saturating_add(size);
                            for (key, regions) in &state.mapping_index {
                                if key.address_space == address_space
                                    && key.location.get() >= start.get()
                                    && key.location.get() < end
                                {
                                    affected.extend(regions.iter().copied());
                                }
                            }
                        }
                        MemoryInvalidationKind::InstructionCache { address_space } => {
                            affected.extend(state.regions.iter().filter_map(|(id, region)| {
                                region
                                    .identity
                                    .keys
                                    .iter()
                                    .any(|key| key.address_space == address_space)
                                    .then_some(*id)
                            }));
                        }
                    }
                }
            }
            if affected.is_empty() && state.compiling_slots == 0 {
                if let Some(performance) = &mut state.performance {
                    performance.record_invalidation(record_counts, history_lost, 0, 0, true);
                }
                state.invalidation_cursor = through;
                return Ok(InvalidationEffect::default());
            }
            let next_revision = state.invalidation_revision.checked_add(1).ok_or_else(|| {
                CacheError::Capacity("domain cache invalidation identities are exhausted".into())
            })?;
            let flights: Vec<_> = state
                .slots
                .values()
                .filter_map(|slot| match slot {
                    CacheSlot::Compiling(flight) => Some(Arc::clone(flight)),
                    CacheSlot::Ready(_) => None,
                })
                .collect();
            let retirement_epoch = state.next_retirement_epoch()?;
            let retired_count = affected.len() as u64;
            let links_before = state.active_links.len();
            for id in affected {
                state.retire(id, retirement_epoch);
            }
            let links_unlinked = links_before.saturating_sub(state.active_links.len()) as u64;
            if let Some(performance) = &mut state.performance {
                performance.record_invalidation(
                    record_counts,
                    history_lost,
                    retired_count,
                    links_unlinked,
                    false,
                );
            }
            state
                .slots
                .retain(|_, slot| matches!(slot, CacheSlot::Ready(_)));
            state.compiling_slots = 0;
            state.invalidation_revision = next_revision;
            state.invalidation_cursor = through;
            (
                flights,
                InvalidationEffect {
                    retired_regions: retired_count,
                },
            )
        };
        for flight in flights {
            flight.cancel(CacheError::Stale);
        }
        self.compilation_ready.notify_all();
        self.reclaim_retired();
        Ok(effect)
    }

    /// Advances the cache cursor only when executable-content records cannot
    /// affect live code and no unpublished compilation can still depend on
    /// them. This is used inside guest write helpers before they request a
    /// native poll, avoiding a second Rust transition for irrelevant writes.
    pub(crate) fn consume_irrelevant_invalidations(
        &self,
        records: &[MemoryInvalidation],
        through: MemoryInvalidationCursor,
    ) -> Result<bool, CacheError> {
        let mut state = lock(&self.state);
        if state.invalidation_cursor >= through {
            return Ok(true);
        }
        if state.compiling_slots != 0 {
            return Ok(false);
        }
        let mut counts = InvalidationRecordCounts::default();
        for record in records
            .iter()
            .filter(|record| record.cursor > state.invalidation_cursor)
        {
            let MemoryInvalidationKind::ExecutableContent { first, second } = record.kind else {
                return Ok(false);
            };
            counts.executable_content = counts.executable_content.saturating_add(1);
            if state.physical_index.contains_key(&first)
                || second.is_some_and(|page| state.physical_index.contains_key(&page))
            {
                return Ok(false);
            }
        }
        if let Some(performance) = &mut state.performance {
            performance.record_invalidation(counts, false, 0, 0, true);
        }
        state.invalidation_cursor = through;
        Ok(true)
    }

    pub(crate) fn begin_shutdown(&self) -> Result<(), CacheError> {
        let flights = lock(&self.state).begin_shutdown()?;
        for flight in flights {
            flight.cancel(CacheError::Cancelled);
        }
        self.compilation_ready.notify_all();
        self.reclaim_retired();
        Ok(())
    }

    pub(crate) fn live_thread_count(&self) -> usize {
        let mut state = lock(&self.state);
        state
            .executors
            .retain(|executor| executor.strong_count() != 0);
        state.executors.len()
    }

    pub(crate) fn performance_snapshot(&self) -> Option<CachePerformanceSnapshot> {
        lock(&self.state)
            .performance
            .as_ref()
            .map(|performance| performance.snapshot.clone())
    }

    pub(crate) fn record_background_failure(&self, error: CacheError) {
        let mut state = lock(&self.state);
        if state.background_failure.is_none() {
            state.background_failure = Some(error);
        }
    }

    pub(crate) fn background_failure(&self) -> Option<CacheError> {
        lock(&self.state).background_failure.clone()
    }

    /// Returns an already-published translation without starting or waiting on
    /// compilation. The light tier uses this to keep executing while another
    /// vCPU may be compiling the same region.
    pub(crate) fn lookup_ready(&self, key: RegionKey) -> Option<Arc<CachedRegion>> {
        let mut state = lock(&self.state);
        let region = match state.slots.get(&key) {
            Some(CacheSlot::Ready(id)) => state
                .regions
                .get(id)
                .filter(|region| region.is_live())
                .cloned(),
            Some(CacheSlot::Compiling(_)) | None => None,
        };
        if region.is_some()
            && let Some(performance) = &mut state.performance
        {
            performance.snapshot.global_ready_hits =
                performance.snapshot.global_ready_hits.saturating_add(1);
        }
        region
    }

    pub(crate) fn request_promotion(
        &self,
        address: usize,
    ) -> Result<Option<PromotionRequest>, CacheError> {
        let promotion_ptr = std::ptr::with_exposed_provenance::<PromotionCell>(address);
        let region_id = unsafe { promotion_ptr.as_ref() }
            .ok_or_else(|| CacheError::Internal("native promotion cell is null".into()))?
            .region_id
            .load(Ordering::Acquire);
        let state = lock(&self.state);
        let Some(region) = state
            .regions
            .get(&region_id)
            .filter(|region| region.is_live())
        else {
            return Ok(None);
        };
        let Some(promotion) = &region.promotion else {
            return Ok(None);
        };
        if Arc::as_ptr(promotion).addr() != address || region.tier != CodeTier::Light {
            return Err(CacheError::Internal(
                "native promotion cell does not belong to its published light-tier region".into(),
            ));
        }
        if !promotion.queue() {
            return Ok(None);
        }
        let region_ir = region.source_ir.as_ref().ok_or_else(|| {
            CacheError::Internal("light-tier region retained no source IR for promotion".into())
        })?;
        let key = region.identity.keys.first().copied().ok_or_else(|| {
            CacheError::Internal("published region has no guest entry identity".into())
        })?;
        Ok(Some(PromotionRequest {
            key,
            baseline_id: region.id,
            address_space: key.address_space,
            mode: key.translation_mode,
            region: Arc::clone(region_ir),
            promotion: Arc::clone(promotion),
        }))
    }

    pub(crate) fn publish_upgrade(
        &self,
        request: &PromotionRequest,
        pending: PendingRegion,
    ) -> Result<Arc<CachedRegion>, CacheError> {
        if pending.tier != CodeTier::Optimized
            || pending.promotion.is_some()
            || pending.source_ir.is_some()
        {
            return Err(CacheError::Internal(
                "optimized publication contains light-tier state".into(),
            ));
        }
        if !pending.identity.keys.contains(&request.key) {
            return Err(CacheError::Internal(
                "optimized region does not own its requested entry key".into(),
            ));
        }
        if pending.compiled.mapped_len() > self.limits.max_live_mapped_bytes
            || pending.ir_operations > self.limits.max_live_ir_operations
        {
            return Err(CacheError::Capacity(
                "one optimized region exceeds the domain cache bounds".into(),
            ));
        }

        let mut state = lock(&self.state);
        if !state.accepting_compilation {
            return Err(CacheError::Cancelled);
        }
        let baseline = state
            .regions
            .get(&request.baseline_id)
            .filter(|region| region.is_live())
            .cloned()
            .ok_or(CacheError::Stale)?;
        if baseline.tier != CodeTier::Light
            || baseline
                .promotion
                .as_ref()
                .is_none_or(|cell| !Arc::ptr_eq(cell, &request.promotion))
            || request.promotion.state.load(Ordering::Acquire) != PROMOTION_COMPILING
        {
            return Err(CacheError::Stale);
        }
        let id = state.next_region_id;
        state.next_region_id = state
            .next_region_id
            .checked_add(1)
            .ok_or_else(|| CacheError::Capacity("domain region identities are exhausted".into()))?;
        let retirement_epoch = state.next_retirement_epoch()?;
        state.retire(request.baseline_id, retirement_epoch);
        while state.regions.len() == self.limits.max_live_segments
            || state
                .live_mapped_bytes
                .checked_add(pending.compiled.mapped_len())
                .is_none_or(|bytes| bytes > self.limits.max_live_mapped_bytes)
            || state
                .live_ir_operations
                .checked_add(pending.ir_operations)
                .is_none_or(|work| work > self.limits.max_live_ir_operations)
        {
            let Some(oldest) = state.retirement_order.pop_front() else {
                return Err(CacheError::Capacity(
                    "domain cache pressure cannot retire an eligible region".into(),
                ));
            };
            state.retire(oldest, retirement_epoch);
            if let Some(performance) = &mut state.performance {
                performance.snapshot.regions_evicted_capacity = performance
                    .snapshot
                    .regions_evicted_capacity
                    .saturating_add(1);
            }
        }

        let region = install_pending_region(&mut state, id, pending);
        request.promotion.mark_optimized();
        if let Some(performance) = &mut state.performance {
            performance.snapshot.regions_published =
                performance.snapshot.regions_published.saturating_add(1);
            performance.snapshot.tier_upgrades =
                performance.snapshot.tier_upgrades.saturating_add(1);
        }
        drop(state);
        self.reclaim_retired();
        Ok(region)
    }

    pub(crate) fn resolve(
        &self,
        key: RegionKey,
        build: impl FnOnce(CompilationCancellation<'_>) -> Result<PendingRegion, CacheError>,
    ) -> Result<Arc<CachedRegion>, CacheError> {
        enum Resolution {
            Ready(Arc<CachedRegion>),
            Wait(Arc<CompilationFlight>),
            Build(Arc<CompilationFlight>),
        }

        let resolution = {
            let mut state = lock(&self.state);
            if !state.accepting_compilation {
                return Err(CacheError::Cancelled);
            }
            let resolution = match state.slots.get(&key) {
                Some(CacheSlot::Ready(id)) => state
                    .regions
                    .get(id)
                    .filter(|region| region.is_live())
                    .cloned()
                    .map(Resolution::Ready)
                    .unwrap_or_else(|| {
                        state.slots.remove(&key);
                        let flight = Arc::new(CompilationFlight::new(state.invalidation_revision));
                        state
                            .slots
                            .insert(key, CacheSlot::Compiling(Arc::clone(&flight)));
                        state.compiling_slots += 1;
                        Resolution::Build(flight)
                    }),
                Some(CacheSlot::Compiling(flight)) => Resolution::Wait(Arc::clone(flight)),
                None => {
                    let flight = Arc::new(CompilationFlight::new(state.invalidation_revision));
                    state
                        .slots
                        .insert(key, CacheSlot::Compiling(Arc::clone(&flight)));
                    state.compiling_slots += 1;
                    Resolution::Build(flight)
                }
            };
            if let Some(performance) = &mut state.performance {
                match &resolution {
                    Resolution::Ready(_) => {
                        performance.snapshot.global_ready_hits =
                            performance.snapshot.global_ready_hits.saturating_add(1);
                    }
                    Resolution::Wait(_) => {
                        performance.snapshot.global_waits =
                            performance.snapshot.global_waits.saturating_add(1);
                    }
                    Resolution::Build(_) => performance.record_build(key),
                }
            }
            resolution
        };

        match resolution {
            Resolution::Ready(region) => Ok(region),
            Resolution::Wait(flight) => flight.wait(),
            Resolution::Build(flight) => {
                let cancellation = flight.cancellation();
                let result = match catch_unwind(AssertUnwindSafe(|| {
                    let _permit = self.acquire_compilation(&flight)?;
                    cancellation
                        .check()
                        .and_then(|()| build(cancellation))
                        .and_then(|pending| self.publish(key, &flight, pending))
                })) {
                    Ok(result) => result,
                    Err(_) => Err(CacheError::Internal(
                        format!("panic was contained during JIT compilation for {key:?}")
                            .into_boxed_str(),
                    )),
                };
                if result.is_err() {
                    let mut state = lock(&self.state);
                    if let Some(performance) = &mut state.performance {
                        match &result {
                            Err(CacheError::Stale) => {
                                performance.snapshot.build_stale =
                                    performance.snapshot.build_stale.saturating_add(1);
                            }
                            Err(_) => {
                                performance.snapshot.build_failures =
                                    performance.snapshot.build_failures.saturating_add(1);
                            }
                            Ok(_) => unreachable!(),
                        }
                    }
                    if matches!(
                        state.slots.get(&key),
                        Some(CacheSlot::Compiling(current)) if Arc::ptr_eq(current, &flight)
                    ) {
                        state.slots.remove(&key);
                        state.compiling_slots = state.compiling_slots.saturating_sub(1);
                    }
                }
                flight.complete(result.clone());
                self.reclaim_retired();
                result
            }
        }
    }

    pub(crate) fn link(
        &self,
        source_id: u64,
        site_index: u32,
        location: LocationDescriptor,
        target: &Arc<CachedRegion>,
    ) -> Result<(LinkKind, LinkOutcome), CacheError> {
        let mut state = lock(&self.state);
        let source = state
            .regions
            .get(&source_id)
            .filter(|region| region.is_live())
            .cloned()
            .ok_or(CacheError::Stale)?;
        if !target.is_live() || !state.regions.contains_key(&target.id) {
            return Err(CacheError::Stale);
        }
        if !target
            .identity
            .keys
            .iter()
            .any(|key| key.location == location)
        {
            return Err(CacheError::Internal(
                "link target does not publish the resolved guest entry".into(),
            ));
        }
        let metadata = source
            .compiled
            .metadata
            .link_sites
            .get(site_index as usize)
            .ok_or_else(|| CacheError::Internal("native exit references no link site".into()))?;
        if metadata.kind == LinkKind::Direct && metadata.direct_target != Some(location) {
            return Err(CacheError::Internal(
                "direct link resolver received an incompatible destination".into(),
            ));
        }
        let ways = if metadata.kind == LinkKind::Direct {
            1
        } else {
            INDIRECT_LINK_WAYS
        };
        let mut empty = None;
        for way in 0..ways {
            let link = LinkRef {
                source: source_id,
                site: site_index,
                way: way as u8,
            };
            match state.active_links.get(&link) {
                Some(active) if active.location == location => {
                    return Ok((metadata.kind, LinkOutcome::AlreadyPresent));
                }
                Some(_) => {}
                None if empty.is_none() => empty = Some(link),
                None => {}
            }
        }
        let (link, replaced, outcome) = if let Some(link) = empty {
            (link, None, LinkOutcome::Published)
        } else {
            let site = source
                .links
                .site(site_index as usize)
                .ok_or_else(|| CacheError::Internal("link table has no requested site".into()))?;
            let link = LinkRef {
                source: source_id,
                site: site_index,
                way: u8::try_from(site.replacement_way(ways))
                    .map_err(|_| CacheError::Internal("link way exceeds u8".into()))?,
            };
            let active = state.active_links.get(&link).copied().ok_or_else(|| {
                CacheError::Internal("full native PIC has an untracked cell".into())
            })?;
            (link, Some(active), LinkOutcome::PicFull)
        };
        let payload = Arc::new(NativeLinkTarget {
            guest_pc: location.pc.get(),
            guest_state: encode_execution_state(location.execution_state),
            reserved: 0,
            region_id: target.id,
            link_table: target.links.base_address(),
            metadata: std::ptr::from_ref(&target.compiled.metadata).addr(),
            entry: target.compiled.entry_address(),
        });
        let target_address = Arc::as_ptr(&payload).cast_mut();
        let cell = source
            .links
            .site(site_index as usize)
            .and_then(|site| site.cell(usize::from(link.way)))
            .ok_or_else(|| {
                CacheError::Internal("link table does not own its metadata site".into())
            })?;
        if let Some(replaced) = replaced {
            let retirement_epoch = state.next_retirement_epoch()?;
            let expected = std::ptr::with_exposed_provenance_mut(replaced.target_address);
            if !cell.replace(expected, target_address) {
                return Err(CacheError::Internal(
                    "domain link cell changed outside the code-cache lock".into(),
                ));
            }
            if let Some(incoming) = state.incoming_links.get_mut(&replaced.target) {
                incoming.remove(&link);
                if incoming.is_empty() {
                    state.incoming_links.remove(&replaced.target);
                }
            }
            let retired = state
                .link_targets
                .remove(&replaced.target_address)
                .expect("every active link owns a retained native payload");
            state
                .retired_batch(retirement_epoch)
                .link_targets
                .push(retired);
        } else if !cell.publish(target_address) {
            return Err(CacheError::Internal(
                "domain link cell changed outside the code-cache lock".into(),
            ));
        }
        state.link_targets.insert(target_address.addr(), payload);
        state.active_links.insert(
            link,
            ActiveLink {
                target: target.id,
                target_address: target_address.addr(),
                location,
            },
        );
        state
            .incoming_links
            .entry(target.id)
            .or_default()
            .insert(link);
        drop(state);
        if outcome == LinkOutcome::PicFull {
            self.reclaim_retired();
        }
        Ok((metadata.kind, outcome))
    }

    pub(crate) fn region_for_exit(&self, id: u64) -> Option<Arc<CachedRegion>> {
        let state = lock(&self.state);
        state.regions.get(&id).cloned().or_else(|| {
            state
                .retired
                .iter()
                .flat_map(|batch| batch.regions.iter())
                .find(|region| region.id == id)
                .cloned()
        })
    }

    fn publish(
        &self,
        requested_key: RegionKey,
        flight: &Arc<CompilationFlight>,
        pending: PendingRegion,
    ) -> Result<Arc<CachedRegion>, CacheError> {
        if !pending.identity.keys.contains(&requested_key) {
            return Err(CacheError::Internal(
                "compiled region does not own its requested entry key".into(),
            ));
        }
        if pending.compiled.mapped_len() > self.limits.max_live_mapped_bytes
            || pending.ir_operations > self.limits.max_live_ir_operations
        {
            return Err(CacheError::Capacity(
                "one compiled region exceeds the domain cache bounds".into(),
            ));
        }

        let mut state = lock(&self.state);
        if !state.accepting_compilation
            || flight.cancellation.load(Ordering::Acquire) != COMPILATION_ACTIVE
            || flight.invalidation_revision != state.invalidation_revision
            || !matches!(
                state.slots.get(&requested_key),
                Some(CacheSlot::Compiling(current)) if Arc::ptr_eq(current, flight)
            )
        {
            if let Some(performance) = &mut state.performance {
                performance.snapshot.publish_stale =
                    performance.snapshot.publish_stale.saturating_add(1);
            }
            return Err(CacheError::Stale);
        }
        while state.regions.len() == self.limits.max_live_segments
            || state
                .live_mapped_bytes
                .checked_add(pending.compiled.mapped_len())
                .is_none_or(|bytes| bytes > self.limits.max_live_mapped_bytes)
            || state
                .live_ir_operations
                .checked_add(pending.ir_operations)
                .is_none_or(|work| work > self.limits.max_live_ir_operations)
        {
            let Some(oldest) = state.retirement_order.pop_front() else {
                return Err(CacheError::Capacity(
                    "domain cache pressure cannot retire an eligible region".into(),
                ));
            };
            let retirement_epoch = state.next_retirement_epoch()?;
            let links_before = state.active_links.len();
            state.retire(oldest, retirement_epoch);
            let links_unlinked = links_before.saturating_sub(state.active_links.len()) as u64;
            if let Some(performance) = &mut state.performance {
                performance.snapshot.regions_evicted_capacity = performance
                    .snapshot
                    .regions_evicted_capacity
                    .saturating_add(1);
                performance.snapshot.links_unlinked_capacity = performance
                    .snapshot
                    .links_unlinked_capacity
                    .saturating_add(links_unlinked);
            }
        }

        let id = state.next_region_id;
        state.next_region_id = state
            .next_region_id
            .checked_add(1)
            .ok_or_else(|| CacheError::Capacity("domain region identities are exhausted".into()))?;
        let link_site_count = pending.compiled.metadata.link_sites.len();
        let region = Arc::new(CachedRegion {
            id,
            live: AtomicBool::new(true),
            compiled: pending.compiled,
            identity: pending.identity,
            ir_operations: pending.ir_operations,
            links: LinkTable::new(link_site_count),
            tier: pending.tier,
            source_ir: pending.source_ir,
            promotion: pending.promotion,
        });

        if let Some(promotion) = &region.promotion {
            promotion.install_region(id);
        }

        for dependency in &region.identity.code_dependencies {
            state
                .physical_index
                .entry(dependency.page)
                .or_default()
                .insert(id);
        }
        for dependency in &region.identity.mapping_dependencies {
            state
                .mapping_index
                .entry(MappingIndexKey {
                    address_space: requested_key.address_space,
                    location: dependency.location.pc,
                    generation: dependency.dependency.mapping_generation,
                })
                .or_default()
                .insert(id);
        }
        state.compiling_slots = state
            .compiling_slots
            .checked_sub(1)
            .ok_or_else(|| CacheError::Internal("compiling-slot accounting underflow".into()))?;
        for entry_key in &region.identity.keys {
            match state.slots.get(entry_key) {
                Some(CacheSlot::Compiling(current)) if !Arc::ptr_eq(current, flight) => {}
                Some(CacheSlot::Ready(_)) => {}
                _ => {
                    state.slots.insert(*entry_key, CacheSlot::Ready(id));
                }
            }
        }
        state
            .slots
            .insert(requested_key, CacheSlot::Ready(region.id));
        state.live_mapped_bytes += region.compiled.mapped_len();
        state.live_ir_operations += region.ir_operations;
        state.retirement_order.push_back(id);
        state.regions.insert(id, Arc::clone(&region));
        let live_regions = state.regions.len() as u64;
        let live_mapped_bytes = state.live_mapped_bytes as u64;
        let cache_slots = state.slots.len() as u64;
        if let Some(performance) = &mut state.performance {
            performance.snapshot.regions_published =
                performance.snapshot.regions_published.saturating_add(1);
            performance.snapshot.peak_live_regions =
                performance.snapshot.peak_live_regions.max(live_regions);
            performance.snapshot.peak_live_mapped_bytes = performance
                .snapshot
                .peak_live_mapped_bytes
                .max(live_mapped_bytes);
            performance.snapshot.peak_cache_slots =
                performance.snapshot.peak_cache_slots.max(cache_slots);
        }
        Ok(region)
    }
}

fn install_pending_region(
    state: &mut CacheState,
    id: RegionId,
    pending: PendingRegion,
) -> Arc<CachedRegion> {
    let link_site_count = pending.compiled.metadata.link_sites.len();
    let region = Arc::new(CachedRegion {
        id,
        live: AtomicBool::new(true),
        compiled: pending.compiled,
        identity: pending.identity,
        ir_operations: pending.ir_operations,
        links: LinkTable::new(link_site_count),
        tier: pending.tier,
        source_ir: pending.source_ir,
        promotion: pending.promotion,
    });
    if let Some(promotion) = &region.promotion {
        promotion.install_region(id);
    }
    for dependency in &region.identity.code_dependencies {
        state
            .physical_index
            .entry(dependency.page)
            .or_default()
            .insert(id);
    }
    let address_space = region.identity.keys[0].address_space;
    for dependency in &region.identity.mapping_dependencies {
        state
            .mapping_index
            .entry(MappingIndexKey {
                address_space,
                location: dependency.location.pc,
                generation: dependency.dependency.mapping_generation,
            })
            .or_default()
            .insert(id);
    }
    for key in &region.identity.keys {
        state.slots.insert(*key, CacheSlot::Ready(id));
    }
    state.live_mapped_bytes += region.compiled.mapped_len();
    state.live_ir_operations += region.ir_operations;
    state.retirement_order.push_back(id);
    state.regions.insert(id, Arc::clone(&region));
    region
}

struct LocalEntry {
    key: RegionKey,
    region: Arc<CachedRegion>,
}

/// Direct-mapped executor-private cache. A valid hit acquires no shared lock.
pub(crate) struct LocalLookupCache {
    entries: Box<[Option<LocalEntry>]>,
}

impl LocalLookupCache {
    pub(crate) fn new() -> Self {
        Self {
            entries: std::iter::repeat_with(|| None)
                .take(LOCAL_LOOKUP_SLOTS)
                .collect(),
        }
    }

    pub(crate) fn lookup(&mut self, key: RegionKey) -> Option<Arc<CachedRegion>> {
        let slot = local_slot(key);
        let entry = self.entries[slot].as_ref()?;
        if entry.key == key && entry.region.is_live() {
            return Some(Arc::clone(&entry.region));
        }
        self.entries[slot] = None;
        None
    }

    pub(crate) fn insert(&mut self, key: RegionKey, region: Arc<CachedRegion>) {
        self.entries[local_slot(key)] = Some(LocalEntry { key, region });
    }

    pub(crate) fn clear(&mut self) {
        self.entries.iter_mut().for_each(|entry| *entry = None);
    }
}

fn local_slot(key: RegionKey) -> usize {
    let mut hasher = DefaultHasher::new();
    key.hash(&mut hasher);
    hasher.finish() as usize & (LOCAL_LOOKUP_SLOTS - 1)
}

fn encode_execution_state(state: ExecutionState) -> u32 {
    match state {
        ExecutionState::A64 => EXECUTION_STATE_A64,
        ExecutionState::A32 => EXECUTION_STATE_A32,
        ExecutionState::T32 => EXECUTION_STATE_T32,
    }
}

fn remove_index<K: Eq + Hash + Copy>(
    index: &mut HashMap<K, BTreeSet<RegionId>>,
    key: K,
    id: RegionId,
) {
    let remove = index.get_mut(&key).is_some_and(|regions| {
        regions.remove(&id);
        regions.is_empty()
    });
    if remove {
        index.remove(&key);
    }
}

fn remove_ordered_index<K: Ord + Copy>(
    index: &mut BTreeMap<K, BTreeSet<RegionId>>,
    key: K,
    id: RegionId,
) {
    let remove = index.get_mut(&key).is_some_and(|regions| {
        regions.remove(&id);
        regions.is_empty()
    });
    if remove {
        index.remove(&key);
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc;
    use std::thread;
    use std::time::{Duration, Instant};

    use nixe_cpu::ir::region::RegionMetadata;
    use nixe_cpu::profile::GuestCpuProfile;
    use nixe_memory::{ContentGeneration, MappingGeneration};

    use super::*;
    use crate::compiler::CompiledRegionMetadata;

    const SPACE: AddressSpaceId = AddressSpaceId::new(7);

    fn location(pc: u64) -> LocationDescriptor {
        LocationDescriptor::new(
            GuestVirtualAddress::new(pc),
            ExecutionState::A64,
            GuestCpuProfile::SWITCH_1_ID,
        )
    }

    fn dependency(page: u64, generation: u64) -> CodePageDependency {
        CodePageDependency {
            page: GuestPhysicalPageId::new(page),
            generation: ContentGeneration::new(generation),
            mapping_generation: MappingGeneration::new(generation),
        }
    }

    fn key(pc: u64, page: u64, generation: u64) -> RegionKey {
        RegionKey::new(
            SPACE,
            location(pc),
            TranslationMode::Baseline,
            dependency(page, generation),
        )
    }

    fn pending(key: RegionKey, extra_dependency: Option<CodePageDependency>) -> PendingRegion {
        pending_with_links(key, extra_dependency, Box::new([]))
    }

    fn pending_with_links(
        key: RegionKey,
        extra_dependency: Option<CodePageDependency>,
        link_sites: Box<[crate::links::LinkSiteMetadata]>,
    ) -> PendingRegion {
        let mut dependencies = vec![key.root_code_mapping];
        if let Some(extra) = extra_dependency {
            dependencies.push(extra);
        }
        let mapping_dependencies = dependencies
            .iter()
            .copied()
            .enumerate()
            .map(|(index, dependency)| MappingDependency {
                location: location(key.location.pc.get() + (index as u64 * 4)),
                dependency,
            })
            .collect();
        PendingRegion {
            compiled: CompiledRegion::for_cache_test(
                CompiledRegionMetadata {
                    start: key.location,
                    sources: Box::new([]),
                    side_exits: Box::new([]),
                    semantic_calls: Box::new([]),
                    native_named_operations: 0,
                    link_sites,
                },
                4096,
            ),
            identity: RegionIdentity {
                keys: Box::new([key]),
                code_dependencies: dependencies.into_boxed_slice(),
                mapping_dependencies,
            },
            ir_operations: 1,
            tier: CodeTier::Optimized,
            source_ir: None,
            promotion: None,
        }
    }

    fn light_pending(key: RegionKey) -> (PendingRegion, Arc<PromotionCell>) {
        let promotion = Arc::new(PromotionCell::new());
        let mut pending = pending(key, None);
        pending.tier = CodeTier::Light;
        pending.source_ir = Some(Arc::new(IrRegion::new(
            RegionMetadata {
                start: key.location,
                guest_byte_count: 4,
                guest_instruction_count: 1,
                ir_operation_count: 1,
                entries: Box::new([]),
                exits: Box::new([]),
                code_dependencies: Box::new([key.root_code_mapping]),
                safepoints: Box::new([]),
            },
            Vec::new(),
        )));
        pending.promotion = Some(Arc::clone(&promotion));
        (pending, promotion)
    }

    #[test]
    fn light_region_promotes_once_and_optimized_publication_replaces_it() {
        let cache = ProcessCodeCache::new(
            JitConfiguration::default().with_performance_report_directory(Some("unused".into())),
        );
        let region_key = key(0x1000, 1, 1);
        let (light_pending_region, promotion) = light_pending(region_key);
        let light = cache
            .resolve(region_key, |_| Ok(light_pending_region))
            .unwrap();
        let request = cache
            .request_promotion(Arc::as_ptr(&promotion).addr())
            .unwrap()
            .expect("the first threshold crossing queues the light region");
        assert!(
            cache
                .request_promotion(Arc::as_ptr(&promotion).addr())
                .unwrap()
                .is_none(),
            "one light version can own only one promotion"
        );
        assert!(request.promotion.begin_compilation());
        let optimized = cache
            .publish_upgrade(&request, pending(region_key, None))
            .unwrap();
        assert!(!light.is_live());
        assert!(optimized.is_live());
        assert_eq!(optimized.tier, CodeTier::Optimized);
        assert_eq!(cache.lookup_ready(region_key).unwrap().id(), optimized.id());
        let snapshot = cache.performance_snapshot().unwrap();
        assert_eq!(snapshot.tier_upgrades, 1);
    }

    #[test]
    fn retiring_a_target_unlinks_before_rebinding_the_same_source_cell() {
        use crate::links::{LinkKind, LinkSiteMetadata};

        let cache = ProcessCodeCache::new(JitConfiguration::default());
        let source_key = key(0x1000, 1, 1);
        let target_location = location(0x2000);
        let target_key = RegionKey::new(
            SPACE,
            target_location,
            TranslationMode::Baseline,
            dependency(2, 1),
        );
        let source = cache
            .resolve(source_key, |_| {
                Ok(pending_with_links(
                    source_key,
                    None,
                    Box::new([LinkSiteMetadata {
                        kind: LinkKind::Direct,
                        direct_target: Some(target_location),
                    }]),
                ))
            })
            .unwrap();
        let first = cache
            .resolve(target_key, |_| Ok(pending(target_key, None)))
            .unwrap();
        assert_eq!(
            cache.link(source.id, 0, target_location, &first).unwrap(),
            (LinkKind::Direct, LinkOutcome::Published)
        );
        assert_eq!(
            cache.link(source.id, 0, target_location, &first).unwrap(),
            (LinkKind::Direct, LinkOutcome::AlreadyPresent)
        );
        let first_address = lock(&cache.state)
            .active_links
            .values()
            .next()
            .unwrap()
            .target_address;

        {
            let mut state = lock(&cache.state);
            let epoch = state.next_retirement_epoch().unwrap();
            state.retire(first.id, epoch);
        }
        {
            let state = lock(&cache.state);
            assert!(state.active_links.is_empty());
            assert!(state.incoming_links.is_empty());
            assert!(source.is_live());
            assert!(!first.is_live());
        }

        let replacement_key = RegionKey::new(
            SPACE,
            target_location,
            TranslationMode::Baseline,
            dependency(2, 2),
        );
        let replacement = cache
            .resolve(replacement_key, |_| Ok(pending(replacement_key, None)))
            .unwrap();
        cache
            .link(source.id, 0, target_location, &replacement)
            .unwrap();
        let state = lock(&cache.state);
        let rebound = state.active_links.values().next().unwrap();
        assert_eq!(rebound.target, replacement.id);
        let _retired_address_may_be_reused = first_address;
        assert_eq!(state.link_targets.len(), 1);
    }

    #[test]
    fn full_indirect_link_sites_replace_a_bounded_way() {
        use crate::links::{INDIRECT_LINK_WAYS, LinkKind, LinkSiteMetadata};

        let cache = Arc::new(ProcessCodeCache::new(JitConfiguration::default()));
        let executor = cache.register_thread();
        let source_key = key(0x3000, 3, 1);
        let source = cache
            .resolve(source_key, |_| {
                Ok(pending_with_links(
                    source_key,
                    None,
                    Box::new([LinkSiteMetadata {
                        kind: LinkKind::Indirect,
                        direct_target: None,
                    }]),
                ))
            })
            .unwrap();
        let mut last_location = None;
        let mut native_guard = None;
        for index in 0..=INDIRECT_LINK_WAYS {
            if index == INDIRECT_LINK_WAYS {
                native_guard = Some(cache.begin_native(&source, &executor).unwrap());
            }
            let target_key = key(0x4000 + index as u64 * 4, 10 + index as u64, 1);
            let target = cache
                .resolve(target_key, |_| Ok(pending(target_key, None)))
                .unwrap();
            let outcome = cache
                .link(source.id, 0, target_key.location, &target)
                .unwrap();
            assert_eq!(
                outcome,
                (
                    LinkKind::Indirect,
                    if index < INDIRECT_LINK_WAYS {
                        LinkOutcome::Published
                    } else {
                        LinkOutcome::PicFull
                    }
                )
            );
            last_location = Some(target_key.location);
        }
        let state = lock(&cache.state);
        assert_eq!(state.active_links.len(), INDIRECT_LINK_WAYS);
        assert_eq!(state.link_targets.len(), INDIRECT_LINK_WAYS);
        assert_eq!(state.retired.len(), 1);
        assert!(
            state
                .active_links
                .values()
                .any(|active| Some(active.location) == last_location)
        );
        drop(state);
        drop(native_guard);
        assert!(lock(&cache.state).retired.is_empty());
    }

    #[test]
    fn hot_key_contains_every_translation_identity_dimension() {
        let root = dependency(1, 1);
        let baseline = RegionKey::new(SPACE, location(0x1000), TranslationMode::Baseline, root);
        assert_ne!(
            baseline,
            RegionKey::new(
                AddressSpaceId::new(8),
                location(0x1000),
                TranslationMode::Baseline,
                root,
            )
        );
        assert_ne!(baseline, key(0x2000, 1, 1));
        assert_ne!(baseline, key(0x1000, 2, 1));
        assert_ne!(baseline, key(0x1000, 1, 2));
        let switch_2 = LocationDescriptor::new(
            GuestVirtualAddress::new(0x1000),
            ExecutionState::A64,
            GuestCpuProfile::SWITCH_2_NATIVE_ID,
        );
        assert_ne!(
            baseline,
            RegionKey::new(SPACE, switch_2, TranslationMode::Baseline, root)
        );
        let t32 = LocationDescriptor::new(
            GuestVirtualAddress::new(0x1000),
            ExecutionState::T32,
            GuestCpuProfile::SWITCH_1_ID,
        );
        assert_ne!(
            baseline,
            RegionKey::new(SPACE, t32, TranslationMode::Baseline, root)
        );
    }

    #[test]
    fn concurrent_misses_compile_one_region() {
        let cache = Arc::new(ProcessCodeCache::new(JitConfiguration::default()));
        let compilation_count = Arc::new(AtomicUsize::new(0));
        let region_key = key(0x1000, 1, 1);
        let mut workers = Vec::new();
        for _ in 0..8 {
            let cache = Arc::clone(&cache);
            let compilation_count = Arc::clone(&compilation_count);
            workers.push(thread::spawn(move || {
                cache
                    .resolve(region_key, |_| {
                        compilation_count.fetch_add(1, Ordering::AcqRel);
                        thread::sleep(Duration::from_millis(10));
                        Ok(pending(region_key, None))
                    })
                    .unwrap()
            }));
        }
        let regions: Vec<_> = workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect();
        assert_eq!(compilation_count.load(Ordering::Acquire), 1);
        assert!(
            regions
                .iter()
                .all(|region| Arc::ptr_eq(region, &regions[0]))
        );
    }

    #[test]
    fn compilation_budget_queues_distinct_flights_and_shutdown_cancels_the_queue() {
        let configuration = JitConfiguration::new(8, 1024 * 1024, 1).unwrap();
        let cache = Arc::new(ProcessCodeCache::new(configuration));
        let first_key = key(0x1000, 1, 1);
        let second_key = key(0x2000, 2, 1);
        let (first_started_tx, first_started_rx) = mpsc::sync_channel(1);
        let (release_first_tx, release_first_rx) = mpsc::sync_channel(1);
        let first_cache = Arc::clone(&cache);
        let first = thread::spawn(move || {
            first_cache.resolve(first_key, |cancellation| {
                first_started_tx.send(()).unwrap();
                release_first_rx.recv().unwrap();
                cancellation.check()?;
                Ok(pending(first_key, None))
            })
        });
        first_started_rx.recv().unwrap();

        let second_builder_calls = Arc::new(AtomicUsize::new(0));
        let second_cache = Arc::clone(&cache);
        let second_calls = Arc::clone(&second_builder_calls);
        let second = thread::spawn(move || {
            second_cache.resolve(second_key, |_| {
                second_calls.fetch_add(1, Ordering::AcqRel);
                Ok(pending(second_key, None))
            })
        });
        let deadline = Instant::now() + Duration::from_secs(1);
        while lock(&cache.state).slots.len() != 2 {
            assert!(
                Instant::now() < deadline,
                "second compilation flight did not enter the bounded queue"
            );
            thread::yield_now();
        }
        assert_eq!(second_builder_calls.load(Ordering::Acquire), 0);

        cache.begin_shutdown().unwrap();
        assert!(matches!(second.join().unwrap(), Err(CacheError::Cancelled)));
        assert_eq!(second_builder_calls.load(Ordering::Acquire), 0);
        release_first_tx.send(()).unwrap();
        assert!(matches!(first.join().unwrap(), Err(CacheError::Cancelled)));
    }

    #[test]
    fn shutdown_cancels_waiters_and_in_progress_compilation_without_waiting() {
        let cache = Arc::new(ProcessCodeCache::new(JitConfiguration::default()));
        let region_key = key(0x1000, 1, 1);
        let (started_tx, started_rx) = mpsc::sync_channel(1);
        let (release_tx, release_rx) = mpsc::sync_channel(1);
        let builder_cache = Arc::clone(&cache);
        let builder = thread::spawn(move || {
            builder_cache.resolve(region_key, |cancellation| {
                started_tx.send(()).unwrap();
                release_rx.recv().unwrap();
                cancellation.check()?;
                Ok(pending(region_key, None))
            })
        });
        started_rx.recv().unwrap();

        let waiter_cache = Arc::clone(&cache);
        let waiter = thread::spawn(move || {
            waiter_cache.resolve(region_key, |_| {
                unreachable!("the existing compilation flight owns the build")
            })
        });

        cache.begin_shutdown().unwrap();
        assert!(matches!(waiter.join().unwrap(), Err(CacheError::Cancelled)));
        release_tx.send(()).unwrap();
        assert!(matches!(
            builder.join().unwrap(),
            Err(CacheError::Cancelled)
        ));
        assert!(matches!(
            cache.resolve(region_key, |_| Ok(pending(region_key, None))),
            Err(CacheError::Cancelled)
        ));
        let state = lock(&cache.state);
        assert!(state.slots.is_empty());
        assert!(state.regions.is_empty());
        assert!(state.retired.is_empty());
    }

    #[test]
    fn invalidation_cancels_compilation_as_retryable_stale_work() {
        let configuration = JitConfiguration::default()
            .with_performance_report_directory(Some("unused-performance-report".into()));
        let cache = Arc::new(ProcessCodeCache::new(configuration));
        let region_key = key(0x1000, 1, 1);
        let (started_tx, started_rx) = mpsc::sync_channel(1);
        let (release_tx, release_rx) = mpsc::sync_channel(1);
        let builder_cache = Arc::clone(&cache);
        let builder = thread::spawn(move || {
            builder_cache.resolve(region_key, |cancellation| {
                started_tx.send(()).unwrap();
                release_rx.recv().unwrap();
                cancellation.check()?;
                Ok(pending(region_key, None))
            })
        });
        started_rx.recv().unwrap();
        cache
            .apply_invalidations(
                &[MemoryInvalidation {
                    cursor: MemoryInvalidationCursor::new(1),
                    kind: MemoryInvalidationKind::ExecutableContent {
                        first: GuestPhysicalPageId::new(1),
                        second: None,
                    },
                }],
                MemoryInvalidationCursor::new(1),
                false,
            )
            .unwrap();
        release_tx.send(()).unwrap();
        assert!(matches!(builder.join().unwrap(), Err(CacheError::Stale)));
        assert!(
            cache
                .resolve(region_key, |_| Ok(pending(region_key, None)))
                .is_ok()
        );
        let snapshot = cache.performance_snapshot().unwrap();
        assert_eq!(snapshot.global_builds, 2);
        assert_eq!(snapshot.build_stale, 1);
        assert_eq!(snapshot.regions_published, 1);
        assert_eq!(snapshot.unique_region_keys, 1);
        assert_eq!(snapshot.recompiled_region_keys, 1);
        assert_eq!(snapshot.repeat_builds, 1);
        assert_eq!(snapshot.invalidation_batches, 1);
        assert_eq!(snapshot.invalidation_executable_content, 1);
    }

    #[test]
    fn irrelevant_invalidation_uses_constant_time_fast_path() {
        let configuration = JitConfiguration::default()
            .with_performance_report_directory(Some("unused-performance-report".into()));
        let cache = ProcessCodeCache::new(configuration);
        let region_key = key(0x1000, 1, 1);
        let region = cache
            .resolve(region_key, |_| Ok(pending(region_key, None)))
            .unwrap();
        let revision_before = lock(&cache.state).invalidation_revision;

        let effect = cache
            .apply_invalidations(
                &[MemoryInvalidation {
                    cursor: MemoryInvalidationCursor::new(1),
                    kind: MemoryInvalidationKind::ExecutableContent {
                        first: GuestPhysicalPageId::new(999),
                        second: None,
                    },
                }],
                MemoryInvalidationCursor::new(1),
                false,
            )
            .unwrap();

        assert_eq!(effect, InvalidationEffect::default());
        assert!(region.is_live());
        let state = lock(&cache.state);
        assert_eq!(state.invalidation_revision, revision_before);
        assert_eq!(state.compiling_slots, 0);
        drop(state);
        let snapshot = cache.performance_snapshot().unwrap();
        assert_eq!(snapshot.fast_irrelevant_invalidations, 1);
    }

    #[test]
    fn guest_write_prefilter_advances_only_irrelevant_executable_content() {
        let cache = ProcessCodeCache::new(JitConfiguration::default());
        let region_key = key(0x1000, 1, 1);
        let region = cache
            .resolve(region_key, |_| Ok(pending(region_key, None)))
            .unwrap();
        let irrelevant = MemoryInvalidation {
            cursor: MemoryInvalidationCursor::new(1),
            kind: MemoryInvalidationKind::ExecutableContent {
                first: GuestPhysicalPageId::new(999),
                second: None,
            },
        };
        assert!(
            cache
                .consume_irrelevant_invalidations(&[irrelevant], MemoryInvalidationCursor::new(1))
                .unwrap()
        );
        assert_eq!(
            lock(&cache.state).invalidation_cursor,
            MemoryInvalidationCursor::new(1)
        );

        let relevant = MemoryInvalidation {
            cursor: MemoryInvalidationCursor::new(2),
            kind: MemoryInvalidationKind::ExecutableContent {
                first: GuestPhysicalPageId::new(1),
                second: None,
            },
        };
        assert!(
            !cache
                .consume_irrelevant_invalidations(&[relevant], MemoryInvalidationCursor::new(2))
                .unwrap()
        );
        let state = lock(&cache.state);
        assert_eq!(state.invalidation_cursor, MemoryInvalidationCursor::new(1));
        assert!(region.is_live());
    }

    #[test]
    fn compilation_panic_completes_the_flight_and_permits_a_later_retry() {
        let cache = ProcessCodeCache::new(JitConfiguration::default());
        let region_key = key(0x1000, 1, 1);
        assert!(matches!(
            cache.resolve(region_key, |_| panic!("injected compilation panic")),
            Err(CacheError::Internal(detail))
                if detail.contains("panic was contained during JIT compilation for RegionKey")
        ));
        assert!(
            cache
                .resolve(region_key, |_| Ok(pending(region_key, None)))
                .is_ok()
        );
    }

    #[test]
    fn pressure_retires_fifo_and_removes_complete_reverse_indexes() {
        let cache = ProcessCodeCache::new(JitConfiguration::new(2, 1024 * 1024, 1).unwrap());
        let first_key = key(0x1000, 1, 1);
        let first_extra = dependency(11, 1);
        let first = cache
            .resolve(first_key, |_| Ok(pending(first_key, Some(first_extra))))
            .unwrap();
        let second_key = key(0x2000, 2, 1);
        cache
            .resolve(second_key, |_| Ok(pending(second_key, None)))
            .unwrap();
        cache.resolve(first_key, |_| unreachable!()).unwrap();
        let third_key = key(0x3000, 3, 1);
        cache
            .resolve(third_key, |_| Ok(pending(third_key, None)))
            .unwrap();

        assert!(
            !first.is_live(),
            "lookup hits do not perturb FIFO retirement"
        );
        let state = lock(&cache.state);
        assert_eq!(state.regions.len(), 2);
        assert!(
            !state
                .physical_index
                .contains_key(&GuestPhysicalPageId::new(1))
        );
        assert!(
            !state
                .physical_index
                .contains_key(&GuestPhysicalPageId::new(11))
        );
        assert_eq!(state.mapping_index.len(), 2);
    }

    #[test]
    fn performance_snapshot_separates_hits_rebuilds_evictions_and_invalidations() {
        let configuration = JitConfiguration::new(2, 1024 * 1024, 1)
            .unwrap()
            .with_performance_report_directory(Some("unused-performance-report".into()));
        let cache = ProcessCodeCache::new(configuration);
        let first = key(0x1000, 1, 1);
        let second = key(0x2000, 2, 1);
        let third = key(0x3000, 3, 1);

        cache.resolve(first, |_| Ok(pending(first, None))).unwrap();
        cache.resolve(first, |_| unreachable!()).unwrap();
        cache
            .resolve(second, |_| Ok(pending(second, None)))
            .unwrap();
        cache.resolve(third, |_| Ok(pending(third, None))).unwrap();
        cache.resolve(first, |_| Ok(pending(first, None))).unwrap();
        cache
            .apply_invalidations(
                &[MemoryInvalidation {
                    cursor: MemoryInvalidationCursor::new(1),
                    kind: MemoryInvalidationKind::ExecutableContent {
                        first: GuestPhysicalPageId::new(3),
                        second: None,
                    },
                }],
                MemoryInvalidationCursor::new(1),
                false,
            )
            .unwrap();

        let snapshot = cache.performance_snapshot().unwrap();
        assert_eq!(snapshot.global_ready_hits, 1);
        assert_eq!(snapshot.global_builds, 4);
        assert_eq!(snapshot.regions_published, 4);
        assert_eq!(snapshot.unique_region_keys, 3);
        assert_eq!(snapshot.recompiled_region_keys, 1);
        assert_eq!(snapshot.repeat_builds, 1);
        assert_eq!(snapshot.regions_evicted_capacity, 2);
        assert_eq!(snapshot.invalidation_batches, 1);
        assert_eq!(snapshot.invalidation_executable_content, 1);
        assert_eq!(snapshot.regions_retired_invalidation, 1);
        assert_eq!(snapshot.peak_live_regions, 2);
    }

    #[test]
    fn physical_content_invalidation_detaches_only_dependent_lookups() {
        let cache = ProcessCodeCache::new(JitConfiguration::default());
        let region_key = key(0x1000, 1, 1);
        let region = cache
            .resolve(region_key, |_| Ok(pending(region_key, None)))
            .unwrap();
        let unrelated_key = key(0x2000, 2, 1);
        let unrelated = cache
            .resolve(unrelated_key, |_| Ok(pending(unrelated_key, None)))
            .unwrap();
        let mut local = LocalLookupCache::new();
        local.insert(region_key, Arc::clone(&region));
        local.insert(unrelated_key, Arc::clone(&unrelated));
        assert!(Arc::ptr_eq(&local.lookup(region_key).unwrap(), &region));

        cache
            .apply_invalidations(
                &[MemoryInvalidation {
                    cursor: MemoryInvalidationCursor::new(1),
                    kind: MemoryInvalidationKind::ExecutableContent {
                        first: GuestPhysicalPageId::new(1),
                        second: None,
                    },
                }],
                MemoryInvalidationCursor::new(1),
                false,
            )
            .unwrap();
        assert!(local.lookup(region_key).is_none());
        assert!(!region.is_live());
        assert!(unrelated.is_live());
        assert!(Arc::ptr_eq(
            &local.lookup(unrelated_key).unwrap(),
            &unrelated
        ));
        assert_eq!(lock(&cache.state).regions.len(), 1);
    }

    #[test]
    fn retired_resources_wait_for_every_active_executor_epoch() {
        let cache = Arc::new(ProcessCodeCache::new(JitConfiguration::default()));
        let first_token = cache.register_thread();
        let second_token = cache.register_thread();
        let region_key = key(0x1000, 1, 1);
        let region = cache
            .resolve(region_key, |_| Ok(pending(region_key, None)))
            .unwrap();
        let first_guard = cache.begin_native(&region, &first_token).unwrap();
        let second_guard = cache.begin_native(&region, &second_token).unwrap();

        cache
            .apply_invalidations(
                &[MemoryInvalidation {
                    cursor: MemoryInvalidationCursor::new(1),
                    kind: MemoryInvalidationKind::ExecutableContent {
                        first: GuestPhysicalPageId::new(1),
                        second: None,
                    },
                }],
                MemoryInvalidationCursor::new(1),
                false,
            )
            .unwrap();
        assert!(!region.is_live());
        assert_eq!(lock(&cache.state).retired.len(), 1);

        drop(first_guard);
        assert_eq!(lock(&cache.state).retired.len(), 1);
        drop(second_guard);
        assert!(lock(&cache.state).retired.is_empty());
    }

    #[test]
    fn lost_invalidation_history_retires_the_complete_domain() {
        let cache = ProcessCodeCache::new(JitConfiguration::default());
        for (pc, page) in [(0x1000, 1), (0x2000, 2)] {
            let region_key = key(pc, page, 1);
            cache
                .resolve(region_key, |_| Ok(pending(region_key, None)))
                .unwrap();
        }
        cache
            .apply_invalidations(&[], MemoryInvalidationCursor::new(9), true)
            .unwrap();
        assert!(lock(&cache.state).regions.is_empty());
    }
}
