//! Complete, immutable code ownership and metadata-first publication. Guest
//! capture/lowering and coordinated mapping changes are wired by Task 3;
//! this boundary consumes their owned image and checks its captured cursor.

use super::directory::{Interval, Table};
use super::registry::{Handle, Registry, Slot};
use super::{DispatchSlot, Error, Lifetime, PreparedStorage, Publication, Reason};
use crate::abi::{
    CheckedCounter, CodeUnitId, CodeVersion, DispatchPayload, EntryContract, ExecutionEpoch,
    ExitStateMap, FamilyVersion, HcqEntry, HcqFamilyId, HostAbi, InstructionKey, LazyFlags,
    MaintenanceSequence, NATIVE_ABI_VERSION, NzcvLocation, PublishedEntry, ValueLocation,
};
use crate::executable::{Accounted, Installed, MetadataLease, SEGMENTS, Tier};
use nixe_cpu::memory::CodePageDependency;
use nixe_memory::MemoryInvalidationCursor;
use std::hash::{BuildHasher, RandomState};
use std::mem::{size_of, size_of_val};
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

mod bridge;
pub(crate) mod dynamic;
mod invalidation;
pub(crate) mod links;
pub(crate) mod patch;
mod reclaim;
pub(crate) use reclaim::Snapshot;
#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Instruction {
    pub key: InstructionKey,
    /// Exact captured A64 instruction bits, not a later guest-memory read.
    pub bits: u32,
}

pub(crate) struct Entry {
    pub key: crate::abi::BlockKey,
    pub canonical_offset: u32,
    pub fast_offset: u32,
    pub contract: EntryContract,
}

pub(crate) struct StateRecord {
    pub native_offset: u32,
    pub state: ExitStateMap,
    pub exit: Option<GuestExit>,
    // Only terminals allocate this payload; do not enlarge every prefault map
    // by the full transfer descriptor. Its bytes share the CodeUnit's budget.
    pub transfer: Option<Box<TerminalTransfer>>,
}

/// Immutable terminal information from lowering and final allocation. The
/// patch starts at StateRecord.native_offset; mutable links belong to Lifetime.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TerminalTransfer {
    pub destination: ValueLocation,
    /// Linkable static dispatch destination, not an observation's constant PC.
    /// Never inferred from allocated machine bytes; dynamic exits use None.
    pub static_target: Option<crate::abi::BlockKey>,
    /// Completed guest prefix: dispatch includes the branch, PRE exits do not.
    pub completed: u16,
    pub patch_bytes: u8,
    /// Canonical fallback after any indirect RSB/PIC operation. A static call
    /// includes its push here because unlinking bypasses the installed bridge.
    /// Call/return slice polls perform their update on a separate path when
    /// necessary; they must neither probe a successor nor repeat an update.
    pub fallback_offset: u32,
    /// Already-charged cold patch. Resume at native_offset, never at the
    /// subtraction preceding it. None for uncheckpointed PRE observations.
    pub poll_offset: Option<u32>,
}

/// Source identity and edge semantics, retained with the physical exit map.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GuestExit {
    pub pc: nixe_memory::GuestVirtualAddress,
    pub kind: EdgeKind,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum EdgeKind {
    Static,
    Taken,
    NotTaken,
    Call,
    Indirect,
    Return,
    SupervisorCall(u16),
    Breakpoint(u16),
    FpSystem(crate::abi::FpSystemOperation),
    FpCompare(crate::abi::FpCompareOperation),
    FpRound(crate::abi::FpRoundOperation),
    FpAdd(crate::abi::FpAddOperation),
    FpDivide(crate::abi::FpDivideOperation),
    VectorFpDivide(crate::abi::VectorFpDivideOperation),
    VectorFpMultiplyElement(crate::abi::VectorFpMultiplyElementOperation),
    FpMultiply(crate::abi::FpMultiplyOperation),
    FpFused(crate::abi::FpFusedOperation),
    FpUnary(crate::abi::FpUnaryOperation),
    FpToInteger(crate::abi::FpToIntegerOperation),
    IntegerToFp(crate::abi::IntegerToFpOperation),
    VectorIntegerToFp(crate::abi::VectorIntegerToFpOperation),
    RuntimeSystem(crate::abi::RuntimeSystemOperation),
    ExclusiveStore(crate::abi::ExclusiveStoreOperation),
    Unsupported,
    InvalidInstruction,
    FragmentLimit,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Access {
    Read,
    Write,
    Atomic,
    /// A host byte read proving coherent RAM, never a guest data read/retry.
    CacheProbe,
}

pub(crate) struct FaultRecord {
    /// One exact instruction interval, supplied by native emission. Never
    /// approximate it with the gap to the next faulting instruction.
    pub native_start: u32,
    pub native_end: u32,
    pub instruction: InstructionKey,
    pub access: Access,
    pub bytes: u8,
    pub subaccess: u16,
    /// Semantic lowering's architectural commit stage, not a native-op count.
    pub commit_stage: u16,
    /// Raw first-element bits at the second access of a pair load (width is
    /// `bytes`). Not an architectural register update: cold completion uses
    /// this value without replaying the read; a guest fault keeps PRE registers.
    pub completed_read: Option<ValueLocation>,
    /// Index in CodeUnit.states; includes deferred NZCV and pending host FPSR.
    pub state_map: u32,
}

pub(crate) struct Input {
    pub identity: EmissionIdentity,
    pub code: Installed,
    pub tier: Tier,
    pub instructions: Box<[Instruction]>,
    pub entries: Box<[Entry]>,
    pub dependencies: Box<[CodePageDependency]>,
    pub cursor: MemoryInvalidationCursor,
    pub states: Box<[StateRecord]>,
    pub faults: Box<[FaultRecord]>,
}

/// Single-use identity reserved before emitting version-bearing native exits.
/// Publication consumes it; abandoned compilations never recycle its numbers.
pub(crate) struct EmissionIdentity {
    process: u64,
    admission: crate::abi::AdmissionEpoch,
    tier: Tier,
    id: CodeUnitId,
    version: CodeVersion,
}
impl EmissionIdentity {
    pub(crate) fn version(&self) -> CodeVersion {
        self.version
    }
}

pub(crate) struct CodeUnit {
    pub id: CodeUnitId,
    pub version: CodeVersion,
    pub abi_version: u32,
    pub input: Input,
    // Cold ownership accounting, never read by generated code or fault lookup.
    baseline_pins: AtomicUsize,
    // Set once before directory/dispatch exposure. Cold native-PC resolution
    // reaches the exact generational registry slot without scanning live units.
    registration: std::sync::OnceLock<UnitHandle>,
}
impl std::ops::Deref for CodeUnit {
    type Target = Input;
    fn deref(&self) -> &Input {
        &self.input
    }
}
impl CodeUnit {
    pub(crate) fn registered_handle(&self) -> Option<UnitHandle> {
        self.registration.get().copied()
    }

    fn entry(&self, entry: &Entry) -> PublishedEntry {
        let base = self.code.allocation.address();
        PublishedEntry {
            unit: self.id,
            version: self.version,
            canonical: NonZeroUsize::new(base + entry.canonical_offset as usize).unwrap(),
            fast: NonZeroUsize::new(base + entry.fast_offset as usize).unwrap(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct UnitHandle(Handle<UnitRecord>, u64);

/// Cold dispatch-to-registry identity. The slot does not acquire an extra
/// strong reference; publication/retirement maintain it with the payload under
/// JIT state. The generational handle resolves ownership without a unit scan.
#[derive(Clone, Copy)]
pub(super) struct UnitEntry {
    unit: UnitHandle,
    index: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Lifecycle {
    Published,
    Superseded,
    Invalidating,
    Unlinked,
    Retired(ExecutionEpoch),
}

struct UnitRecord {
    code: Arc<Accounted<CodeUnit>>,
    static_sites: Accounted<Box<[links::SourceSite]>>,
    published: ExecutionEpoch,
    lifecycle: Lifecycle,
    slots: Accounted<Box<[Handle<DispatchSlot>]>>,
    family: Option<Handle<Arc<Accounted<Family>>>>,
    retirement: Option<(Reason, MaintenanceSequence)>,
    retirement_next: Option<Handle<UnitRecord>>,
    // Memory safety work may join an already queued eviction/tier cutover.
    // Keep both acknowledgements tied to this exact unit until unlink.
    invalidation: Option<MaintenanceSequence>,
    detached_epoch: Option<ExecutionEpoch>,
    detached_table: Option<Arc<Accounted<Table>>>,
    outgoing: Option<Handle<links::Link>>,
    incoming: Option<Handle<links::Link>>,
    pic_outgoing: Option<super::pic::Site>,
    pic_incoming: Option<super::pic::Site>,
}

/// Intrusive pending-only lists: membership is exactly `retirement.is_some()`.
/// Links live in the accounted registry slots, so queuing never allocates and
/// draining never searches resident units or holes left by reclaimed units.
#[derive(Default)]
struct Retirements {
    lcq: Option<Handle<UnitRecord>>,
    hcq: Option<Handle<UnitRecord>>,
}
impl Retirements {
    fn head_mut(&mut self, tier: Tier) -> &mut Option<Handle<UnitRecord>> {
        match tier {
            Tier::Lcq => &mut self.lcq,
            Tier::Hcq => &mut self.hcq,
        }
    }

    fn next(&self) -> Option<Handle<UnitRecord>> {
        // Release active HCQ baseline promises before selecting LCQ.
        self.hcq.or(self.lcq)
    }
}
impl UnitRecord {
    fn queue_retirement(
        &mut self,
        handle: Handle<UnitRecord>,
        pending: &mut Retirements,
        reason: Reason,
        sequence: MaintenanceSequence,
    ) {
        if self.retirement.is_none() {
            let head = pending.head_mut(self.code.tier);
            self.retirement_next = *head;
            *head = Some(handle);
        }
        self.retirement = Some((reason, sequence));
    }
}

struct Family {
    id: HcqFamilyId,
    version: FamilyVersion,
    unit: Arc<Accounted<CodeUnit>>,
    // Active family ownership pins baseline code. The code registry also keeps
    // each unit until epoch/ref reclamation; these are not raw entry promises.
    baselines: Box<[BaselinePin]>,
}

struct BaselinePin(Arc<Accounted<CodeUnit>>);
impl BaselinePin {
    // Only create new pins under state while the unit is eligible. A pin owns
    // an Arc, so the count cannot exceed Arc's own checked reference bound.
    fn new(unit: &Arc<Accounted<CodeUnit>>) -> Self {
        let unit = Arc::clone(unit);
        unit.baseline_pins.fetch_add(1, Ordering::Relaxed);
        Self(unit)
    }
}
impl std::ops::Deref for BaselinePin {
    type Target = Arc<Accounted<CodeUnit>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl Drop for BaselinePin {
    fn drop(&mut self) {
        self.0.baseline_pins.fetch_sub(1, Ordering::Relaxed);
    }
}

#[derive(Clone, Copy)]
struct Dependency {
    page: CodePageDependency,
    unit: UnitHandle,
}
struct DependencyIndex {
    entries: hashbrown::HashTable<Dependency>,
    hash: RandomState,
}
impl DependencyIndex {
    fn new(capacity: usize) -> Self {
        Self {
            entries: hashbrown::HashTable::with_capacity(capacity),
            hash: RandomState::new(),
        }
    }
    fn insert(&mut self, dependency: Dependency) {
        // Hash physical identity so invalidation can find all virtual aliases;
        // retain the exact mapping generation in each association. Associations
        // are inserted once per unit/page and removed only at reclamation.
        self.entries.insert_unique(
            self.hash.hash_one(dependency.page.page),
            dependency,
            |dependency| self.hash.hash_one(dependency.page.page),
        );
    }
    fn for_page(
        &self,
        page: nixe_memory::GuestPhysicalPageId,
    ) -> impl Iterator<Item = &Dependency> {
        self.entries
            .iter_hash(self.hash.hash_one(page))
            .filter(move |dependency| dependency.page.page == page)
    }
}

struct RetiredTable {
    table: Arc<Accounted<Table>>,
    epoch: ExecutionEpoch,
}

pub(super) struct Units {
    records: Registry<UnitRecord>,
    // Includes retired/detached records until their last reference and epoch
    // permit removal. Directory-table presence alone is not sufficient.
    segment_records: [usize; SEGMENTS],
    segment_retired: [usize; SEGMENTS],
    links: links::Links,
    static_sites: links::StaticSites,
    retirements: Retirements,
    families: Registry<Arc<Accounted<Family>>>,
    ids: CheckedCounter<CodeUnitId>,
    versions: CheckedCounter<CodeVersion>,
    family_ids: CheckedCounter<HcqFamilyId>,
    family_versions: CheckedCounter<FamilyVersion>,
    tables: [Option<Arc<Accounted<Table>>>; SEGMENTS],
    retired_tables: Vec<RetiredTable>,
    dependencies: DependencyIndex,
    record_storage: Option<MetadataLease>,
    family_storage: Option<MetadataLease>,
    retired_storage: Option<MetadataLease>,
    dependency_storage: Option<MetadataLease>,
    static_site_storage: Option<MetadataLease>,
    hcq_failure: Option<Error>,
    collecting: bool,
    decommitting: [bool; SEGMENTS],
    shutdown_finished: bool,
}
impl Default for Units {
    fn default() -> Self {
        Self {
            records: Registry::default(),
            segment_records: [0; SEGMENTS],
            segment_retired: [0; SEGMENTS],
            links: links::Links::default(),
            static_sites: links::StaticSites::new(0),
            retirements: Retirements::default(),
            families: Registry::default(),
            ids: CheckedCounter::default(),
            versions: CheckedCounter::default(),
            family_ids: CheckedCounter::default(),
            family_versions: CheckedCounter::default(),
            tables: std::array::from_fn(|_| None),
            retired_tables: Vec::new(),
            dependencies: DependencyIndex::new(0),
            record_storage: None,
            family_storage: None,
            retired_storage: None,
            dependency_storage: None,
            static_site_storage: None,
            hcq_failure: None,
            collecting: false,
            decommitting: [false; SEGMENTS],
            shutdown_finished: false,
        }
    }
}
impl Units {
    // O(pending units), O(1) when empty. Sequences need not follow list order:
    // repeated requests retain old invalidation work alongside another reason.
    pub(super) fn pending_retirement(&self, reason: Reason, sequence: MaintenanceSequence) -> bool {
        (reason == Reason::Shutdown && !self.shutdown_finished)
            || [self.retirements.hcq, self.retirements.lcq]
                .into_iter()
                .any(|mut next| {
                    while let Some(handle) = next {
                        let record = self.records.get(handle).unwrap();
                        if (reason == Reason::MappingChange
                            && record
                                .invalidation
                                .is_some_and(|pending| pending <= sequence))
                            || record.retirement.is_some_and(|(kind, pending)| {
                                kind == reason && pending <= sequence
                            })
                        {
                            return true;
                        }
                        next = record.retirement_next;
                    }
                    false
                })
    }

    // Selection and removal hold the same state lock. Failed unlinks leave
    // the head queued; cancellation/success removes it before slot reclamation.
    fn finish_retirement(&mut self, handle: Handle<UnitRecord>) {
        let record = self.records.get_mut(handle).unwrap();
        let head = self.retirements.head_mut(record.code.tier);
        debug_assert_eq!(*head, Some(handle));
        *head = record.retirement_next.take();
        record.retirement = None;
        record.invalidation = None;
    }

    pub(super) fn mark_shutdown(&mut self, sequence: MaintenanceSequence) {
        for (handle, record) in self.records.iter_mut() {
            if !matches!(record.lifecycle, Lifecycle::Retired(_)) {
                record.lifecycle = Lifecycle::Invalidating;
                record.queue_retirement(handle, &mut self.retirements, Reason::Shutdown, sequence);
            }
        }
    }
    fn overlaps_family(&self, instructions: &[Instruction]) -> bool {
        self.families.values().any(|family| {
            family
                .unit
                .instructions
                .iter()
                .any(|old| instructions.iter().any(|new| new.key == old.key))
        })
    }
}

fn recipe_bytes(recipe: &LazyFlags<ValueLocation>) -> usize {
    match recipe {
        LazyFlags::Conditional { when_true, .. } => {
            size_of_val(&**when_true) + recipe_bytes(when_true)
        }
        _ => 0,
    }
}
fn nzcv_bytes(nzcv: &NzcvLocation) -> usize {
    match nzcv {
        NzcvLocation::Deferred(recipe) => recipe_bytes(recipe),
        _ => 0,
    }
}
impl Input {
    fn metadata_bytes(&self) -> usize {
        // Installed already charges its own inline storage and backend metadata.
        size_of::<Accounted<CodeUnit>>() - size_of::<Installed>()
            + 2 * size_of::<usize>()
            + size_of_val(&*self.instructions)
            + size_of_val(&*self.entries)
            + size_of_val(&*self.dependencies)
            + size_of_val(&*self.states)
            + size_of_val(&*self.faults)
            + self
                .entries
                .iter()
                .map(|entry| {
                    size_of_val(&*entry.contract.bindings) + nzcv_bytes(&entry.contract.nzcv)
                })
                .sum::<usize>()
            + self
                .states
                .iter()
                .map(|map| {
                    size_of_val(&*map.state.bindings)
                        + nzcv_bytes(&map.state.nzcv)
                        + map
                            .transfer
                            .as_ref()
                            .map_or(0, |transfer| size_of_val(&**transfer))
                })
                .sum::<usize>()
    }

    fn validate(&self, process: &Lifetime, publications: &[Publication<'_>]) -> Result<(), Error> {
        let fail = Error::InvalidUnit;
        if !self.code.allocation.belongs_to(&process.cache) {
            return Err(fail(
                "executable allocation belongs to a different process cache",
            ));
        }
        if self.code.allocation.tier != self.tier {
            return Err(fail("unit tier differs from its executable allocation"));
        }
        if self.identity.process != process.identity || self.identity.tier != self.tier {
            return Err(fail("emission identity belongs to another process or tier"));
        }
        if self.entries.is_empty()
            || self.instructions.is_empty()
            || publications.len() != self.entries.len()
        {
            return Err(fail(
                "unit needs an instruction image and one publication per entry",
            ));
        }
        let semantic = self.entries[0].key;
        for (i, instruction) in self.instructions.iter().enumerate() {
            let key = instruction.key.block_key();
            if semantic.at(key.pc) != Some(key)
                || self.instructions[..i]
                    .iter()
                    .any(|old| old.key == instruction.key)
            {
                return Err(fail(
                    "duplicate instruction or mixed semantic address spaces",
                ));
            }
        }
        for (i, (entry, publication)) in self.entries.iter().zip(publications).enumerate() {
            if !std::ptr::eq(publication.process, process) || publication.key != entry.key {
                return Err(Error::StalePublication);
            }
            if !self
                .instructions
                .iter()
                .any(|instruction| instruction.key.block_key() == entry.key)
                || self.entries[..i].iter().any(|other| other.key == entry.key)
            {
                return Err(fail(
                    "entry is duplicate or absent from the instruction image",
                ));
            }
            for offset in [entry.canonical_offset, entry.fast_offset] {
                if offset as usize >= self.code.allocation.len()
                    || (self.code.metadata.abi == HostAbi::Aarch64 && offset % 4 != 0)
                {
                    return Err(fail("entry offset is outside code or unaligned"));
                }
            }
            if entry.contract.abi != self.code.metadata.abi
                || !self
                    .code
                    .metadata
                    .entries
                    .iter()
                    .any(|(_, offset)| *offset == entry.fast_offset)
            {
                return Err(fail(
                    "fast entry does not match the final backend ABI/label",
                ));
            }
            entry.contract.validate().map_err(fail)?;
        }
        for (i, dependency) in self.dependencies.iter().enumerate() {
            if self.dependencies[..i].contains(dependency) {
                return Err(fail("duplicate code dependency"));
            }
        }
        for (index, map) in self.states.iter().enumerate() {
            if map.state.abi != self.code.metadata.abi
                || map.state.site.source != self.identity.version
                || map.state.site.state_map as usize != index
                || map.native_offset as usize >= self.code.allocation.len()
            {
                return Err(fail("invalid semantic state-map identity, ABI or offset"));
            }
            let backend = self
                .code
                .metadata
                .states
                .iter()
                .chain(self.code.metadata.faults.iter())
                .find(|backend| !backend.entry && backend.offset == map.native_offset)
                .ok_or_else(|| fail("semantic state map has no final backend boundary"))?;
            if let Some(transfer) = &map.transfer {
                let patch_bytes = if map.state.abi == HostAbi::X86_64 {
                    8
                } else {
                    4
                };
                if map.exit.is_none()
                    || !transfer.destination.valid(map.state.abi, 8)
                    || transfer.completed > 2048
                    || transfer.patch_bytes != patch_bytes
                    || transfer.patch_bytes != backend.patch_bytes
                    || transfer.poll_offset != backend.poll.map(|poll| poll.offset)
                    || backend
                        .poll
                        .is_some_and(|poll| poll.completed != transfer.completed)
                    || !map.native_offset.is_multiple_of(u32::from(patch_bytes))
                    || map.native_offset as usize + usize::from(patch_bytes)
                        > self.code.allocation.len()
                    || transfer.fallback_offset as usize >= self.code.allocation.len()
                    || (map.state.abi == HostAbi::Aarch64 && transfer.fallback_offset % 4 != 0)
                    || transfer.static_target.is_some_and(|target| {
                        self.instructions[0].key.block_key().at(target.pc) != Some(target)
                            || !map.exit.is_some_and(|exit| {
                                matches!(
                                    exit.kind,
                                    EdgeKind::Static
                                        | EdgeKind::Taken
                                        | EdgeKind::NotTaken
                                        | EdgeKind::Call
                                        | EdgeKind::FragmentLimit
                                )
                            })
                    })
                {
                    return Err(fail("invalid terminal transfer contract"));
                }
            }
            map.state.validate().map_err(fail)?;
        }
        let static_exits = self
            .states
            .iter()
            .filter(|map| {
                map.transfer
                    .as_ref()
                    .is_some_and(|transfer| transfer.static_target.is_some())
            })
            .count();
        if static_exits > self.code.allocation.island_count() {
            return Err(fail(
                "static exits exceed the source's reserved island capacity",
            ));
        }
        if self.code.metadata.states.iter().any(|backend| {
            backend.patch_bytes != 0
                && !self
                    .states
                    .iter()
                    .any(|map| map.native_offset == backend.offset)
        }) {
            return Err(fail("backend exit has no semantic state map"));
        }
        for (i, fault) in self.faults.iter().enumerate() {
            if fault.native_start >= fault.native_end
                || fault.native_end as usize > self.code.allocation.len()
                || (i != 0 && self.faults[i - 1].native_end > fault.native_start)
                || !matches!(fault.bytes, 1 | 2 | 4 | 8 | 16)
                || !self
                    .instructions
                    .iter()
                    .any(|instruction| instruction.key == fault.instruction)
            {
                return Err(fail("invalid/overlapping fault interval or guest access"));
            }
            let map = self
                .states
                .get(fault.state_map as usize)
                .ok_or(fail("missing prefault state map"))?;
            if let Some(location) = fault.completed_read
                && (fault.access != Access::Read
                    || fault.subaccess != 1
                    || fault.commit_stage != 0
                    || !matches!(fault.bytes, 4 | 8 | 16)
                    || !location.valid(self.code.metadata.abi, fault.bytes))
            {
                return Err(fail("invalid retained pair-read location or stage"));
            }
            if map.native_offset != fault.native_start
                || !self.code.metadata.faults.iter().any(|map| {
                    map.offset == fault.native_start
                        && map.offset.checked_add(u32::from(map.fault_bytes))
                            == Some(fault.native_end)
                })
            {
                return Err(fail(
                    "prefault map does not name the exact final faulting instruction",
                ));
            }
            if self.code.metadata.abi == HostAbi::Aarch64
                && (fault.native_start % 4 != 0 || fault.native_end - fault.native_start != 4)
            {
                return Err(fail(
                    "AArch64 fault interval must name one aligned instruction",
                ));
            }
            if self.code.metadata.abi == HostAbi::X86_64
                && fault.native_end - fault.native_start > 15
            {
                return Err(fail("x86-64 fault interval exceeds one instruction"));
            }
        }
        if self.code.metadata.faults.iter().any(|map| {
            !self
                .faults
                .iter()
                .any(|fault| fault.native_start == map.offset)
        }) {
            return Err(fail("backend fault has no semantic record"));
        }
        Ok(())
    }
}

impl Lifetime {
    pub(crate) fn begin_unit(&self, tier: Tier) -> Result<EmissionIdentity, Error> {
        let mut state = self.lock();
        let admission = state.open()?;
        if tier == Tier::Hcq
            && let Some(error) = state.units.hcq_failure
        {
            return Err(error);
        }
        let result = state.units.ids.next_id();
        let id = self.publication_identity(&mut state, result, tier)?;
        let result = state.units.versions.next_id();
        let version = self.publication_identity(&mut state, result, tier)?;
        Ok(EmissionIdentity {
            process: self.identity,
            admission,
            tier,
            id,
            version,
        })
    }

    fn publication_failure(&self, state: &mut super::State, error: Error, tier: Tier) {
        if tier == Tier::Hcq {
            // An optimizer identity limit disables further HCQ publication,
            // not execution of the installed baseline. A required LCQ/lifecycle
            // operation still fails closed if its own counter is exhausted.
            state.units.hcq_failure.get_or_insert(error);
        } else {
            self.fail(state, error);
        }
    }

    fn publication_identity<T: crate::abi::Identity>(
        &self,
        state: &mut super::State,
        result: Result<T, crate::abi::IdentityExhausted>,
        tier: Tier,
    ) -> Result<T, Error> {
        result
            .map_err(Error::Exhausted)
            .inspect_err(|error| self.publication_failure(state, *error, tier))
    }

    /// Prepare complete output without holding state during allocation or table
    /// construction. `cursor` is the memory authority's stable notification:
    /// the captured image must have been frozen at Input.cursor, and future
    /// mapping/content transitions must serialize through this coordinator.
    /// Synthetic units can have no guest dependencies; no guest fetch occurs here.
    pub(crate) fn prepare_unit<'a>(
        &'a self,
        publications: &[Publication<'a>],
        input: Input,
        cursor: &'a AtomicU64,
    ) -> Result<PreparedUnit<'a>, Error> {
        input.validate(self, publications)?;
        self.collect_tables()?;
        let static_sites = input.source_sites();
        let bytes = size_of_val(&*static_sites);
        let static_sites = self.cache.account(static_sites, bytes, input.tier)?;
        self.grow_units(input.tier, input.dependencies.len(), static_sites.len())?;
        self.reserve_publication_links(input.tier, static_sites.len(), &input.entries)?;
        let publications = publications.to_vec().into_boxed_slice();
        let bytes = size_of_val(&*publications);
        let publications = self.cache.account(publications, bytes, input.tier)?;
        let payloads = Vec::with_capacity(publications.len());
        let bytes = payloads.capacity() * size_of::<DispatchPayload>();
        let mut payloads = self.cache.account(payloads, bytes, input.tier)?;
        let baselines = Vec::with_capacity(if input.tier == Tier::Hcq {
            input.instructions.len() + publications.len()
        } else {
            0
        });
        let bytes = baselines.capacity() * size_of::<BaselinePin>();
        let mut baselines = self.cache.account(baselines, bytes, input.tier)?;
        let segment = input.code.allocation.segment;
        let generation = input.code.allocation.generation;
        // All vectors used below have charged storage before the state lock.
        let (id, version, family_identity, table) = {
            let mut state = self.lock();
            for publication in publications.iter() {
                state.validate(publication)?;
            }
            if input.identity.admission != state.admission {
                return Err(Error::StalePublication);
            }
            if input.tier == Tier::Hcq
                && let Some(error) = state.units.hcq_failure
            {
                return Err(error);
            }
            if cursor.load(Ordering::Acquire) != input.cursor.get() {
                return Err(Error::StalePublication);
            }
            if input.tier == Tier::Hcq {
                // Family discovery/reshape is later work. Initial family
                // publication already rejects overlap and pins each baseline.
                if state.units.overlaps_family(&input.instructions) {
                    return Err(Error::InvalidUnit(
                        "HCQ overlap requires coordinated family replacement",
                    ));
                }
                for instruction in &input.instructions {
                    let record = state
                        .units
                        .records
                        .values()
                        .find(|record| {
                            record.code.tier == Tier::Lcq
                                && record.lifecycle == Lifecycle::Published
                                && record.code.instructions.iter().any(|old| {
                                    old.key == instruction.key && old.bits == instruction.bits
                                })
                        })
                        .ok_or(Error::InvalidUnit(
                            "HCQ instruction has no matching resident LCQ image",
                        ))?;
                    if !baselines
                        .iter()
                        .any(|unit: &BaselinePin| unit.id == record.code.id)
                    {
                        baselines.value.push(BaselinePin::new(&record.code));
                    }
                }
            }
            for publication in publications.iter() {
                let old = state.dispatch.get(publication.slot).unwrap().snapshot();
                if input.tier == Tier::Hcq {
                    let lcq = old
                        .lcq()
                        .ok_or(Error::InvalidUnit("HCQ entry has no resident LCQ baseline"))?;
                    let owner = state.dispatch.get(publication.slot).unwrap().owners[0]
                        .ok_or(Error::StalePublication)?;
                    let record = state
                        .units
                        .records
                        .get(owner.unit.0)
                        .filter(|record| {
                            record.code.id == lcq.unit
                                && record.code.version == lcq.version
                                && record.lifecycle == Lifecycle::Published
                        })
                        .ok_or(Error::StalePublication)?;
                    if !baselines
                        .iter()
                        .any(|unit: &BaselinePin| unit.id == record.code.id)
                    {
                        baselines.value.push(BaselinePin::new(&record.code));
                    }
                }
                payloads.value.push(old);
            }
            let id = input.identity.id;
            let version = input.identity.version;
            let identity = if input.tier == Tier::Hcq {
                let result = state.units.family_ids.next_id();
                let id = self.publication_identity(&mut state, result, input.tier)?;
                let result = state.units.family_versions.next_id();
                Some((
                    id,
                    self.publication_identity(&mut state, result, input.tier)?,
                ))
            } else {
                None
            };
            // Reserve fresh payload identities before constructing immutable boxes.
            for payload in &mut payloads.value {
                let result = state.reachabilities.next_id();
                let reachability = self.publication_identity(&mut state, result, input.tier)?;
                *payload = DispatchPayload::new(reachability, payload.lcq(), payload.hcq());
            }
            (id, version, identity, state.units.tables[segment].clone())
        };
        if let Some(table) = &table
            && table.generation != generation
        {
            return Err(Error::StalePublication);
        }
        let tier = input.tier;
        let bytes = input.metadata_bytes();
        let unit = Arc::new(self.cache.account(
            CodeUnit {
                id,
                version,
                abi_version: NATIVE_ABI_VERSION,
                input,
                baseline_pins: AtomicUsize::new(0),
                registration: std::sync::OnceLock::new(),
            },
            bytes,
            tier,
        )?);
        let mut intervals = table
            .as_ref()
            .map_or_else(Vec::new, |old| old.intervals.to_vec());
        intervals.push(Interval {
            start: unit.code.allocation.address(),
            end: unit.code.allocation.address() + unit.code.allocation.len(),
            unit: &unit.value,
        });
        intervals.sort_unstable_by_key(|interval| interval.start);
        if intervals.windows(2).any(|pair| pair[0].end > pair[1].start) {
            return Err(Error::InvalidUnit(
                "native unit intervals overlap published code",
            ));
        }
        let bytes = size_of::<Accounted<Table>>()
            + 2 * size_of::<usize>()
            + intervals.capacity() * size_of::<Interval>();
        let next_table = Arc::new(self.cache.account(
            Table {
                generation,
                intervals,
            },
            bytes,
            tier,
        )?);
        let family = if let Some((id, version)) = family_identity {
            let Accounted {
                value: baselines,
                charge,
            } = baselines;
            let baselines = baselines.into_boxed_slice();
            let bytes =
                size_of::<Accounted<Family>>() + 2 * size_of::<usize>() + size_of_val(&*baselines);
            let family = Arc::new(self.cache.account(
                Family {
                    id,
                    version,
                    unit: Arc::clone(&unit),
                    baselines,
                },
                bytes,
                tier,
            )?);
            drop(charge);
            Some(family)
        } else {
            None
        };
        let payload_boxes = unit
            .entries
            .iter()
            .zip(payloads.iter())
            .map(|(entry, old)| {
                let (lcq, hcq) = match &family {
                    Some(family) => (
                        old.lcq(),
                        Some(HcqEntry {
                            entry: unit.entry(entry),
                            family: family.id,
                            family_version: family.version,
                        }),
                    ),
                    None => (Some(unit.entry(entry)), old.hcq()),
                };
                self.cache
                    .account(
                        DispatchPayload::new(old.reachability(), lcq, hcq),
                        size_of::<Accounted<DispatchPayload>>(),
                        tier,
                    )
                    .map(|payload| Some(Box::new(payload)))
            })
            .collect::<Result<Box<[_]>, _>>()?;
        // Publication consumes/returns payload boxes in place; no allocation or
        // destructor which could take the cache lock runs inside its state lock.
        let bytes = size_of_val(&*payload_boxes);
        let payload_boxes = self.cache.account(payload_boxes, bytes, tier)?;
        let slots: Box<[_]> = publications
            .iter()
            .map(|publication| publication.slot)
            .collect();
        let bytes = size_of_val(&*slots);
        let slots = self.cache.account(slots, bytes, tier)?;
        Ok(PreparedUnit {
            process: self,
            publications,
            cursor,
            unit: Some(unit),
            family,
            previous_table: table,
            table: Some(next_table),
            payloads: payload_boxes,
            slots: Some(slots),
            static_sites: Some(static_sites),
        })
    }

    fn grow_units(
        &self,
        tier: Tier,
        dependency_count: usize,
        static_count: usize,
    ) -> Result<(), Error> {
        loop {
            let (records, families, retired, dependencies, static_sites) = {
                let state = self.lock();
                state.open()?;
                let records = if state.units.records.has_space() {
                    0
                } else {
                    state.units.records.capacity().saturating_mul(2).max(16)
                };
                let families = if tier == Tier::Lcq || state.units.families.has_space() {
                    0
                } else {
                    state.units.families.capacity().saturating_mul(2).max(16)
                };
                let retired =
                    if state.units.retired_tables.len() < state.units.retired_tables.capacity() {
                        0
                    } else {
                        state
                            .units
                            .retired_tables
                            .capacity()
                            .saturating_mul(2)
                            .max(16)
                    };
                let needed = state
                    .units
                    .dependencies
                    .entries
                    .len()
                    .checked_add(dependency_count)
                    .ok_or(Error::Capacity("dependency index size overflow"))?;
                let dependencies = if needed <= state.units.dependencies.entries.capacity() {
                    0
                } else {
                    needed
                        .max(
                            state
                                .units
                                .dependencies
                                .entries
                                .capacity()
                                .saturating_mul(2),
                        )
                        .max(16)
                };
                let sites = &state.units.static_sites.entries;
                let needed = sites
                    .len()
                    .checked_add(static_count)
                    .ok_or(Error::Capacity("static source index size overflow"))?;
                let static_sites = if needed <= sites.capacity() {
                    0
                } else {
                    needed.max(sites.capacity().saturating_mul(2)).max(16)
                };
                if records == 0
                    && families == 0
                    && retired == 0
                    && dependencies == 0
                    && static_sites == 0
                {
                    return Ok(());
                }
                (records, families, retired, dependencies, static_sites)
            };
            let records = Vec::with_capacity(records);
            let bytes = records.capacity() * size_of::<Slot<UnitRecord>>();
            let mut records = PreparedStorage::for_tier(records, bytes, &self.cache, tier)?;
            let families = Vec::with_capacity(families);
            let bytes = families.capacity() * size_of::<Slot<Arc<Accounted<Family>>>>();
            let mut families = PreparedStorage::for_tier(families, bytes, &self.cache, tier)?;
            let retired = Vec::with_capacity(retired);
            let bytes = retired.capacity() * size_of::<RetiredTable>();
            let mut retired = PreparedStorage::for_tier(retired, bytes, &self.cache, tier)?;
            let dependencies = DependencyIndex::new(dependencies);
            let bytes = dependencies.entries.allocation_size();
            let mut dependencies =
                PreparedStorage::for_tier(dependencies, bytes, &self.cache, tier)?;
            let static_sites = links::StaticSites::new(static_sites);
            let bytes = static_sites.entries.allocation_size();
            let mut static_sites =
                PreparedStorage::for_tier(static_sites, bytes, &self.cache, tier)?;
            let mut state = self.lock();
            state.open()?;
            if static_sites.value.entries.capacity() > state.units.static_sites.entries.capacity() {
                for site in state.units.static_sites.entries.drain() {
                    static_sites.value.insert(site);
                }
                std::mem::swap(&mut static_sites.value, &mut state.units.static_sites);
                std::mem::swap(
                    &mut static_sites.charge,
                    &mut state.units.static_site_storage,
                );
            }
            if dependencies.value.entries.capacity() > state.units.dependencies.entries.capacity() {
                for dependency in state.units.dependencies.entries.drain() {
                    dependencies.value.insert(dependency);
                }
                std::mem::swap(&mut dependencies.value, &mut state.units.dependencies);
                std::mem::swap(
                    &mut dependencies.charge,
                    &mut state.units.dependency_storage,
                );
            }
            if records.value.capacity() > state.units.records.capacity() {
                state.units.records.grow(&mut records.value);
                std::mem::swap(&mut records.charge, &mut state.units.record_storage);
            }
            if families.value.capacity() > state.units.families.capacity() {
                state.units.families.grow(&mut families.value);
                std::mem::swap(&mut families.charge, &mut state.units.family_storage);
            }
            if retired.value.capacity() > state.units.retired_tables.capacity() {
                retired.value.append(&mut state.units.retired_tables);
                std::mem::swap(&mut retired.value, &mut state.units.retired_tables);
                std::mem::swap(&mut retired.charge, &mut state.units.retired_storage);
            }
        }
    }

    /// Cold snapshot collection. Records named by any still-readable table
    /// remain in the unit registry until the directory grace period completes.
    pub(crate) fn collect_tables(&self) -> Result<usize, Error> {
        let mut count = 0;
        loop {
            let removed = {
                let mut state = self.lock();
                state.healthy()?;
                let Some(index) = state
                    .units
                    .retired_tables
                    .iter()
                    .position(|table| state.quiescent(table.epoch))
                else {
                    return Ok(count);
                };
                state.units.retired_tables.swap_remove(index)
            };
            drop(removed);
            count += 1;
        }
    }
}

fn same_snapshot<T>(left: &Option<Arc<T>>, right: &Option<Arc<T>>) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => Arc::ptr_eq(left, right),
        (None, None) => true,
        _ => false,
    }
}

pub(crate) struct PreparedUnit<'a> {
    process: &'a Lifetime,
    publications: Accounted<Box<[Publication<'a>]>>,
    cursor: &'a AtomicU64,
    unit: Option<Arc<Accounted<CodeUnit>>>,
    family: Option<Arc<Accounted<Family>>>,
    previous_table: Option<Arc<Accounted<Table>>>,
    table: Option<Arc<Accounted<Table>>>,
    payloads: Accounted<Box<[Option<OwnedPayload>]>>,
    slots: Option<Accounted<Box<[Handle<DispatchSlot>]>>>,
    static_sites: Option<Accounted<Box<[links::SourceSite]>>>,
}
type OwnedPayload = Box<Accounted<DispatchPayload>>;
impl PreparedUnit<'_> {
    pub(crate) fn publish(mut self) -> Result<UnitHandle, Error> {
        let process = self.process;
        let mut state = process.lock();
        for publication in self.publications.iter() {
            state.validate(publication)?;
        }
        let unit = Arc::clone(self.unit.as_ref().unwrap());
        let tier = unit.tier;
        if tier == Tier::Hcq
            && let Some(error) = state.units.hcq_failure
        {
            return Err(error);
        }
        let segment = unit.code.allocation.segment;
        if self.cursor.load(Ordering::Acquire) != unit.cursor.get()
            || state.units.decommitting[segment]
            || !state.units.records.has_space()
            || (self.family.is_some() && !state.units.families.has_space())
            || !same_snapshot(&state.units.tables[segment], &self.previous_table)
            || state.units.dependencies.entries.capacity() - state.units.dependencies.entries.len()
                < unit.dependencies.len()
            || state.units.static_sites.entries.capacity() - state.units.static_sites.entries.len()
                < self.static_sites.as_ref().unwrap().len()
            || (self.previous_table.is_some()
                && state.units.retired_tables.len() == state.units.retired_tables.capacity())
        {
            return Err(Error::StalePublication);
        }
        let handle = state
            .units
            .records
            .next_handle()
            .inspect_err(|error| process.publication_failure(&mut state, *error, tier))?;
        if self.family.is_some() {
            state
                .units
                .families
                .next_handle()
                .inspect_err(|error| process.publication_failure(&mut state, *error, tier))?;
            if state.units.overlaps_family(&unit.instructions) {
                return Err(Error::StalePublication);
            }
        }
        // Resolve and validate all outgoing roots before mutation. New sources
        // and their links become registered together under this same lock.
        let source = UnitHandle(handle, process.identity);
        let link_count = self
            .check_outgoing_links(&state, source, &unit)?
            .checked_add(self.check_waiting_links(&state, source, &unit)?)
            .ok_or(Error::Capacity("publication link count overflow"))?;
        state
            .units
            .check_publication_link_capacity(link_count)
            .inspect_err(|error| {
                if matches!(error, Error::Exhausted(_)) {
                    process.publication_failure(&mut state, *error, tier);
                }
            })?;
        // Reserve epochs and any cutover request before reachable mutation.
        // The retirement stamp covers ANY
        // invocations that could have loaded the replaced table, including
        // readers executing another unit in the same segment.
        let retired = state.execution;
        let result = state.executions.next_id();
        let next_epoch = process.publication_identity(&mut state, result, tier)?;
        let cutover = if tier == Tier::Lcq
            && self.publications.iter().any(|publication| {
                state
                    .dispatch
                    .get(publication.slot)
                    .unwrap()
                    .snapshot()
                    .lcq()
                    .is_some()
            }) {
            Some(
                process
                    .request_locked(&mut state, Reason::TierCutover)?
                    .sequence,
            )
        } else {
            None
        };
        let link_sequence = if link_count != 0 {
            Some(
                process
                    .request_locked(&mut state, Reason::LinkPatch)?
                    .sequence,
            )
        } else {
            None
        };

        for page in &*unit.dependencies {
            state.units.dependencies.insert(Dependency {
                page: *page,
                unit: UnitHandle(handle, process.identity),
            });
        }

        let record = UnitRecord {
            code: self.unit.take().unwrap(),
            static_sites: self.static_sites.take().unwrap(),
            published: retired,
            lifecycle: Lifecycle::Published,
            slots: self.slots.take().unwrap(),
            family: None,
            retirement: None,
            retirement_next: None,
            invalidation: None,
            detached_epoch: None,
            detached_table: None,
            outgoing: None,
            incoming: None,
            pic_outgoing: None,
            pic_incoming: None,
        };
        let inserted = state
            .units
            .records
            .insert(&mut Some(record))
            .expect("validated unit insertion");
        debug_assert_eq!(inserted, handle);
        state.units.segment_records[segment] += 1;
        unit.registration
            .set(source)
            .expect("single-use unit publication");
        // Discovery and the source payload share one publication point. These
        // weak associations neither expose code nor retain target ownership.
        state
            .units
            .insert_static_source(UnitHandle(handle, process.identity));
        if let Some(sequence) = link_sequence {
            self.insert_outgoing_links(&mut state, source, &unit, sequence);
            self.insert_waiting_links(&mut state, source, &unit, sequence);
        }
        if self.family.is_some() {
            let family = state
                .units
                .families
                .insert(&mut self.family)
                .expect("validated family insertion");
            state.units.records.get_mut(handle).unwrap().family = Some(family);
        }
        // Swap table owners before the signal-visible pointer. Old tables
        // remain owned by the epoch retire list and any compiler preparation.
        std::mem::swap(&mut state.units.tables[segment], &mut self.table);
        if let Some(table) = self.table.take() {
            state.units.retired_tables.push(RetiredTable {
                table,
                epoch: retired,
            });
        }
        unsafe {
            process.directory.publish(
                segment,
                Arc::as_ptr(state.units.tables[segment].as_ref().unwrap()),
            );
        }
        state.execution = next_epoch;
        for (index, (publication, payload)) in self
            .publications
            .iter()
            .zip(self.payloads.value.iter_mut())
            .enumerate()
        {
            let slot = state.dispatch.get_mut(publication.slot).unwrap();
            slot.units += 1;
            let previous = slot.owners;
            let mut owners = previous;
            owners[if tier == Tier::Lcq { 0 } else { 1 }] = Some(UnitEntry {
                unit: UnitHandle(handle, process.identity),
                index,
            });
            let old = slot.replace(payload.take().unwrap(), owners);
            if let Some(sequence) = cutover
                && let Some(old) = previous[0]
            {
                let handle = old.unit.0;
                let units = &mut state.units;
                let record = units.records.get_mut(handle).unwrap();
                record.lifecycle = Lifecycle::Superseded;
                record.queue_retirement(
                    handle,
                    &mut units.retirements,
                    Reason::TierCutover,
                    sequence,
                );
            }
            *payload = Some(old);
        }
        process.changed.notify_all();
        // No publication can fail here. Old payload owners drop only
        // after unlocking state; the unit registry now owns all executable bytes.
        drop(state);
        Ok(UnitHandle(handle, process.identity))
    }
}
