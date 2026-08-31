//! Linux native-fault runtime shared by CPU execution frontends.
//!
//! The signal handler performs only bounded slot lookup, attribution, context
//! capture, and register redirection. Emulator policy runs after `sigreturn`
//! on a preallocated dispatcher stack.

#![cfg(target_os = "linux")]

use std::cell::UnsafeCell;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Display, Formatter};
use std::mem::{ManuallyDrop, MaybeUninit, size_of};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use nixe_cpu::memory::{
    CpuMemory, DataAccessFault, DataAccessKind, DirectFaultResolution, MemoryAccess,
    MemoryAccessClass, MemoryAccessSize, MemoryOrdering, MemoryValue,
};
use nixe_memory::{
    AddressSpaceId, DIRECT_PAGE_SIZE, DirectAddressSpaceView, DirectStoreControl,
    GuestVirtualAddress,
};

const MAX_WORKER_SLOTS: usize = 128;
const SIGNAL_STACK_SIZE: usize = 64 * 1024;
const DISPATCH_STACK_SIZE: usize = 64 * 1024;
// Linux UAPI `siginfo.h` values not exported by every libc target module.
const LINUX_SEGV_MAPERR: i32 = 1;
const LINUX_SEGV_ACCERR: i32 = 2;
#[cfg(target_arch = "x86_64")]
const MAX_X86_FPSTATE_SIZE: usize = 64 * 1024;
#[cfg(target_arch = "x86_64")]
const FP_XSTATE_MAGIC1: u32 = 0x4650_5853;
#[cfg(target_arch = "x86_64")]
const FP_XSTATE_SW_BYTES_OFFSET: usize = 464;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HostIntegerRegister {
    pub encoding: u8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeMemoryAccessKind {
    Read,
    Write,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeMemoryAccess {
    pub address_space: AddressSpaceId,
    pub guest_pc: GuestVirtualAddress,
    pub kind: NativeMemoryAccessKind,
    pub size: u8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeFaultCompletion {
    None,
    IntegerLoad {
        register: u8,
        signed: bool,
        destination_bits: u8,
    },
    IntegerPairLoad {
        first_register: u8,
        second_register: u8,
        signed: bool,
        destination_bits: u8,
        access_index: u8,
        writeback_register: u8,
        writeback_offset: i16,
        writeback: bool,
    },
    IntegerStore {
        register: u8,
    },
    VectorLoad {
        register: u8,
    },
    VectorPairLoad {
        first_register: u8,
        second_register: u8,
        access_index: u8,
        writeback_register: u8,
        writeback_offset: i16,
        writeback: bool,
    },
    VectorStore {
        register: u8,
    },
}

/// One exact native instruction interval which may access a direct arena.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeFaultSite {
    pub native_start: usize,
    pub native_end: usize,
    pub access: NativeMemoryAccess,
    pub completion: NativeFaultCompletion,
    /// Effective guest address retained in Cranelift's reserved pinned register.
    /// Register retaining the effective guest address. Fixed stubs publish
    /// their dynamic address through `StubCall` and therefore use `None`.
    pub guest_address: Option<HostIntegerRegister>,
}

/// Immutable metadata for one finalized native function.
#[derive(Clone, Debug)]
pub struct NativeFaultRegion {
    pub native_start: usize,
    pub native_end: usize,
    pub sites: Arc<[NativeFaultSite]>,
}

/// Append-only native-PC attribution registry.
///
/// Published regions and their sites are immutable. Publication installs each
/// region in a fixed open-addressed native-page index with release stores, so
/// signal-context lookup needs only acquire loads and never allocates, locks,
/// or observes partially initialized metadata. Native code is process-lifetime
/// in the JIT; retaining retired metadata here gives linked executions the same
/// lifetime.
pub struct NativeFaultRegistry {
    /// Fixed open-addressed index. One entry is published for every native
    /// page intersected by a region; duplicate page keys are intentional when
    /// several small regions share one allocator page.
    slots: Box<[AtomicPtr<NativeFaultRegion>]>,
    region_capacity: usize,
    owned: Mutex<OwnedFaultRegions>,
}

struct OwnedFaultRegions {
    regions: Vec<Arc<NativeFaultRegion>>,
    ranges: BTreeMap<usize, usize>,
}

impl NativeFaultRegistry {
    pub fn new(mut regions: Vec<NativeFaultRegion>) -> Result<Self, FaultRuntimeError> {
        regions.sort_unstable_by_key(|region| region.native_start);
        let registry = Self::with_capacity(regions.len().max(1))?;
        for region in regions {
            registry.publish(Arc::new(region))?;
        }
        Ok(registry)
    }

    pub fn with_capacity(capacity: usize) -> Result<Self, FaultRuntimeError> {
        if capacity == 0 {
            return Err(FaultRuntimeError::new(
                "native fault registry capacity must be nonzero",
            ));
        }
        let index_capacity = capacity
            .checked_mul(4)
            .and_then(usize::checked_next_power_of_two)
            .ok_or_else(|| FaultRuntimeError::new("native fault index capacity overflows"))?;
        let slots = (0..index_capacity.max(4))
            .map(|_| AtomicPtr::new(std::ptr::null_mut()))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Ok(Self {
            slots,
            region_capacity: capacity,
            owned: Mutex::new(OwnedFaultRegions {
                regions: Vec::with_capacity(capacity.min(4096)),
                ranges: BTreeMap::new(),
            }),
        })
    }

    pub fn publish(&self, region: Arc<NativeFaultRegion>) -> Result<(), FaultRuntimeError> {
        validate_region(&region)?;
        let mut owned = self.owned.lock().unwrap_or_else(PoisonError::into_inner);
        let index = owned.regions.len();
        if index == self.region_capacity {
            return Err(FaultRuntimeError::new(
                "native fault registry capacity is exhausted",
            ));
        }
        let overlaps_predecessor = owned
            .ranges
            .range(..=region.native_start)
            .next_back()
            .is_some_and(|(_, end)| *end > region.native_start);
        let overlaps_successor = owned
            .ranges
            .range(region.native_start..)
            .next()
            .is_some_and(|(start, _)| *start < region.native_end);
        if overlaps_predecessor || overlaps_successor {
            return Err(FaultRuntimeError::new(
                "native fault regions overlap an already published range",
            ));
        }
        let mut selected = BTreeSet::new();
        let first_page = region.native_start & !(DIRECT_PAGE_SIZE - 1);
        let last_page = (region.native_end - 1) & !(DIRECT_PAGE_SIZE - 1);
        let mut page = first_page;
        loop {
            let slot = self
                .vacant_index_slot(page, &selected)
                .ok_or_else(|| FaultRuntimeError::new("native fault page index is exhausted"))?;
            selected.insert(slot);
            if page == last_page {
                break;
            }
            page = page
                .checked_add(DIRECT_PAGE_SIZE)
                .ok_or_else(|| FaultRuntimeError::new("native fault page range overflows"))?;
        }
        owned.ranges.insert(region.native_start, region.native_end);
        owned.regions.push(Arc::clone(&region));
        for slot in selected {
            self.slots[slot].store(Arc::as_ptr(&region).cast_mut(), Ordering::Release);
        }
        Ok(())
    }

    fn find(&self, native_pc: usize) -> Option<&NativeFaultSite> {
        let page = native_pc & !(DIRECT_PAGE_SIZE - 1);
        let start = fault_page_hash(page) & (self.slots.len() - 1);
        for probe in 0..self.slots.len() {
            let index = start.wrapping_add(probe) & (self.slots.len() - 1);
            let pointer = self.slots[index].load(Ordering::Acquire);
            if pointer.is_null() {
                return None;
            }
            let region = unsafe { &*pointer };
            if native_pc < region.native_start || native_pc >= region.native_end {
                continue;
            }
            let site_index = region
                .sites
                .partition_point(|site| site.native_start <= native_pc)
                .checked_sub(1)?;
            let site = region.sites.get(site_index)?;
            return (native_pc < site.native_end).then_some(site);
        }
        None
    }

    fn vacant_index_slot(&self, page: usize, selected: &BTreeSet<usize>) -> Option<usize> {
        let start = fault_page_hash(page) & (self.slots.len() - 1);
        for probe in 0..self.slots.len() {
            let index = start.wrapping_add(probe) & (self.slots.len() - 1);
            if self.slots[index].load(Ordering::Acquire).is_null() && !selected.contains(&index) {
                return Some(index);
            }
        }
        None
    }
}

fn fault_page_hash(page: usize) -> usize {
    (page / DIRECT_PAGE_SIZE).wrapping_mul(0x9e37_79b9_7f4a_7c15_usize)
}

fn validate_region(region: &NativeFaultRegion) -> Result<(), FaultRuntimeError> {
    if region.native_start >= region.native_end
        || region.sites.iter().any(|site| {
            site.native_start < region.native_start
                || site.native_start >= site.native_end
                || site.native_end > region.native_end
        })
        || region
            .sites
            .windows(2)
            .any(|pair| pair[0].native_end > pair[1].native_start)
    {
        return Err(FaultRuntimeError::new(
            "native fault registry contains invalid or overlapping ranges",
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum FaultDisposition {
    Retry = 0,
    Escape = 1,
    Fatal = 2,
}

pub type FaultDispatcher =
    unsafe extern "C" fn(*mut libc::c_void, *mut CapturedFault) -> FaultDisposition;
pub type NativeGateway = unsafe extern "C" fn(*mut libc::c_void, usize);

/// One native entry call made while a fault-attribution snapshot is active.
#[derive(Clone, Copy)]
pub struct NativeInvocation {
    pub gateway: NativeGateway,
    pub context: *mut libc::c_void,
    pub entry: usize,
}

/// Fault information made available only on the normal dispatcher stack.
pub struct CapturedFault {
    slot: NonNull<FaultSlot>,
}

impl CapturedFault {
    #[must_use]
    pub fn signal(&self) -> i32 {
        unsafe { self.slot.as_ref() }.signal.load(Ordering::Relaxed)
    }

    #[must_use]
    pub fn fault_address(&self) -> usize {
        unsafe { self.slot.as_ref() }
            .fault_address
            .load(Ordering::Relaxed)
    }

    #[must_use]
    pub fn native_pc(&self) -> usize {
        unsafe { self.slot.as_ref() }
            .native_pc
            .load(Ordering::Relaxed)
    }

    #[must_use]
    pub fn site(&self) -> &NativeFaultSite {
        let site = unsafe { self.slot.as_ref() }.site.load(Ordering::Acquire);
        assert!(
            !site.is_null(),
            "an attributed fault retains its immutable site"
        );
        unsafe { &*site }
    }

    /// Reads one integer register from the captured host frame.
    pub fn read_host_integer(
        &self,
        register: HostIntegerRegister,
    ) -> Result<u64, FaultRuntimeError> {
        let slot = unsafe { self.slot.as_ref() };
        let context = unsafe { &*(*slot.context.get()).as_ptr() };
        read_host_integer(context, register)
    }
}

#[derive(Debug)]
#[repr(C)]
struct FaultSlot {
    resume: UnsafeCell<ResumeRecord>,
    tid: AtomicI32,
    active: AtomicBool,
    dispatching: AtomicBool,
    retry_escape: AtomicBool,
    arena_base: AtomicUsize,
    arena_guard_end: AtomicUsize,
    registry: AtomicPtr<NativeFaultRegistry>,
    dispatcher: AtomicUsize,
    opaque: AtomicPtr<libc::c_void>,
    escape_sp: AtomicUsize,
    escape_pc: AtomicUsize,
    dispatcher_stack_top: AtomicUsize,
    signal: AtomicI32,
    fault_address: AtomicUsize,
    native_pc: AtomicUsize,
    site: AtomicPtr<NativeFaultSite>,
    context: UnsafeCell<MaybeUninit<libc::ucontext_t>>,
    #[cfg(target_arch = "x86_64")]
    fpstate: UnsafeCell<AlignedFpState>,
}

#[cfg(target_arch = "x86_64")]
#[repr(align(64))]
struct AlignedFpState([u8; MAX_X86_FPSTATE_SIZE]);

#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
struct ResumeRecord {
    pc: usize,
    rax: usize,
    r11: usize,
    rdi: usize,
    rsi: usize,
    rflags: usize,
    fpstate: usize,
}

unsafe impl Sync for FaultSlot {}
unsafe impl Send for FaultSlot {}

impl FaultSlot {
    fn new() -> Self {
        Self {
            resume: UnsafeCell::new(ResumeRecord {
                pc: 0,
                rax: 0,
                r11: 0,
                rdi: 0,
                rsi: 0,
                rflags: 0,
                fpstate: 0,
            }),
            tid: AtomicI32::new(0),
            active: AtomicBool::new(false),
            dispatching: AtomicBool::new(false),
            retry_escape: AtomicBool::new(false),
            arena_base: AtomicUsize::new(0),
            arena_guard_end: AtomicUsize::new(0),
            registry: AtomicPtr::new(std::ptr::null_mut()),
            dispatcher: AtomicUsize::new(0),
            opaque: AtomicPtr::new(std::ptr::null_mut()),
            escape_sp: AtomicUsize::new(0),
            escape_pc: AtomicUsize::new(0),
            dispatcher_stack_top: AtomicUsize::new(0),
            signal: AtomicI32::new(0),
            fault_address: AtomicUsize::new(0),
            native_pc: AtomicUsize::new(0),
            site: AtomicPtr::new(std::ptr::null_mut()),
            context: UnsafeCell::new(MaybeUninit::uninit()),
            #[cfg(target_arch = "x86_64")]
            fpstate: UnsafeCell::new(AlignedFpState([0; MAX_X86_FPSTATE_SIZE])),
        }
    }
}

static SLOTS: OnceLock<Box<[FaultSlot]>> = OnceLock::new();
static SLOT_POINTER: AtomicPtr<FaultSlot> = AtomicPtr::new(std::ptr::null_mut());
static SLOT_COUNT: AtomicUsize = AtomicUsize::new(0);
static PREVIOUS: OnceLock<PreviousHandlers> = OnceLock::new();
static INSTALLED: OnceLock<Result<(), FaultRuntimeError>> = OnceLock::new();

struct PreviousHandlers {
    segv: libc::sigaction,
    bus: libc::sigaction,
}

/// Installs process-wide signal capture once.
pub fn install() -> Result<(), FaultRuntimeError> {
    INSTALLED
        .get_or_init(install_once)
        .as_ref()
        .map(|_| ())
        .map_err(Clone::clone)
}

fn install_once() -> Result<(), FaultRuntimeError> {
    let slots = (0..MAX_WORKER_SLOTS)
        .map(|_| FaultSlot::new())
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let slots = SLOTS.get_or_init(|| slots);
    SLOT_POINTER.store(slots.as_ptr().cast_mut(), Ordering::Release);
    SLOT_COUNT.store(slots.len(), Ordering::Release);

    let mut previous_segv = MaybeUninit::<libc::sigaction>::uninit();
    let mut previous_bus = MaybeUninit::<libc::sigaction>::uninit();
    let action = signal_action();
    if unsafe { libc::sigaction(libc::SIGSEGV, &action, previous_segv.as_mut_ptr()) } != 0 {
        return Err(FaultRuntimeError::last(
            "SIGSEGV handler installation failed",
        ));
    }
    if unsafe { libc::sigaction(libc::SIGBUS, &action, previous_bus.as_mut_ptr()) } != 0 {
        let _ =
            unsafe { libc::sigaction(libc::SIGSEGV, previous_segv.as_ptr(), std::ptr::null_mut()) };
        return Err(FaultRuntimeError::last(
            "SIGBUS handler installation failed",
        ));
    }
    PREVIOUS
        .set(PreviousHandlers {
            segv: unsafe { previous_segv.assume_init() },
            bus: unsafe { previous_bus.assume_init() },
        })
        .map_err(|_| FaultRuntimeError::new("previous signal handlers were already recorded"))?;
    Ok(())
}

fn signal_action() -> libc::sigaction {
    let mut action = unsafe { MaybeUninit::<libc::sigaction>::zeroed().assume_init() };
    action.sa_sigaction = signal_handler as *const () as usize;
    action.sa_flags = libc::SA_SIGINFO | libc::SA_ONSTACK | libc::SA_NODEFER;
    unsafe { libc::sigemptyset(&mut action.sa_mask) };
    action
}

/// Per-host-worker registration and preallocated recovery stacks.
pub struct WorkerFaultContext {
    slot: NonNull<FaultSlot>,
    signal_stack: ManuallyDrop<GuardedStack>,
    dispatch_stack: ManuallyDrop<GuardedStack>,
    previous_stack: libc::stack_t,
    tid: i32,
}

// SAFETY: methods reject use from any TID other than the one which registered
// the context. If a safe caller nevertheless moves and drops a registered
// context on another TID, `Drop` deliberately retains its stacks and slot so
// the original thread's installed alternate stack can never dangle.
unsafe impl Send for WorkerFaultContext {}

struct GuardedStack {
    mapping: NonNull<libc::c_void>,
    mapping_size: usize,
    usable: NonNull<u8>,
    usable_size: usize,
}

impl GuardedStack {
    fn new(usable_size: usize) -> Result<Self, FaultRuntimeError> {
        if usable_size == 0 || !usable_size.is_multiple_of(DIRECT_PAGE_SIZE) {
            return Err(FaultRuntimeError::new(
                "guarded stack size is not a nonzero host-page multiple",
            ));
        }
        let mapping_size = usable_size
            .checked_add(DIRECT_PAGE_SIZE * 2)
            .ok_or_else(|| FaultRuntimeError::new("guarded stack reservation overflows"))?;
        let mapping = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                mapping_size,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if mapping == libc::MAP_FAILED {
            return Err(FaultRuntimeError::last("guarded stack reservation failed"));
        }
        let mapping = NonNull::new(mapping)
            .ok_or_else(|| FaultRuntimeError::new("guarded stack reservation returned null"))?;
        let usable = unsafe { mapping.cast::<u8>().add(DIRECT_PAGE_SIZE) };
        if unsafe {
            libc::mprotect(
                usable.as_ptr().cast(),
                usable_size,
                libc::PROT_READ | libc::PROT_WRITE,
            )
        } != 0
        {
            let error = FaultRuntimeError::last("guarded stack publication failed");
            let _ = unsafe { libc::munmap(mapping.as_ptr(), mapping_size) };
            return Err(error);
        }
        Ok(Self {
            mapping,
            mapping_size,
            usable,
            usable_size,
        })
    }

    fn stack_t(&self) -> libc::stack_t {
        libc::stack_t {
            ss_sp: self.usable.as_ptr().cast(),
            ss_flags: 0,
            ss_size: self.usable_size,
        }
    }

    fn top(&self) -> usize {
        self.usable.as_ptr().addr() + self.usable_size
    }
}

impl Drop for GuardedStack {
    fn drop(&mut self) {
        let _ = unsafe { libc::munmap(self.mapping.as_ptr(), self.mapping_size) };
    }
}

impl WorkerFaultContext {
    pub fn register() -> Result<Self, FaultRuntimeError> {
        install()?;
        let tid = current_tid();
        let signal_stack = GuardedStack::new(SIGNAL_STACK_SIZE)?;
        let dispatch_stack = GuardedStack::new(DISPATCH_STACK_SIZE)?;
        let slots = SLOTS
            .get()
            .ok_or_else(|| FaultRuntimeError::new("fault slots are not installed"))?;
        let slot = slots
            .iter()
            .find(|slot| {
                slot.tid
                    .compare_exchange(0, tid, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
            })
            .ok_or_else(|| FaultRuntimeError::new("native fault worker capacity is exhausted"))?;

        let mut previous_stack = unsafe { MaybeUninit::<libc::stack_t>::zeroed().assume_init() };
        let stack = signal_stack.stack_t();
        if unsafe { libc::sigaltstack(&stack, &mut previous_stack) } != 0 {
            slot.tid.store(0, Ordering::Release);
            return Err(FaultRuntimeError::last(
                "worker alternate signal stack installation failed",
            ));
        }
        let dispatch_top = dispatch_stack.top();
        slot.dispatcher_stack_top
            .store(dispatch_top & !15, Ordering::Release);
        Ok(Self {
            slot: NonNull::from(slot),
            signal_stack: ManuallyDrop::new(signal_stack),
            dispatch_stack: ManuallyDrop::new(dispatch_stack),
            previous_stack,
            tid,
        })
    }

    #[must_use]
    pub fn registered_tid(&self) -> i32 {
        self.tid
    }

    /// Executes one native gateway with an immutable attribution snapshot.
    ///
    /// # Safety
    ///
    /// `context`, `entry`, `gateway`, dispatcher opaque data, and every
    /// registry pointer must remain valid until this call returns.
    pub unsafe fn invoke(
        &mut self,
        arena: DirectAddressSpaceView,
        registry: &Arc<NativeFaultRegistry>,
        dispatcher: FaultDispatcher,
        opaque: *mut libc::c_void,
        invocation: NativeInvocation,
    ) -> Result<InvocationOutcome, FaultRuntimeError> {
        unsafe { self.begin_batch(arena, registry, dispatcher, opaque) }?;
        let escaped = unsafe {
            nixe_direct_memory_invoke(
                self.slot.as_ptr().cast(),
                invocation.context,
                invocation.entry,
                invocation.gateway,
            )
        };
        self.end_batch()?;
        Ok(invocation_outcome(escaped, unsafe { self.slot.as_ref() }))
    }

    /// Publishes one stable arena/registry/dispatcher snapshot for a batch of
    /// fixed scalar accesses, normally one interpreter slice.
    ///
    /// # Safety
    ///
    /// `registry`, `opaque`, the arena and everything reachable from the
    /// dispatcher must outlive the matching [`Self::end_batch`].
    pub unsafe fn begin_batch(
        &mut self,
        arena: DirectAddressSpaceView,
        registry: &Arc<NativeFaultRegistry>,
        dispatcher: FaultDispatcher,
        opaque: *mut libc::c_void,
    ) -> Result<(), FaultRuntimeError> {
        if current_tid() != self.tid {
            return Err(FaultRuntimeError::new(
                "native fault context was invoked from a different host TID",
            ));
        }
        let end = arena
            .base
            .checked_add(arena.address_space_size)
            .ok_or_else(|| FaultRuntimeError::new("direct arena end overflows"))?;
        let guard_end = end
            .checked_add(DIRECT_PAGE_SIZE)
            .ok_or_else(|| FaultRuntimeError::new("direct arena guard end overflows"))?;
        let slot = unsafe { self.slot.as_ref() };
        if slot.active.load(Ordering::Acquire) {
            return Err(FaultRuntimeError::new(
                "native fault context is already active",
            ));
        }
        slot.arena_base.store(arena.base, Ordering::Relaxed);
        slot.arena_guard_end.store(guard_end, Ordering::Relaxed);
        slot.registry
            .store(Arc::as_ptr(registry).cast_mut(), Ordering::Relaxed);
        slot.dispatcher
            .store(dispatcher as usize, Ordering::Relaxed);
        slot.opaque.store(opaque, Ordering::Relaxed);
        slot.site.store(std::ptr::null_mut(), Ordering::Relaxed);
        slot.retry_escape.store(false, Ordering::Relaxed);
        if slot
            .active
            .compare_exchange(false, true, Ordering::Release, Ordering::Acquire)
            .is_err()
        {
            return Err(FaultRuntimeError::new(
                "native fault context became active during publication",
            ));
        }
        Ok(())
    }

    /// Ends a previously published fixed-access batch.
    pub fn end_batch(&mut self) -> Result<(), FaultRuntimeError> {
        if current_tid() != self.tid {
            return Err(FaultRuntimeError::new(
                "native fault context batch ended from a different host TID",
            ));
        }
        let slot = unsafe { self.slot.as_ref() };
        if !slot.active.load(Ordering::Acquire) {
            return Err(FaultRuntimeError::new(
                "native fault context batch is not active",
            ));
        }
        slot.active.store(false, Ordering::Release);
        slot.registry.store(std::ptr::null_mut(), Ordering::Relaxed);
        slot.dispatcher.store(0, Ordering::Relaxed);
        slot.opaque.store(std::ptr::null_mut(), Ordering::Relaxed);
        Ok(())
    }

    /// Invokes one fixed scalar stub while a batch snapshot is active.
    ///
    /// # Safety
    ///
    /// `context` must point to the stub call layout and `entry` must be one of
    /// the immutable functions registered in the active registry.
    pub unsafe fn invoke_scalar_in_batch(
        &mut self,
        context: *mut libc::c_void,
        entry: usize,
    ) -> Result<InvocationOutcome, FaultRuntimeError> {
        let slot = unsafe { self.slot.as_ref() };
        if !slot.active.load(Ordering::Acquire) {
            return Err(FaultRuntimeError::new(
                "native fault context scalar batch is not active",
            ));
        }
        let escaped =
            unsafe { nixe_direct_scalar_invoke(self.slot.as_ptr().cast(), context, entry) };
        Ok(invocation_outcome(escaped, slot))
    }
}

fn invocation_outcome(escaped: u32, slot: &FaultSlot) -> InvocationOutcome {
    if escaped == 0 {
        InvocationOutcome::Returned
    } else if slot.retry_escape.swap(false, Ordering::AcqRel) {
        InvocationOutcome::Retry
    } else {
        InvocationOutcome::Escaped
    }
}

impl Drop for WorkerFaultContext {
    fn drop(&mut self) {
        if current_tid() != self.tid {
            return;
        }
        let slot = unsafe { self.slot.as_ref() };
        slot.active.store(false, Ordering::Release);
        slot.registry.store(std::ptr::null_mut(), Ordering::Relaxed);
        slot.dispatcher.store(0, Ordering::Relaxed);
        slot.opaque.store(std::ptr::null_mut(), Ordering::Relaxed);
        if unsafe { libc::sigaltstack(&self.previous_stack, std::ptr::null_mut()) } != 0 {
            return;
        }
        slot.dispatcher_stack_top.store(0, Ordering::Release);
        slot.tid.store(0, Ordering::Release);
        unsafe {
            ManuallyDrop::drop(&mut self.signal_stack);
            ManuallyDrop::drop(&mut self.dispatch_stack);
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InvocationOutcome {
    Returned,
    /// The native access was made visible, but this architecture requires the
    /// caller to re-enter it instead of restoring a partial libc `ucontext`.
    Retry,
    Escaped,
}

/// Failure returned by the fixed interpreter direct-memory frontend.
#[derive(Debug)]
pub enum DirectScalarAccessError {
    DataFault(DataAccessFault),
    Backend(Box<str>),
    Runtime(FaultRuntimeError),
}

impl Display for DirectScalarAccessError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DataFault(fault) => write!(formatter, "direct scalar data fault: {fault:?}"),
            Self::Backend(detail) => formatter.write_str(detail),
            Self::Runtime(error) => Display::fmt(error, formatter),
        }
    }
}

impl std::error::Error for DirectScalarAccessError {}

/// Per-worker fixed-stub frontend used by the reference interpreter.
///
/// Binding selects this object once for a LinuxDirect process. Accesses contain
/// no page-table walk or permission test; host protection remains the access
/// authority, while the compact store control publishes the temporary page
/// generation required before the raw-store cutover.
pub struct DirectScalarFrontend {
    worker: Option<WorkerFaultContext>,
    arena: DirectAddressSpaceView,
    address_space: AddressSpaceId,
    dispatcher_context: ManuallyDrop<Box<ScalarDispatcherContext>>,
    batch_active: bool,
}

#[derive(Default)]
struct ScalarDispatcherContext {
    current_call: AtomicPtr<StubCall>,
}

impl DirectScalarFrontend {
    /// # Safety
    ///
    /// The arena and every canonical object referenced by its store controls
    /// must remain alive and unchanged for every call made through this
    /// frontend. Higher-level CPU frontends validate the view against the
    /// currently borrowed `CpuMemory` before beginning each slice.
    pub unsafe fn new(
        arena: DirectAddressSpaceView,
        address_space: AddressSpaceId,
    ) -> Result<Self, FaultRuntimeError> {
        Ok(Self {
            worker: None,
            arena,
            address_space,
            dispatcher_context: ManuallyDrop::new(Box::new(ScalarDispatcherContext::default())),
            batch_active: false,
        })
    }

    /// Publishes the stable fault snapshot once for an interpreter slice.
    pub fn begin_slice(&mut self) -> Result<(), FaultRuntimeError> {
        if self.batch_active {
            return Err(FaultRuntimeError::new(
                "direct scalar interpreter slice is already active",
            ));
        }
        let arena = self.arena;
        let registry = scalar_stub_registry()?;
        let opaque = std::ptr::from_ref(self.dispatcher_context.as_ref())
            .cast_mut()
            .cast();
        let worker = self.worker()?;
        unsafe { worker.begin_batch(arena, registry, dispatch_scalar_stub_fault, opaque) }?;
        self.batch_active = true;
        Ok(())
    }

    /// Clears the slice snapshot before the process can release its arena.
    pub fn end_slice(&mut self) -> Result<(), FaultRuntimeError> {
        if !self.batch_active {
            return Err(FaultRuntimeError::new(
                "direct scalar interpreter slice is not active",
            ));
        }
        self.worker
            .as_mut()
            .expect("an active scalar slice owns a worker")
            .end_batch()?;
        self.batch_active = false;
        Ok(())
    }

    pub fn read(
        &mut self,
        memory: &dyn CpuMemory,
        guest_pc: GuestVirtualAddress,
        address: GuestVirtualAddress,
        access: MemoryAccess,
    ) -> Result<MemoryValue, DirectScalarAccessError> {
        let size = validate_scalar_access(access)?;
        let pointer = self
            .direct_pointer(address, size)
            .unwrap_or_else(|| self.poison_page());
        let mut call = StubCall::new(
            pointer,
            0,
            memory,
            self.address_space,
            guest_pc,
            address,
            access,
            DataAccessKind::Read,
        );
        self.invoke(&mut call, scalar_stub(size, DataAccessKind::Read)?)?;
        call.finish()?;
        Ok(MemoryValue::from_bits(size, u128::from(call.output)))
    }

    pub fn write(
        &mut self,
        memory: &dyn CpuMemory,
        guest_pc: GuestVirtualAddress,
        address: GuestVirtualAddress,
        access: MemoryAccess,
        value: MemoryValue,
    ) -> Result<(), DirectScalarAccessError> {
        self.write_inner(memory, guest_pc, address, access, value, || {})
    }

    fn write_inner(
        &mut self,
        memory: &dyn CpuMemory,
        guest_pc: GuestVirtualAddress,
        address: GuestVirtualAddress,
        access: MemoryAccess,
        value: MemoryValue,
        before_invoke: impl FnOnce(),
    ) -> Result<(), DirectScalarAccessError> {
        let size = validate_scalar_access(access)?;
        if value.size() != size {
            return Err(DirectScalarAccessError::Backend(
                "direct scalar store value does not match its access width".into(),
            ));
        }
        let publication = self.store_publication(address, size);
        // A store which did not acquire publication ownership must fault even
        // if another writer arms the real alias before this stub executes.
        // The poison guard makes that earlier decision stable; registered
        // write faults always complete exactly once through canonical memory.
        let pointer = publication.as_ref().map_or_else(
            || self.poison_page(),
            |_| {
                self.direct_pointer(address, size)
                    .expect("a publication token proves a confined direct pointer")
            },
        );
        let mut call = StubCall::new(
            pointer,
            value.bits() as u64,
            memory,
            self.address_space,
            guest_pc,
            address,
            access,
            DataAccessKind::Write,
        );
        before_invoke();
        let invocation = self.invoke_raw(&mut call, scalar_stub(size, DataAccessKind::Write)?);
        match (publication, &invocation) {
            (Some(publication), Ok(InvocationOutcome::Returned)) => publication.commit(),
            (Some(publication), _) => publication.abort(),
            (None, _) => {}
        }
        invocation.map_err(DirectScalarAccessError::Runtime)?;
        call.finish()
    }

    fn invoke(
        &mut self,
        call: &mut StubCall,
        entry: usize,
    ) -> Result<InvocationOutcome, DirectScalarAccessError> {
        self.invoke_raw(call, entry)
            .map_err(DirectScalarAccessError::Runtime)
    }

    fn invoke_raw(
        &mut self,
        call: &mut StubCall,
        entry: usize,
    ) -> Result<InvocationOutcome, FaultRuntimeError> {
        // Complete every fallible preparation before publishing the stack-owned
        // call to the dispatcher. A failed worker registration must not leave
        // a dangling `current_call` behind.
        let registry = if self.batch_active {
            None
        } else {
            Some(scalar_stub_registry()?)
        };
        if self.worker.is_none() {
            self.worker()?;
        }
        let call_pointer = std::ptr::from_mut(call);
        if self
            .dispatcher_context
            .current_call
            .compare_exchange(
                std::ptr::null_mut(),
                call_pointer,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
        {
            return Err(FaultRuntimeError::new(
                "direct scalar frontend already has an active call",
            ));
        }
        let outcome = loop {
            let outcome = if self.batch_active {
                let worker = self
                    .worker
                    .as_mut()
                    .expect("an active scalar slice owns a worker");
                unsafe { worker.invoke_scalar_in_batch(call_pointer.cast(), entry) }
            } else {
                let arena = self.arena;
                let opaque = std::ptr::from_ref(self.dispatcher_context.as_ref())
                    .cast_mut()
                    .cast();
                let worker = self
                    .worker
                    .as_mut()
                    .expect("a prepared scalar call owns a worker");
                unsafe {
                    worker.invoke(
                        arena,
                        registry.expect("an unbatched scalar call prepared its registry"),
                        dispatch_scalar_stub_fault,
                        opaque,
                        NativeInvocation {
                            gateway: scalar_stub_gateway,
                            context: call_pointer.cast(),
                            entry,
                        },
                    )
                }
            }?;
            if outcome != InvocationOutcome::Retry {
                break Ok(outcome);
            }
            if call.kind == DataAccessKind::Write {
                break Err(FaultRuntimeError::new(
                    "a direct scalar write unexpectedly requested native retry",
                ));
            }
        };
        self.dispatcher_context
            .current_call
            .store(std::ptr::null_mut(), Ordering::Release);
        outcome
    }

    fn worker(&mut self) -> Result<&mut WorkerFaultContext, FaultRuntimeError> {
        if self.worker.is_none() {
            self.worker = Some(WorkerFaultContext::register()?);
        }
        let worker = self
            .worker
            .as_mut()
            .expect("direct scalar worker was initialized");
        if worker.registered_tid() != current_tid() {
            return Err(FaultRuntimeError::new(
                "direct scalar frontend moved to a different host TID after first use",
            ));
        }
        Ok(worker)
    }

    fn direct_pointer(
        &self,
        address: GuestVirtualAddress,
        size: MemoryAccessSize,
    ) -> Option<usize> {
        let bytes = size.bytes();
        let last = address.get().checked_add((bytes - 1) as u64)?;
        if last >= self.arena.address_space_size as u64
            || !address.get().is_multiple_of(bytes as u64)
        {
            return None;
        }
        self.arena.base.checked_add(address.get() as usize)
    }

    fn poison_page(&self) -> usize {
        self.arena.base + self.arena.address_space_size
    }

    fn store_publication(
        &self,
        address: GuestVirtualAddress,
        size: MemoryAccessSize,
    ) -> Option<DirectStorePublication> {
        self.direct_pointer(address, size)?;
        let page = address.get() as usize / DIRECT_PAGE_SIZE;
        let slot = unsafe { &*(self.arena.store_controls as *const AtomicUsize).add(page) };
        let control = slot.load(Ordering::Acquire) as *const DirectStoreControl;
        let control = unsafe { control.as_ref()? };
        let armed = unsafe { atomic_at(control.write_armed_address) };
        if armed.load(Ordering::Acquire) == 0 {
            return None;
        }
        let generation = unsafe { atomic_at(control.generation_address) };
        if generation.load(Ordering::Acquire) == u64::MAX {
            return None;
        }
        DirectStorePublication::acquire(*control)
    }
}

impl Drop for DirectScalarFrontend {
    fn drop(&mut self) {
        if self.batch_active {
            let Some(worker) = self.worker.as_ref() else {
                return;
            };
            if worker.registered_tid() != current_tid() || self.end_slice().is_err() {
                // The worker context retains its stacks and slot on a foreign
                // TID. Retain the published opaque dispatcher with them.
                return;
            }
        }
        unsafe { ManuallyDrop::drop(&mut self.dispatcher_context) };
    }
}

fn validate_scalar_access(
    access: MemoryAccess,
) -> Result<MemoryAccessSize, DirectScalarAccessError> {
    if access.ordering != MemoryOrdering::Relaxed
        || access.class != MemoryAccessClass::Normal
        || access.size == MemoryAccessSize::Quadword
    {
        return Err(DirectScalarAccessError::Backend(
            "direct scalar stubs accept only relaxed ordinary 1/2/4/8-byte accesses".into(),
        ));
    }
    Ok(access.size)
}

unsafe fn atomic_at(address: usize) -> &'static AtomicU64 {
    unsafe { &*(address as *const AtomicU64) }
}

struct DirectStorePublication {
    control: DirectStoreControl,
    sequence: u64,
}

impl DirectStorePublication {
    fn acquire(control: DirectStoreControl) -> Option<Self> {
        let sequence = unsafe { atomic_at(control.write_sequence_address) };
        let mut observed = sequence.load(Ordering::Acquire);
        loop {
            if observed & 1 != 0 {
                std::hint::spin_loop();
                observed = sequence.load(Ordering::Acquire);
                continue;
            }
            match sequence.compare_exchange_weak(
                observed,
                observed.wrapping_add(1),
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(current) => observed = current,
            }
        }
        let armed = unsafe { atomic_at(control.write_armed_address) };
        let exhausted =
            unsafe { atomic_at(control.generation_address) }.load(Ordering::Acquire) == u64::MAX;
        if armed.load(Ordering::Acquire) == 0 || exhausted {
            sequence.store(observed, Ordering::Release);
            return None;
        }
        Some(Self {
            control,
            sequence: observed,
        })
    }

    fn commit(self) {
        unsafe { atomic_at(self.control.generation_address) }.fetch_add(1, Ordering::Release);
        self.finish();
    }

    fn abort(self) {
        self.finish();
    }

    fn finish(self) {
        unsafe { atomic_at(self.control.write_sequence_address) }
            .store(self.sequence.wrapping_add(2), Ordering::Release);
    }
}

#[repr(C)]
struct StubCall {
    pointer: usize,
    value: u64,
    output: u64,
    memory: *const dyn CpuMemory,
    address_space: AddressSpaceId,
    guest_pc: GuestVirtualAddress,
    address: GuestVirtualAddress,
    access: MemoryAccess,
    kind: DataAccessKind,
    data_fault: Option<DataAccessFault>,
    backend_error: Option<Box<str>>,
}

impl StubCall {
    #[allow(clippy::too_many_arguments)]
    fn new(
        pointer: usize,
        value: u64,
        memory: &dyn CpuMemory,
        address_space: AddressSpaceId,
        guest_pc: GuestVirtualAddress,
        address: GuestVirtualAddress,
        access: MemoryAccess,
        kind: DataAccessKind,
    ) -> Self {
        let memory = unsafe { std::mem::transmute::<&dyn CpuMemory, *const dyn CpuMemory>(memory) };
        Self {
            pointer,
            value,
            output: 0,
            memory,
            address_space,
            guest_pc,
            address,
            access,
            kind,
            data_fault: None,
            backend_error: None,
        }
    }

    fn finish(&mut self) -> Result<(), DirectScalarAccessError> {
        if let Some(fault) = self.data_fault.take() {
            return Err(DirectScalarAccessError::DataFault(fault));
        }
        if let Some(detail) = self.backend_error.take() {
            return Err(DirectScalarAccessError::Backend(detail));
        }
        Ok(())
    }
}

unsafe extern "C" fn scalar_stub_gateway(context: *mut libc::c_void, entry: usize) {
    let stub =
        unsafe { std::mem::transmute::<usize, unsafe extern "C" fn(*mut libc::c_void)>(entry) };
    unsafe { stub(context) };
}

unsafe extern "C" fn dispatch_scalar_stub_fault(
    opaque: *mut libc::c_void,
    _fault: *mut CapturedFault,
) -> FaultDisposition {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let context = unsafe { &*opaque.cast::<ScalarDispatcherContext>() };
        let call = context.current_call.load(Ordering::Acquire);
        if call.is_null() {
            return FaultDisposition::Fatal;
        }
        let call = unsafe { &mut *call };
        let memory = unsafe { &*call.memory };
        match memory.resolve_direct_fault(
            call.address_space,
            call.address,
            call.access.size,
            call.kind,
        ) {
            DirectFaultResolution::Retry => FaultDisposition::Retry,
            DirectFaultResolution::Checked => {
                match call.kind {
                    DataAccessKind::Read => {
                        match memory.read(call.address_space, call.address, call.access) {
                            Ok(result) => {
                                call.output = result.value.bits() as u64;
                            }
                            Err(fault) => {
                                call.data_fault = Some(fault);
                            }
                        }
                    }
                    DataAccessKind::Write => match memory.complete_direct_write_fault(
                        call.address_space,
                        call.address,
                        call.access,
                        MemoryValue::from_bits(call.access.size, u128::from(call.value)),
                    ) {
                        Ok(_) => {}
                        Err(fault) => {
                            call.data_fault = Some(fault);
                        }
                    },
                }
                FaultDisposition::Escape
            }
            DirectFaultResolution::Fatal(detail) => {
                call.backend_error = Some(detail);
                FaultDisposition::Escape
            }
        }
    }))
    .unwrap_or(FaultDisposition::Fatal)
}

fn scalar_stub(
    size: MemoryAccessSize,
    kind: DataAccessKind,
) -> Result<usize, DirectScalarAccessError> {
    let entry = match (kind, size) {
        (DataAccessKind::Read, MemoryAccessSize::Byte) => function_address(nixe_direct_stub_read_1),
        (DataAccessKind::Read, MemoryAccessSize::Halfword) => {
            function_address(nixe_direct_stub_read_2)
        }
        (DataAccessKind::Read, MemoryAccessSize::Word) => function_address(nixe_direct_stub_read_4),
        (DataAccessKind::Read, MemoryAccessSize::Doubleword) => {
            function_address(nixe_direct_stub_read_8)
        }
        (DataAccessKind::Write, MemoryAccessSize::Byte) => {
            function_address(nixe_direct_stub_write_1)
        }
        (DataAccessKind::Write, MemoryAccessSize::Halfword) => {
            function_address(nixe_direct_stub_write_2)
        }
        (DataAccessKind::Write, MemoryAccessSize::Word) => {
            function_address(nixe_direct_stub_write_4)
        }
        (DataAccessKind::Write, MemoryAccessSize::Doubleword) => {
            function_address(nixe_direct_stub_write_8)
        }
        (_, MemoryAccessSize::Quadword) => {
            return Err(DirectScalarAccessError::Backend(
                "direct scalar stubs do not support 128-bit accesses".into(),
            ));
        }
    };
    Ok(entry)
}

fn function_address(function: unsafe extern "C" fn(*mut libc::c_void)) -> usize {
    function as *const () as usize
}

fn scalar_stub_registry() -> Result<&'static Arc<NativeFaultRegistry>, FaultRuntimeError> {
    static REGISTRY: OnceLock<Result<Arc<NativeFaultRegistry>, FaultRuntimeError>> =
        OnceLock::new();
    REGISTRY
        .get_or_init(|| {
            NativeFaultRegistry::new(vec![
                scalar_stub_region(
                    function_address(nixe_direct_stub_read_1),
                    std::ptr::addr_of!(nixe_direct_stub_read_1_end).addr(),
                    std::ptr::addr_of!(nixe_direct_stub_read_1_fault).addr(),
                    std::ptr::addr_of!(nixe_direct_stub_read_1_after).addr(),
                    DataAccessKind::Read,
                    1,
                ),
                scalar_stub_region(
                    function_address(nixe_direct_stub_read_2),
                    std::ptr::addr_of!(nixe_direct_stub_read_2_end).addr(),
                    std::ptr::addr_of!(nixe_direct_stub_read_2_fault).addr(),
                    std::ptr::addr_of!(nixe_direct_stub_read_2_after).addr(),
                    DataAccessKind::Read,
                    2,
                ),
                scalar_stub_region(
                    function_address(nixe_direct_stub_read_4),
                    std::ptr::addr_of!(nixe_direct_stub_read_4_end).addr(),
                    std::ptr::addr_of!(nixe_direct_stub_read_4_fault).addr(),
                    std::ptr::addr_of!(nixe_direct_stub_read_4_after).addr(),
                    DataAccessKind::Read,
                    4,
                ),
                scalar_stub_region(
                    function_address(nixe_direct_stub_read_8),
                    std::ptr::addr_of!(nixe_direct_stub_read_8_end).addr(),
                    std::ptr::addr_of!(nixe_direct_stub_read_8_fault).addr(),
                    std::ptr::addr_of!(nixe_direct_stub_read_8_after).addr(),
                    DataAccessKind::Read,
                    8,
                ),
                scalar_stub_region(
                    function_address(nixe_direct_stub_write_1),
                    std::ptr::addr_of!(nixe_direct_stub_write_1_end).addr(),
                    std::ptr::addr_of!(nixe_direct_stub_write_1_fault).addr(),
                    std::ptr::addr_of!(nixe_direct_stub_write_1_after).addr(),
                    DataAccessKind::Write,
                    1,
                ),
                scalar_stub_region(
                    function_address(nixe_direct_stub_write_2),
                    std::ptr::addr_of!(nixe_direct_stub_write_2_end).addr(),
                    std::ptr::addr_of!(nixe_direct_stub_write_2_fault).addr(),
                    std::ptr::addr_of!(nixe_direct_stub_write_2_after).addr(),
                    DataAccessKind::Write,
                    2,
                ),
                scalar_stub_region(
                    function_address(nixe_direct_stub_write_4),
                    std::ptr::addr_of!(nixe_direct_stub_write_4_end).addr(),
                    std::ptr::addr_of!(nixe_direct_stub_write_4_fault).addr(),
                    std::ptr::addr_of!(nixe_direct_stub_write_4_after).addr(),
                    DataAccessKind::Write,
                    4,
                ),
                scalar_stub_region(
                    function_address(nixe_direct_stub_write_8),
                    std::ptr::addr_of!(nixe_direct_stub_write_8_end).addr(),
                    std::ptr::addr_of!(nixe_direct_stub_write_8_fault).addr(),
                    std::ptr::addr_of!(nixe_direct_stub_write_8_after).addr(),
                    DataAccessKind::Write,
                    8,
                ),
            ])
            .map(Arc::new)
        })
        .as_ref()
        .map_err(Clone::clone)
}

fn scalar_stub_region(
    native_start: usize,
    native_end: usize,
    site_start: usize,
    site_end: usize,
    kind: DataAccessKind,
    size: u8,
) -> NativeFaultRegion {
    NativeFaultRegion {
        native_start,
        native_end,
        sites: Arc::from([NativeFaultSite {
            native_start: site_start,
            native_end: site_end,
            access: NativeMemoryAccess {
                address_space: AddressSpaceId::new(0),
                guest_pc: GuestVirtualAddress::new(0),
                kind: match kind {
                    DataAccessKind::Read => NativeMemoryAccessKind::Read,
                    DataAccessKind::Write => NativeMemoryAccessKind::Write,
                },
                size,
            },
            completion: NativeFaultCompletion::None,
            guest_address: None,
        }]),
    }
}

unsafe extern "C" {
    fn nixe_direct_stub_read_1(call: *mut libc::c_void);
    fn nixe_direct_stub_read_2(call: *mut libc::c_void);
    fn nixe_direct_stub_read_4(call: *mut libc::c_void);
    fn nixe_direct_stub_read_8(call: *mut libc::c_void);
    fn nixe_direct_stub_write_1(call: *mut libc::c_void);
    fn nixe_direct_stub_write_2(call: *mut libc::c_void);
    fn nixe_direct_stub_write_4(call: *mut libc::c_void);
    fn nixe_direct_stub_write_8(call: *mut libc::c_void);
    static nixe_direct_stub_read_1_fault: u8;
    static nixe_direct_stub_read_1_after: u8;
    static nixe_direct_stub_read_1_end: u8;
    static nixe_direct_stub_read_2_fault: u8;
    static nixe_direct_stub_read_2_after: u8;
    static nixe_direct_stub_read_2_end: u8;
    static nixe_direct_stub_read_4_fault: u8;
    static nixe_direct_stub_read_4_after: u8;
    static nixe_direct_stub_read_4_end: u8;
    static nixe_direct_stub_read_8_fault: u8;
    static nixe_direct_stub_read_8_after: u8;
    static nixe_direct_stub_read_8_end: u8;
    static nixe_direct_stub_write_1_fault: u8;
    static nixe_direct_stub_write_1_after: u8;
    static nixe_direct_stub_write_1_end: u8;
    static nixe_direct_stub_write_2_fault: u8;
    static nixe_direct_stub_write_2_after: u8;
    static nixe_direct_stub_write_2_end: u8;
    static nixe_direct_stub_write_4_fault: u8;
    static nixe_direct_stub_write_4_after: u8;
    static nixe_direct_stub_write_4_end: u8;
    static nixe_direct_stub_write_8_fault: u8;
    static nixe_direct_stub_write_8_after: u8;
    static nixe_direct_stub_write_8_end: u8;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FaultRuntimeError(Box<str>);

impl FaultRuntimeError {
    fn new(detail: impl Into<Box<str>>) -> Self {
        Self(detail.into())
    }

    fn last(operation: &str) -> Self {
        Self(format!("{operation}: {}", std::io::Error::last_os_error()).into_boxed_str())
    }
}

impl Display for FaultRuntimeError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for FaultRuntimeError {}

unsafe extern "C" {
    fn nixe_direct_memory_invoke(
        slot: *mut libc::c_void,
        context: *mut libc::c_void,
        entry: usize,
        gateway: NativeGateway,
    ) -> u32;
    fn nixe_direct_scalar_invoke(
        slot: *mut libc::c_void,
        context: *mut libc::c_void,
        entry: usize,
    ) -> u32;
    fn nixe_direct_fault_landing_pad();
    fn nixe_direct_escape_now(pc: usize, sp: usize) -> !;
    #[cfg(target_arch = "x86_64")]
    fn nixe_direct_retry_trampoline();
}

#[unsafe(no_mangle)]
unsafe extern "C" fn nixe_direct_prepare_escape(
    slot: *mut FaultSlot,
    stack_pointer: usize,
    escape_pc: usize,
) {
    let slot = unsafe { &*slot };
    slot.escape_sp.store(stack_pointer, Ordering::Relaxed);
    slot.escape_pc.store(escape_pc, Ordering::Release);
}

#[unsafe(no_mangle)]
unsafe extern "C" fn nixe_direct_fault_dispatch(slot: *mut FaultSlot) -> ! {
    let slot = unsafe { &*slot };
    let dispatcher = slot.dispatcher.load(Ordering::Acquire);
    if dispatcher == 0 {
        fatal_signal(slot.signal.load(Ordering::Relaxed));
    }
    let dispatcher = unsafe { std::mem::transmute::<usize, FaultDispatcher>(dispatcher) };
    let mut fault = CapturedFault {
        slot: unsafe { NonNull::new_unchecked(slot as *const FaultSlot as *mut FaultSlot) },
    };
    let disposition = unsafe { dispatcher(slot.opaque.load(Ordering::Relaxed), &mut fault) };
    match disposition {
        FaultDisposition::Retry => {
            #[cfg(target_arch = "x86_64")]
            {
                let context = unsafe { &mut *(*slot.context.get()).as_mut_ptr() };
                prepare_retry_context(context, slot);
                slot.dispatching.store(false, Ordering::Release);
            }
            #[cfg(target_arch = "aarch64")]
            {
                // glibc's AArch64 setcontext intentionally restores only a
                // subset of volatile GPR/SIMD state. Escaping lets the caller
                // re-enter the exact registered access without relying on a
                // partial context restore.
                slot.retry_escape.store(true, Ordering::Release);
                slot.dispatching.store(false, Ordering::Release);
                unsafe {
                    nixe_direct_escape_now(
                        slot.escape_pc.load(Ordering::Acquire),
                        slot.escape_sp.load(Ordering::Acquire),
                    )
                }
            }
        }
        FaultDisposition::Escape => {
            slot.dispatching.store(false, Ordering::Release);
            unsafe {
                nixe_direct_escape_now(
                    slot.escape_pc.load(Ordering::Acquire),
                    slot.escape_sp.load(Ordering::Acquire),
                )
            }
        }
        FaultDisposition::Fatal => {
            fatal_signal(slot.signal.load(Ordering::Relaxed));
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        let context = unsafe { (*slot.context.get()).as_ptr() };
        if unsafe { libc::setcontext(context) } != 0 {
            fatal_signal(slot.signal.load(Ordering::Relaxed));
        }
        unsafe { std::hint::unreachable_unchecked() }
    }
}

unsafe extern "C" fn signal_handler(
    signal: i32,
    info: *mut libc::siginfo_t,
    context: *mut libc::c_void,
) {
    if info.is_null()
        || context.is_null()
        || !accepted_memory_fault_code(signal, unsafe { (*info).si_code })
    {
        unsafe { chain_or_reraise(signal, info, context) };
    }
    let tid = current_tid();
    let pointer = SLOT_POINTER.load(Ordering::Acquire);
    let count = SLOT_COUNT.load(Ordering::Acquire);
    let mut selected = None;
    for index in 0..count {
        let slot = unsafe { &*pointer.add(index) };
        if slot.tid.load(Ordering::Acquire) == tid {
            selected = Some(slot);
            break;
        }
    }
    let Some(slot) = selected else {
        unsafe { chain_or_reraise(signal, info, context) };
    };
    if !slot.active.load(Ordering::Acquire) {
        unsafe { chain_or_reraise(signal, info, context) };
    }
    if slot
        .dispatching
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        unsafe { chain_or_reraise(signal, info, context) };
    }
    let fault_address = unsafe { (*info).si_addr().addr() };
    let context = context.cast::<libc::ucontext_t>();
    let native_pc = unsafe { context_pc(&*context) };
    let arena_base = slot.arena_base.load(Ordering::Relaxed);
    let arena_guard_end = slot.arena_guard_end.load(Ordering::Relaxed);
    let registry = slot.registry.load(Ordering::Acquire);
    let site = (!registry.is_null())
        .then(|| unsafe { &*registry }.find(native_pc))
        .flatten();
    if fault_address < arena_base || fault_address >= arena_guard_end {
        slot.dispatching.store(false, Ordering::Release);
        unsafe { chain_or_reraise(signal, info, context.cast()) };
    }
    let Some(site) = site else {
        slot.dispatching.store(false, Ordering::Release);
        unsafe { chain_or_reraise(signal, info, context.cast()) };
    };
    unsafe {
        copy_signal_bytes(
            context.cast(),
            (*slot.context.get()).as_mut_ptr().cast(),
            size_of::<libc::ucontext_t>(),
        );
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let source = (*context).uc_mcontext.fpregs;
        let Some(fpstate_size) = x86_fpstate_size(source) else {
            slot.dispatching.store(false, Ordering::Release);
            chain_or_reraise(signal, info, context.cast());
        };
        copy_signal_bytes(
            source.cast(),
            (*slot.fpstate.get()).0.as_mut_ptr().cast(),
            fpstate_size,
        );
        (*(*slot.context.get()).as_mut_ptr()).uc_mcontext.fpregs =
            (*slot.fpstate.get()).0.as_mut_ptr().cast();
        let resume = &mut *slot.resume.get();
        resume.pc = (*context).uc_mcontext.gregs[libc::REG_RIP as usize] as usize;
        resume.rax = (*context).uc_mcontext.gregs[libc::REG_RAX as usize] as usize;
        resume.r11 = (*context).uc_mcontext.gregs[libc::REG_R11 as usize] as usize;
        resume.rdi = (*context).uc_mcontext.gregs[libc::REG_RDI as usize] as usize;
        resume.rsi = (*context).uc_mcontext.gregs[libc::REG_RSI as usize] as usize;
        resume.rflags = (*context).uc_mcontext.gregs[libc::REG_EFL as usize] as usize;
        resume.fpstate = (*slot.fpstate.get()).0.as_ptr().addr();
    }
    slot.signal.store(signal, Ordering::Relaxed);
    slot.fault_address.store(fault_address, Ordering::Relaxed);
    slot.native_pc.store(native_pc, Ordering::Relaxed);
    slot.site.store(
        site as *const NativeFaultSite as *mut NativeFaultSite,
        Ordering::Release,
    );
    let dispatch_top = slot.dispatcher_stack_top.load(Ordering::Acquire);
    unsafe { redirect_to_landing(&mut *context, slot, dispatch_top) };
}

const fn accepted_memory_fault_code(signal: i32, code: i32) -> bool {
    match signal {
        libc::SIGSEGV => matches!(code, LINUX_SEGV_MAPERR | LINUX_SEGV_ACCERR),
        libc::SIGBUS => matches!(code, libc::BUS_ADRALN | libc::BUS_ADRERR | libc::BUS_OBJERR),
        _ => false,
    }
}

#[cfg(target_arch = "x86_64")]
unsafe fn x86_fpstate_size(source: *mut libc::_libc_fpstate) -> Option<usize> {
    if source.is_null() {
        return None;
    }
    let bytes = source.cast::<u8>();
    let magic =
        unsafe { std::ptr::read_unaligned(bytes.add(FP_XSTATE_SW_BYTES_OFFSET).cast::<u32>()) };
    let size = if magic == FP_XSTATE_MAGIC1 {
        unsafe {
            std::ptr::read_unaligned(bytes.add(FP_XSTATE_SW_BYTES_OFFSET + 4).cast::<u32>())
                as usize
        }
    } else {
        size_of::<libc::_libc_fpstate>()
    };
    (size >= size_of::<libc::_libc_fpstate>() && size <= MAX_X86_FPSTATE_SIZE).then_some(size)
}

fn current_tid() -> i32 {
    unsafe { libc::syscall(libc::SYS_gettid) as i32 }
}

#[inline(always)]
unsafe fn copy_signal_bytes(source: *const u8, destination: *mut u8, size: usize) {
    for offset in 0..size {
        unsafe {
            destination
                .add(offset)
                .write_volatile(source.add(offset).read_volatile());
        }
    }
}

unsafe fn chain_or_reraise(
    signal: i32,
    info: *mut libc::siginfo_t,
    context: *mut libc::c_void,
) -> ! {
    let previous = PREVIOUS.get();
    let action = previous.map(|previous| {
        if signal == libc::SIGBUS {
            &previous.bus
        } else {
            &previous.segv
        }
    });
    if let Some(action) = action {
        let handler = action.sa_sigaction;
        if handler != libc::SIG_DFL && handler != libc::SIG_IGN {
            if action.sa_flags & libc::SA_SIGINFO != 0 {
                let callback = unsafe {
                    std::mem::transmute::<
                        usize,
                        unsafe extern "C" fn(i32, *mut libc::siginfo_t, *mut libc::c_void),
                    >(handler)
                };
                unsafe { callback(signal, info, context) };
            } else {
                let callback =
                    unsafe { std::mem::transmute::<usize, unsafe extern "C" fn(i32)>(handler) };
                unsafe { callback(signal) };
            }
        }
    }
    fatal_signal(signal)
}

fn fatal_signal(signal: i32) -> ! {
    let mut action = unsafe { MaybeUninit::<libc::sigaction>::zeroed().assume_init() };
    action.sa_sigaction = libc::SIG_DFL;
    unsafe { libc::sigemptyset(&mut action.sa_mask) };
    let _ = unsafe { libc::sigaction(signal, &action, std::ptr::null_mut()) };
    let pid = unsafe { libc::getpid() };
    let tid = current_tid();
    let _ = unsafe { libc::syscall(libc::SYS_tgkill, pid, tid, signal) };
    unsafe { libc::_exit(128 + signal) }
}

#[cfg(target_arch = "x86_64")]
fn context_pc(context: &libc::ucontext_t) -> usize {
    context.uc_mcontext.gregs[libc::REG_RIP as usize] as usize
}

#[cfg(target_arch = "x86_64")]
fn set_context_pc_sp(context: &mut libc::ucontext_t, pc: usize, sp: usize) {
    context.uc_mcontext.gregs[libc::REG_RIP as usize] = pc as libc::greg_t;
    context.uc_mcontext.gregs[libc::REG_RSP as usize] = sp as libc::greg_t;
}

#[cfg(target_arch = "x86_64")]
fn prepare_retry_context(context: &mut libc::ucontext_t, slot: &FaultSlot) {
    context.uc_mcontext.gregs[libc::REG_RIP as usize] =
        nixe_direct_retry_trampoline as *const () as libc::greg_t;
    context.uc_mcontext.gregs[libc::REG_RDI as usize] = slot.resume.get().addr() as libc::greg_t;
}

#[cfg(target_arch = "x86_64")]
unsafe fn redirect_to_landing(
    context: &mut libc::ucontext_t,
    slot: &FaultSlot,
    dispatch_top: usize,
) {
    set_context_pc_sp(
        context,
        nixe_direct_fault_landing_pad as *const () as usize,
        (dispatch_top & !15).wrapping_sub(8),
    );
    context.uc_mcontext.gregs[libc::REG_RDI as usize] = slot as *const FaultSlot as libc::greg_t;
}

#[cfg(target_arch = "x86_64")]
fn read_host_integer(
    context: &libc::ucontext_t,
    register: HostIntegerRegister,
) -> Result<u64, FaultRuntimeError> {
    let index = x86_greg_index(register.encoding).ok_or_else(|| {
        FaultRuntimeError::new("fault metadata names an unsupported x86 integer register")
    })?;
    Ok(context.uc_mcontext.gregs[index] as u64)
}

#[cfg(target_arch = "x86_64")]
fn x86_greg_index(encoding: u8) -> Option<usize> {
    Some(match encoding {
        0 => libc::REG_RAX,
        1 => libc::REG_RCX,
        2 => libc::REG_RDX,
        3 => libc::REG_RBX,
        4 => libc::REG_RSP,
        5 => libc::REG_RBP,
        6 => libc::REG_RSI,
        7 => libc::REG_RDI,
        8 => libc::REG_R8,
        9 => libc::REG_R9,
        10 => libc::REG_R10,
        11 => libc::REG_R11,
        12 => libc::REG_R12,
        13 => libc::REG_R13,
        14 => libc::REG_R14,
        15 => libc::REG_R15,
        _ => return None,
    } as usize)
}

#[cfg(target_arch = "aarch64")]
fn context_pc(context: &libc::ucontext_t) -> usize {
    context.uc_mcontext.pc as usize
}

#[cfg(target_arch = "aarch64")]
fn set_context_pc_sp(context: &mut libc::ucontext_t, pc: usize, sp: usize) {
    context.uc_mcontext.pc = pc as u64;
    context.uc_mcontext.sp = sp as u64;
}

#[cfg(target_arch = "aarch64")]
unsafe fn redirect_to_landing(
    context: &mut libc::ucontext_t,
    slot: &FaultSlot,
    dispatch_top: usize,
) {
    set_context_pc_sp(
        context,
        nixe_direct_fault_landing_pad as *const () as usize,
        dispatch_top & !15,
    );
    context.uc_mcontext.regs[0] = slot as *const FaultSlot as u64;
}

#[cfg(target_arch = "aarch64")]
fn read_host_integer(
    context: &libc::ucontext_t,
    register: HostIntegerRegister,
) -> Result<u64, FaultRuntimeError> {
    if register.encoding >= 31 {
        return Err(FaultRuntimeError::new(
            "fault metadata names an unsupported AArch64 integer register",
        ));
    }
    Ok(context.uc_mcontext.regs[register.encoding as usize])
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
compile_error!("nixe-cpu-direct-memory supports Linux x86-64 and AArch64 hosts");

#[cfg(target_arch = "x86_64")]
core::arch::global_asm!(
    r#"
    .text
    .globl nixe_direct_memory_invoke
    .type nixe_direct_memory_invoke,@function
nixe_direct_memory_invoke:
    push rbp
    mov rbp,rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp,8
    mov r12,rdi
    mov r13,rsi
    mov r14,rdx
    mov r15,rcx
    mov rdi,r12
    mov rsi,rsp
    lea rdx,[rip+.Ldirect_escape]
    call nixe_direct_prepare_escape
    mov rdi,r13
    mov rsi,r14
    call r15
    xor eax,eax
    jmp .Ldirect_return
.Ldirect_escape:
    mov eax,1
.Ldirect_return:
    add rsp,8
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
    .size nixe_direct_memory_invoke,.-nixe_direct_memory_invoke

    .globl nixe_direct_scalar_invoke
    .type nixe_direct_scalar_invoke,@function
nixe_direct_scalar_invoke:
    push rbp
    mov rbp,rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp,8
    mov r12,rsi
    mov r13,rdx
    mov rsi,rsp
    lea rdx,[rip+.Ldirect_scalar_escape]
    call nixe_direct_prepare_escape
    mov rdi,r12
    call r13
    xor eax,eax
    jmp .Ldirect_scalar_return
.Ldirect_scalar_escape:
    mov eax,1
.Ldirect_scalar_return:
    add rsp,8
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
    .size nixe_direct_scalar_invoke,.-nixe_direct_scalar_invoke

    .globl nixe_direct_fault_landing_pad
    .type nixe_direct_fault_landing_pad,@function
nixe_direct_fault_landing_pad:
    sub rsp,8
    call nixe_direct_fault_dispatch
    ud2
    .size nixe_direct_fault_landing_pad,.-nixe_direct_fault_landing_pad

    .globl nixe_direct_escape_now
    .type nixe_direct_escape_now,@function
nixe_direct_escape_now:
    mov rsp,rsi
    jmp rdi
    .size nixe_direct_escape_now,.-nixe_direct_escape_now

    .globl nixe_direct_retry_trampoline
    .type nixe_direct_retry_trampoline,@function
nixe_direct_retry_trampoline:
    mov rsi,[rdi+48]
    fxrstor64 [rsi]
    push qword ptr [rdi+40]
    popfq
    push qword ptr [rdi]
    mov rax,[rdi+8]
    mov r11,[rdi+16]
    mov rsi,[rdi+32]
    mov rdi,[rdi+24]
    ret
    .size nixe_direct_retry_trampoline,.-nixe_direct_retry_trampoline
"#
);

#[cfg(target_arch = "x86_64")]
core::arch::global_asm!(
    r#"
    .text
    .macro DIRECT_READ name
    .globl nixe_direct_stub_read_\name
    .type nixe_direct_stub_read_\name,@function
nixe_direct_stub_read_\name:
    mov rax,[rdi]
    .globl nixe_direct_stub_read_\name\()_fault
nixe_direct_stub_read_\name\()_fault:
    .if \name == 1
    movzx eax,byte ptr [rax]
    .elseif \name == 2
    movzx eax,word ptr [rax]
    .elseif \name == 4
    mov eax,dword ptr [rax]
    .else
    mov rax,qword ptr [rax]
    .endif
    .globl nixe_direct_stub_read_\name\()_after
nixe_direct_stub_read_\name\()_after:
    mov [rdi+16],rax
    ret
    .globl nixe_direct_stub_read_\name\()_end
nixe_direct_stub_read_\name\()_end:
    .size nixe_direct_stub_read_\name,.-nixe_direct_stub_read_\name
    .endm

    DIRECT_READ 1
    DIRECT_READ 2
    DIRECT_READ 4
    DIRECT_READ 8

    .macro DIRECT_WRITE name
    .globl nixe_direct_stub_write_\name
    .type nixe_direct_stub_write_\name,@function
nixe_direct_stub_write_\name:
    mov rax,[rdi]
    mov rdx,[rdi+8]
    .globl nixe_direct_stub_write_\name\()_fault
nixe_direct_stub_write_\name\()_fault:
    .if \name == 1
    mov byte ptr [rax],dl
    .elseif \name == 2
    mov word ptr [rax],dx
    .elseif \name == 4
    mov dword ptr [rax],edx
    .else
    mov qword ptr [rax],rdx
    .endif
    .globl nixe_direct_stub_write_\name\()_after
nixe_direct_stub_write_\name\()_after:
    ret
    .globl nixe_direct_stub_write_\name\()_end
nixe_direct_stub_write_\name\()_end:
    .size nixe_direct_stub_write_\name,.-nixe_direct_stub_write_\name
    .endm

    DIRECT_WRITE 1
    DIRECT_WRITE 2
    DIRECT_WRITE 4
    DIRECT_WRITE 8
"#
);

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;
    use std::os::unix::process::ExitStatusExt;
    use std::process::Command;
    use std::sync::Barrier;

    use nixe_cpu::memory::{ExecutionMemory, MemoryPermissions};
    use nixe_memory::{
        CanonicalBackingPage, CanonicalBackingStore, CanonicalRangeTranslator, ContentGeneration,
        DirectArena, DirectBackendPolicy, DirectMapRequest, DirectProtectRequest, DirectProtection,
        GuestPhysicalPageId,
    };

    #[repr(C)]
    struct SyntheticContext {
        address: *const u8,
        observed: u8,
    }

    #[inline(never)]
    #[cfg(target_arch = "x86_64")]
    unsafe extern "C" fn faulting_read(context: *mut libc::c_void) {
        let context = unsafe { &mut *context.cast::<SyntheticContext>() };
        let observed: u8;
        unsafe {
            core::arch::asm!(
                "mov {observed}, byte ptr [{address}]",
                observed = out(reg_byte) observed,
                address = in(reg) context.address,
                options(nostack, readonly),
            );
        }
        context.observed = observed;
    }

    #[inline(never)]
    #[cfg(target_arch = "aarch64")]
    unsafe extern "C" fn faulting_read(context: *mut libc::c_void) {
        let context = unsafe { &mut *context.cast::<SyntheticContext>() };
        let observed: u64;
        unsafe {
            core::arch::asm!(
                "ldrb {observed:w}, [{address}]",
                observed = out(reg) observed,
                address = in(reg) context.address,
                options(nostack, readonly),
            );
        }
        context.observed = observed as u8;
    }

    unsafe extern "C" fn gateway(context: *mut libc::c_void, entry: usize) {
        let entry =
            unsafe { std::mem::transmute::<usize, unsafe extern "C" fn(*mut libc::c_void)>(entry) };
        unsafe { entry(context) };
    }

    unsafe extern "C" fn retry(
        opaque: *mut libc::c_void,
        _fault: *mut CapturedFault,
    ) -> FaultDisposition {
        let arena = unsafe { &*opaque.cast::<DirectArena>() };
        arena
            .protect_ranges(&[DirectProtectRequest {
                guest_address: 0x1000,
                size: 4096,
                protection: DirectProtection::Read,
            }])
            .unwrap();
        FaultDisposition::Retry
    }

    unsafe extern "C" fn escape(
        _opaque: *mut libc::c_void,
        _fault: *mut CapturedFault,
    ) -> FaultDisposition {
        FaultDisposition::Escape
    }

    #[cfg(target_arch = "x86_64")]
    unsafe extern "C" fn nested_fault(
        _opaque: *mut libc::c_void,
        _fault: *mut CapturedFault,
    ) -> FaultDisposition {
        unsafe {
            core::arch::asm!(
                "mov al,byte ptr [0]",
                out("al") _,
                options(nostack, readonly),
            );
        }
        FaultDisposition::Fatal
    }

    #[cfg(target_arch = "aarch64")]
    unsafe extern "C" fn nested_fault(
        _opaque: *mut libc::c_void,
        _fault: *mut CapturedFault,
    ) -> FaultDisposition {
        unsafe {
            core::arch::asm!(
                "mov x9,xzr",
                "ldrb w10,[x9]",
                out("x9") _,
                out("x10") _,
                options(nostack, readonly),
            );
        }
        FaultDisposition::Fatal
    }

    unsafe extern "C" fn chained_exit(
        _signal: i32,
        _info: *mut libc::siginfo_t,
        _context: *mut libc::c_void,
    ) {
        unsafe { libc::_exit(77) }
    }

    fn fixture() -> (DirectArena, Arc<NativeFaultRegistry>) {
        let store = CanonicalBackingStore::allocate().unwrap();
        let page = CanonicalBackingPage::initialized(
            &store,
            GuestPhysicalPageId::new(1),
            &vec![0x5a; 4096],
            ContentGeneration::INITIAL,
        )
        .unwrap();
        let backing = page.direct_backing().unwrap();
        let arena = DirectArena::new(0x4000).unwrap();
        arena
            .map_pages(&[DirectMapRequest {
                guest_address: 0x1000,
                backing: &backing,
                protection: DirectProtection::None,
            }])
            .unwrap();
        let start = faulting_read as *const () as usize;
        let site = NativeFaultSite {
            native_start: start,
            native_end: start + 256,
            access: NativeMemoryAccess {
                address_space: AddressSpaceId::new(1),
                guest_pc: GuestVirtualAddress::new(0x8000),
                kind: NativeMemoryAccessKind::Read,
                size: 1,
            },
            completion: NativeFaultCompletion::None,
            guest_address: None,
        };
        let registry = NativeFaultRegistry::new(vec![NativeFaultRegion {
            native_start: start,
            native_end: start + 256,
            sites: Arc::from([site]),
        }])
        .unwrap();
        (arena, Arc::new(registry))
    }

    #[test]
    fn attributed_fault_dispatches_after_sigreturn_and_retries_once() {
        let (arena, registry) = fixture();
        let view = arena.view();
        let mut context = SyntheticContext {
            address: (view.base + 0x1000) as *const u8,
            observed: 0,
        };
        let mut worker = WorkerFaultContext::register().unwrap();
        let outcome = loop {
            let outcome = unsafe {
                worker.invoke(
                    view,
                    &registry,
                    retry,
                    std::ptr::from_ref(&arena).cast_mut().cast(),
                    NativeInvocation {
                        gateway,
                        context: std::ptr::from_mut(&mut context).cast(),
                        entry: faulting_read as *const () as usize,
                    },
                )
            }
            .unwrap();
            if outcome != InvocationOutcome::Retry {
                break outcome;
            }
        };
        assert_eq!(outcome, InvocationOutcome::Returned);
        assert_eq!(context.observed, 0x5a);
    }

    #[test]
    fn attributed_fault_can_escape_without_retrying_the_access() {
        let (arena, registry) = fixture();
        let view = arena.view();
        let mut context = SyntheticContext {
            address: (view.base + 0x1000) as *const u8,
            observed: 0,
        };
        let mut worker = WorkerFaultContext::register().unwrap();
        let outcome = unsafe {
            worker.invoke(
                view,
                &registry,
                escape,
                std::ptr::null_mut(),
                NativeInvocation {
                    gateway,
                    context: std::ptr::from_mut(&mut context).cast(),
                    entry: faulting_read as *const () as usize,
                },
            )
        }
        .unwrap();
        assert_eq!(outcome, InvocationOutcome::Escaped);
        assert_eq!(context.observed, 0);
    }

    #[test]
    fn attributed_sigbus_is_distinguished_and_can_escape() {
        let (arena, registry) = fixture();
        let view = arena.view();
        let name = CString::new("nixe-direct-sigbus-test").unwrap();
        let fd = unsafe { libc::memfd_create(name.as_ptr(), libc::MFD_CLOEXEC) };
        assert!(fd >= 0);
        assert_eq!(
            unsafe { libc::ftruncate(fd, DIRECT_PAGE_SIZE as libc::off_t) },
            0
        );
        let target = (view.base + 0x1000) as *mut libc::c_void;
        let mapped = unsafe {
            libc::mmap(
                target,
                DIRECT_PAGE_SIZE,
                libc::PROT_READ,
                libc::MAP_SHARED | libc::MAP_FIXED,
                fd,
                0,
            )
        };
        assert_eq!(mapped, target);
        assert_eq!(unsafe { libc::ftruncate(fd, 0) }, 0);
        let mut context = SyntheticContext {
            address: target.cast(),
            observed: 0,
        };
        let mut worker = WorkerFaultContext::register().unwrap();
        let outcome = unsafe {
            worker.invoke(
                view,
                &registry,
                escape,
                std::ptr::null_mut(),
                NativeInvocation {
                    gateway,
                    context: std::ptr::from_mut(&mut context).cast(),
                    entry: faulting_read as *const () as usize,
                },
            )
        }
        .unwrap();
        assert_eq!(outcome, InvocationOutcome::Escaped);
        assert_eq!(unsafe { libc::close(fd) }, 0);
    }

    #[test]
    fn dropping_a_worker_unregisters_its_tid_before_slot_reuse() {
        let worker = WorkerFaultContext::register().unwrap();
        let tid = worker.registered_tid();
        assert!(
            SLOTS
                .get()
                .unwrap()
                .iter()
                .any(|slot| { slot.tid.load(Ordering::Acquire) == tid })
        );
        drop(worker);
        assert!(
            SLOTS
                .get()
                .unwrap()
                .iter()
                .all(|slot| { slot.tid.load(Ordering::Acquire) != tid })
        );
        let replacement = WorkerFaultContext::register().unwrap();
        assert_eq!(replacement.registered_tid(), tid);
    }

    #[test]
    fn dropping_an_active_scalar_frontend_clears_its_worker_snapshot() {
        let (arena, _) = fixture();
        let mut frontend =
            unsafe { DirectScalarFrontend::new(arena.view(), AddressSpaceId::new(7)) }.unwrap();
        frontend.begin_slice().unwrap();
        let tid = frontend.worker.as_ref().unwrap().registered_tid();

        drop(frontend);

        assert!(
            SLOTS
                .get()
                .unwrap()
                .iter()
                .all(|slot| slot.tid.load(Ordering::Acquire) != tid)
        );
        let replacement = WorkerFaultContext::register().unwrap();
        assert_eq!(replacement.registered_tid(), tid);
    }

    #[test]
    fn native_fault_registry_accepts_arbitrary_order_and_rejects_overlap_and_exhaustion() {
        let registry = NativeFaultRegistry::with_capacity(2).unwrap();
        registry
            .publish(Arc::new(fake_region(0x2000, 0x2100)))
            .unwrap();
        registry
            .publish(Arc::new(fake_region(0x1000, 0x1100)))
            .unwrap();
        assert!(registry.find(0x1018).is_some());
        assert!(registry.find(0x2018).is_some());
        assert!(
            registry
                .publish(Arc::new(fake_region(0x1080, 0x1180)))
                .is_err()
        );
        assert!(
            registry
                .publish(Arc::new(fake_region(0x3000, 0x3100)))
                .is_err()
        );

        let invalid = NativeFaultRegion {
            native_start: 0x2000,
            native_end: 0x2100,
            sites: Arc::from([fake_site(0x2010, 0x2050), fake_site(0x2040, 0x2060)]),
        };
        assert!(NativeFaultRegistry::new(vec![invalid]).is_err());
    }

    #[test]
    fn store_publication_revalidates_armed_after_acquiring_the_page_sequence() {
        struct Controls {
            sequence: AtomicU64,
            generation: AtomicU64,
            armed: AtomicU64,
        }

        let controls = Arc::new(Controls {
            sequence: AtomicU64::new(1),
            generation: AtomicU64::new(7),
            armed: AtomicU64::new(1),
        });
        let control = DirectStoreControl {
            write_sequence_address: std::ptr::from_ref(&controls.sequence).addr(),
            generation_address: std::ptr::from_ref(&controls.generation).addr(),
            write_armed_address: std::ptr::from_ref(&controls.armed).addr(),
        };
        let started = Arc::new(Barrier::new(2));
        let worker_started = Arc::clone(&started);
        let worker = std::thread::spawn(move || {
            worker_started.wait();
            DirectStorePublication::acquire(control).is_none()
        });

        started.wait();
        controls.armed.store(0, Ordering::Release);
        controls.sequence.store(0, Ordering::Release);

        assert!(worker.join().unwrap());
        assert_eq!(controls.sequence.load(Ordering::Acquire), 0);
        assert_eq!(controls.generation.load(Ordering::Acquire), 7);
    }

    #[test]
    fn registry_readers_observe_only_complete_published_regions() {
        const REGIONS: usize = 256;
        let registry = Arc::new(NativeFaultRegistry::with_capacity(REGIONS).unwrap());
        let complete = Arc::new(AtomicBool::new(false));
        let reader_registry = Arc::clone(&registry);
        let reader_complete = Arc::clone(&complete);
        let reader = std::thread::spawn(move || {
            while !reader_complete.load(Ordering::Acquire) {
                for index in 0..REGIONS {
                    let start = 0x10_0000 + index * 0x100;
                    if let Some(site) = reader_registry.find(start + 0x18) {
                        assert_eq!(site.native_start, start + 0x10);
                        assert_eq!(site.native_end, start + 0x20);
                    }
                }
            }
        });
        for index in 0..REGIONS {
            let start = 0x10_0000 + index * 0x100;
            registry
                .publish(Arc::new(fake_region(start, start + 0x100)))
                .unwrap();
        }
        complete.store(true, Ordering::Release);
        reader.join().unwrap();
        for index in 0..REGIONS {
            let start = 0x10_0000 + index * 0x100;
            assert!(registry.find(start + 0x18).is_some());
        }
    }

    fn fake_region(start: usize, end: usize) -> NativeFaultRegion {
        NativeFaultRegion {
            native_start: start,
            native_end: end,
            sites: Arc::from([fake_site(start + 0x10, start + 0x20)]),
        }
    }

    fn fake_site(start: usize, end: usize) -> NativeFaultSite {
        NativeFaultSite {
            native_start: start,
            native_end: end,
            access: NativeMemoryAccess {
                address_space: AddressSpaceId::new(1),
                guest_pc: GuestVirtualAddress::new(0x8000),
                kind: NativeMemoryAccessKind::Read,
                size: 1,
            },
            completion: NativeFaultCompletion::None,
            guest_address: None,
        }
    }

    #[test]
    fn signal_code_filter_accepts_only_kernel_memory_faults() {
        assert!(accepted_memory_fault_code(libc::SIGSEGV, LINUX_SEGV_MAPERR));
        assert!(accepted_memory_fault_code(libc::SIGSEGV, LINUX_SEGV_ACCERR));
        assert!(accepted_memory_fault_code(libc::SIGBUS, libc::BUS_ADRERR));
        assert!(!accepted_memory_fault_code(libc::SIGSEGV, libc::SI_USER));
        assert!(!accepted_memory_fault_code(libc::SIGBUS, libc::SI_TKILL));
        assert!(!accepted_memory_fault_code(libc::SIGILL, 1));
    }

    #[test]
    fn fatal_fault_subprocess_entry() {
        let Ok(case) = std::env::var("NIXE_DIRECT_FATAL_CASE") else {
            return;
        };
        if case == "alternate_stack_guard" {
            let stack = GuardedStack::new(SIGNAL_STACK_SIZE).unwrap();
            unsafe { stack.usable.as_ptr().sub(1).write_volatile(0) };
            panic!("alternate-stack guard write unexpectedly returned");
        }
        if case == "unrelated_sigbus" {
            install().unwrap();
            unsafe { libc::raise(libc::SIGBUS) };
            panic!("unrelated SIGBUS unexpectedly returned");
        }
        if case == "chain" {
            let mut action = unsafe { MaybeUninit::<libc::sigaction>::zeroed().assume_init() };
            action.sa_sigaction = chained_exit as *const () as usize;
            action.sa_flags = libc::SA_SIGINFO;
            unsafe { libc::sigemptyset(&mut action.sa_mask) };
            assert_eq!(
                unsafe { libc::sigaction(libc::SIGSEGV, &action, std::ptr::null_mut()) },
                0,
            );
        }
        let (arena, default_registry) = fixture();
        let view = arena.view();
        let mut context = SyntheticContext {
            address: if case == "outside_address" || case == "chain" {
                std::ptr::null()
            } else {
                (view.base + 0x1000) as *const u8
            },
            observed: 0,
        };
        let registry = if case == "outside_pc" {
            Arc::new(
                NativeFaultRegistry::new(vec![NativeFaultRegion {
                    native_start: faulting_read as *const () as usize,
                    native_end: faulting_read as *const () as usize + 256,
                    sites: Arc::from([]),
                }])
                .unwrap(),
            )
        } else {
            default_registry
        };
        let dispatcher = if case == "nested" {
            nested_fault
        } else {
            escape
        };
        let mut worker = WorkerFaultContext::register().unwrap();
        let _ = unsafe {
            worker.invoke(
                view,
                &registry,
                dispatcher,
                std::ptr::null_mut(),
                NativeInvocation {
                    gateway,
                    context: std::ptr::from_mut(&mut context).cast(),
                    entry: faulting_read as *const () as usize,
                },
            )
        };
        panic!("fatal signal case unexpectedly returned");
    }

    #[test]
    fn unrelated_and_nested_faults_remain_fatal_and_previous_handlers_chain() {
        let executable = std::env::current_exe().unwrap();
        for case in [
            "outside_address",
            "outside_pc",
            "nested",
            "alternate_stack_guard",
            "unrelated_sigbus",
            "chain",
        ] {
            let status = Command::new(&executable)
                .args([
                    "--exact",
                    "tests::fatal_fault_subprocess_entry",
                    "--nocapture",
                ])
                .env("NIXE_DIRECT_FATAL_CASE", case)
                .status()
                .unwrap();
            if case == "chain" {
                assert_eq!(status.code(), Some(77));
            } else {
                let expected = if case == "unrelated_sigbus" {
                    libc::SIGBUS
                } else {
                    libc::SIGSEGV
                };
                assert_eq!(status.signal(), Some(expected), "case={case}");
            }
        }
    }

    #[test]
    fn concurrent_first_writers_each_complete_once_and_share_one_armed_page() {
        let mut memory = ExecutionMemory::new();
        let space = AddressSpaceId::new(1);
        let address = GuestVirtualAddress::new(0x1000);
        let page = GuestPhysicalPageId::new(1);
        assert!(memory.add_ram_page(page));
        assert!(memory.map_page(space, address, page, MemoryPermissions::READ_WRITE));
        memory
            .bind_cpu_memory_backend(space, 0x4000, DirectBackendPolicy::Required)
            .unwrap();
        let range = memory
            .translate_canonical_range(space, address, 1, MemoryPermissions::READ)
            .unwrap();
        let cpu_writes = nixe_memory::CanonicalCpuWriteDependency::capture(&range).unwrap();
        let view = memory.direct_address_space_view(space).unwrap();
        let memory = Arc::new(memory);
        let barrier = Arc::new(Barrier::new(2));

        let workers = [0x11_u8, 0x22]
            .into_iter()
            .map(|value| {
                let memory = Arc::clone(&memory);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    let mut direct = unsafe { DirectScalarFrontend::new(view, space) }.unwrap();
                    direct
                        .write_inner(
                            memory.as_ref(),
                            GuestVirtualAddress::new(0x8000),
                            address,
                            MemoryAccess::normal(MemoryAccessSize::Byte),
                            MemoryValue::U8(value),
                            || {
                                barrier.wait();
                            },
                        )
                        .unwrap();
                })
            })
            .collect::<Vec<_>>();
        for worker in workers {
            worker.join().unwrap();
        }

        assert!(!cpu_writes.remains_current());
        let MemoryValue::U8(observed) = memory
            .read(space, address, MemoryAccess::normal(MemoryAccessSize::Byte))
            .unwrap()
            .value
        else {
            unreachable!()
        };
        assert!(matches!(observed, 0x11 | 0x22));
        let control_slot = unsafe {
            &*(view.store_controls as *const AtomicUsize)
                .add(address.get() as usize / DIRECT_PAGE_SIZE)
        };
        assert_ne!(control_slot.load(Ordering::Acquire), 0);
    }
}

#[cfg(target_arch = "aarch64")]
core::arch::global_asm!(
    r#"
    .text
    .globl nixe_direct_memory_invoke
    .type nixe_direct_memory_invoke,%function
nixe_direct_memory_invoke:
    stp x29,x30,[sp,#-16]!
    mov x29,sp
    stp x19,x20,[sp,#-16]!
    stp x21,x22,[sp,#-16]!
    stp x23,x24,[sp,#-16]!
    stp x25,x26,[sp,#-16]!
    stp x27,x28,[sp,#-16]!
    mov x19,x0
    mov x20,x1
    mov x21,x2
    mov x22,x3
    mov x0,x19
    mov x1,sp
    adr x2,.Ldirect_escape
    bl nixe_direct_prepare_escape
    mov x0,x20
    mov x1,x21
    blr x22
    mov w0,#0
    b .Ldirect_return
.Ldirect_escape:
    mov w0,#1
.Ldirect_return:
    ldp x27,x28,[sp],#16
    ldp x25,x26,[sp],#16
    ldp x23,x24,[sp],#16
    ldp x21,x22,[sp],#16
    ldp x19,x20,[sp],#16
    ldp x29,x30,[sp],#16
    ret
    .size nixe_direct_memory_invoke,.-nixe_direct_memory_invoke

    .globl nixe_direct_scalar_invoke
    .type nixe_direct_scalar_invoke,%function
nixe_direct_scalar_invoke:
    stp x29,x30,[sp,#-16]!
    mov x29,sp
    stp x19,x20,[sp,#-16]!
    stp x21,x22,[sp,#-16]!
    stp x23,x24,[sp,#-16]!
    stp x25,x26,[sp,#-16]!
    stp x27,x28,[sp,#-16]!
    mov x19,x1
    mov x20,x2
    mov x1,sp
    adr x2,.Ldirect_scalar_escape
    bl nixe_direct_prepare_escape
    mov x0,x19
    blr x20
    mov w0,#0
    b .Ldirect_scalar_return
.Ldirect_scalar_escape:
    mov w0,#1
.Ldirect_scalar_return:
    ldp x27,x28,[sp],#16
    ldp x25,x26,[sp],#16
    ldp x23,x24,[sp],#16
    ldp x21,x22,[sp],#16
    ldp x19,x20,[sp],#16
    ldp x29,x30,[sp],#16
    ret
    .size nixe_direct_scalar_invoke,.-nixe_direct_scalar_invoke

    .globl nixe_direct_fault_landing_pad
    .type nixe_direct_fault_landing_pad,%function
nixe_direct_fault_landing_pad:
    bl nixe_direct_fault_dispatch
    brk #0
    .size nixe_direct_fault_landing_pad,.-nixe_direct_fault_landing_pad

    .globl nixe_direct_escape_now
    .type nixe_direct_escape_now,%function
nixe_direct_escape_now:
    mov sp,x1
    br x0
    .size nixe_direct_escape_now,.-nixe_direct_escape_now
"#
);

#[cfg(target_arch = "aarch64")]
core::arch::global_asm!(
    r#"
    .text
    .macro DIRECT_READ name, instruction, result
    .globl nixe_direct_stub_read_\name
    .type nixe_direct_stub_read_\name,%function
nixe_direct_stub_read_\name:
    ldr x9,[x0]
    .globl nixe_direct_stub_read_\name\()_fault
nixe_direct_stub_read_\name\()_fault:
    \instruction \result,[x9]
    .globl nixe_direct_stub_read_\name\()_after
nixe_direct_stub_read_\name\()_after:
    str x10,[x0,#16]
    ret
    .globl nixe_direct_stub_read_\name\()_end
nixe_direct_stub_read_\name\()_end:
    .size nixe_direct_stub_read_\name,.-nixe_direct_stub_read_\name
    .endm

    DIRECT_READ 1, ldrb, w10
    DIRECT_READ 2, ldrh, w10
    DIRECT_READ 4, ldr, w10
    DIRECT_READ 8, ldr, x10

    .macro DIRECT_WRITE name, instruction, source
    .globl nixe_direct_stub_write_\name
    .type nixe_direct_stub_write_\name,%function
nixe_direct_stub_write_\name:
    ldr x9,[x0]
    ldr x10,[x0,#8]
    .globl nixe_direct_stub_write_\name\()_fault
nixe_direct_stub_write_\name\()_fault:
    \instruction \source,[x9]
    .globl nixe_direct_stub_write_\name\()_after
nixe_direct_stub_write_\name\()_after:
    ret
    .globl nixe_direct_stub_write_\name\()_end
nixe_direct_stub_write_\name\()_end:
    .size nixe_direct_stub_write_\name,.-nixe_direct_stub_write_\name
    .endm

    DIRECT_WRITE 1, strb, w10
    DIRECT_WRITE 2, strh, w10
    DIRECT_WRITE 4, str, w10
    DIRECT_WRITE 8, str, x10
"#
);
