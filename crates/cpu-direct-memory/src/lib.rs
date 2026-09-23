//! Linux native-fault runtime shared by CPU execution frontends.
//!
//! The signal handler performs only bounded slot lookup, fixed-stub attribution,
//! context capture, and register redirection. Epoch-owned JIT attribution and
//! emulator policy run after `sigreturn` on a preallocated dispatcher stack.

#![cfg(target_os = "linux")]

mod fatal;

use std::cell::UnsafeCell;
use std::fmt::{Display, Formatter};
use std::mem::offset_of;
use std::mem::{ManuallyDrop, MaybeUninit, size_of};
use std::ptr::NonNull;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicPtr, AtomicU64, AtomicUsize, Ordering};

use nixe_cpu::memory::{
    CpuMemory, DataAccessFault, DataAccessKind, DirectFaultResolution, MemoryAccess,
    MemoryAccessClass, MemoryAccessSize, MemoryOrdering, MemoryValue,
};
use nixe_memory::{AddressSpaceId, DIRECT_PAGE_SIZE, DirectAddressSpaceView, GuestVirtualAddress};

const MAX_WORKER_SLOTS: usize = 128;
const MAX_UNCHANGED_RETRIES: usize = 8;
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
    pub element_index: u8,
}

/// One exact native instruction interval which may access a direct arena.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeFaultSite {
    pub native_start: usize,
    pub native_end: usize,
    pub access: NativeMemoryAccess,
}

/// Immutable metadata for one finalized native function.
#[derive(Clone, Debug)]
struct NativeFaultRegion {
    native_start: usize,
    native_end: usize,
    sites: Box<[NativeFaultSite]>,
}

/// Immutable attribution for fixed native memory stubs, published as a whole
/// before a worker becomes active. JIT code uses its own epoch-owned directory.
/// Signal-context lookup only reads this table: no locks, allocations or
/// incremental publication, and no retained retired code.
struct NativeFaultRegistry {
    regions: Box<[NativeFaultRegion]>,
}

impl NativeFaultRegistry {
    fn new(mut regions: Vec<NativeFaultRegion>) -> Result<Self, FaultRuntimeError> {
        regions.sort_unstable_by_key(|region| region.native_start);
        for region in &regions {
            validate_region(region)?;
        }
        if regions
            .windows(2)
            .any(|pair| pair[0].native_end > pair[1].native_start)
        {
            return Err(FaultRuntimeError::new("native fault regions overlap"));
        }
        Ok(Self {
            regions: regions.into_boxed_slice(),
        })
    }

    fn find(&self, native_pc: usize) -> Option<&NativeFaultSite> {
        let index = self
            .regions
            .partition_point(|region| region.native_start <= native_pc)
            .checked_sub(1)?;
        let region = &self.regions[index];
        if native_pc >= region.native_end {
            return None;
        }
        let site_index = region
            .sites
            .partition_point(|site| site.native_start <= native_pc)
            .checked_sub(1)?;
        let site = &region.sites[site_index];
        (native_pc < site.native_end).then_some(site)
    }
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
    /// Epoch-owned directory lookup did not attribute the captured native PC.
    FatalUnattributed = 3,
    /// The dispatcher caught a panic; unwinding must not cross native assembly.
    FatalPanic = 4,
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

mod captured;
pub use captured::CapturedFault;

#[derive(Debug)]
#[repr(C)]
struct FaultSlot {
    #[cfg(target_arch = "x86_64")]
    resume: UnsafeCell<ResumeRecord>,
    tid: AtomicI32,
    active: AtomicBool,
    dispatching: AtomicBool,
    arena_base: AtomicUsize,
    arena_guard_end: AtomicUsize,
    registry: AtomicPtr<NativeFaultRegistry>,
    dispatcher_fp_control: AtomicU64,
    dispatcher_fp_status: AtomicU64,
    dispatcher: AtomicUsize,
    opaque: AtomicPtr<libc::c_void>,
    escape_sp: AtomicUsize,
    escape_pc: AtomicUsize,
    dispatcher_stack_top: AtomicUsize,
    signal: AtomicI32,
    fault_address: AtomicUsize,
    native_pc: AtomicUsize,
    site: AtomicPtr<NativeFaultSite>,
    retry_pc: AtomicUsize,
    retry_address: AtomicUsize,
    retry_count: AtomicUsize,
    context: UnsafeCell<MaybeUninit<libc::ucontext_t>>,
    #[cfg(target_arch = "aarch64")]
    signal_frame: AtomicUsize,
    #[cfg(target_arch = "aarch64")]
    signal_context: AtomicUsize,
    #[cfg(target_arch = "x86_64")]
    fpstate: UnsafeCell<AlignedFpState>,
}

#[cfg(target_arch = "x86_64")]
#[repr(align(64))]
struct AlignedFpState([u8; MAX_X86_FPSTATE_SIZE]);

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
struct ResumeRecord {
    rax: usize,
    rbx: usize,
    rcx: usize,
    rdx: usize,
    rbp: usize,
    rsp: usize,
    rsi: usize,
    rdi: usize,
    r8: usize,
    r9: usize,
    r10: usize,
    r11: usize,
    r12: usize,
    r13: usize,
    r14: usize,
    r15: usize,
    pc: usize,
    rflags: usize,
    fpstate: usize,
    xstate_features: usize,
}

unsafe impl Sync for FaultSlot {}
unsafe impl Send for FaultSlot {}

impl FaultSlot {
    fn new() -> Self {
        Self {
            #[cfg(target_arch = "x86_64")]
            resume: UnsafeCell::new(ResumeRecord {
                rax: 0,
                rbx: 0,
                rcx: 0,
                rdx: 0,
                rbp: 0,
                rsp: 0,
                rsi: 0,
                rdi: 0,
                r8: 0,
                r9: 0,
                r10: 0,
                r11: 0,
                r12: 0,
                r13: 0,
                r14: 0,
                r15: 0,
                pc: 0,
                rflags: 0,
                fpstate: 0,
                xstate_features: 0,
            }),
            tid: AtomicI32::new(0),
            active: AtomicBool::new(false),
            dispatching: AtomicBool::new(false),
            arena_base: AtomicUsize::new(0),
            arena_guard_end: AtomicUsize::new(0),
            registry: AtomicPtr::new(std::ptr::null_mut()),
            dispatcher_fp_control: AtomicU64::new(0),
            dispatcher_fp_status: AtomicU64::new(0),
            dispatcher: AtomicUsize::new(0),
            opaque: AtomicPtr::new(std::ptr::null_mut()),
            escape_sp: AtomicUsize::new(0),
            escape_pc: AtomicUsize::new(0),
            dispatcher_stack_top: AtomicUsize::new(0),
            signal: AtomicI32::new(0),
            fault_address: AtomicUsize::new(0),
            native_pc: AtomicUsize::new(0),
            site: AtomicPtr::new(std::ptr::null_mut()),
            retry_pc: AtomicUsize::new(0),
            retry_address: AtomicUsize::new(0),
            retry_count: AtomicUsize::new(0),
            context: UnsafeCell::new(MaybeUninit::uninit()),
            #[cfg(target_arch = "aarch64")]
            signal_frame: AtomicUsize::new(0),
            #[cfg(target_arch = "aarch64")]
            signal_context: AtomicUsize::new(0),
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

/// OS-thread owner shared by sequential CPU slices, regardless of their process
/// or backend. Construct on the executing worker; registration is lazy. Retiring
/// a process does not unregister the host's alternate signal stack.
#[derive(Default)]
pub struct NativeWorker {
    faults: Option<WorkerFaultContext>,
    _thread_bound: std::marker::PhantomData<std::rc::Rc<()>>,
}

impl NativeWorker {
    pub fn faults(&mut self) -> Result<&mut WorkerFaultContext, FaultRuntimeError> {
        if self.faults.is_none() {
            self.faults = Some(WorkerFaultContext::register()?);
        }
        Ok(self.faults.as_mut().expect("worker registration succeeded"))
    }

    /// Explicit OS-worker teardown reports restoration failures. Drop remains
    /// best-effort and retains the registration's resources on failure.
    pub fn finish(&mut self) -> Result<(), FaultRuntimeError> {
        if let Some(faults) = &mut self.faults {
            faults.unregister()?;
        }
        self.faults = None;
        Ok(())
    }
}

/// Per-host-worker registration and preallocated recovery stacks.
/// The installed alternate stack and its owner must stay on their OS thread.
/// This makes native entry/exit thread-affine without per-invocation gettid.
///
/// ```compile_fail
/// fn require_send<T: Send>() {}
/// require_send::<nixe_cpu_direct_memory::WorkerFaultContext>();
/// ```
pub struct WorkerFaultContext {
    slot: NonNull<FaultSlot>,
    signal_stack: ManuallyDrop<GuardedStack>,
    dispatch_stack: ManuallyDrop<GuardedStack>,
    previous_stack: libc::stack_t,
    tid: i32,
    escaped: bool,
    _thread_bound: std::marker::PhantomData<std::rc::Rc<()>>,
}

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
        let slots = SLOTS
            .get()
            .ok_or_else(|| FaultRuntimeError::new("fault slots are not installed"))?;
        // Nested registrations cannot be safely retired in arbitrary process
        // order: the saved previous stack may belong to an already freed owner.
        if slots
            .iter()
            .any(|slot| slot.tid.load(Ordering::Acquire) == tid)
        {
            return Err(FaultRuntimeError::new(
                "native fault context already registered on this host TID; share its NativeWorker",
            ));
        }
        let signal_stack = GuardedStack::new(SIGNAL_STACK_SIZE)?;
        let dispatch_stack = GuardedStack::new(DISPATCH_STACK_SIZE)?;
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
            escaped: false,
            _thread_bound: std::marker::PhantomData,
        })
    }

    /// Original registration TID, or zero after successful unregistration.
    #[must_use]
    pub fn registered_tid(&self) -> i32 {
        self.tid
    }

    /// Release this worker's registration on its original OS thread. On
    /// failure the stacks/slot remain owned so teardown can report or retry;
    /// they must never be freed while an alternate stack may still name them.
    /// Repeated successful teardown is harmless; native use afterward fails.
    pub fn unregister(&mut self) -> Result<(), FaultRuntimeError> {
        if self.tid == 0 {
            return Ok(());
        }
        if current_tid() != self.tid {
            return Err(FaultRuntimeError::new(
                "native fault worker unregistered from a different host TID",
            ));
        }
        let slot = unsafe { self.slot.as_ref() };
        slot.active.store(false, Ordering::Release);
        slot.registry.store(std::ptr::null_mut(), Ordering::Relaxed);
        slot.dispatcher.store(0, Ordering::Relaxed);
        slot.opaque.store(std::ptr::null_mut(), Ordering::Relaxed);
        if unsafe { libc::sigaltstack(&self.previous_stack, std::ptr::null_mut()) } != 0 {
            return Err(FaultRuntimeError::last(
                "worker alternate signal stack restoration failed",
            ));
        }
        slot.dispatcher_stack_top.store(0, Ordering::Release);
        slot.tid.store(0, Ordering::Release);
        self.tid = 0;
        self.escaped = false;
        unsafe {
            ManuallyDrop::drop(&mut self.signal_stack);
            ManuallyDrop::drop(&mut self.dispatch_stack);
        }
        Ok(())
    }

    /// Borrow the last escaped machine image on the original worker. The
    /// borrow prevents slot reuse until cold reconstruction has finished.
    /// The execution owner must separately retain the code epoch and frame.
    pub fn escaped_fault(&mut self) -> Result<CapturedFault<'_>, FaultRuntimeError> {
        if current_tid() != self.tid
            || !self.escaped
            || unsafe { self.slot.as_ref() }.active.load(Ordering::Acquire)
        {
            return Err(FaultRuntimeError::new("no escaped fault on this worker"));
        }
        Ok(CapturedFault {
            slot: self.slot,
            site: None,
            lifetime: std::marker::PhantomData,
        })
    }

    /// Capture faults without installing a second native-PC registry. The
    /// dispatcher attributes `CapturedFault::native_pc()` through the execution
    /// owner's already-protected directory, after returning from the signal.
    /// Fixed interpreter stubs publish a batch snapshot for each slice.
    /// The landing leaf installs the invocation owner's saved caller FP state
    /// before entering Rust. Retry restores the untouched captured guest state;
    /// it neither commits FPSR nor changes the frontend's FP ownership fields.
    ///
    /// # Safety
    ///
    /// The invocation, arena, opaque data and all code/metadata reachable by the
    /// dispatcher must remain valid until return. The caller must publish its
    /// execution epoch before reading the native entry, and retain it throughout
    /// capture, resolution and retry. The dispatcher must reject unattributed
    /// PCs with a fatal disposition, never infer attribution from an arena
    /// address alone. `FatalUnattributed` identifies this diagnostic precisely.
    /// It must not use `CapturedFault::site`, unwind, or return `Retry` without
    /// repairing the captured access. `Escape` skips gateway Rust frames: those
    /// frames must own no destructors, and the caller must restore FP and guest
    /// state explicitly before announcing quiescence.
    /// `caller_fp` is the owner's saved host [control, status]: x86 MXCSR split
    /// into control/status bits, or AArch64 FPCR/FPSR. It must be valid for this
    /// host, saved on this worker, and safe for ordinary dispatcher Rust work.
    pub unsafe fn invoke_captured(
        &mut self,
        arena: DirectAddressSpaceView,
        caller_fp: [u64; 2],
        dispatcher: FaultDispatcher,
        opaque: *mut libc::c_void,
        invocation: NativeInvocation,
    ) -> Result<InvocationOutcome, FaultRuntimeError> {
        unsafe {
            self.begin_capture(
                arena,
                std::ptr::null_mut(),
                Some(caller_fp),
                dispatcher,
                opaque,
            )
        }?;
        unsafe { self.invoke_active(invocation) }
    }

    unsafe fn invoke_active(
        &mut self,
        invocation: NativeInvocation,
    ) -> Result<InvocationOutcome, FaultRuntimeError> {
        let escaped = unsafe {
            nixe_direct_memory_invoke(
                self.slot.as_ptr().cast(),
                invocation.context,
                invocation.entry,
                invocation.gateway,
            )
        };
        self.end_batch()?;
        self.escaped = escaped != 0;
        Ok(invocation_outcome(escaped, unsafe { self.slot.as_ref() }))
    }

    /// Publishes one stable arena/registry/dispatcher snapshot for a batch of
    /// fixed direct accesses, normally one interpreter slice.
    ///
    /// # Safety
    ///
    /// `registry`, `opaque`, the arena and everything reachable from the
    /// dispatcher must outlive the matching [`Self::end_batch`].
    unsafe fn begin_batch(
        &mut self,
        arena: DirectAddressSpaceView,
        registry: &NativeFaultRegistry,
        dispatcher: FaultDispatcher,
        opaque: *mut libc::c_void,
    ) -> Result<(), FaultRuntimeError> {
        unsafe {
            self.begin_capture(
                arena,
                std::ptr::from_ref(registry).cast_mut(),
                None,
                dispatcher,
                opaque,
            )
        }
    }

    unsafe fn begin_capture(
        &mut self,
        arena: DirectAddressSpaceView,
        registry: *mut NativeFaultRegistry,
        caller_fp: Option<[u64; 2]>,
        dispatcher: FaultDispatcher,
        opaque: *mut libc::c_void,
    ) -> Result<(), FaultRuntimeError> {
        if self.tid == 0 {
            return Err(FaultRuntimeError::new(
                "native fault context was invoked after unregistration",
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
        if let Some([control, status]) = caller_fp {
            #[cfg(target_arch = "x86_64")]
            let control = control | status;
            slot.dispatcher_fp_control.store(control, Ordering::Relaxed);
            slot.dispatcher_fp_status.store(status, Ordering::Relaxed);
        }
        self.escaped = false;
        slot.arena_base.store(arena.base, Ordering::Relaxed);
        slot.arena_guard_end.store(guard_end, Ordering::Relaxed);
        slot.registry.store(registry, Ordering::Relaxed);
        slot.dispatcher
            .store(dispatcher as usize, Ordering::Relaxed);
        slot.opaque.store(opaque, Ordering::Relaxed);
        slot.site.store(std::ptr::null_mut(), Ordering::Relaxed);
        slot.retry_count.store(0, Ordering::Relaxed);
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
    fn end_batch(&mut self) -> Result<(), FaultRuntimeError> {
        if self.tid == 0 {
            return Err(FaultRuntimeError::new(
                "native fault context batch ended after unregistration",
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

    /// Invokes one fixed memory stub while a batch snapshot is active.
    ///
    /// # Safety
    ///
    /// `context` must point to the stub call layout and `entry` must be one of
    /// the immutable functions registered in the active registry.
    unsafe fn invoke_stub_in_batch(
        &mut self,
        context: *mut libc::c_void,
        entry: usize,
    ) -> Result<InvocationOutcome, FaultRuntimeError> {
        let slot = unsafe { self.slot.as_ref() };
        if self.tid == 0 || !slot.active.load(Ordering::Acquire) {
            return Err(FaultRuntimeError::new(
                "native fault context stub batch is not active",
            ));
        }
        let escaped = unsafe { nixe_direct_stub_invoke(self.slot.as_ptr().cast(), context, entry) };
        self.escaped = escaped != 0;
        Ok(invocation_outcome(escaped, slot))
    }
}

fn invocation_outcome(escaped: u32, slot: &FaultSlot) -> InvocationOutcome {
    let outcome = if escaped == 0 {
        InvocationOutcome::Returned
    } else {
        InvocationOutcome::Escaped
    };
    slot.retry_count.store(0, Ordering::Relaxed);
    outcome
}

impl Drop for WorkerFaultContext {
    fn drop(&mut self) {
        // Explicit callers use unregister to observe failure. Drop preserves
        // the previous leak-on-failure behavior, never a dangling signal stack.
        let _ = self.unregister();
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InvocationOutcome {
    Returned,
    Escaped,
}

/// Failure returned by the fixed interpreter direct-memory frontend.
#[derive(Debug)]
pub enum DirectMemoryAccessError {
    DataFault(DataAccessFault),
    Backend(Box<str>),
    Runtime(FaultRuntimeError),
}

impl Display for DirectMemoryAccessError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DataFault(fault) => write!(formatter, "direct memory data fault: {fault:?}"),
            Self::Backend(detail) => formatter.write_str(detail),
            Self::Runtime(error) => Display::fmt(error, formatter),
        }
    }
}

impl std::error::Error for DirectMemoryAccessError {}

/// Process-bound fixed-stub frontend used by the reference interpreter.
///
/// Binding selects this object once for a LinuxDirect process. Accesses contain
/// no page-table walk or permission test; host protection remains the access
/// authority. Recoverable first-write and visibility faults retry the exact
/// native load or store after the shared page transition completes.
pub struct DirectMemoryFrontend {
    arena: DirectAddressSpaceView,
    address_space: AddressSpaceId,
    dispatcher_context: Box<MemoryDispatcherContext>,
}

/// Active interpreter slice borrowing, never owning, the OS-worker registration.
/// Neither the frontend nor worker can be reused until the snapshot is cleared.
pub struct DirectMemorySlice<'a> {
    frontend: &'a mut DirectMemoryFrontend,
    worker: &'a mut WorkerFaultContext,
    _thread_bound: std::marker::PhantomData<std::rc::Rc<()>>,
}

struct MemoryDispatcherContext {
    current_call: AtomicPtr<StubCall>,
    arena_base: usize,
    address_space: AddressSpaceId,
}

impl DirectMemoryFrontend {
    /// # Safety
    ///
    /// The arena must remain alive and unchanged for every call made through
    /// this frontend. Higher-level CPU frontends validate the view against the
    /// currently borrowed `CpuMemory` before beginning each slice.
    pub unsafe fn new(
        arena: DirectAddressSpaceView,
        address_space: AddressSpaceId,
    ) -> Result<Self, FaultRuntimeError> {
        Ok(Self {
            arena,
            address_space,
            dispatcher_context: Box::new(MemoryDispatcherContext {
                current_call: AtomicPtr::new(std::ptr::null_mut()),
                arena_base: arena.base,
                address_space,
            }),
        })
    }

    /// Publishes the stable fault snapshot once for an interpreter slice.
    pub fn begin_slice<'a>(
        &'a mut self,
        worker: &'a mut NativeWorker,
    ) -> Result<DirectMemorySlice<'a>, FaultRuntimeError> {
        let arena = self.arena;
        let registry = memory_stub_registry()?;
        let opaque = std::ptr::from_ref(self.dispatcher_context.as_ref())
            .cast_mut()
            .cast();
        let worker = worker.faults()?;
        unsafe { worker.begin_batch(arena, registry, dispatch_memory_stub_fault, opaque) }?;
        Ok(DirectMemorySlice {
            frontend: self,
            worker,
            _thread_bound: std::marker::PhantomData,
        })
    }
}

impl DirectMemorySlice<'_> {
    pub fn read(
        &mut self,
        memory: &dyn CpuMemory,
        address: GuestVirtualAddress,
        access: MemoryAccess,
    ) -> Result<MemoryValue, DirectMemoryAccessError> {
        let size = validate_direct_access(access)?;
        let Some(pointer) = self.frontend.direct_pointer(address, size) else {
            return memory
                .read(self.frontend.address_space, address, access)
                .map(|result| result.value)
                .map_err(DirectMemoryAccessError::DataFault);
        };
        let mut call = StubCall::new(pointer, 0, memory);
        self.invoke(&mut call, memory_stub(size, DataAccessKind::Read))?;
        call.finish()?;
        Ok(MemoryValue::from_bits(size, call.output))
    }

    pub fn write(
        &mut self,
        memory: &dyn CpuMemory,
        address: GuestVirtualAddress,
        access: MemoryAccess,
        value: MemoryValue,
    ) -> Result<(), DirectMemoryAccessError> {
        let size = validate_direct_access(access)?;
        if value.size() != size {
            return Err(DirectMemoryAccessError::Backend(
                "direct store value does not match its access width".into(),
            ));
        }
        let Some(pointer) = self.frontend.direct_pointer(address, size) else {
            return memory
                .write(self.frontend.address_space, address, access, value)
                .map(|_| ())
                .map_err(DirectMemoryAccessError::DataFault);
        };
        let mut call = StubCall::new(pointer, value.bits(), memory);
        self.invoke(&mut call, memory_stub(size, DataAccessKind::Write))?;
        call.finish()
    }

    fn invoke(
        &mut self,
        call: &mut StubCall,
        entry: usize,
    ) -> Result<InvocationOutcome, DirectMemoryAccessError> {
        let call_pointer = std::ptr::from_mut(call);
        self.frontend
            .dispatcher_context
            .current_call
            .store(call_pointer, Ordering::Release);
        let outcome = unsafe { self.worker.invoke_stub_in_batch(call_pointer.cast(), entry) };
        self.frontend
            .dispatcher_context
            .current_call
            .store(std::ptr::null_mut(), Ordering::Release);
        outcome.map_err(DirectMemoryAccessError::Runtime)
    }
}

impl Drop for DirectMemorySlice<'_> {
    fn drop(&mut self) {
        self.worker
            .end_batch()
            .expect("a borrowed direct-memory slice ends on its OS worker");
    }
}

impl DirectMemoryFrontend {
    fn direct_pointer(
        &self,
        address: GuestVirtualAddress,
        size: MemoryAccessSize,
    ) -> Option<usize> {
        let address = address.get();
        let bytes = size.bytes();
        if address >= self.arena.address_space_size as u64 {
            return None;
        }
        if bytes != 1 {
            let last = address.checked_add((bytes - 1) as u64)?;
            let page_offset = address & (DIRECT_PAGE_SIZE as u64 - 1);
            if last >= self.arena.address_space_size as u64
                || page_offset > (DIRECT_PAGE_SIZE - bytes) as u64
            {
                return None;
            }
        }
        self.arena.base.checked_add(address as usize)
    }
}

fn validate_direct_access(
    access: MemoryAccess,
) -> Result<MemoryAccessSize, DirectMemoryAccessError> {
    if access.ordering != MemoryOrdering::Relaxed || access.class != MemoryAccessClass::Normal {
        return Err(DirectMemoryAccessError::Backend(
            "direct stubs accept only relaxed ordinary accesses".into(),
        ));
    }
    Ok(access.size)
}

#[repr(C)]
struct StubCall {
    pointer: usize,
    value: u128,
    output: u128,
    memory: *const dyn CpuMemory,
    data_fault: Option<DataAccessFault>,
    backend_error: Option<Box<str>>,
}

impl StubCall {
    fn new(pointer: usize, value: u128, memory: &dyn CpuMemory) -> Self {
        let memory = unsafe { std::mem::transmute::<&dyn CpuMemory, *const dyn CpuMemory>(memory) };
        Self {
            pointer,
            value,
            output: 0,
            memory,
            data_fault: None,
            backend_error: None,
        }
    }

    fn finish(&mut self) -> Result<(), DirectMemoryAccessError> {
        if let Some(fault) = self.data_fault.take() {
            return Err(DirectMemoryAccessError::DataFault(fault));
        }
        if let Some(detail) = self.backend_error.take() {
            return Err(DirectMemoryAccessError::Backend(detail));
        }
        Ok(())
    }
}

unsafe extern "C" fn dispatch_memory_stub_fault(
    opaque: *mut libc::c_void,
    fault: *mut CapturedFault,
) -> FaultDisposition {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let context = unsafe { &*opaque.cast::<MemoryDispatcherContext>() };
        let call = context.current_call.load(Ordering::Acquire);
        if call.is_null() {
            return FaultDisposition::Fatal;
        }
        let call = unsafe { &mut *call };
        let site = unsafe { &*fault }.site();
        let size = match site.access.size {
            1 => MemoryAccessSize::Byte,
            2 => MemoryAccessSize::Halfword,
            4 => MemoryAccessSize::Word,
            8 => MemoryAccessSize::Doubleword,
            16 => MemoryAccessSize::Quadword,
            _ => return FaultDisposition::Fatal,
        };
        if site.access.element_index != 0 {
            return FaultDisposition::Fatal;
        }
        let kind = match site.access.kind {
            NativeMemoryAccessKind::Read => DataAccessKind::Read,
            NativeMemoryAccessKind::Write => DataAccessKind::Write,
        };
        let memory = unsafe { &*call.memory };
        let Some(guest_address) = call.pointer.checked_sub(context.arena_base) else {
            return FaultDisposition::Fatal;
        };
        match memory.resolve_direct_fault(
            context.address_space,
            GuestVirtualAddress::new(guest_address as u64),
            size,
            kind,
        ) {
            DirectFaultResolution::Retry => FaultDisposition::Retry,
            DirectFaultResolution::Cold => {
                // Fixed stubs enter only after checked page-local RAM eligibility.
                call.backend_error = Some(
                    "eligible fixed memory stub unexpectedly requires typed completion".into(),
                );
                FaultDisposition::Escape
            }
            DirectFaultResolution::Fault(fault) => {
                call.data_fault = Some(fault);
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

fn memory_stub(size: MemoryAccessSize, kind: DataAccessKind) -> usize {
    match (kind, size) {
        (DataAccessKind::Read, MemoryAccessSize::Byte) => function_address(nixe_direct_stub_read_1),
        (DataAccessKind::Read, MemoryAccessSize::Halfword) => {
            function_address(nixe_direct_stub_read_2)
        }
        (DataAccessKind::Read, MemoryAccessSize::Word) => function_address(nixe_direct_stub_read_4),
        (DataAccessKind::Read, MemoryAccessSize::Doubleword) => {
            function_address(nixe_direct_stub_read_8)
        }
        (DataAccessKind::Read, MemoryAccessSize::Quadword) => {
            function_address(nixe_direct_stub_read_16)
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
        (DataAccessKind::Write, MemoryAccessSize::Quadword) => {
            function_address(nixe_direct_stub_write_16)
        }
    }
}

fn function_address(function: unsafe extern "C" fn(*mut libc::c_void)) -> usize {
    function as *const () as usize
}

fn memory_stub_registry() -> Result<&'static NativeFaultRegistry, FaultRuntimeError> {
    static REGISTRY: OnceLock<Result<NativeFaultRegistry, FaultRuntimeError>> = OnceLock::new();
    REGISTRY
        .get_or_init(|| {
            NativeFaultRegistry::new(vec![
                memory_stub_region(
                    function_address(nixe_direct_stub_read_1),
                    std::ptr::addr_of!(nixe_direct_stub_read_1_end).addr(),
                    std::ptr::addr_of!(nixe_direct_stub_read_1_fault).addr(),
                    std::ptr::addr_of!(nixe_direct_stub_read_1_after).addr(),
                    DataAccessKind::Read,
                    1,
                ),
                memory_stub_region(
                    function_address(nixe_direct_stub_read_2),
                    std::ptr::addr_of!(nixe_direct_stub_read_2_end).addr(),
                    std::ptr::addr_of!(nixe_direct_stub_read_2_fault).addr(),
                    std::ptr::addr_of!(nixe_direct_stub_read_2_after).addr(),
                    DataAccessKind::Read,
                    2,
                ),
                memory_stub_region(
                    function_address(nixe_direct_stub_read_4),
                    std::ptr::addr_of!(nixe_direct_stub_read_4_end).addr(),
                    std::ptr::addr_of!(nixe_direct_stub_read_4_fault).addr(),
                    std::ptr::addr_of!(nixe_direct_stub_read_4_after).addr(),
                    DataAccessKind::Read,
                    4,
                ),
                memory_stub_region(
                    function_address(nixe_direct_stub_read_8),
                    std::ptr::addr_of!(nixe_direct_stub_read_8_end).addr(),
                    std::ptr::addr_of!(nixe_direct_stub_read_8_fault).addr(),
                    std::ptr::addr_of!(nixe_direct_stub_read_8_after).addr(),
                    DataAccessKind::Read,
                    8,
                ),
                memory_stub_region(
                    function_address(nixe_direct_stub_read_16),
                    std::ptr::addr_of!(nixe_direct_stub_read_16_end).addr(),
                    std::ptr::addr_of!(nixe_direct_stub_read_16_fault).addr(),
                    std::ptr::addr_of!(nixe_direct_stub_read_16_after).addr(),
                    DataAccessKind::Read,
                    16,
                ),
                memory_stub_region(
                    function_address(nixe_direct_stub_write_1),
                    std::ptr::addr_of!(nixe_direct_stub_write_1_end).addr(),
                    std::ptr::addr_of!(nixe_direct_stub_write_1_fault).addr(),
                    std::ptr::addr_of!(nixe_direct_stub_write_1_after).addr(),
                    DataAccessKind::Write,
                    1,
                ),
                memory_stub_region(
                    function_address(nixe_direct_stub_write_2),
                    std::ptr::addr_of!(nixe_direct_stub_write_2_end).addr(),
                    std::ptr::addr_of!(nixe_direct_stub_write_2_fault).addr(),
                    std::ptr::addr_of!(nixe_direct_stub_write_2_after).addr(),
                    DataAccessKind::Write,
                    2,
                ),
                memory_stub_region(
                    function_address(nixe_direct_stub_write_4),
                    std::ptr::addr_of!(nixe_direct_stub_write_4_end).addr(),
                    std::ptr::addr_of!(nixe_direct_stub_write_4_fault).addr(),
                    std::ptr::addr_of!(nixe_direct_stub_write_4_after).addr(),
                    DataAccessKind::Write,
                    4,
                ),
                memory_stub_region(
                    function_address(nixe_direct_stub_write_8),
                    std::ptr::addr_of!(nixe_direct_stub_write_8_end).addr(),
                    std::ptr::addr_of!(nixe_direct_stub_write_8_fault).addr(),
                    std::ptr::addr_of!(nixe_direct_stub_write_8_after).addr(),
                    DataAccessKind::Write,
                    8,
                ),
                memory_stub_region(
                    function_address(nixe_direct_stub_write_16),
                    std::ptr::addr_of!(nixe_direct_stub_write_16_end).addr(),
                    std::ptr::addr_of!(nixe_direct_stub_write_16_fault).addr(),
                    std::ptr::addr_of!(nixe_direct_stub_write_16_after).addr(),
                    DataAccessKind::Write,
                    16,
                ),
            ])
        })
        .as_ref()
        .map_err(Clone::clone)
}

fn memory_stub_region(
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
        sites: Box::from([NativeFaultSite {
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
                element_index: 0,
            },
        }]),
    }
}

unsafe extern "C" {
    fn nixe_direct_stub_read_1(call: *mut libc::c_void);
    fn nixe_direct_stub_read_2(call: *mut libc::c_void);
    fn nixe_direct_stub_read_4(call: *mut libc::c_void);
    fn nixe_direct_stub_read_8(call: *mut libc::c_void);
    fn nixe_direct_stub_read_16(call: *mut libc::c_void);
    fn nixe_direct_stub_write_1(call: *mut libc::c_void);
    fn nixe_direct_stub_write_2(call: *mut libc::c_void);
    fn nixe_direct_stub_write_4(call: *mut libc::c_void);
    fn nixe_direct_stub_write_8(call: *mut libc::c_void);
    fn nixe_direct_stub_write_16(call: *mut libc::c_void);
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
    static nixe_direct_stub_read_16_fault: u8;
    static nixe_direct_stub_read_16_after: u8;
    static nixe_direct_stub_read_16_end: u8;
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
    static nixe_direct_stub_write_16_fault: u8;
    static nixe_direct_stub_write_16_after: u8;
    static nixe_direct_stub_write_16_end: u8;
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
    fn nixe_direct_stub_invoke(
        slot: *mut libc::c_void,
        context: *mut libc::c_void,
        entry: usize,
    ) -> u32;
    fn nixe_direct_fault_landing_pad();
    fn nixe_direct_escape_now(pc: usize, sp: usize) -> !;
    #[cfg(target_arch = "x86_64")]
    fn nixe_direct_retry_trampoline();
    #[cfg(target_arch = "aarch64")]
    fn nixe_direct_retry_signal_frame(frame: usize) -> !;
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
    // Lock order for every production dispatcher is deliberately one-way:
    // immutable captured/site data -> memory mapping snapshot -> backing/page
    // transition -> mapping revalidation/protection publication. The signal
    // handler owns none of those locks, and fault-time resolution never tries
    // to acquire the exclusive execution gate while its caller holds a shared
    // native-execution lease.
    let slot = unsafe { &*slot };
    // Epoch-owned mappings are monotonic during a retry. Repeating the same
    // native PC and failing byte after a claimed repair is fatal BEFORE asking
    // policy to repair again. Fixed stubs retain their existing retry bound.
    if slot.registry.load(Ordering::Relaxed).is_null()
        && slot.retry_count.load(Ordering::Relaxed) != 0
        && slot.retry_pc.load(Ordering::Relaxed) == slot.native_pc.load(Ordering::Relaxed)
        && slot.retry_address.load(Ordering::Relaxed) == slot.fault_address.load(Ordering::Relaxed)
    {
        fatal::terminate(slot, fatal::Reason::RetryWithoutProgress);
    }
    let dispatcher = slot.dispatcher.load(Ordering::Acquire);
    if dispatcher == 0 {
        fatal::terminate(slot, fatal::Reason::MissingDispatcher);
    }
    let dispatcher = unsafe { std::mem::transmute::<usize, FaultDispatcher>(dispatcher) };
    let mut fault = CapturedFault {
        slot: unsafe { NonNull::new_unchecked(slot as *const FaultSlot as *mut FaultSlot) },
        site: unsafe { slot.site.load(Ordering::Acquire).as_ref() },
        lifetime: std::marker::PhantomData,
    };
    let disposition = unsafe { dispatcher(slot.opaque.load(Ordering::Relaxed), &mut fault) };
    match disposition {
        FaultDisposition::Retry => {
            let native_pc = slot.native_pc.load(Ordering::Relaxed);
            let fault_address = slot.fault_address.load(Ordering::Relaxed);
            let repeated = slot.retry_pc.load(Ordering::Relaxed) == native_pc
                && slot.retry_address.load(Ordering::Relaxed) == fault_address;
            let attempts = if repeated {
                slot.retry_count.fetch_add(1, Ordering::Relaxed) + 1
            } else {
                slot.retry_pc.store(native_pc, Ordering::Relaxed);
                slot.retry_address.store(fault_address, Ordering::Relaxed);
                slot.retry_count.store(1, Ordering::Relaxed);
                1
            };
            if attempts > MAX_UNCHANGED_RETRIES {
                fatal::terminate(slot, fatal::Reason::RetryLimit);
            }
            #[cfg(target_arch = "x86_64")]
            {
                let context = unsafe { &mut *(*slot.context.get()).as_mut_ptr() };
                prepare_retry_context(context, slot);
                slot.dispatching.store(false, Ordering::Release);
            }
            #[cfg(target_arch = "aarch64")]
            {
                let signal_frame = slot.signal_frame.load(Ordering::Relaxed);
                let signal_context = slot.signal_context.load(Ordering::Relaxed);
                if signal_frame == 0 || signal_context == 0 {
                    fatal::terminate(slot, fatal::Reason::MissingSignalFrame);
                }
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        (*slot.context.get()).as_ptr(),
                        signal_context as *mut libc::ucontext_t,
                        1,
                    );
                }
                slot.dispatching.store(false, Ordering::Release);
                unsafe { nixe_direct_retry_signal_frame(signal_frame) }
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
            fatal::terminate(slot, fatal::Reason::DispatcherRejected);
        }
        FaultDisposition::FatalUnattributed => {
            fatal::terminate(slot, fatal::Reason::UnattributedPc);
        }
        FaultDisposition::FatalPanic => {
            fatal::terminate(slot, fatal::Reason::DispatcherPanicked);
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        let context = unsafe { (*slot.context.get()).as_ptr() };
        if unsafe { libc::setcontext(context) } != 0 {
            fatal::terminate(slot, fatal::Reason::ContextRestoreFailed);
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
    let fault_address = unsafe { (*info).si_addr().addr() };
    let native_pc = unsafe { context_pc(&*context.cast::<libc::ucontext_t>()) };
    if slot
        .dispatching
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        fatal::report(signal, native_pc, fault_address, fatal::Reason::NestedFault);
        unsafe { chain_or_reraise(signal, info, context) };
    }
    let context = context.cast::<libc::ucontext_t>();
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
    if !registry.is_null() && site.is_none() {
        slot.dispatching.store(false, Ordering::Release);
        unsafe { chain_or_reraise(signal, info, context.cast()) };
    }
    unsafe {
        copy_signal_bytes(
            context.cast(),
            (*slot.context.get()).as_mut_ptr().cast(),
            size_of::<libc::ucontext_t>(),
        );
    }
    #[cfg(target_arch = "aarch64")]
    {
        // Linux places siginfo first in the AArch64 rt_sigframe and passes a
        // pointer to its embedded ucontext separately. Retaining both raw
        // addresses lets the cold dispatcher restore that exact kernel frame.
        slot.signal_frame.store(info.addr(), Ordering::Relaxed);
        slot.signal_context.store(context.addr(), Ordering::Relaxed);
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let source = (*context).uc_mcontext.fpregs;
        let Some((fpstate_size, xstate_features)) = x86_fpstate(source) else {
            fatal::report(
                signal,
                native_pc,
                fault_address,
                fatal::Reason::UnsupportedFpState,
            );
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
        resume.rax = (*context).uc_mcontext.gregs[libc::REG_RAX as usize] as usize;
        resume.rbx = (*context).uc_mcontext.gregs[libc::REG_RBX as usize] as usize;
        resume.rcx = (*context).uc_mcontext.gregs[libc::REG_RCX as usize] as usize;
        resume.rdx = (*context).uc_mcontext.gregs[libc::REG_RDX as usize] as usize;
        resume.rbp = (*context).uc_mcontext.gregs[libc::REG_RBP as usize] as usize;
        resume.rsp = (*context).uc_mcontext.gregs[libc::REG_RSP as usize] as usize;
        resume.rsi = (*context).uc_mcontext.gregs[libc::REG_RSI as usize] as usize;
        resume.rdi = (*context).uc_mcontext.gregs[libc::REG_RDI as usize] as usize;
        resume.r8 = (*context).uc_mcontext.gregs[libc::REG_R8 as usize] as usize;
        resume.r9 = (*context).uc_mcontext.gregs[libc::REG_R9 as usize] as usize;
        resume.r10 = (*context).uc_mcontext.gregs[libc::REG_R10 as usize] as usize;
        resume.r11 = (*context).uc_mcontext.gregs[libc::REG_R11 as usize] as usize;
        resume.r12 = (*context).uc_mcontext.gregs[libc::REG_R12 as usize] as usize;
        resume.r13 = (*context).uc_mcontext.gregs[libc::REG_R13 as usize] as usize;
        resume.r14 = (*context).uc_mcontext.gregs[libc::REG_R14 as usize] as usize;
        resume.r15 = (*context).uc_mcontext.gregs[libc::REG_R15 as usize] as usize;
        resume.pc = (*context).uc_mcontext.gregs[libc::REG_RIP as usize] as usize;
        resume.rflags = (*context).uc_mcontext.gregs[libc::REG_EFL as usize] as usize;
        resume.fpstate = (*slot.fpstate.get()).0.as_ptr().addr();
        resume.xstate_features = xstate_features;
    }
    slot.signal.store(signal, Ordering::Relaxed);
    slot.fault_address.store(fault_address, Ordering::Relaxed);
    slot.native_pc.store(native_pc, Ordering::Relaxed);
    slot.site.store(
        site.map_or(std::ptr::null_mut(), |site| {
            std::ptr::from_ref(site).cast_mut()
        }),
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
unsafe fn x86_fpstate(source: *mut libc::_libc_fpstate) -> Option<(usize, usize)> {
    if source.is_null() {
        return None;
    }
    let bytes = source.cast::<u8>();
    let magic =
        unsafe { std::ptr::read_unaligned(bytes.add(FP_XSTATE_SW_BYTES_OFFSET).cast::<u32>()) };
    let (size, xstate_features) = if magic == FP_XSTATE_MAGIC1 {
        unsafe {
            (
                std::ptr::read_unaligned(bytes.add(FP_XSTATE_SW_BYTES_OFFSET + 4).cast::<u32>())
                    as usize,
                std::ptr::read_unaligned(bytes.add(FP_XSTATE_SW_BYTES_OFFSET + 8).cast::<u64>())
                    as usize,
            )
        }
    } else {
        (size_of::<libc::_libc_fpstate>(), 0)
    };
    (size >= size_of::<libc::_libc_fpstate>() && size <= MAX_X86_FPSTATE_SIZE)
        .then_some((size, xstate_features))
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

    .globl nixe_direct_stub_invoke
    .type nixe_direct_stub_invoke,@function
nixe_direct_stub_invoke:
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
    lea rdx,[rip+.Ldirect_stub_escape]
    call nixe_direct_prepare_escape
    mov rdi,r12
    call r13
    xor eax,eax
    jmp .Ldirect_stub_return
.Ldirect_stub_escape:
    mov eax,1
.Ldirect_stub_return:
    add rsp,8
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
    .size nixe_direct_stub_invoke,.-nixe_direct_stub_invoke

    .globl nixe_direct_fault_landing_pad
    .type nixe_direct_fault_landing_pad,@function
nixe_direct_fault_landing_pad:
    cmp qword ptr [rdi+{registry}],0
    jne 1f
    ldmxcsr [rdi+{dispatcher_fp_control}]
1:
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
    mov rsi,[rdi+{fpstate}]
    mov rax,[rdi+{xstate_features}]
    test rax,rax
    jz 1f
    mov rdx,rax
    shr rdx,32
    xrstor64 [rsi]
    jmp 2f
1:
    fxrstor64 [rsi]
2:
    push qword ptr [rdi+{rflags}]
    popfq
    mov rsp,[rdi+{rsp}]
    push qword ptr [rdi+{pc}]
    mov rax,[rdi+{rax}]
    mov rbx,[rdi+{rbx}]
    mov rcx,[rdi+{rcx}]
    mov rdx,[rdi+{rdx}]
    mov rbp,[rdi+{rbp}]
    mov rsi,[rdi+{rsi}]
    mov r8,[rdi+{r8}]
    mov r9,[rdi+{r9}]
    mov r10,[rdi+{r10}]
    mov r11,[rdi+{r11}]
    mov r12,[rdi+{r12}]
    mov r13,[rdi+{r13}]
    mov r14,[rdi+{r14}]
    mov r15,[rdi+{r15}]
    mov rdi,[rdi+{rdi}]
    ret
    .size nixe_direct_retry_trampoline,.-nixe_direct_retry_trampoline
"#,
    rax = const offset_of!(ResumeRecord, rax),
    rbx = const offset_of!(ResumeRecord, rbx),
    rcx = const offset_of!(ResumeRecord, rcx),
    rdx = const offset_of!(ResumeRecord, rdx),
    rbp = const offset_of!(ResumeRecord, rbp),
    rsp = const offset_of!(ResumeRecord, rsp),
    rsi = const offset_of!(ResumeRecord, rsi),
    rdi = const offset_of!(ResumeRecord, rdi),
    r8 = const offset_of!(ResumeRecord, r8),
    r9 = const offset_of!(ResumeRecord, r9),
    r10 = const offset_of!(ResumeRecord, r10),
    r11 = const offset_of!(ResumeRecord, r11),
    r12 = const offset_of!(ResumeRecord, r12),
    r13 = const offset_of!(ResumeRecord, r13),
    r14 = const offset_of!(ResumeRecord, r14),
    r15 = const offset_of!(ResumeRecord, r15),
    pc = const offset_of!(ResumeRecord, pc),
    rflags = const offset_of!(ResumeRecord, rflags),
    fpstate = const offset_of!(ResumeRecord, fpstate),
    xstate_features = const offset_of!(ResumeRecord, xstate_features),
    registry = const offset_of!(FaultSlot, registry),
    dispatcher_fp_control = const offset_of!(FaultSlot, dispatcher_fp_control),
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
    .if \name == 16
    movdqu xmm0,xmmword ptr [rax]
    .elseif \name == 1
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
    .if \name == 16
    movdqu xmmword ptr [rdi+{output}],xmm0
    .else
    mov [rdi+{output}],rax
    .endif
    ret
    .globl nixe_direct_stub_read_\name\()_end
nixe_direct_stub_read_\name\()_end:
    .size nixe_direct_stub_read_\name,.-nixe_direct_stub_read_\name
    .endm

    DIRECT_READ 1
    DIRECT_READ 2
    DIRECT_READ 4
    DIRECT_READ 8
    DIRECT_READ 16

    .macro DIRECT_WRITE name
    .globl nixe_direct_stub_write_\name
    .type nixe_direct_stub_write_\name,@function
nixe_direct_stub_write_\name:
    mov rax,[rdi]
    .if \name == 16
    movdqu xmm0,xmmword ptr [rdi+{value}]
    .else
    mov rdx,[rdi+{value}]
    .endif
    .globl nixe_direct_stub_write_\name\()_fault
nixe_direct_stub_write_\name\()_fault:
    .if \name == 16
    movdqu xmmword ptr [rax],xmm0
    .elseif \name == 1
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
    DIRECT_WRITE 16
"#,
    value = const offset_of!(StubCall, value),
    output = const offset_of!(StubCall, output),
);

#[cfg(all(test, target_arch = "x86_64"))]
core::arch::global_asm!(
    r#"
    .text
    .globl nixe_x86_retry_probe
    .type nixe_x86_retry_probe,@function
nixe_x86_retry_probe:
    push rbp
    push rbx
    push r12
    push r13
    push r14
    push r15
    mov rbx,rdi
    mov qword ptr [rbx+8],0
    mov rbp,0x2122232425262728
    mov r12,0x3132333435363738
    mov r13,0x4142434445464748
    mov r14,0x5152535455565758
    mov r15,0x6162636465666768
    mov rcx,0x1112131415161718
    mov rdx,0x7172737475767778
    mov rsi,0x8182838485868788
    mov r8,0x9192939495969798
    mov r9,0xa1a2a3a4a5a6a7a8
    mov r10,0xb1b2b3b4b5b6b7b8
    mov r11,0xc1c2c3c4c5c6c7c8
    vpcmpeqd ymm0,ymm0,ymm0
    cmp rbx,rbx
    stc
    mov rax,[rbx]
    .globl nixe_x86_retry_probe_fault
nixe_x86_retry_probe_fault:
    mov al,byte ptr [rax]
    pushfq
    pop rax
    and eax,0x41
    cmp eax,0x41
    jne 9f
    mov rax,0x1112131415161718
    cmp rcx,rax
    jne 9f
    mov rax,0x7172737475767778
    cmp rdx,rax
    jne 9f
    mov rax,0x8182838485868788
    cmp rsi,rax
    jne 9f
    cmp rdi,rbx
    jne 9f
    mov rax,0x9192939495969798
    cmp r8,rax
    jne 9f
    mov rax,0xa1a2a3a4a5a6a7a8
    cmp r9,rax
    jne 9f
    mov rax,0xb1b2b3b4b5b6b7b8
    cmp r10,rax
    jne 9f
    mov rax,0xc1c2c3c4c5c6c7c8
    cmp r11,rax
    jne 9f
    mov rax,0x2122232425262728
    cmp rbp,rax
    jne 9f
    mov rax,0x3132333435363738
    cmp r12,rax
    jne 9f
    mov rax,0x4142434445464748
    cmp r13,rax
    jne 9f
    mov rax,0x5152535455565758
    cmp r14,rax
    jne 9f
    mov rax,0x6162636465666768
    cmp r15,rax
    jne 9f
    vpmovmskb eax,ymm0
    cmp eax,-1
    jne 9f
    mov qword ptr [rbx+8],1
9:
    vzeroupper
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
    .globl nixe_x86_retry_probe_end
nixe_x86_retry_probe_end:
    .size nixe_x86_retry_probe,.-nixe_x86_retry_probe

    .globl nixe_x86_retry_store_probe
    .type nixe_x86_retry_store_probe,@function
nixe_x86_retry_store_probe:
    mov rax,[rdi]
    mov rcx,1
    .globl nixe_x86_retry_store_probe_fault
nixe_x86_retry_store_probe_fault:
    lock xadd qword ptr [rax],rcx
    ret
    .globl nixe_x86_retry_store_probe_end
nixe_x86_retry_store_probe_end:
    .size nixe_x86_retry_store_probe,.-nixe_x86_retry_store_probe
"#
);

#[cfg(all(test, target_arch = "aarch64"))]
core::arch::global_asm!(
    r#"
    .text
    .globl nixe_aarch64_retry_probe
    .type nixe_aarch64_retry_probe,%function
nixe_aarch64_retry_probe:
    stp x29,x30,[sp,#-16]!
    stp x19,x20,[sp,#-16]!
    stp x21,x22,[sp,#-16]!
    stp x23,x24,[sp,#-16]!
    stp x25,x26,[sp,#-16]!
    stp x27,x28,[sp,#-16]!
    stp q8,q9,[sp,#-32]!
    stp q10,q11,[sp,#-32]!
    stp q12,q13,[sp,#-32]!
    stp q14,q15,[sp,#-32]!
    mov x19,x0
    mrs x0,fpcr
    mrs x1,fpsr
    stp x0,x1,[sp,#-16]!
    ldr x20,[x19]
    str xzr,[x19,#8]
    mov x1,#1
    mov x2,#2
    mov x3,#3
    mov x4,#4
    mov x5,#5
    mov x6,#6
    mov x7,#7
    mov x8,#8
    mov x9,#9
    mov x10,#10
    mov x11,#11
    mov x12,#12
    mov x13,#13
    mov x14,#14
    mov x15,#15
    mov x16,#16
    mov x17,#17
    mov x18,#18
    mov x21,#21
    mov x22,#22
    mov x23,#23
    mov x24,#24
    mov x25,#25
    mov x26,#26
    mov x27,#27
    mov x28,#28
    movi v0.16b,#0x5a
    movi v8.16b,#0x87
    movi v31.16b,#0xa5
    mov x0,#0x400000
    msr fpcr,x0
    mov x0,#0x81
    msr fpsr,x0
    cmp xzr,xzr
    .globl nixe_aarch64_retry_probe_fault
nixe_aarch64_retry_probe_fault:
    ldrb w0,[x20]
    mrs x0,nzcv
    lsr x0,x0,#28
    cmp x0,#6
    b.ne 9f
    cmp x1,#1
    b.ne 9f
    cmp x2,#2
    b.ne 9f
    cmp x3,#3
    b.ne 9f
    cmp x4,#4
    b.ne 9f
    cmp x5,#5
    b.ne 9f
    cmp x6,#6
    b.ne 9f
    cmp x7,#7
    b.ne 9f
    cmp x8,#8
    b.ne 9f
    cmp x9,#9
    b.ne 9f
    cmp x10,#10
    b.ne 9f
    cmp x11,#11
    b.ne 9f
    cmp x12,#12
    b.ne 9f
    cmp x13,#13
    b.ne 9f
    cmp x14,#14
    b.ne 9f
    cmp x15,#15
    b.ne 9f
    cmp x16,#16
    b.ne 9f
    cmp x17,#17
    b.ne 9f
    cmp x18,#18
    b.ne 9f
    cmp x21,#21
    b.ne 9f
    cmp x22,#22
    b.ne 9f
    cmp x23,#23
    b.ne 9f
    cmp x24,#24
    b.ne 9f
    cmp x25,#25
    b.ne 9f
    cmp x26,#26
    b.ne 9f
    cmp x27,#27
    b.ne 9f
    cmp x28,#28
    b.ne 9f
    umov w0,v0.b[0]
    cmp w0,#0x5a
    b.ne 9f
    umov w0,v8.b[15]
    cmp w0,#0x87
    b.ne 9f
    umov w0,v31.b[7]
    cmp w0,#0xa5
    b.ne 9f
    mrs x0,fpcr
    mov x1,#0x400000
    cmp x0,x1
    b.ne 9f
    mrs x0,fpsr
    cmp x0,#0x81
    b.ne 9f
    mov x0,#1
    str x0,[x19,#8]
9:
    ldp x0,x1,[sp],#16
    msr fpcr,x0
    msr fpsr,x1
    ldp q14,q15,[sp],#32
    ldp q12,q13,[sp],#32
    ldp q10,q11,[sp],#32
    ldp q8,q9,[sp],#32
    ldp x27,x28,[sp],#16
    ldp x25,x26,[sp],#16
    ldp x23,x24,[sp],#16
    ldp x21,x22,[sp],#16
    ldp x19,x20,[sp],#16
    ldp x29,x30,[sp],#16
    ret
    .globl nixe_aarch64_retry_probe_end
nixe_aarch64_retry_probe_end:
    .size nixe_aarch64_retry_probe,.-nixe_aarch64_retry_probe

    .globl nixe_aarch64_retry_store_probe
    .type nixe_aarch64_retry_store_probe,%function
nixe_aarch64_retry_store_probe:
    ldr x1,[x0]
    mov x2,#0x1357
    .globl nixe_aarch64_retry_store_probe_fault
nixe_aarch64_retry_store_probe_fault:
    str x2,[x1]
    ret
    .globl nixe_aarch64_retry_store_probe_end
nixe_aarch64_retry_store_probe_end:
    .size nixe_aarch64_retry_store_probe,.-nixe_aarch64_retry_store_probe
"#
);

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;
    use std::os::unix::process::ExitStatusExt;
    use std::process::Command;
    use std::sync::Arc;
    use std::sync::Barrier;

    use nixe_cpu::memory::{ExecutionMemory, MemoryPermissions};
    use nixe_memory::{
        CanonicalBackingPage, CanonicalBackingStore, CanonicalRangeTranslator, ContentGeneration,
        CpuVisibilityRequest, DeviceAccessDeclaration, DeviceVisibilityPoint,
        DeviceVisibilityRequest, DirectArena, DirectBackendPolicy, DirectMapRequest,
        DirectProtectRequest, DirectProtection, GuestPhysicalPageId, NonCpuDeviceId,
        VisibilityCoordinator, VisibilityCoordinatorError, VisibilityState,
    };

    struct DeviceWriteback {
        bytes: Box<[u8]>,
    }

    impl VisibilityCoordinator for DeviceWriteback {
        fn make_device_visible(
            &self,
            _request: DeviceVisibilityRequest,
            _canonical_bytes: &[u8],
        ) -> Result<(), VisibilityCoordinatorError> {
            Ok(())
        }

        fn make_cpu_visible(
            &self,
            _request: CpuVisibilityRequest,
        ) -> Result<Box<[u8]>, VisibilityCoordinatorError> {
            Ok(self.bytes.clone())
        }
    }

    #[repr(C)]
    struct SyntheticContext {
        address: *const u8,
        observed: u8,
    }

    #[cfg(target_arch = "x86_64")]
    #[repr(C)]
    struct X86RetryContext {
        address: *const u8,
        preserved: u64,
    }

    #[cfg(target_arch = "aarch64")]
    #[repr(C)]
    struct Aarch64RetryContext {
        address: *const u8,
        preserved: u64,
    }

    #[cfg(target_arch = "x86_64")]
    unsafe extern "C" {
        fn nixe_x86_retry_probe(context: *mut libc::c_void);
        static nixe_x86_retry_probe_fault: u8;
        static nixe_x86_retry_probe_end: u8;
        fn nixe_x86_retry_store_probe(context: *mut libc::c_void);
        static nixe_x86_retry_store_probe_fault: u8;
        static nixe_x86_retry_store_probe_end: u8;
    }

    #[cfg(target_arch = "aarch64")]
    unsafe extern "C" {
        fn nixe_aarch64_retry_probe(context: *mut libc::c_void);
        static nixe_aarch64_retry_probe_fault: u8;
        static nixe_aarch64_retry_probe_end: u8;
        fn nixe_aarch64_retry_store_probe(context: *mut libc::c_void);
        static nixe_aarch64_retry_store_probe_fault: u8;
        static nixe_aarch64_retry_store_probe_end: u8;
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

    unsafe extern "C" fn retry_write(
        opaque: *mut libc::c_void,
        _fault: *mut CapturedFault,
    ) -> FaultDisposition {
        let arena = unsafe { &*opaque.cast::<DirectArena>() };
        arena
            .protect_ranges(&[DirectProtectRequest {
                guest_address: 0x1000,
                size: 4096,
                protection: DirectProtection::ReadWrite,
            }])
            .unwrap();
        FaultDisposition::Retry
    }

    struct ConcurrentRetry {
        arena: Arc<DirectArena>,
        faults: Barrier,
    }

    unsafe extern "C" fn retry_concurrently(
        opaque: *mut libc::c_void,
        _fault: *mut CapturedFault,
    ) -> FaultDisposition {
        let retry = unsafe { &*opaque.cast::<ConcurrentRetry>() };
        retry.faults.wait();
        retry
            .arena
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

    unsafe extern "C" fn retry_without_progress(
        _opaque: *mut libc::c_void,
        _fault: *mut CapturedFault,
    ) -> FaultDisposition {
        FaultDisposition::Retry
    }

    unsafe extern "C" fn captured_retry_without_progress(
        opaque: *mut libc::c_void,
        _fault: *mut CapturedFault,
    ) -> FaultDisposition {
        let calls = unsafe { &mut *opaque.cast::<usize>() };
        if *calls != 0 {
            // Distinguish an incorrect second policy invocation from the
            // runtime's required SIGSEGV failure before invoking policy again.
            unsafe { libc::_exit(78) };
        }
        *calls += 1;
        FaultDisposition::Retry
    }

    unsafe extern "C" fn reject_fault(
        _: *mut libc::c_void,
        _: *mut CapturedFault,
    ) -> FaultDisposition {
        FaultDisposition::Fatal
    }

    unsafe extern "C" fn reject_unattributed(
        _: *mut libc::c_void,
        _: *mut CapturedFault,
    ) -> FaultDisposition {
        FaultDisposition::FatalUnattributed
    }

    unsafe extern "C" fn panicking_dispatcher(
        _: *mut libc::c_void,
        _: *mut CapturedFault,
    ) -> FaultDisposition {
        std::panic::catch_unwind(|| panic!("test dispatcher panic"))
            .unwrap_or(FaultDisposition::FatalPanic)
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
                element_index: 0,
            },
        };
        let registry = NativeFaultRegistry::new(vec![NativeFaultRegion {
            native_start: start,
            native_end: start + 256,
            sites: Box::from([site]),
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
        let outcome = unsafe {
            worker
                .begin_batch(
                    view,
                    &registry,
                    retry,
                    std::ptr::from_ref(&arena).cast_mut().cast(),
                )
                .unwrap();
            worker.invoke_active(NativeInvocation {
                gateway,
                context: std::ptr::from_mut(&mut context).cast(),
                entry: faulting_read as *const () as usize,
            })
        }
        .unwrap();
        assert_eq!(outcome, InvocationOutcome::Returned);
        assert_eq!(context.observed, 0x5a);
    }

    #[test]
    fn simultaneous_workers_retry_their_native_load_after_one_page_transition() {
        let (arena, registry) = fixture();
        let arena = Arc::new(arena);
        let view = arena.view();
        let retry = Arc::new(ConcurrentRetry {
            arena: Arc::clone(&arena),
            faults: Barrier::new(3),
        });
        let workers = (0..2)
            .map(|_| {
                let registry = Arc::clone(&registry);
                let retry = Arc::clone(&retry);
                std::thread::spawn(move || {
                    let mut context = SyntheticContext {
                        address: (view.base + 0x1000) as *const u8,
                        observed: 0,
                    };
                    let mut worker = WorkerFaultContext::register().unwrap();
                    let outcome = unsafe {
                        worker
                            .begin_batch(
                                view,
                                &registry,
                                retry_concurrently,
                                Arc::as_ptr(&retry).cast_mut().cast(),
                            )
                            .unwrap();
                        worker.invoke_active(NativeInvocation {
                            gateway,
                            context: std::ptr::from_mut(&mut context).cast(),
                            entry: faulting_read as *const () as usize,
                        })
                    }
                    .unwrap();
                    (outcome, context.observed)
                })
            })
            .collect::<Vec<_>>();

        retry.faults.wait();
        for worker in workers {
            let (outcome, observed) = worker.join().unwrap();
            assert_eq!(outcome, InvocationOutcome::Returned);
            assert_eq!(observed, 0x5a);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn x86_retry_restores_gprs_flags_and_avx_state() {
        if !std::arch::is_x86_feature_detected!("avx") {
            return;
        }
        let (arena, _) = fixture();
        let view = arena.view();
        let start = nixe_x86_retry_probe as *const () as usize;
        let registry = Arc::new(
            NativeFaultRegistry::new(vec![NativeFaultRegion {
                native_start: start,
                native_end: std::ptr::addr_of!(nixe_x86_retry_probe_end).addr(),
                sites: Box::from([NativeFaultSite {
                    native_start: std::ptr::addr_of!(nixe_x86_retry_probe_fault).addr(),
                    native_end: std::ptr::addr_of!(nixe_x86_retry_probe_fault).addr() + 1,
                    access: NativeMemoryAccess {
                        address_space: AddressSpaceId::new(1),
                        guest_pc: GuestVirtualAddress::new(0x8000),
                        kind: NativeMemoryAccessKind::Read,
                        size: 1,
                        element_index: 0,
                    },
                }]),
            }])
            .unwrap(),
        );
        let mut context = X86RetryContext {
            address: (view.base + 0x1000) as *const u8,
            preserved: 0,
        };
        let mut worker = WorkerFaultContext::register().unwrap();
        let outcome = unsafe {
            worker
                .begin_batch(
                    view,
                    &registry,
                    retry,
                    std::ptr::from_ref(&arena).cast_mut().cast(),
                )
                .unwrap();
            worker.invoke_active(NativeInvocation {
                gateway,
                context: std::ptr::from_mut(&mut context).cast(),
                entry: start,
            })
        }
        .unwrap();

        assert_eq!(outcome, InvocationOutcome::Returned);
        assert_eq!(context.preserved, 1);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn x86_retry_executes_the_faulting_store_once() {
        let (arena, _) = fixture();
        let view = arena.view();
        let start = nixe_x86_retry_store_probe as *const () as usize;
        let registry = Arc::new(
            NativeFaultRegistry::new(vec![NativeFaultRegion {
                native_start: start,
                native_end: std::ptr::addr_of!(nixe_x86_retry_store_probe_end).addr(),
                sites: Box::from([NativeFaultSite {
                    native_start: std::ptr::addr_of!(nixe_x86_retry_store_probe_fault).addr(),
                    native_end: std::ptr::addr_of!(nixe_x86_retry_store_probe_fault).addr() + 1,
                    access: NativeMemoryAccess {
                        address_space: AddressSpaceId::new(1),
                        guest_pc: GuestVirtualAddress::new(0x8000),
                        kind: NativeMemoryAccessKind::Write,
                        size: 8,
                        element_index: 0,
                    },
                }]),
            }])
            .unwrap(),
        );
        let address = view.base + 0x1000;
        let mut context = X86RetryContext {
            address: address as *const u8,
            preserved: 0,
        };
        let mut worker = WorkerFaultContext::register().unwrap();
        let outcome = unsafe {
            worker
                .begin_batch(
                    view,
                    &registry,
                    retry_write,
                    std::ptr::from_ref(&arena).cast_mut().cast(),
                )
                .unwrap();
            worker.invoke_active(NativeInvocation {
                gateway,
                context: std::ptr::from_mut(&mut context).cast(),
                entry: start,
            })
        }
        .unwrap();

        assert_eq!(outcome, InvocationOutcome::Returned);
        assert_eq!(
            unsafe { (address as *const u64).read_volatile() },
            u64::from_le_bytes([0x5a; 8]) + 1,
        );
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn aarch64_retry_restores_gprs_nzcv_fp_and_simd_state() {
        let (arena, _) = fixture();
        let view = arena.view();
        let start = nixe_aarch64_retry_probe as *const () as usize;
        let registry = Arc::new(
            NativeFaultRegistry::new(vec![NativeFaultRegion {
                native_start: start,
                native_end: std::ptr::addr_of!(nixe_aarch64_retry_probe_end).addr(),
                sites: Box::from([NativeFaultSite {
                    native_start: std::ptr::addr_of!(nixe_aarch64_retry_probe_fault).addr(),
                    native_end: std::ptr::addr_of!(nixe_aarch64_retry_probe_fault).addr() + 4,
                    access: NativeMemoryAccess {
                        address_space: AddressSpaceId::new(1),
                        guest_pc: GuestVirtualAddress::new(0x8000),
                        kind: NativeMemoryAccessKind::Read,
                        size: 1,
                        element_index: 0,
                    },
                }]),
            }])
            .unwrap(),
        );
        let mut context = Aarch64RetryContext {
            address: (view.base + 0x1000) as *const u8,
            preserved: 0,
        };
        let mut worker = WorkerFaultContext::register().unwrap();
        let outcome = unsafe {
            worker
                .begin_batch(
                    view,
                    &registry,
                    retry,
                    std::ptr::from_ref(&arena).cast_mut().cast(),
                )
                .unwrap();
            worker.invoke_active(NativeInvocation {
                gateway,
                context: std::ptr::from_mut(&mut context).cast(),
                entry: start,
            })
        }
        .unwrap();

        assert_eq!(outcome, InvocationOutcome::Returned);
        assert_eq!(context.preserved, 1);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn aarch64_retry_executes_the_faulting_store_in_place() {
        let (arena, _) = fixture();
        let view = arena.view();
        let start = nixe_aarch64_retry_store_probe as *const () as usize;
        let registry = Arc::new(
            NativeFaultRegistry::new(vec![NativeFaultRegion {
                native_start: start,
                native_end: std::ptr::addr_of!(nixe_aarch64_retry_store_probe_end).addr(),
                sites: Box::from([NativeFaultSite {
                    native_start: std::ptr::addr_of!(nixe_aarch64_retry_store_probe_fault).addr(),
                    native_end: std::ptr::addr_of!(nixe_aarch64_retry_store_probe_fault).addr() + 4,
                    access: NativeMemoryAccess {
                        address_space: AddressSpaceId::new(1),
                        guest_pc: GuestVirtualAddress::new(0x8000),
                        kind: NativeMemoryAccessKind::Write,
                        size: 8,
                        element_index: 0,
                    },
                }]),
            }])
            .unwrap(),
        );
        let address = view.base + 0x1000;
        let mut context = Aarch64RetryContext {
            address: address as *const u8,
            preserved: 0,
        };
        let mut worker = WorkerFaultContext::register().unwrap();
        let outcome = unsafe {
            worker
                .begin_batch(
                    view,
                    &registry,
                    retry_write,
                    std::ptr::from_ref(&arena).cast_mut().cast(),
                )
                .unwrap();
            worker.invoke_active(NativeInvocation {
                gateway,
                context: std::ptr::from_mut(&mut context).cast(),
                entry: start,
            })
        }
        .unwrap();

        assert_eq!(outcome, InvocationOutcome::Returned);
        assert_eq!(unsafe { (address as *const u64).read_volatile() }, 0x1357);
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
            worker
                .begin_batch(view, &registry, escape, std::ptr::null_mut())
                .unwrap();
            worker.invoke_active(NativeInvocation {
                gateway,
                context: std::ptr::from_mut(&mut context).cast(),
                entry: faulting_read as *const () as usize,
            })
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
            worker
                .begin_batch(view, &registry, escape, std::ptr::null_mut())
                .unwrap();
            worker.invoke_active(NativeInvocation {
                gateway,
                context: std::ptr::from_mut(&mut context).cast(),
                entry: faulting_read as *const () as usize,
            })
        }
        .unwrap();
        assert_eq!(outcome, InvocationOutcome::Escaped);
        drop(registry);
        {
            let fault = worker.escaped_fault().unwrap();
            assert_eq!(fault.fault_address(), target.addr());
            assert!(
                fault.site.is_none(),
                "escaped image cannot borrow the old registry"
            );
            assert!(fault.native_pc() >= faulting_read as *const () as usize);
            assert!(fault.integer(0).is_some());
            assert!(fault.integer(32).is_none());
            assert!(fault.vector(0).is_some());
            assert!(fault.vector(32).is_none());
            assert!(fault.fp().is_some());
        }
        unsafe {
            worker.begin_batch(
                view,
                memory_stub_registry().unwrap(),
                escape,
                std::ptr::null_mut(),
            )
        }
        .unwrap();
        assert!(worker.escaped_fault().is_err());
        worker.end_batch().unwrap();
        assert!(
            worker.escaped_fault().is_err(),
            "a new batch invalidates the old capture"
        );
        assert_eq!(unsafe { libc::close(fd) }, 0);
    }

    #[test]
    fn a_second_registration_cannot_replace_the_live_workers_signal_stack() {
        let mut first = NativeWorker::default();
        let mut second = NativeWorker::default();
        let tid = first.faults().unwrap().registered_tid();
        assert!(
            second
                .faults()
                .err()
                .unwrap()
                .to_string()
                .contains("already registered")
        );
        assert!(second.faults.is_none());
        assert_eq!(first.faults().unwrap().registered_tid(), tid);
        first.finish().unwrap();
        assert_eq!(second.faults().unwrap().registered_tid(), tid);
        second.finish().unwrap();
    }

    #[test]
    fn explicit_worker_unregistration_is_idempotent_and_cannot_clear_a_reused_slot() {
        let mut worker = WorkerFaultContext::register().unwrap();
        let tid = worker.registered_tid();
        worker.unregister().unwrap();
        assert_eq!(worker.registered_tid(), 0);
        assert!(worker.escaped_fault().is_err());
        let replacement = WorkerFaultContext::register().unwrap();
        worker.unregister().unwrap();
        drop(worker);
        assert_eq!(replacement.registered_tid(), tid);
        assert_eq!(
            unsafe { replacement.slot.as_ref() }
                .tid
                .load(Ordering::Acquire),
            tid
        );
    }

    #[test]
    fn unregistered_worker_cannot_touch_a_replacement_batch() {
        let (arena, _) = fixture();
        let mut worker = WorkerFaultContext::register().unwrap();
        worker.unregister().unwrap();
        let mut replacement = WorkerFaultContext::register().unwrap();
        // Reuse the original slot and leave the replacement snapshot active.
        let slot = unsafe { replacement.slot.as_ref() };
        slot.active.store(true, Ordering::Release);
        assert!(worker.end_batch().is_err());
        assert!(
            unsafe {
                worker.begin_capture(
                    arena.view(),
                    std::ptr::null_mut(),
                    None,
                    reject_fault,
                    std::ptr::null_mut(),
                )
            }
            .is_err()
        );
        assert!(slot.active.load(Ordering::Acquire));
        replacement.end_batch().unwrap();
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
    fn dropping_a_direct_slice_clears_the_snapshot_but_retains_the_os_worker() {
        let (arena, _) = fixture();
        let mut worker = NativeWorker::default();
        assert!(worker.faults.is_none());
        let mut frontend =
            unsafe { DirectMemoryFrontend::new(arena.view(), AddressSpaceId::new(7)) }.unwrap();
        let slice = frontend.begin_slice(&mut worker).unwrap();
        let tid = slice.worker.registered_tid();
        let slot = slice.worker.slot;
        assert!(unsafe { slot.as_ref() }.active.load(Ordering::Acquire));
        drop(slice);
        assert!(!unsafe { slot.as_ref() }.active.load(Ordering::Acquire));
        drop(frontend);
        assert_eq!(unsafe { slot.as_ref() }.tid.load(Ordering::Acquire), tid);
        worker.finish().unwrap();
        assert!(worker.faults.is_none());
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
    fn native_fault_registry_sorts_regions_and_rejects_overlapping_metadata() {
        let registry = NativeFaultRegistry::new(vec![
            fake_region(0x2000, 0x2100),
            fake_region(0x1000, 0x1100),
        ])
        .unwrap();
        assert!(registry.find(0x1018).is_some());
        assert!(registry.find(0x2018).is_some());
        assert!(
            NativeFaultRegistry::new(vec![
                fake_region(0x1000, 0x1100),
                fake_region(0x1080, 0x1180),
            ])
            .is_err()
        );
        let invalid = NativeFaultRegion {
            native_start: 0x2000,
            native_end: 0x2100,
            sites: Box::from([fake_site(0x2010, 0x2050), fake_site(0x2040, 0x2060)]),
        };
        assert!(NativeFaultRegistry::new(vec![invalid]).is_err());
        assert!(NativeFaultRegistry::new(vec![fake_region(0x2000, 0x2000)]).is_err());
    }

    #[test]
    fn direct_frontend_retries_the_faulting_stub_after_gpu_writeback() {
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
            .translate_canonical_range(
                space,
                address,
                DIRECT_PAGE_SIZE as u64,
                MemoryPermissions::READ,
            )
            .unwrap();
        let mut bytes = vec![0; DIRECT_PAGE_SIZE];
        bytes[9] = 0xa5;
        let coordinator: Arc<dyn VisibilityCoordinator> = Arc::new(DeviceWriteback {
            bytes: bytes.into_boxed_slice(),
        });
        let declaration = DeviceAccessDeclaration::write(
            NonCpuDeviceId::new(4),
            DeviceVisibilityPoint::new(11),
            DeviceVisibilityPoint::new(12),
        )
        .unwrap();
        range
            .prepare_device_access(declaration, Arc::clone(&coordinator))
            .unwrap();
        range
            .publish_device_write(declaration, Arc::clone(&coordinator))
            .unwrap();
        assert!(matches!(
            range.segments()[0].visibility_state(),
            VisibilityState::GpuNewer { .. }
        ));

        let mut frontend = unsafe {
            DirectMemoryFrontend::new(memory.direct_address_space_view(space).unwrap(), space)
        }
        .unwrap();
        let mut worker = NativeWorker::default();
        let mut frontend = frontend.begin_slice(&mut worker).unwrap();
        let value = frontend
            .read(
                &memory,
                GuestVirtualAddress::new(address.get() + 9),
                MemoryAccess::normal(MemoryAccessSize::Byte),
            )
            .unwrap();

        assert_eq!(value, MemoryValue::U8(0xa5));
        assert_eq!(
            range.segments()[0].visibility_state(),
            VisibilityState::Clean
        );
    }

    #[test]
    fn direct_frontend_round_trips_one_native_quadword() {
        let mut memory = ExecutionMemory::new();
        let space = AddressSpaceId::new(1);
        let address = GuestVirtualAddress::new(0x1010);
        let page = GuestPhysicalPageId::new(1);
        assert!(memory.add_ram_page(page));
        assert!(memory.map_page(
            space,
            GuestVirtualAddress::new(0x1000),
            page,
            MemoryPermissions::READ_WRITE,
        ));
        memory
            .bind_cpu_memory_backend(space, 0x4000, DirectBackendPolicy::Required)
            .unwrap();
        let mut frontend = unsafe {
            DirectMemoryFrontend::new(memory.direct_address_space_view(space).unwrap(), space)
        }
        .unwrap();
        let mut worker = NativeWorker::default();
        let mut frontend = frontend.begin_slice(&mut worker).unwrap();
        let value = MemoryValue::U128(0x0011_2233_4455_6677_8899_aabb_ccdd_eeff);

        frontend
            .write(
                &memory,
                address,
                MemoryAccess::normal(MemoryAccessSize::Quadword),
                value,
            )
            .unwrap();
        assert_eq!(
            frontend
                .read(
                    &memory,
                    address,
                    MemoryAccess::normal(MemoryAccessSize::Quadword),
                )
                .unwrap(),
            value,
        );
    }

    #[test]
    fn direct_frontend_confines_each_native_stub_to_one_page() {
        let (arena, _) = fixture();
        let frontend =
            unsafe { DirectMemoryFrontend::new(arena.view(), AddressSpaceId::new(1)) }.unwrap();
        let page_end = DIRECT_PAGE_SIZE as u64;

        assert!(
            frontend
                .direct_pointer(
                    GuestVirtualAddress::new(page_end - 1),
                    MemoryAccessSize::Byte,
                )
                .is_some()
        );
        assert!(
            frontend
                .direct_pointer(
                    GuestVirtualAddress::new(page_end - 1),
                    MemoryAccessSize::Halfword,
                )
                .is_none()
        );
        assert!(
            frontend
                .direct_pointer(
                    GuestVirtualAddress::new(page_end - 16),
                    MemoryAccessSize::Quadword,
                )
                .is_some()
        );
        assert!(
            frontend
                .direct_pointer(
                    GuestVirtualAddress::new(page_end - 15),
                    MemoryAccessSize::Quadword,
                )
                .is_none()
        );
        assert!(
            frontend
                .direct_pointer(GuestVirtualAddress::new(0x4000), MemoryAccessSize::Byte)
                .is_none()
        );
    }

    #[test]
    fn immutable_registry_attributes_only_exact_sites_across_pages_and_gaps() {
        let registry = NativeFaultRegistry::new(vec![
            NativeFaultRegion {
                native_start: 0x1000,
                native_end: 0x4000,
                sites: Box::from([
                    fake_site(0x1010, 0x1020),
                    fake_site(0x1ff8, 0x2008),
                    fake_site(0x3010, 0x3020),
                ]),
            },
            fake_region(0x4000, 0x4100),
        ])
        .unwrap();
        for (start, end) in [
            (0x1010, 0x1020),
            (0x1ff8, 0x2008),
            (0x3010, 0x3020),
            (0x4010, 0x4020),
        ] {
            for pc in [start, end - 1] {
                assert_eq!(registry.find(pc), Some(&fake_site(start, end)));
            }
            assert!(registry.find(start - 1).is_none());
            assert!(registry.find(end).is_none());
        }
        for pc in [0, 0xfff, 0x1000, 0x3fff, 0x4000, 0x4100, usize::MAX] {
            assert!(registry.find(pc).is_none());
        }
        assert!(
            NativeFaultRegistry::new(Vec::new())
                .unwrap()
                .find(0)
                .is_none()
        );
    }

    fn fake_region(start: usize, end: usize) -> NativeFaultRegion {
        NativeFaultRegion {
            native_start: start,
            native_end: end,
            sites: Box::from([fake_site(start + 0x10, start + 0x20)]),
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
                element_index: 0,
            },
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
        if case == "closed_diagnostic_pipe" {
            let mut pipe = [0; 2];
            assert_eq!(unsafe { libc::pipe(pipe.as_mut_ptr()) }, 0);
            assert_eq!(
                unsafe { libc::dup2(pipe[1], libc::STDERR_FILENO) },
                libc::STDERR_FILENO
            );
            unsafe {
                libc::close(pipe[0]);
                libc::close(pipe[1]);
                libc::signal(libc::SIGPIPE, libc::SIG_DFL);
            }
        }
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
                    sites: Box::from([]),
                }])
                .unwrap(),
            )
        } else {
            default_registry
        };
        let dispatcher = match case.as_str() {
            "nested" => nested_fault,
            "retry_livelock" => retry_without_progress,
            "captured_retry_livelock" => captured_retry_without_progress,
            "captured_unattributed" => reject_unattributed,
            "captured_panic" => panicking_dispatcher,
            "dispatcher_fatal" | "closed_diagnostic_pipe" => reject_fault,
            _ => escape,
        };
        let mut worker = WorkerFaultContext::register().unwrap();
        if case.starts_with("captured_") {
            let mut calls = 0_usize;
            #[cfg(target_arch = "x86_64")]
            let caller_fp = {
                let mut mxcsr = 0_u32;
                unsafe {
                    core::arch::asm!("stmxcsr [{}]", in(reg) &mut mxcsr, options(nostack, preserves_flags))
                };
                [u64::from(mxcsr), 0]
            };
            #[cfg(target_arch = "aarch64")]
            let caller_fp = {
                let control: u64;
                let status: u64;
                unsafe {
                    core::arch::asm!("mrs {},fpcr", "mrs {},fpsr", out(reg) control, out(reg) status, options(nostack, preserves_flags))
                };
                [control, status]
            };
            let _ = unsafe {
                worker.invoke_captured(
                    view,
                    caller_fp,
                    dispatcher,
                    std::ptr::from_mut(&mut calls).cast(),
                    NativeInvocation {
                        gateway,
                        context: std::ptr::from_mut(&mut context).cast(),
                        entry: faulting_read as *const () as usize,
                    },
                )
            };
            panic!("fatal captured fault unexpectedly returned");
        }
        let _ = unsafe {
            worker
                .begin_batch(view, &registry, dispatcher, std::ptr::null_mut())
                .unwrap();
            worker.invoke_active(NativeInvocation {
                gateway,
                context: std::ptr::from_mut(&mut context).cast(),
                entry: faulting_read as *const () as usize,
            })
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
            "retry_livelock",
            "captured_retry_livelock",
            "captured_unattributed",
            "captured_panic",
            "dispatcher_fatal",
            "closed_diagnostic_pipe",
            "alternate_stack_guard",
            "unrelated_sigbus",
            "chain",
        ] {
            let output = Command::new(&executable)
                .args([
                    "--exact",
                    "tests::fatal_fault_subprocess_entry",
                    "--nocapture",
                ])
                .env("NIXE_DIRECT_FATAL_CASE", case)
                .output()
                .unwrap();
            let status = output.status;
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
            let diagnostic = String::from_utf8_lossy(&output.stderr);
            let reason = match case {
                "nested" => Some("nested-dispatch-fault"),
                "retry_livelock" => Some("retry-limit"),
                "captured_retry_livelock" => Some("retry-without-progress"),
                "captured_unattributed" => Some("unattributed-native-pc"),
                "captured_panic" => Some("dispatcher-panicked"),
                "dispatcher_fatal" => Some("dispatcher-rejected"),
                _ => None,
            };
            if let Some(reason) = reason {
                let line = diagnostic
                    .lines()
                    .find(|line| line.starts_with("nixe native fault: "))
                    .unwrap_or_else(|| panic!("missing diagnostic for {case}: {diagnostic}"));
                assert!(
                    line.contains(&format!("reason={reason} signal={}", libc::SIGSEGV)),
                    "{line}"
                );
                for name in ["native_pc=0x", "address=0x"] {
                    let value = line
                        .split_once(name)
                        .unwrap()
                        .1
                        .split_whitespace()
                        .next()
                        .unwrap();
                    assert_eq!(value.len(), 16);
                    assert!(usize::from_str_radix(value, 16).is_ok());
                }
                assert!(!line.contains("native_pc=0x0000000000000000"));
            } else {
                assert!(
                    !diagnostic.contains("nixe native fault:"),
                    "case={case}: {diagnostic}"
                );
            }
        }
    }

    #[test]
    fn concurrent_first_writers_make_progress_on_one_physical_page() {
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
        let barrier = Arc::new(Barrier::new(3));

        let workers = [0x11_u8, 0x22]
            .into_iter()
            .map(|value| {
                let memory = Arc::clone(&memory);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    let mut direct = unsafe { DirectMemoryFrontend::new(view, space) }.unwrap();
                    let mut worker = NativeWorker::default();
                    let mut direct = direct.begin_slice(&mut worker).unwrap();
                    barrier.wait();
                    direct
                        .write(
                            memory.as_ref(),
                            address,
                            MemoryAccess::normal(MemoryAccessSize::Byte),
                            MemoryValue::U8(value),
                        )
                        .unwrap();
                })
            })
            .collect::<Vec<_>>();
        barrier.wait();
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

    .globl nixe_direct_stub_invoke
    .type nixe_direct_stub_invoke,%function
nixe_direct_stub_invoke:
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
    adr x2,.Ldirect_stub_escape
    bl nixe_direct_prepare_escape
    mov x0,x19
    blr x20
    mov w0,#0
    b .Ldirect_stub_return
.Ldirect_stub_escape:
    mov w0,#1
.Ldirect_stub_return:
    ldp x27,x28,[sp],#16
    ldp x25,x26,[sp],#16
    ldp x23,x24,[sp],#16
    ldp x21,x22,[sp],#16
    ldp x19,x20,[sp],#16
    ldp x29,x30,[sp],#16
    ret
    .size nixe_direct_stub_invoke,.-nixe_direct_stub_invoke

    .globl nixe_direct_fault_landing_pad
    .type nixe_direct_fault_landing_pad,%function
nixe_direct_fault_landing_pad:
    ldr x9,[x0,#{registry}]
    cbnz x9,1f
    ldr x9,[x0,#{dispatcher_fp_control}]
    msr fpcr,x9
    ldr x9,[x0,#{dispatcher_fp_status}]
    msr fpsr,x9
1:
    bl nixe_direct_fault_dispatch
    brk #0
    .size nixe_direct_fault_landing_pad,.-nixe_direct_fault_landing_pad

    .globl nixe_direct_escape_now
    .type nixe_direct_escape_now,%function
nixe_direct_escape_now:
    mov sp,x1
    br x0
    .size nixe_direct_escape_now,.-nixe_direct_escape_now

    .globl nixe_direct_retry_signal_frame
    .type nixe_direct_retry_signal_frame,%function
nixe_direct_retry_signal_frame:
    mov sp,x0
    mov x8,#{sys_rt_sigreturn}
    svc #0
    brk #0
    .size nixe_direct_retry_signal_frame,.-nixe_direct_retry_signal_frame
"#
    ,
    sys_rt_sigreturn = const libc::SYS_rt_sigreturn,
    registry = const offset_of!(FaultSlot, registry),
    dispatcher_fp_control = const offset_of!(FaultSlot, dispatcher_fp_control),
    dispatcher_fp_status = const offset_of!(FaultSlot, dispatcher_fp_status),
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
    .if \name == 16
    str q10,[x0,#{output}]
    .else
    str x10,[x0,#{output}]
    .endif
    ret
    .globl nixe_direct_stub_read_\name\()_end
nixe_direct_stub_read_\name\()_end:
    .size nixe_direct_stub_read_\name,.-nixe_direct_stub_read_\name
    .endm

    DIRECT_READ 1, ldrb, w10
    DIRECT_READ 2, ldrh, w10
    DIRECT_READ 4, ldr, w10
    DIRECT_READ 8, ldr, x10
    DIRECT_READ 16, ldr, q10

    .macro DIRECT_WRITE name, instruction, source
    .globl nixe_direct_stub_write_\name
    .type nixe_direct_stub_write_\name,%function
nixe_direct_stub_write_\name:
    ldr x9,[x0]
    .if \name == 16
    ldr q10,[x0,#{value}]
    .else
    ldr x10,[x0,#{value}]
    .endif
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
    DIRECT_WRITE 16, str, q10
"#,
    value = const offset_of!(StubCall, value),
    output = const offset_of!(StubCall, output),
);
