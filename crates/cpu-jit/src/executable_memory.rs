//! Bounded executable-memory ownership and platform publication.
//!
//! The mapping in this module accepts finalized native code only. Mutable JIT
//! metadata and link tables are ordinary Rust allocations and must never be
//! placed in this arena.

use std::collections::BTreeMap;
use std::fmt;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::{Duration, Instant};

const DEFAULT_MAX_BYTES: usize = 1024 * 1024 * 1024;
const DEFAULT_MAX_SEGMENTS: usize = 65_536;

pub(crate) type SharedExecutableMemory = Arc<Mutex<ExecutableMemory>>;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct PublicationMetrics {
    pub(crate) total_ns: u64,
    pub(crate) lock_wait_ns: u64,
    pub(crate) allocation_ns: u64,
    pub(crate) zero_copy_ns: u64,
    pub(crate) protection_ns: u64,
    pub(crate) instruction_cache_ns: u64,
}

impl PublicationMetrics {
    fn add_platform(&mut self, other: Self) {
        self.allocation_ns = self.allocation_ns.saturating_add(other.allocation_ns);
        self.zero_copy_ns = self.zero_copy_ns.saturating_add(other.zero_copy_ns);
        self.protection_ns = self.protection_ns.saturating_add(other.protection_ns);
        self.instruction_cache_ns = self
            .instruction_cache_ns
            .saturating_add(other.instruction_cache_ns);
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ErrorKind {
    HostUnavailable,
    PrivilegeUnavailable,
    PlatformUnsupported,
    InvalidRequest,
    Exhausted,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ExecutableMemoryError {
    kind: ErrorKind,
    detail: Box<str>,
}

impl ExecutableMemoryError {
    fn new(kind: ErrorKind, detail: impl Into<Box<str>>) -> Self {
        Self {
            kind,
            detail: detail.into(),
        }
    }

    pub(crate) fn detail(&self) -> &str {
        &self.detail
    }

    #[cfg(test)]
    fn kind(&self) -> ErrorKind {
        self.kind
    }
}

impl fmt::Display for ExecutableMemoryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.detail)
    }
}

#[derive(Clone, Copy, Debug)]
struct Limits {
    max_bytes: usize,
    max_segments: usize,
}

impl Limits {
    const DEFAULT: Self = Self {
        max_bytes: DEFAULT_MAX_BYTES,
        max_segments: DEFAULT_MAX_SEGMENTS,
    };
}

#[derive(Clone, Copy, Debug)]
struct PublicationPlan {
    offset: usize,
    code_len: usize,
    mapped_len: usize,
    reused: bool,
}

#[derive(Debug)]
struct AllocationState {
    page_size: usize,
    limits: Limits,
    high_water: usize,
    live_segments: usize,
    free: BTreeMap<usize, usize>,
    poisoned: bool,
}

impl AllocationState {
    fn new(page_size: usize, limits: Limits) -> Result<Self, ExecutableMemoryError> {
        if page_size == 0 || !page_size.is_power_of_two() {
            return Err(ExecutableMemoryError::new(
                ErrorKind::HostUnavailable,
                "host reported an invalid executable-memory page size",
            ));
        }
        if limits.max_bytes < page_size || !limits.max_bytes.is_multiple_of(page_size) {
            return Err(ExecutableMemoryError::new(
                ErrorKind::InvalidRequest,
                "executable-memory byte limit must be a nonzero page multiple",
            ));
        }
        if limits.max_segments == 0 {
            return Err(ExecutableMemoryError::new(
                ErrorKind::InvalidRequest,
                "executable-memory segment limit must be nonzero",
            ));
        }
        Ok(Self {
            page_size,
            limits,
            high_water: 0,
            live_segments: 0,
            free: BTreeMap::new(),
            poisoned: false,
        })
    }

    fn plan(
        &mut self,
        code_len: usize,
        alignment: usize,
    ) -> Result<PublicationPlan, ExecutableMemoryError> {
        if self.live_segments == self.limits.max_segments {
            return Err(ExecutableMemoryError::new(
                ErrorKind::Exhausted,
                "executable-memory segment limit exhausted",
            ));
        }
        if self.poisoned {
            return Err(ExecutableMemoryError::new(
                ErrorKind::HostUnavailable,
                "executable-memory owner is unavailable after a failed publication",
            ));
        }
        if code_len == 0 {
            return Err(ExecutableMemoryError::new(
                ErrorKind::InvalidRequest,
                "cannot publish an empty native-code segment",
            ));
        }
        if alignment == 0 || !alignment.is_power_of_two() || alignment > self.page_size {
            return Err(ExecutableMemoryError::new(
                ErrorKind::InvalidRequest,
                "native-code alignment must be a power of two no larger than one page",
            ));
        }
        let mapped_len = align_up(code_len, self.page_size).ok_or_else(|| {
            ExecutableMemoryError::new(
                ErrorKind::Exhausted,
                "native-code segment size overflowed executable-memory accounting",
            )
        })?;
        if let Some((offset, available)) = self.free.iter().find_map(|(offset, available)| {
            (*available >= mapped_len).then_some((*offset, *available))
        }) {
            self.free.remove(&offset);
            if available > mapped_len {
                self.free
                    .insert(offset + mapped_len, available - mapped_len);
            }
            return Ok(PublicationPlan {
                offset,
                code_len,
                mapped_len,
                reused: true,
            });
        }
        let end = self.high_water.checked_add(mapped_len).ok_or_else(|| {
            ExecutableMemoryError::new(
                ErrorKind::Exhausted,
                "executable-memory allocation overflowed its bounded arena",
            )
        })?;
        if end > self.limits.max_bytes {
            return Err(ExecutableMemoryError::new(
                ErrorKind::Exhausted,
                "executable-memory byte limit exhausted",
            ));
        }
        Ok(PublicationPlan {
            offset: self.high_water,
            code_len,
            mapped_len,
            reused: false,
        })
    }

    fn commit(&mut self, plan: PublicationPlan) {
        if !plan.reused {
            self.high_water += plan.mapped_len;
        }
        self.live_segments += 1;
    }

    fn release(&mut self, offset: usize, mapped_len: usize) {
        self.live_segments = self
            .live_segments
            .checked_sub(1)
            .expect("every executable allocation is released exactly once");
        let mut start = offset;
        let mut len = mapped_len;
        if let Some((&previous, &previous_len)) = self.free.range(..offset).next_back()
            && previous + previous_len == offset
        {
            self.free.remove(&previous);
            start = previous;
            len += previous_len;
        }
        if let Some((&next, &next_len)) = self.free.range(start..).next()
            && start + len == next
        {
            self.free.remove(&next);
            len += next_len;
        }
        self.free.insert(start, len);
    }
}

#[derive(Clone, Debug)]
pub(crate) struct PublishedCode {
    // Store the immutable process address as an integer. The allocation
    // handle below owns the arena range and keeps it live; reconstructing a
    // pointer happens only at the native-call/publication boundary.
    pub(crate) address: usize,
    #[cfg(test)]
    len: usize,
    pub(crate) mapped_len: usize,
    _allocation: Arc<PublishedAllocation>,
}

#[derive(Debug)]
struct PublishedAllocation {
    owner: Weak<Mutex<ExecutableMemory>>,
    offset: usize,
    mapped_len: usize,
}

impl Drop for PublishedAllocation {
    fn drop(&mut self) {
        if let Some(owner) = self.owner.upgrade() {
            owner
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .state
                .release(self.offset, self.mapped_len);
        }
    }
}

/// The process-wide owner of the only executable mapping used by the JIT.
///
/// Page granularity keeps every live RX publication disjoint from writable
/// bytes. Retired pages return to the allocator only after cache quiescence.
pub(crate) struct ExecutableMemory {
    arena: platform::Arena,
    state: AllocationState,
}

impl ExecutableMemory {
    fn new(limits: Limits) -> Result<Self, ExecutableMemoryError> {
        let arena = platform::Arena::new(limits.max_bytes)?;
        let state = AllocationState::new(arena.page_size(), limits)?;
        Ok(Self { arena, state })
    }

    fn publish_raw(
        &mut self,
        code: &[u8],
        alignment: usize,
    ) -> Result<(NonNull<u8>, usize, usize, usize, PublicationMetrics), ExecutableMemoryError> {
        let allocation_started = Instant::now();
        let plan = self.state.plan(code.len(), alignment)?;
        let allocation_ns = duration_ns(allocation_started.elapsed());
        let (address, mut metrics) = match self.arena.publish(plan, code) {
            Ok(publication) => publication,
            Err(error) => {
                // A platform transition can fail after touching the reserved
                // page. Never allow that page to be reused or mutated.
                self.state.poisoned = true;
                return Err(error);
            }
        };
        self.state.commit(plan);
        metrics.allocation_ns = metrics.allocation_ns.saturating_add(allocation_ns);
        Ok((address, code.len(), plan.offset, plan.mapped_len, metrics))
    }

    #[cfg(test)]
    unsafe fn bytes(&self, code: &PublishedCode) -> &[u8] {
        // SAFETY: the returned range was committed by this live arena and is
        // immutable for the arena's lifetime.
        unsafe { std::slice::from_raw_parts(code.address as *const u8, code.len) }
    }
}

pub(crate) fn publish_code(
    owner: &SharedExecutableMemory,
    code: &[u8],
    alignment: usize,
) -> Result<(PublishedCode, PublicationMetrics), ExecutableMemoryError> {
    let total_started = Instant::now();
    let lock_started = Instant::now();
    let mut guard = owner.lock().map_err(|_| {
        ExecutableMemoryError::new(
            ErrorKind::HostUnavailable,
            "executable-memory owner lock was poisoned",
        )
    })?;
    let lock_wait_ns = duration_ns(lock_started.elapsed());
    let (address, code_len, offset, mapped_len, platform_metrics) =
        guard.publish_raw(code, alignment)?;
    drop(guard);
    let mut metrics = PublicationMetrics {
        lock_wait_ns,
        total_ns: duration_ns(total_started.elapsed()),
        ..PublicationMetrics::default()
    };
    metrics.add_platform(platform_metrics);
    #[cfg(not(test))]
    let _ = code_len;
    Ok((
        PublishedCode {
            address: address.as_ptr().addr(),
            #[cfg(test)]
            len: code_len,
            mapped_len,
            _allocation: Arc::new(PublishedAllocation {
                owner: Arc::downgrade(owner),
                offset,
                mapped_len,
            }),
        },
        metrics,
    ))
}

fn duration_ns(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

/// Returns one shared owner per process while allowing it to be reclaimed when
/// every provider, domain, and executor has gone away. The weak registry also
/// prevents concurrent providers from creating multiple MAP_JIT regions on
/// macOS.
pub(crate) fn process_executable_memory() -> Result<SharedExecutableMemory, ExecutableMemoryError> {
    if !cfg!(any(target_arch = "x86_64", target_arch = "aarch64")) {
        return Err(ExecutableMemoryError::new(
            ErrorKind::PlatformUnsupported,
            format!(
                "JIT executable memory supports only x86-64 and AArch64 hosts, not {}",
                std::env::consts::ARCH
            ),
        ));
    }
    static OWNER: OnceLock<Mutex<Weak<Mutex<ExecutableMemory>>>> = OnceLock::new();

    let registry = OWNER.get_or_init(|| Mutex::new(Weak::new()));
    let mut weak = registry.lock().map_err(|_| {
        ExecutableMemoryError::new(
            ErrorKind::HostUnavailable,
            "executable-memory owner registry was poisoned",
        )
    })?;
    if let Some(owner) = weak.upgrade() {
        return Ok(owner);
    }
    let owner = Arc::new(Mutex::new(ExecutableMemory::new(Limits::DEFAULT)?));
    *weak = Arc::downgrade(&owner);
    Ok(owner)
}

const fn align_up(value: usize, alignment: usize) -> Option<usize> {
    match value.checked_add(alignment - 1) {
        Some(value) => Some(value & !(alignment - 1)),
        None => None,
    }
}

#[cfg(all(unix, not(target_vendor = "apple")))]
mod platform {
    use std::ptr::NonNull;
    use std::time::Instant;

    use super::{
        ErrorKind, ExecutableMemoryError, PublicationMetrics, PublicationPlan, duration_ns,
    };

    pub(super) struct Arena {
        base: NonNull<u8>,
        len: usize,
        page_size: usize,
    }

    // SAFETY: the mapping is exclusively mutated through `&mut Arena` before
    // publication; published pages are immutable. Moving ownership is safe.
    unsafe impl Send for Arena {}

    impl Arena {
        pub(super) fn new(len: usize) -> Result<Self, ExecutableMemoryError> {
            // SAFETY: sysconf has no pointer preconditions.
            let reported_page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
            let page_size = usize::try_from(reported_page_size).map_err(|_| {
                ExecutableMemoryError::new(
                    ErrorKind::HostUnavailable,
                    "host did not report an executable-memory page size",
                )
            })?;
            // SAFETY: the anonymous mapping has no file descriptor or caller
            // memory lifetime. It is released by Drop.
            let pointer = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    len,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    -1,
                    0,
                )
            };
            if pointer == libc::MAP_FAILED {
                return Err(last_os_error("could not reserve bounded JIT memory"));
            }
            Ok(Self {
                // SAFETY: MAP_FAILED was rejected and a successful nonempty
                // mmap cannot return null.
                base: unsafe { NonNull::new_unchecked(pointer.cast()) },
                len,
                page_size,
            })
        }

        pub(super) const fn page_size(&self) -> usize {
            self.page_size
        }

        pub(super) fn publish(
            &mut self,
            plan: PublicationPlan,
            code: &[u8],
        ) -> Result<(NonNull<u8>, PublicationMetrics), ExecutableMemoryError> {
            // SAFETY: AllocationState confines the range to this mapping.
            let destination = unsafe { self.base.as_ptr().add(plan.offset) };
            let mut metrics = PublicationMetrics::default();
            let protection_started = Instant::now();
            if plan.reused
                && unsafe {
                    libc::mprotect(
                        destination.cast(),
                        plan.mapped_len,
                        libc::PROT_READ | libc::PROT_WRITE,
                    )
                } != 0
            {
                return Err(last_os_error(
                    "could not reopen retired JIT pages for reuse",
                ));
            }
            metrics.protection_ns = duration_ns(protection_started.elapsed());
            let zero_copy_started = Instant::now();
            unsafe { std::ptr::write_bytes(destination, 0, plan.mapped_len) };
            unsafe { std::ptr::copy_nonoverlapping(code.as_ptr(), destination, plan.code_len) };
            metrics.zero_copy_ns = duration_ns(zero_copy_started.elapsed());
            // Linux/Unix write-to-execute transition contract:
            // https://man7.org/linux/man-pages/man2/mprotect.2.html
            // SAFETY: destination and mapped_len are page-aligned and wholly
            // contained in the live mapping.
            let protection_started = Instant::now();
            if unsafe {
                libc::mprotect(
                    destination.cast(),
                    plan.mapped_len,
                    libc::PROT_READ | libc::PROT_EXEC,
                )
            } != 0
            {
                return Err(last_os_error("could not seal JIT code read-execute"));
            }
            metrics.protection_ns = metrics
                .protection_ns
                .saturating_add(duration_ns(protection_started.elapsed()));
            // SAFETY: the range was initialized above and has just been made
            // executable. No address escapes until synchronization completes.
            let instruction_cache_started = Instant::now();
            unsafe { synchronize_instruction_cache(destination, plan.code_len) };
            metrics.instruction_cache_ns = duration_ns(instruction_cache_started.elapsed());
            // SAFETY: mmap returned a non-null base and the offset is in range.
            Ok((unsafe { NonNull::new_unchecked(destination) }, metrics))
        }
    }

    impl Drop for Arena {
        fn drop(&mut self) {
            // SAFETY: this pair exactly matches the successful mmap in new.
            let _ = unsafe { libc::munmap(self.base.as_ptr().cast(), self.len) };
        }
    }

    fn last_os_error(operation: &str) -> ExecutableMemoryError {
        let error = std::io::Error::last_os_error();
        let kind = match error.raw_os_error() {
            Some(libc::EACCES | libc::EPERM) => ErrorKind::PrivilegeUnavailable,
            _ => ErrorKind::HostUnavailable,
        };
        ExecutableMemoryError::new(kind, format!("{operation}: {error}"))
    }

    #[cfg(target_arch = "x86_64")]
    unsafe fn synchronize_instruction_cache(_start: *mut u8, _len: usize) {
        // x86-64 has coherent instruction and data caches. The compiler fence
        // prevents Rust operations from moving across the publication point.
        std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
    }

    #[cfg(target_arch = "aarch64")]
    unsafe fn synchronize_instruction_cache(start: *mut u8, len: usize) {
        // Arm's required self-modifying-code sequence and CTR_EL0 line-size
        // fields: https://developer.arm.com/community/arm-community-blogs/b/
        // architectures-and-processors-blog/posts/caches-self-modifying-code-implementing-clear-cache
        let ctr: usize;
        // SAFETY: reading CTR_EL0 is permitted at EL0 on supported hosts.
        unsafe { core::arch::asm!("mrs {ctr}, ctr_el0", ctr = out(reg) ctr) };
        let data_line = 4usize << ((ctr >> 16) & 0xf);
        let instruction_line = 4usize << (ctr & 0xf);
        let end = start as usize + len;
        if ctr & (1 << 28) == 0 {
            let mut address = (start as usize) & !(data_line - 1);
            while address < end {
                // SAFETY: DC CVAU accepts any address in the initialized range.
                unsafe { core::arch::asm!("dc cvau, {address}", address = in(reg) address) };
                address += data_line;
            }
        }
        // SAFETY: required completion barrier before I-cache invalidation.
        unsafe { core::arch::asm!("dsb ish") };
        if ctr & (1 << 29) == 0 {
            let mut address = (start as usize) & !(instruction_line - 1);
            while address < end {
                // SAFETY: IC IVAU accepts any address in the initialized range.
                unsafe { core::arch::asm!("ic ivau, {address}", address = in(reg) address) };
                address += instruction_line;
            }
        }
        // SAFETY: required completion and context-synchronization barriers.
        unsafe { core::arch::asm!("dsb ish", "isb") };
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    unsafe fn synchronize_instruction_cache(_start: *mut u8, _len: usize) {}
}

#[cfg(target_os = "macos")]
mod platform {
    use std::ffi::{CStr, c_char, c_void};
    use std::ptr::NonNull;
    use std::time::Instant;

    use super::{
        ErrorKind, ExecutableMemoryError, PublicationMetrics, PublicationPlan, duration_ns,
    };

    type WriteProtect = unsafe extern "C" fn(i32);

    pub(super) struct Arena {
        base: NonNull<u8>,
        len: usize,
        page_size: usize,
        write_protect: WriteProtect,
    }

    // SAFETY: MAP_JIT write permission is thread-local, and every mutation is
    // serialized through the owning Mutex and completed before publication.
    unsafe impl Send for Arena {}

    impl Arena {
        pub(super) fn new(len: usize) -> Result<Self, ExecutableMemoryError> {
            // Apple's JIT porting contract defines MAP_JIT, the per-thread
            // write-protection toggle, and the callback-only allowlist mode:
            // https://developer.apple.com/documentation/apple-silicon/porting-just-in-time-compilers-to-apple-silicon
            if entitlement_enabled(c"com.apple.security.cs.jit-write-allowlist")? {
                return Err(ExecutableMemoryError::new(
                    ErrorKind::PrivilegeUnavailable,
                    "the macOS JIT write-allowlist entitlement requires a callback-based publisher that this JIT does not use",
                ));
            }
            let write_protect = resolve_write_protect()?;
            // SAFETY: sysconf has no pointer preconditions.
            let reported_page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
            let page_size = usize::try_from(reported_page_size).map_err(|_| {
                ExecutableMemoryError::new(
                    ErrorKind::HostUnavailable,
                    "macOS did not report an executable-memory page size",
                )
            })?;
            // MAP_JIT plus pthread_jit_write_protect_np is Apple's supported
            // W^X mechanism. Hardened processes without allow-jit are rejected
            // here rather than failing after engine selection.
            let pointer = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    len,
                    libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
                    libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_JIT,
                    -1,
                    0,
                )
            };
            if pointer == libc::MAP_FAILED {
                return Err(last_os_error("macOS rejected the MAP_JIT arena"));
            }
            // SAFETY: enables execute and disables write access for this thread.
            unsafe { write_protect(1) };
            Ok(Self {
                base: unsafe { NonNull::new_unchecked(pointer.cast()) },
                len,
                page_size,
                write_protect,
            })
        }

        pub(super) const fn page_size(&self) -> usize {
            self.page_size
        }

        pub(super) fn publish(
            &mut self,
            plan: PublicationPlan,
            code: &[u8],
        ) -> Result<(NonNull<u8>, PublicationMetrics), ExecutableMemoryError> {
            let destination = unsafe { self.base.as_ptr().add(plan.offset) };
            let mut metrics = PublicationMetrics::default();
            // SAFETY: temporarily disables execute and enables write access on
            // this thread only. The pointer is not published during this span.
            let protection_started = Instant::now();
            unsafe { (self.write_protect)(0) };
            metrics.protection_ns = duration_ns(protection_started.elapsed());
            let zero_copy_started = Instant::now();
            unsafe { std::ptr::write_bytes(destination, 0, plan.mapped_len) };
            unsafe { std::ptr::copy_nonoverlapping(code.as_ptr(), destination, plan.code_len) };
            metrics.zero_copy_ns = duration_ns(zero_copy_started.elapsed());
            // SAFETY: restores execute-only publication before the address can
            // leave this owner.
            let protection_started = Instant::now();
            unsafe { (self.write_protect)(1) };
            metrics.protection_ns = metrics
                .protection_ns
                .saturating_add(duration_ns(protection_started.elapsed()));
            // Apple's documented instruction-cache API:
            // https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man3/sys_cache_control.3.html
            // SAFETY: the range is initialized and not yet published.
            let instruction_cache_started = Instant::now();
            unsafe { sys_icache_invalidate(destination.cast(), plan.code_len) };
            metrics.instruction_cache_ns = duration_ns(instruction_cache_started.elapsed());
            Ok((unsafe { NonNull::new_unchecked(destination) }, metrics))
        }
    }

    impl Drop for Arena {
        fn drop(&mut self) {
            let _ = unsafe { libc::munmap(self.base.as_ptr().cast(), self.len) };
        }
    }

    fn resolve_write_protect() -> Result<WriteProtect, ExecutableMemoryError> {
        let symbol =
            unsafe { libc::dlsym(libc::RTLD_DEFAULT, c"pthread_jit_write_protect_np".as_ptr()) };
        if symbol.is_null() {
            return Err(ExecutableMemoryError::new(
                ErrorKind::PlatformUnsupported,
                "macOS does not provide pthread_jit_write_protect_np",
            ));
        }
        // SAFETY: dlsym returned the documented function with this signature.
        Ok(unsafe { std::mem::transmute::<*mut c_void, WriteProtect>(symbol) })
    }

    fn entitlement_enabled(name: &CStr) -> Result<bool, ExecutableMemoryError> {
        let task = unsafe { SecTaskCreateFromSelf(std::ptr::null()) };
        let task = NonNull::new(task.cast_mut()).ok_or_else(|| {
            ExecutableMemoryError::new(
                ErrorKind::HostUnavailable,
                "could not inspect macOS JIT entitlements",
            )
        })?;
        let key =
            unsafe { CFStringCreateWithCString(std::ptr::null(), name.as_ptr(), 0x0800_0100) };
        let Some(key) = NonNull::new(key.cast_mut()) else {
            unsafe { CFRelease(task.as_ptr()) };
            return Err(ExecutableMemoryError::new(
                ErrorKind::HostUnavailable,
                "could not construct a macOS JIT entitlement key",
            ));
        };
        let mut error: *const c_void = std::ptr::null();
        let value =
            unsafe { SecTaskCopyValueForEntitlement(task.as_ptr(), key.as_ptr(), &mut error) };
        let enabled = !value.is_null()
            && unsafe { CFGetTypeID(value) == CFBooleanGetTypeID() }
            && unsafe { CFBooleanGetValue(value) != 0 };
        unsafe {
            if !value.is_null() {
                CFRelease(value);
            }
            if !error.is_null() {
                CFRelease(error);
            }
            CFRelease(key.as_ptr());
            CFRelease(task.as_ptr());
        }
        Ok(enabled)
    }

    fn last_os_error(operation: &str) -> ExecutableMemoryError {
        let error = std::io::Error::last_os_error();
        let kind = match error.raw_os_error() {
            Some(libc::EACCES | libc::EPERM) => ErrorKind::PrivilegeUnavailable,
            _ => ErrorKind::HostUnavailable,
        };
        ExecutableMemoryError::new(kind, format!("{operation}: {error}"))
    }

    #[link(name = "Security", kind = "framework")]
    unsafe extern "C" {
        fn SecTaskCreateFromSelf(allocator: *const c_void) -> *const c_void;
        fn SecTaskCopyValueForEntitlement(
            task: *const c_void,
            entitlement: *const c_void,
            error: *mut *const c_void,
        ) -> *const c_void;
    }

    #[link(name = "CoreFoundation", kind = "framework")]
    unsafe extern "C" {
        fn CFStringCreateWithCString(
            allocator: *const c_void,
            bytes: *const c_char,
            encoding: u32,
        ) -> *const c_void;
        fn CFGetTypeID(value: *const c_void) -> usize;
        fn CFBooleanGetTypeID() -> usize;
        fn CFBooleanGetValue(value: *const c_void) -> u8;
        fn CFRelease(value: *const c_void);
    }

    unsafe extern "C" {
        fn sys_icache_invalidate(start: *mut c_void, len: usize);
    }
}

#[cfg(windows)]
mod platform {
    use std::ptr::NonNull;
    use std::time::Instant;

    use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_EXECUTE_READ, PAGE_NOACCESS, PAGE_READWRITE,
        VirtualAlloc, VirtualFree, VirtualProtect,
    };
    use windows_sys::Win32::System::SystemInformation::{GetSystemInfo, SYSTEM_INFO};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    use super::{
        ErrorKind, ExecutableMemoryError, PublicationMetrics, PublicationPlan, duration_ns,
    };

    pub(super) struct Arena {
        base: NonNull<u8>,
        len: usize,
        page_size: usize,
    }

    // SAFETY: publication is serialized and every published page is immutable.
    unsafe impl Send for Arena {}

    impl Arena {
        pub(super) fn new(len: usize) -> Result<Self, ExecutableMemoryError> {
            let mut info = unsafe { std::mem::zeroed::<SYSTEM_INFO>() };
            unsafe { GetSystemInfo(&mut info) };
            let page_size = info.dwPageSize as usize;
            let pointer =
                unsafe { VirtualAlloc(std::ptr::null(), len, MEM_RESERVE, PAGE_NOACCESS) };
            let base = NonNull::new(pointer.cast())
                .ok_or_else(|| last_os_error("could not reserve bounded JIT memory"))?;
            Ok(Self {
                base,
                len,
                page_size,
            })
        }

        pub(super) const fn page_size(&self) -> usize {
            self.page_size
        }

        pub(super) fn publish(
            &mut self,
            plan: PublicationPlan,
            code: &[u8],
        ) -> Result<(NonNull<u8>, PublicationMetrics), ExecutableMemoryError> {
            let destination = unsafe { self.base.as_ptr().add(plan.offset) };
            let mut metrics = PublicationMetrics::default();
            // Windows requires writable allocation followed by VirtualProtect,
            // then an explicit FlushInstructionCache before execution:
            // https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-virtualalloc
            let allocation_started = Instant::now();
            if plan.reused {
                let mut previous = PAGE_NOACCESS;
                if unsafe {
                    VirtualProtect(
                        destination.cast(),
                        plan.mapped_len,
                        PAGE_READWRITE,
                        &mut previous,
                    )
                } == 0
                {
                    return Err(last_os_error(
                        "could not reopen retired JIT pages for reuse",
                    ));
                }
            } else {
                let committed = unsafe {
                    VirtualAlloc(
                        destination.cast(),
                        plan.mapped_len,
                        MEM_COMMIT,
                        PAGE_READWRITE,
                    )
                };
                if committed.is_null() {
                    return Err(last_os_error("could not commit writable JIT pages"));
                }
            }
            metrics.allocation_ns = duration_ns(allocation_started.elapsed());
            let zero_copy_started = Instant::now();
            unsafe { std::ptr::write_bytes(destination, 0, plan.mapped_len) };
            unsafe { std::ptr::copy_nonoverlapping(code.as_ptr(), destination, plan.code_len) };
            metrics.zero_copy_ns = duration_ns(zero_copy_started.elapsed());
            let mut previous = PAGE_NOACCESS;
            // https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-flushinstructioncache
            let protection_started = Instant::now();
            if unsafe {
                VirtualProtect(
                    destination.cast(),
                    plan.mapped_len,
                    PAGE_EXECUTE_READ,
                    &mut previous,
                )
            } == 0
            {
                return Err(last_os_error("could not seal JIT code read-execute"));
            }
            metrics.protection_ns = duration_ns(protection_started.elapsed());
            let instruction_cache_started = Instant::now();
            if unsafe {
                FlushInstructionCache(GetCurrentProcess(), destination.cast(), plan.code_len)
            } == 0
            {
                return Err(last_os_error("could not synchronize the instruction cache"));
            }
            metrics.instruction_cache_ns = duration_ns(instruction_cache_started.elapsed());
            Ok((unsafe { NonNull::new_unchecked(destination) }, metrics))
        }
    }

    impl Drop for Arena {
        fn drop(&mut self) {
            let _ = unsafe { VirtualFree(self.base.as_ptr().cast(), 0, MEM_RELEASE) };
        }
    }

    fn last_os_error(operation: &str) -> ExecutableMemoryError {
        let error = std::io::Error::last_os_error();
        let kind = match error.raw_os_error() {
            Some(5) => ErrorKind::PrivilegeUnavailable,
            _ => ErrorKind::HostUnavailable,
        };
        ExecutableMemoryError::new(kind, format!("{operation}: {error}"))
    }
}

#[cfg(not(any(all(unix, not(target_vendor = "apple")), target_os = "macos", windows)))]
mod platform {
    use std::ptr::NonNull;

    use super::{ErrorKind, ExecutableMemoryError, PublicationMetrics, PublicationPlan};

    pub(super) struct Arena;

    impl Arena {
        pub(super) fn new(_len: usize) -> Result<Self, ExecutableMemoryError> {
            Err(ExecutableMemoryError::new(
                ErrorKind::PlatformUnsupported,
                "JIT executable memory supports x86-64 and AArch64 hosts on Unix, macOS, and Windows",
            ))
        }

        pub(super) const fn page_size(&self) -> usize {
            0
        }

        pub(super) fn publish(
            &mut self,
            _plan: PublicationPlan,
            _code: &[u8],
        ) -> Result<(NonNull<u8>, PublicationMetrics), ExecutableMemoryError> {
            unreachable!("unsupported platforms cannot construct an arena")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocation_state_enforces_byte_segment_and_alignment_bounds_atomically() {
        let mut state = AllocationState::new(
            4096,
            Limits {
                max_bytes: 8192,
                max_segments: 2,
            },
        )
        .unwrap();
        assert_eq!(state.plan(1, 16).unwrap().mapped_len, 4096);
        let first = state.plan(1, 16).unwrap();
        state.commit(first);
        let before = (state.high_water, state.live_segments, state.free.clone());

        assert_eq!(
            state.plan(0, 1).unwrap_err().kind(),
            ErrorKind::InvalidRequest
        );
        assert_eq!(
            state.plan(1, 3).unwrap_err().kind(),
            ErrorKind::InvalidRequest
        );
        assert_eq!(
            state.plan(4097, 1).unwrap_err().kind(),
            ErrorKind::Exhausted
        );
        assert_eq!(
            (state.high_water, state.live_segments, state.free.clone()),
            before
        );

        let second = state.plan(4096, 4096).unwrap();
        state.commit(second);
        assert_eq!(state.plan(1, 1).unwrap_err().kind(), ErrorKind::Exhausted);
    }

    #[test]
    fn invalid_limits_and_overflow_are_rejected() {
        assert!(AllocationState::new(0, Limits::DEFAULT).is_err());
        assert!(
            AllocationState::new(
                4096,
                Limits {
                    max_bytes: 1,
                    max_segments: 1
                }
            )
            .is_err()
        );
        let mut state = AllocationState::new(
            4096,
            Limits {
                max_bytes: usize::MAX & !4095,
                max_segments: 1,
            },
        )
        .unwrap();
        assert_eq!(
            state.plan(usize::MAX, 1).unwrap_err().kind(),
            ErrorKind::Exhausted
        );
    }

    #[test]
    fn released_ranges_coalesce_and_are_reused_before_arena_growth() {
        let mut state = AllocationState::new(
            4096,
            Limits {
                max_bytes: 12_288,
                max_segments: 3,
            },
        )
        .unwrap();
        let first = state.plan(1, 1).unwrap();
        state.commit(first);
        let second = state.plan(1, 1).unwrap();
        state.commit(second);
        state.release(first.offset, first.mapped_len);
        state.release(second.offset, second.mapped_len);

        let reused = state.plan(8192, 1).unwrap();
        assert!(reused.reused);
        assert_eq!(reused.offset, 0);
        assert_eq!(state.high_water, 8192);
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[test]
    fn platform_publication_is_immutable_readable_and_executable() {
        let owner = process_executable_memory().expect("host supports executable memory");

        // Fixed host-ABI bytes isolate executable-memory publication from the
        // compiler tested elsewhere.
        #[cfg(target_arch = "x86_64")]
        let code_bytes: &[u8] = &[0xb8, 42, 0, 0, 0, 0xc3]; // mov eax, 42; ret
        #[cfg(target_arch = "aarch64")]
        let code_bytes: &[u8] = &[
            0x40, 0x05, 0x80, 0x52, // mov w0, #42
            0xc0, 0x03, 0x5f, 0xd6, // ret
        ];

        let (first, _) = publish_code(&owner, code_bytes, 16).unwrap();
        let (second, _) = publish_code(&owner, &[0], 16).unwrap();
        let locked = owner.lock().unwrap();
        assert_eq!(unsafe { locked.bytes(&first) }, code_bytes);
        assert!(second.address.abs_diff(first.address) >= locked.state.page_size);
        #[cfg(target_os = "linux")]
        assert_linux_mapping_is_read_execute(first.address);
        drop(locked);

        // SAFETY: the fixture uses the host ABI, was published by the platform
        // path above, and takes no arguments or borrowed state.
        let function = unsafe {
            std::mem::transmute::<*mut u8, unsafe extern "C" fn() -> u32>(
                std::ptr::with_exposed_provenance_mut(first.address),
            )
        };
        assert_eq!(unsafe { function() }, 42);
    }

    #[cfg(target_os = "linux")]
    fn assert_linux_mapping_is_read_execute(address: usize) {
        let maps = std::fs::read_to_string("/proc/self/maps").unwrap();
        let permissions = maps
            .lines()
            .find_map(|line| {
                let mut fields = line.split_whitespace();
                let (start, end) = fields.next()?.split_once('-')?;
                let start = usize::from_str_radix(start, 16).ok()?;
                let end = usize::from_str_radix(end, 16).ok()?;
                (start <= address && address < end).then(|| fields.next().unwrap())
            })
            .expect("published address appears in the process memory map");
        assert!(permissions.starts_with("r-x"), "permissions={permissions}");
    }
}
