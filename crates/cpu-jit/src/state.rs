//! Exclusive ownership of a guest's complete context, either saved or native.
//!
//! Mirrors the stopped-core access used by Yuzu/Eden. Moving this value through
//! a scheduler lease transfers register authority, not a second register copy.
//! https://git.eden-emu.dev/eden-emu/eden/raw/commit/67bada77f8a43a90da2e94e89b8e7da73c256989/src/core/arm/dynarmic/arm_dynarmic_64.cpp
use crate::{
    JitError, Native, ffi,
    metrics::{Counter, record},
};
use nixe_cpu::state::{
    Nzcv, RegisterContext,
    a64::{A64Register, A64State},
};
use nixe_memory::GuestVirtualAddress;
use std::sync::{Arc, atomic::Ordering};

const SP: u32 = 31;
const PC: u32 = 32;
const TPIDR: u32 = 33;
const TPIDRRO: u32 = 34;
const NZCV: u32 = 35;
const FPCR: u32 = 36;
const FPSR: u32 = 37;

struct NativeOwner {
    core: Arc<Native>,
}
impl Drop for NativeOwner {
    fn drop(&mut self) {
        self.core.context_owned.store(false, Ordering::Release);
    }
}
enum Storage {
    Saved,
    Native(NativeOwner),
    Unavailable,
}

/// One movable, exclusive register owner. It cannot expose saved bytes while
/// native state is authoritative. Accessors require possession of this owner;
/// it is absent from the runtime table throughout native execution.
pub struct ThreadState {
    storage: Storage,
    // Reuse the save area on switches. Its contents are inaccessible when
    // Native or Unavailable; it is neither a snapshot nor a second authority.
    saved: Box<A64State>,
}
impl Default for ThreadState {
    fn default() -> Self {
        A64State::default().into()
    }
}
impl From<A64State> for ThreadState {
    fn from(state: A64State) -> Self {
        Self {
            storage: Storage::Saved,
            saved: Box::new(state),
        }
    }
}
impl ThreadState {
    #[must_use]
    #[inline]
    pub fn is_resident(&self) -> bool {
        matches!(self.storage, Storage::Native(_))
    }
    #[must_use]
    #[inline]
    pub fn is_available(&self) -> bool {
        !matches!(self.storage, Storage::Unavailable)
    }
    pub(crate) fn discard(&mut self) {
        self.storage = Storage::Unavailable;
    }
    /// Save on eviction/migration; repeated same-thread entries never call this.
    #[inline]
    pub fn materialize(&mut self) {
        match &self.storage {
            Storage::Native(owner) => unsafe {
                record(Counter::ContextSaves, 1);
                ffi::nixe_dynarmic_save(owner.core.pointer, self.saved.as_mut());
            },
            Storage::Saved => return,
            Storage::Unavailable => panic!("architectural state is unavailable"),
        }
        self.storage = Storage::Saved;
    }
    /// A scheduler switch/migration ends the old core's local reservation.
    #[inline]
    pub fn leave_core(&mut self) {
        if let Storage::Native(owner) = &self.storage {
            unsafe { ffi::nixe_dynarmic_clear_exclusive(owner.core.pointer) };
        }
        self.materialize();
    }
    /// Owned full snapshot for an actual context consumer; leaves residency intact.
    #[must_use]
    pub fn snapshot(&self) -> A64State {
        match &self.storage {
            Storage::Saved => self.saved.as_ref().clone(),
            Storage::Native(owner) => {
                let mut state = A64State::default();
                record(Counter::ContextSaves, 1);
                unsafe { ffi::nixe_dynarmic_save(owner.core.pointer, &mut state) };
                state
            }
            Storage::Unavailable => panic!("architectural state is unavailable"),
        }
    }
    pub fn saved_mut(&mut self) -> &mut A64State {
        self.materialize();
        self.saved.as_mut()
    }
    pub(crate) fn select(&mut self, core: &Arc<Native>) -> Result<(), JitError> {
        match &self.storage {
            Storage::Native(owner) if Arc::ptr_eq(&owner.core, core) => return Ok(()),
            Storage::Unavailable => {
                return Err(JitError("architectural state is unavailable".into()));
            }
            _ => {}
        }
        // A different destination may not replace an unexported guest. The
        // runtime evicts its old occupant before dispatching a new lease.
        // The unique JitThread is the sole claimant. Other owners can only
        // release, after their final register access; there are no competing
        // claimants requiring an atomic read/modify/write on every switch.
        if core.context_owned.load(Ordering::Acquire) {
            return Err(JitError(
                "Dynarmic core still owns another guest context".into(),
            ));
        }
        core.context_owned.store(true, Ordering::Release);
        self.leave_core();
        record(Counter::ContextLoads, 1);
        unsafe { ffi::nixe_dynarmic_load(core.pointer, self.saved.as_ref()) };
        self.storage = Storage::Native(NativeOwner { core: core.clone() });
        Ok(())
    }
    fn read(&self, reg: u32, saved: impl FnOnce(&A64State) -> u64) -> u64 {
        match &self.storage {
            Storage::Saved => saved(self.saved.as_ref()),
            Storage::Native(owner) => unsafe { ffi::nixe_dynarmic_read(owner.core.pointer, reg) },
            Storage::Unavailable => panic!("architectural state is unavailable"),
        }
    }
    fn write(&mut self, reg: u32, value: u64, saved: impl FnOnce(&mut A64State)) {
        match &self.storage {
            Storage::Saved => saved(self.saved.as_mut()),
            Storage::Native(owner) => unsafe {
                ffi::nixe_dynarmic_write(owner.core.pointer, reg, value)
            },
            Storage::Unavailable => panic!("architectural state is unavailable"),
        }
    }
    #[must_use]
    #[inline]
    pub fn read_x(&self, register: A64Register) -> u64 {
        let reg = match register {
            A64Register::General(x) => u32::from(x.index()),
            A64Register::StackPointer => SP,
            A64Register::Zero => return 0,
        };
        self.read(reg, |state| state.read_x(register))
    }
    #[inline]
    pub fn write_x(&mut self, register: A64Register, value: u64) {
        let reg = match register {
            A64Register::General(x) => u32::from(x.index()),
            A64Register::StackPointer => SP,
            A64Register::Zero => return,
        };
        self.write(reg, value, |state| state.write_x(register, value));
    }
    #[must_use]
    #[inline]
    pub fn read_w(&self, register: A64Register) -> u32 {
        self.read_x(register) as u32
    }
    #[inline]
    pub fn write_w(&mut self, register: A64Register, value: u32) {
        self.write_x(register, u64::from(value));
    }
    #[must_use]
    #[inline]
    pub fn pc(&self) -> u64 {
        self.read(PC, A64State::pc)
    }
    #[inline]
    pub fn set_pc(&mut self, value: u64) {
        self.write(PC, value, |state| state.set_pc(value));
    }
    #[must_use]
    #[inline]
    pub fn nzcv(&self) -> Nzcv {
        Nzcv::from_bits(self.read(NZCV, |state| u64::from(state.nzcv().bits())) as u32)
    }
    #[inline]
    pub fn set_nzcv(&mut self, value: Nzcv) {
        self.write(NZCV, u64::from(value.bits()), |state| state.set_nzcv(value));
    }
    #[must_use]
    #[inline]
    pub fn vector(&self, index: u8) -> Option<u128> {
        if index >= 32 {
            return None;
        }
        match &self.storage {
            Storage::Saved => self.saved.vector(index),
            Storage::Native(owner) => {
                let mut lanes = [0u64; 2];
                unsafe {
                    ffi::nixe_dynarmic_read_vector(
                        owner.core.pointer,
                        u32::from(index),
                        lanes.as_mut_ptr(),
                    )
                };
                Some(u128::from(lanes[0]) | (u128::from(lanes[1]) << 64))
            }
            Storage::Unavailable => panic!("architectural state is unavailable"),
        }
    }
    #[inline]
    pub fn set_vector(&mut self, index: u8, value: u128) -> bool {
        if index >= 32 {
            return false;
        }
        match &self.storage {
            Storage::Saved => self.saved.set_vector(index, value),
            Storage::Native(owner) => {
                let lanes = [value as u64, (value >> 64) as u64];
                unsafe {
                    ffi::nixe_dynarmic_write_vector(
                        owner.core.pointer,
                        u32::from(index),
                        lanes.as_ptr(),
                    )
                };
                true
            }
            Storage::Unavailable => panic!("architectural state is unavailable"),
        }
    }
    #[must_use]
    #[inline]
    pub fn fpcr(&self) -> u32 {
        self.read(FPCR, |state| u64::from(state.fpcr())) as u32
    }
    #[inline]
    pub fn set_fpcr(&mut self, value: u32) {
        self.write(FPCR, u64::from(value), |state| state.set_fpcr(value));
    }
    #[must_use]
    #[inline]
    pub fn fpsr(&self) -> u32 {
        self.read(FPSR, |state| u64::from(state.fpsr())) as u32
    }
    #[inline]
    pub fn set_fpsr(&mut self, value: u32) {
        self.write(FPSR, u64::from(value), |state| state.set_fpsr(value));
    }
    #[must_use]
    #[inline]
    pub fn tpidr_el0(&self) -> u64 {
        self.read(TPIDR, A64State::tpidr_el0)
    }
    #[inline]
    pub fn set_tpidr_el0(&mut self, value: u64) {
        self.write(TPIDR, value, |state| state.set_tpidr_el0(value));
    }
    #[must_use]
    #[inline]
    pub fn tpidrro_el0(&self) -> u64 {
        self.read(TPIDRRO, A64State::tpidrro_el0)
    }
    #[inline]
    pub fn set_tpidrro_el0_from_runtime(&mut self, value: u64) {
        self.write(TPIDRRO, value, |state| {
            state.set_tpidrro_el0_from_runtime(value)
        });
    }
    #[must_use]
    pub fn register_context(&self) -> RegisterContext {
        // Diagnostic consumers request this explicitly; ordinary SVCs do not.
        let x = std::array::from_fn(|i| {
            self.read_x(A64Register::General(
                nixe_cpu::state::a64::A64GeneralRegister::new(i as u8).unwrap(),
            ))
        });
        RegisterContext {
            x,
            sp: self.read_x(A64Register::StackPointer),
            pc: GuestVirtualAddress::new(self.pc()),
            nzcv: self.nzcv(),
        }
    }
}
impl std::fmt::Debug for ThreadState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.storage {
            Storage::Unavailable => f.write_str("Unavailable"),
            _ => f
                .debug_tuple(if self.is_resident() {
                    "Native"
                } else {
                    "Saved"
                })
                .field(&self.snapshot())
                .finish(),
        }
    }
}
impl Clone for ThreadState {
    // Cloning is an explicit immutable snapshot, never a second native owner.
    fn clone(&self) -> Self {
        self.snapshot().into()
    }
}
impl PartialEq for ThreadState {
    fn eq(&self, other: &Self) -> bool {
        self.snapshot() == other.snapshot()
    }
}
impl Eq for ThreadState {}
