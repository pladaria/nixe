//! Guest-thread return predictions, independent of vCPU PIC ownership.
//! Entries contain guest keys only: retaining this state never pins native code.

use crate::abi::{BlockKey, FpSpecialization};
use nixe_cpu::platform::TargetPlatform;

pub(crate) const CAPACITY: usize = 16;

/// Scalar encoding for native loads/stores; no dependency on Rust enum layout.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct Continuation {
    pub pc: u64,
    pub address_space: u64,
    pub profile: u64,
    pub platform: u64,
    pub fp: u64,
}

impl From<BlockKey> for Continuation {
    fn from(key: BlockKey) -> Self {
        Self {
            pc: key.pc.get(),
            address_space: key.address_space.get(),
            profile: key.profile.get(),
            platform: match key.platform {
                TargetPlatform::Switch1 => 0,
                TargetPlatform::Switch2 => 1,
            },
            fp: match key.fp {
                FpSpecialization::Dynamic => 0,
                FpSpecialization::Exact(fpcr) => (1_u64 << 32) | u64::from(fpcr),
            },
        }
    }
}

/// Owned by the scheduled guest thread, never by `JitThread` (a process/vCPU).
/// The scheduler moves its allocation with architectural state and lends it
/// exclusively to one execution lease. Cloning copies predictions, not owners.
#[repr(C)]
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ReturnStack {
    pub(crate) entries: [Continuation; CAPACITY],
    /// Next insertion slot, always in 0..16 (four significant bits).
    pub(crate) head: u32,
    /// Occupied entries, always in 0..=16 (five significant bits).
    pub(crate) depth: u32,
}

impl ReturnStack {
    /// A chain has one execution key. Changing it at canonical admission
    /// invalidates the whole prediction chain, not native code ownership.
    pub(crate) fn prepare(&mut self, key: BlockKey) {
        if self.depth != 0 {
            let top = self.entries[(self.head.wrapping_sub(1) & 15) as usize];
            let expected = Continuation::from(key);
            if top.address_space != expected.address_space
                || top.profile != expected.profile
                || top.platform != expected.platform
                || top.fp != expected.fp
            {
                self.clear();
            }
        }
    }

    /// Discard predictions after an external architectural-state edit. Old
    /// scalar entries need not be overwritten: depth alone defines membership.
    pub fn clear(&mut self) {
        self.head = 0;
        self.depth = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nixe_cpu::profile::{CpuProfileId, ProcessCpuContext};
    use nixe_memory::{AddressSpaceId, GuestVirtualAddress};
    use std::mem::offset_of;

    #[test]
    fn admission_keeps_continuations_but_clears_changed_execution_keys() {
        let cpu = ProcessCpuContext::new(TargetPlatform::Switch1, AddressSpaceId::new(1));
        let key = BlockKey::new(
            cpu,
            GuestVirtualAddress::new(0x1000),
            FpSpecialization::Dynamic,
        )
        .unwrap();
        for head in [0, 3, 15] {
            let mut stack = ReturnStack {
                entries: [Continuation::from(key); CAPACITY],
                head,
                depth: 16,
            };
            let before = stack.clone();
            stack.prepare(key.at(GuestVirtualAddress::new(0x2000)).unwrap());
            assert_eq!(stack, before); // A normal branch is not a context change.
            for changed in [
                BlockKey {
                    address_space: AddressSpaceId::new(2),
                    ..key
                },
                BlockKey {
                    profile: CpuProfileId::new(key.profile.get() + 1),
                    ..key
                },
                BlockKey {
                    platform: TargetPlatform::Switch2,
                    ..key
                },
                BlockKey {
                    fp: FpSpecialization::Exact(0),
                    ..key
                },
            ] {
                stack = before.clone();
                stack.prepare(changed);
                assert_eq!((stack.head, stack.depth), (0, 0));
            }
        }
    }

    #[test]
    fn frame_borrows_the_guest_stack_without_copying_it() {
        use crate::abi::{NativeFrame, PollBudget};
        use nixe_cpu::state::a64::A64State;
        let mut stack = ReturnStack::default();
        let pointer = std::ptr::from_mut(&mut stack);
        let mut state = A64State::default();
        {
            let frame = NativeFrame::new(&mut state, PollBudget::new(4096, 64).unwrap())
                .with_return_stack(&mut stack);
            assert_eq!(frame.return_stack, pointer);
            // Model native stores through the frame's exclusive borrow. No
            // other reference to the stack is used until the frame is dropped.
            unsafe {
                (*frame.return_stack).head = 1;
                (*frame.return_stack).depth = 1;
                (*frame.return_stack).entries[0].pc = 0x1234;
            }
        }
        assert_eq!(
            (stack.head, stack.depth, stack.entries[0].pc),
            (1, 1, 0x1234)
        );
    }

    #[test]
    fn return_stack_layout_and_full_key_encoding() {
        assert_eq!(size_of::<Continuation>(), 40);
        assert_eq!(offset_of!(Continuation, pc), 0);
        assert_eq!(offset_of!(Continuation, address_space), 8);
        assert_eq!(offset_of!(Continuation, profile), 16);
        assert_eq!(offset_of!(Continuation, platform), 24);
        assert_eq!(offset_of!(Continuation, fp), 32);
        assert_eq!(offset_of!(ReturnStack, head), 640);
        assert_eq!(offset_of!(ReturnStack, depth), 644);
        assert_eq!(size_of::<ReturnStack>(), 648);
        let cpu = ProcessCpuContext::new(TargetPlatform::Switch1, AddressSpaceId::new(1));
        let key = BlockKey::new(
            cpu,
            GuestVirtualAddress::new(0x1000),
            FpSpecialization::Dynamic,
        )
        .unwrap();
        let baseline = Continuation::from(key);
        for other in [
            BlockKey {
                pc: GuestVirtualAddress::new(0x2000),
                ..key
            },
            BlockKey {
                address_space: AddressSpaceId::new(2),
                ..key
            },
            BlockKey {
                profile: CpuProfileId::new(key.profile.get() + 1),
                ..key
            },
            BlockKey {
                platform: TargetPlatform::Switch2,
                ..key
            },
            BlockKey {
                fp: FpSpecialization::Exact(0),
                ..key
            },
            BlockKey {
                fp: FpSpecialization::Exact(u32::MAX),
                ..key
            },
        ] {
            assert_ne!(baseline, Continuation::from(other));
        }
        let mut stack = ReturnStack::default();
        stack.entries[15] = baseline;
        stack.head = 0;
        stack.depth = 16;
        let mut copied = stack.clone();
        copied.clear();
        assert_eq!((copied.head, copied.depth), (0, 0));
        assert_eq!(copied.entries[15], baseline);
        assert_eq!(stack.depth, 16);
    }
}
