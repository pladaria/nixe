//! Bounded executable snapshots, independent of a particular translator.

use super::{CodePageDependency, FetchedCode, InstructionMemory};
use crate::error::InstructionFetchFault;
use nixe_memory::{
    AddressSpaceId, ContentGeneration, ExecutableObservation, GuestVirtualAddress,
    MemoryInvalidationCursor,
};
use std::num::NonZeroU16;

/// Memory authority capable of copying a coherent demanded instruction image.
pub trait ExecutableMemory: InstructionMemory {
    /// The caller holds no execution lease, JIT epoch or memory lock. `stop`
    /// only classifies the copied word; it must not call memory or retain state
    /// across calls (tracking or device reconciliation may restart capture). No word after
    /// the first stop is fetched. A fetch fault retains the preceding prefix.
    /// Arming dirty tracking can stop a bound engine. Publication must validate
    /// the captured dependencies through that engine's mutation coordinator.
    fn capture_instructions(
        &self,
        space: AddressSpaceId,
        start: GuestVirtualAddress,
        limit: NonZeroU16,
        stop: &dyn Fn(GuestVirtualAddress, u32) -> bool,
    ) -> InstructionImage;

    /// Cold revalidation, not a substitute for coordinating mutations with
    /// publication. Checks the memory owner and exact captured mappings/content
    /// stamps; an unrelated invalidation elsewhere does not stale this image.
    /// Translators without pinned, coordinator-validated input units must also
    /// guard the check-to-publication race with the captured invalidation cursor.
    fn image_is_current(&self, image: &InstructionImage) -> bool;
}

/// Owned immutable bytes and dependencies. Translators never reread live code
/// while decoding/lowering this image. The memory stamp remains opaque to them.
pub struct InstructionImage {
    pub(super) space: AddressSpaceId,
    pub(super) start: GuestVirtualAddress,
    pub(super) words: Box<[FetchedCode<u32>]>,
    pub(super) fault: Option<InstructionFetchFault>,
    pub(super) cursor: MemoryInvalidationCursor,
    pub(super) pages: Vec<Page>,
    pub(super) owner: std::sync::Arc<nixe_memory::MemoryInvalidationLog>,
}

pub(super) struct Page {
    pub address: GuestVirtualAddress,
    pub dependency: CodePageDependency,
    pub stamp: Stamp,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Stamp {
    Synthetic(ContentGeneration),
    Canonical(ExecutableObservation),
}

impl InstructionImage {
    /// First demanded instruction address.
    pub fn start(&self) -> GuestVirtualAddress {
        self.start
    }
    /// Captured words, in guest address order.
    pub fn words(&self) -> &[FetchedCode<u32>] {
        &self.words
    }
    /// Fault at the first demanded but unreadable word, if any.
    pub fn fault(&self) -> Option<&InstructionFetchFault> {
        self.fault.as_ref()
    }
    /// Publication revalidation cursor captured under memory protection.
    pub fn cursor(&self) -> MemoryInvalidationCursor {
        self.cursor
    }
    /// Exact virtual/physical mapping dependencies, including alias distinctions.
    pub fn dependencies(&self) -> impl Iterator<Item = CodePageDependency> + '_ {
        self.pages.iter().map(|page| page.dependency)
    }
}

pub(super) fn copy_words(
    start: GuestVirtualAddress,
    limit: NonZeroU16,
    stop: &dyn Fn(GuestVirtualAddress, u32) -> bool,
    mut fetch: impl FnMut(GuestVirtualAddress) -> Result<FetchedCode<u32>, InstructionFetchFault>,
) -> (Box<[FetchedCode<u32>]>, Option<InstructionFetchFault>) {
    let mut words = Vec::with_capacity(usize::from(limit.get()));
    let mut pc = start;
    for _ in 0..limit.get() {
        match fetch(pc) {
            Ok(word) => {
                words.push(word);
                if stop(pc, word.bits) {
                    break;
                }
            }
            Err(fault) => return (words.into_boxed_slice(), Some(fault)),
        }
        pc = GuestVirtualAddress::new(pc.get().wrapping_add(4));
    }
    (words.into_boxed_slice(), None)
}
