//! Process-wide negative-result storage. All identities are weak, generational
//! handles. Installation must validate evidence under the caller's state guard;
//! this index does not turn retained storage into valid compiler input.

use super::*;
use crate::executable::Cache;
use crate::lifetime::registry::{Registry, Slot};
use crate::sampling::BoundaryKey;
use std::collections::HashSet;
use std::hash::{BuildHasher, RandomState};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(in crate::lifetime) struct Key {
    pub source: BlockKey,
    pub boundary: BoundaryKey,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::lifetime) enum Rejection {
    Unchanged,
    Disconnected,
    InstructionLimit,
    BackendRejected,
}

/// Participant generations and inspected LCQ inputs use their unit identity;
/// endpoint publication/slot reuse uses the exact dispatch identity.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(in crate::lifetime) enum Owner {
    Unit(UnitHandle),
    Dispatch(Handle<DispatchSlot>),
    // A no-op preserves one family's entire membership. Its required entry
    // set also depends on incoming roots, independently of the code version.
    Entries(UnitHandle),
    // Includes currently unowned PCs; demand and ownership affect selection.
    SelectionPage(SelectionPage),
    // Backend shape also depends on incoming roots, including LCQ-only pages.
    EntryPage(SelectionPage),
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(in crate::lifetime) struct SelectionPage {
    space: nixe_memory::AddressSpaceId,
    page: u64,
}

impl SelectionPage {
    pub fn span(key: BlockKey, instructions: usize) -> impl ExactSizeIterator<Item = Self> {
        let start = Self::of(key);
        let size = nixe_memory::DIRECT_PAGE_SIZE as u64;
        let count = if instructions == 0 {
            0
        } else {
            (key.pc.get() % size + instructions as u64 * 4).div_ceil(size) as usize
        };
        (0..count).map(move |offset| Self {
            space: start.space,
            page: start.page + offset as u64,
        })
    }

    pub fn of(key: BlockKey) -> Self {
        Self {
            space: key.address_space,
            page: key.pc.get() / nixe_memory::DIRECT_PAGE_SIZE as u64,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Link {
    record: Handle<Box<Record>>,
    association: usize,
}

struct Association {
    owner: Owner,
    previous: Option<Link>,
    next: Option<Link>,
}

#[derive(Debug)]
struct Head {
    owner: Owner,
    first: Link,
}

pub(in crate::lifetime) struct Record {
    key: Key,
    pub reason: Rejection,
    pub cursor: MemoryInvalidationCursor,
    associations: Box<[Association]>,
    // Intrusive garbage queue: invalidation allocates nothing under JIT state.
    removed_next: Option<Box<Record>>,
    // Storage drops before returning its charge, outside JIT state.
    _charge: MetadataLease,
}

impl Record {
    /// Called outside state. Deduplicate evidence, never retain code or graphs.
    pub fn prepare(
        cache: &Arc<Cache>,
        key: Key,
        reason: Rejection,
        cursor: MemoryInvalidationCursor,
        owners: impl IntoIterator<Item = Owner>,
    ) -> Result<Box<Self>, Error> {
        Self::prepare_reserved(
            cache.charge_metadata(size_of::<Self>(), Tier::Hcq)?,
            key,
            reason,
            cursor,
            owners,
        )
    }

    pub fn prepare_reserved(
        mut charge: MetadataLease,
        key: Key,
        reason: Rejection,
        cursor: MemoryInvalidationCursor,
        owners: impl IntoIterator<Item = Owner>,
    ) -> Result<Box<Self>, Error> {
        let mut seen = HashSet::new();
        let associations: Box<[_]> = owners
            .into_iter()
            .filter(|owner| seen.insert(*owner))
            .map(|owner| Association {
                owner,
                previous: None,
                next: None,
            })
            .collect();
        if associations.is_empty() {
            return Err(Error::InvalidUnit(
                "negative result needs invalidation evidence",
            ));
        }
        charge.grow(size_of_val(&*associations), Tier::Hcq)?;
        Ok(Box::new(Self {
            key,
            reason,
            cursor,
            associations,
            removed_next: None,
            _charge: charge,
        }))
    }
}

impl Drop for Record {
    fn drop(&mut self) {
        // Teardown may own a long garbage chain; do not recurse on the stack.
        let mut next = self.removed_next.take();
        while let Some(mut record) = next {
            next = record.removed_next.take();
        }
    }
}

pub(in crate::lifetime) struct Storage {
    records: Vec<Slot<Box<Record>>>,
    keys: hashbrown::HashTable<Handle<Box<Record>>>,
    heads: hashbrown::HashTable<Head>,
    charge: Option<MetadataLease>,
}

impl Storage {
    /// Prepare actual requested capacity outside state. No worst-case table or
    /// dependency array is allocated for an idle process or for each worker.
    pub fn prepare(cache: &Arc<Cache>, records: usize, owners: usize) -> Result<Self, Error> {
        let records = Vec::with_capacity(records);
        let keys = hashbrown::HashTable::with_capacity(records.capacity());
        let heads = hashbrown::HashTable::with_capacity(owners);
        let bytes = records
            .capacity()
            .checked_mul(size_of::<Slot<Box<Record>>>())
            .and_then(|bytes| bytes.checked_add(keys.allocation_size()))
            .and_then(|bytes| bytes.checked_add(heads.allocation_size()))
            .ok_or(Error::Capacity("negative index size overflow"))?;
        let charge = Some(cache.charge_metadata(bytes, Tier::Hcq)?);
        Ok(Self {
            records,
            keys,
            heads,
            charge,
        })
    }
}

pub(in crate::lifetime) struct Index {
    records: Registry<Box<Record>>,
    keys: hashbrown::HashTable<Handle<Box<Record>>>,
    heads: hashbrown::HashTable<Head>,
    hash: RandomState,
    removed: Option<Box<Record>>,
    charge: Option<MetadataLease>,
    // Running reshape workers own these slots, not visible negative results.
    reserved: usize,
}

impl Index {
    /// Cold publication/unlink event. Other semantic specializations on the
    /// same page conservatively retry, with no per-instruction watch table.
    pub(in crate::lifetime) fn invalidate_selection(&mut self, page: SelectionPage) {
        if !self.heads.is_empty() {
            self.invalidate(Owner::SelectionPage(page));
        }
    }

    pub fn new() -> Self {
        Self {
            records: Registry::default(),
            keys: hashbrown::HashTable::new(),
            heads: hashbrown::HashTable::new(),
            hash: RandomState::new(),
            removed: None,
            charge: None,
            reserved: 0,
        }
    }

    /// One header/key slot per worker, not a worst-case graph/evidence array.
    /// The live-record count comes from the key table: no registry/free-list scan.
    pub fn reservation_growth(&self) -> Result<Option<(usize, usize)>, Error> {
        let needed = self
            .keys
            .len()
            .checked_add(self.reserved)
            .and_then(|count| count.checked_add(1))
            .ok_or(Error::Capacity("negative reservation count overflow"))?;
        if needed <= self.records.capacity() && needed <= self.keys.capacity() {
            return Ok(None);
        }
        let records = self
            .records
            .capacity()
            .max(self.keys.capacity())
            .checked_mul(2)
            .unwrap_or(needed)
            .max(needed)
            .max(4);
        Ok(Some((records, self.heads.capacity())))
    }

    /// Caller owns a charged header before reserving. Evidence storage is added
    /// after discovery, before compilation; it is not guessed up front.
    pub fn reserve_record(&mut self) -> Result<(), Error> {
        if self.reservation_growth()?.is_some() {
            return Err(Error::Capacity("negative result slot must be prepared"));
        }
        self.records.next_handle()?;
        self.reserved += 1;
        Ok(())
    }

    pub fn release_record(&mut self) {
        assert_ne!(self.reserved, 0);
        self.reserved -= 1;
    }

    /// Keep handles/backlinks stable. Return replaced storage in `spare` for
    /// destruction after unlocking. A stale capacity plan mutates nothing.
    pub fn grow(&mut self, spare: &mut Storage) -> Result<(), Error> {
        if spare.records.capacity() < self.records.capacity()
            || spare.keys.capacity() < self.keys.capacity()
            || spare.heads.capacity() < self.heads.capacity()
            || !spare.records.is_empty()
            || !spare.keys.is_empty()
            || !spare.heads.is_empty()
        {
            return Err(Error::Capacity("negative index capacity plan is stale"));
        }
        self.records.grow(&mut spare.records);
        for handle in self.keys.drain() {
            spare.keys.insert_unique(
                self.hash.hash_one(self.records.get(handle).unwrap().key),
                handle,
                |handle| self.hash.hash_one(self.records.get(*handle).unwrap().key),
            );
        }
        for head in self.heads.drain() {
            spare
                .heads
                .insert_unique(self.hash.hash_one(head.owner), head, |head| {
                    self.hash.hash_one(head.owner)
                });
        }
        std::mem::swap(&mut self.keys, &mut spare.keys);
        std::mem::swap(&mut self.heads, &mut spare.heads);
        std::mem::swap(&mut self.charge, &mut spare.charge);
        Ok(())
    }

    pub fn get(&self, key: Key) -> Option<&Record> {
        let handle = self.keys.find(self.hash.hash_one(key), |handle| {
            self.records.get(*handle).unwrap().key == key
        })?;
        self.records.get(*handle).map(|record| &**record)
    }

    pub fn evidence_growth(&self, record: &Record) -> Result<Option<(usize, usize)>, Error> {
        let missing = record
            .associations
            .iter()
            .filter(|association| self.head(association.owner).is_none())
            .count();
        let needed = self
            .heads
            .len()
            .checked_add(missing)
            .ok_or(Error::Capacity("negative owner count overflow"))?;
        Ok((needed > self.heads.capacity()).then(|| {
            (
                self.records.capacity(),
                needed.max(self.heads.capacity().saturating_mul(2)).max(4),
            )
        }))
    }

    /// Caller revalidates all evidence first. False means a matching result
    /// already exists; neither duplicates nor pressure evict a valid negative.
    /// Failure leaves the prepared owner with the caller for out-of-lock drop.
    pub fn insert(&mut self, prepared: &mut Option<Box<Record>>) -> Result<bool, Error> {
        self.insert_inner(prepared, false)
    }

    pub fn insert_reserved(&mut self, prepared: &mut Option<Box<Record>>) -> Result<bool, Error> {
        assert_ne!(self.reserved, 0);
        let inserted = self.insert_inner(prepared, true)?;
        if inserted {
            self.release_record();
        }
        Ok(inserted)
    }

    fn insert_inner(
        &mut self,
        prepared: &mut Option<Box<Record>>,
        reserved: bool,
    ) -> Result<bool, Error> {
        let record = prepared.as_ref().expect("prepared negative owner");
        if self.get(record.key).is_some() {
            return Ok(false);
        }
        // Unreserved installation cannot steal a running worker's result slot.
        if !reserved && self.reservation_growth()?.is_some() {
            return Err(Error::Capacity("negative result slots are reserved"));
        }
        self.records.check_insertions(1)?;
        let missing = record
            .associations
            .iter()
            .filter(|association| self.head(association.owner).is_none())
            .count();
        if self.keys.len() == self.keys.capacity()
            || missing > self.heads.capacity() - self.heads.len()
        {
            return Err(Error::Capacity("negative index storage must be prepared"));
        }
        let key = record.key;
        let count = record.associations.len();
        let handle = self.records.insert(prepared)?;
        // No fallible operation or allocation after the first mutation.
        self.keys
            .insert_unique(self.hash.hash_one(key), handle, |handle| {
                self.hash.hash_one(self.records.get(*handle).unwrap().key)
            });
        for index in 0..count {
            let link = Link {
                record: handle,
                association: index,
            };
            let owner = self.association(link).owner;
            let next = self.head(owner);
            self.association_mut(link).next = next;
            if let Some(next) = next {
                self.association_mut(next).previous = Some(link);
            }
            if let Some(head) = self
                .heads
                .find_mut(self.hash.hash_one(owner), |head| head.owner == owner)
            {
                head.first = link;
            } else {
                self.heads.insert_unique(
                    self.hash.hash_one(owner),
                    Head { owner, first: link },
                    |head| self.hash.hash_one(head.owner),
                );
            }
        }
        Ok(true)
    }

    /// Visits only affected records. Each reverse association detaches in O(1),
    /// so total work is proportional to their evidence, not all cached results.
    pub fn invalidate(&mut self, owner: Owner) {
        if self.heads.is_empty() {
            return;
        }
        while let Some(link) = self.head(owner) {
            self.remove(link.record);
        }
    }

    pub fn invalidate_unit(&mut self, unit: UnitHandle) {
        self.invalidate(Owner::Unit(unit));
        self.invalidate(Owner::Entries(unit));
    }

    /// History loss and shutdown discard all evidence. Drain each hash table
    /// once, rather than repeatedly searching a sparse table for its next key.
    pub fn invalidate_all(&mut self) {
        self.heads.clear();
        for handle in self.keys.drain() {
            let mut removed = self.records.remove(handle).unwrap();
            removed.removed_next = self.removed.take();
            self.removed = Some(removed);
        }
    }

    fn remove(&mut self, handle: Handle<Box<Record>>) {
        let record = self.records.get(handle).unwrap();
        let key = record.key;
        let count = record.associations.len();
        for association in 0..count {
            let node = self.association(Link {
                record: handle,
                association,
            });
            let (owner, previous, next) = (node.owner, node.previous, node.next);
            if let Some(previous) = previous {
                self.association_mut(previous).next = next;
            } else if let Some(next) = next {
                self.heads
                    .find_mut(self.hash.hash_one(owner), |head| head.owner == owner)
                    .unwrap()
                    .first = next;
            } else {
                self.heads
                    .find_entry(self.hash.hash_one(owner), |head| head.owner == owner)
                    .unwrap()
                    .remove();
            }
            if let Some(next) = next {
                self.association_mut(next).previous = previous;
            }
        }
        self.keys
            .find_entry(self.hash.hash_one(key), |value| *value == handle)
            .unwrap()
            .remove();
        let mut removed = self.records.remove(handle).unwrap();
        removed.removed_next = self.removed.take();
        self.removed = Some(removed);
    }

    /// Take under state; destroy the returned owner after unlocking. Charges
    /// remain live until the real storage is freed, including during pressure.
    pub fn take_removed(&mut self) -> Option<Box<Record>> {
        self.removed.take()
    }

    fn head(&self, owner: Owner) -> Option<Link> {
        self.heads
            .find(self.hash.hash_one(owner), |head| head.owner == owner)
            .map(|head| head.first)
    }
    fn association(&self, link: Link) -> &Association {
        &self.records.get(link.record).unwrap().associations[link.association]
    }
    fn association_mut(&mut self, link: Link) -> &mut Association {
        &mut self.records.get_mut(link.record).unwrap().associations[link.association]
    }
}

impl Units {
    /// Point lookup for a changed incoming root/demand. No per-instruction
    /// negative associations: no-op evidence covers the whole current family;
    /// backend entry shape uses page watches even without an HCQ owner.
    pub(in crate::lifetime) fn invalidate_entry_negatives(&mut self, key: BlockKey) {
        if self.negatives.heads.is_empty() {
            return;
        }
        self.negatives
            .invalidate(Owner::EntryPage(SelectionPage::of(key)));
        let Some(family) = InstructionKey::new(key).and_then(|key| self.family_owners.get(key))
        else {
            return;
        };
        let unit = *self
            .families
            .get(family)
            .unwrap()
            .unit
            .registration
            .get()
            .unwrap();
        self.negatives.invalidate(Owner::Entries(unit));
    }
}

#[cfg(test)]
mod tests;
