//! Direct GPU page entries protected by the address space mapping lock.
const LEAF_SHIFT: u32 = 21;
const PAGE_SHIFT: u32 = 12;
const LEAF_PAGES: usize = 1 << (LEAF_SHIFT - PAGE_SHIFT);

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Entry {
    pub allocation: nixe_memory::AllocationId,
    pub physical_page: u32,
    pub slot: u32,
}
type Leaf = [Entry; LEAF_PAGES];
#[derive(Debug)]
pub(super) struct PageTable {
    leaves: Box<[Option<Box<Leaf>>]>,
}
impl PageTable {
    pub fn new(address_bits: u8) -> Self {
        Self {
            leaves: (0..1usize << (u32::from(address_bits) - LEAF_SHIFT))
                .map(|_| None)
                .collect(),
        }
    }
    #[inline]
    pub fn get(&self, address: u64) -> Option<Entry> {
        let leaf = self
            .leaves
            .get((address >> LEAF_SHIFT) as usize)?
            .as_ref()?;
        let entry = leaf[((address >> PAGE_SHIFT) as usize) & (LEAF_PAGES - 1)];
        (entry.slot != 0).then_some(entry)
    }
    pub fn map(&mut self, address: u64, bytes: u64, mut entry: Entry) {
        debug_assert_eq!((address | bytes) & ((1 << PAGE_SHIFT) - 1), 0);
        for offset in (0..bytes).step_by(1 << PAGE_SHIFT) {
            let at = address + offset;
            let leaf = self.leaves[(at >> LEAF_SHIFT) as usize]
                .get_or_insert_with(|| Box::new([Entry::default(); LEAF_PAGES]));
            leaf[((at >> PAGE_SHIFT) as usize) & (LEAF_PAGES - 1)] = entry;
            entry.physical_page += 1;
        }
    }
    pub fn unmap(&mut self, address: u64, bytes: u64) {
        for offset in (0..bytes).step_by(1 << PAGE_SHIFT) {
            let at = address + offset;
            if let Some(leaf) = &mut self.leaves[(at >> LEAF_SHIFT) as usize] {
                leaf[((at >> PAGE_SHIFT) as usize) & (LEAF_PAGES - 1)] = Entry::default();
            }
        }
    }
    pub fn metadata_bytes(&self) -> usize {
        std::mem::size_of_val(self.leaves.as_ref())
            + self.leaves.iter().filter(|leaf| leaf.is_some()).count() * std::mem::size_of::<Leaf>()
    }
}
