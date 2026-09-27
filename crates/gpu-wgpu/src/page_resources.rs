//! Reverse index of immutable resource bindings by canonical page.

use std::collections::HashMap;

use nixe_gpu::{BackendResourceCreateInfo, BackendResourceHandle};
use nixe_memory::{CanonicalBackingRange, CanonicalPageId};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum PageBinding {
    Buffer {
        offset: u64,
        page_offset: u64,
        size: u64,
    },
    Image {
        binding: usize,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct PageResource {
    pub handle: BackendResourceHandle,
    pub binding: PageBinding,
}

#[derive(Default)]
pub(super) struct PageResources(HashMap<CanonicalPageId, Vec<PageResource>>);

impl PageResources {
    pub fn get(&self, page: CanonicalPageId) -> &[PageResource] {
        self.0.get(&page).map_or(&[], Vec::as_slice)
    }

    pub fn clear(&mut self) {
        self.0.clear();
    }

    pub fn insert(&mut self, handle: BackendResourceHandle, info: &BackendResourceCreateInfo) {
        visit_bindings(info, |range, image| {
            let mut offset = 0;
            for segment in range.segments() {
                let binding = match image {
                    Some(binding) => PageBinding::Image { binding },
                    None => PageBinding::Buffer {
                        offset,
                        page_offset: segment.offset(),
                        size: segment.size(),
                    },
                };
                let entry = PageResource { handle, binding };
                let entries = self.0.entry(segment.page()).or_default();
                if !entries.contains(&entry) {
                    entries.push(entry);
                }
                offset += segment.size();
            }
        });
    }

    pub fn remove(&mut self, handle: BackendResourceHandle, info: &BackendResourceCreateInfo) {
        visit_bindings(info, |range, _| {
            for segment in range.segments() {
                if let Some(entries) = self.0.get_mut(&segment.page()) {
                    entries.retain(|entry| entry.handle != handle);
                    if entries.is_empty() {
                        self.0.remove(&segment.page());
                    }
                }
            }
        });
    }
}

fn visit_bindings(
    info: &BackendResourceCreateInfo,
    mut visit: impl FnMut(&CanonicalBackingRange, Option<usize>),
) {
    match info {
        BackendResourceCreateInfo::Buffer {
            view: Some(view), ..
        } => visit(view.backing().range(), None),
        BackendResourceCreateInfo::Image {
            view: Some(view), ..
        } => {
            for (index, binding) in view.bindings().iter().enumerate() {
                visit(binding.backing().range(), Some(index));
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nixe_gpu::{
        BackendInstanceId, BackendResourceKind, BackingView, BufferDescription, BufferId,
        BufferView, GpuAllocationDescription, GpuAllocationId,
    };
    use nixe_memory::{CanonicalAllocation, MemoryPermissions};

    #[test]
    fn index_keeps_exact_page_spans_and_separates_slot_generations() {
        let allocation = CanonicalAllocation::zeroed(0x3000, 0x1000).unwrap();
        let range = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap()
            .snapshot_subrange(0x800, 0x1800)
            .unwrap();
        let first_page = range.segments()[0].page();
        let second_page = range.segments()[1].page();
        let description = BufferDescription::new(0x1900).unwrap();
        let backing = BackingView::new(
            GpuAllocationId::new(1),
            GpuAllocationDescription::new(0x3000, 4).unwrap(),
            0x800,
            range,
        )
        .unwrap();
        let info = BackendResourceCreateInfo::Buffer {
            id: BufferId::new(1),
            description,
            view: Some(BufferView::new(BufferId::new(1), description, 0x100, backing).unwrap()),
        };
        let old = BackendResourceHandle::new(
            BackendInstanceId::new(1),
            3,
            1,
            BackendResourceKind::Buffer,
        );
        let new = BackendResourceHandle::new(
            BackendInstanceId::new(1),
            3,
            2,
            BackendResourceKind::Buffer,
        );
        let mut index = PageResources::default();
        index.insert(old, &info);
        index.insert(new, &info);
        assert_eq!(
            index.get(first_page),
            &[
                PageResource {
                    handle: old,
                    binding: PageBinding::Buffer {
                        offset: 0,
                        page_offset: 0x800,
                        size: 0x800
                    }
                },
                PageResource {
                    handle: new,
                    binding: PageBinding::Buffer {
                        offset: 0,
                        page_offset: 0x800,
                        size: 0x800
                    }
                }
            ]
        );
        assert_eq!(
            index.get(second_page)[0].binding,
            PageBinding::Buffer {
                offset: 0x800,
                page_offset: 0,
                size: 0x1000
            }
        );
        index.remove(old, &info);
        assert_eq!(index.get(first_page).len(), 1);
        assert_eq!(index.get(first_page)[0].handle, new);
        assert_eq!(index.get(second_page).len(), 1);
        index.remove(new, &info);
        assert!(index.0.is_empty());
    }
}
