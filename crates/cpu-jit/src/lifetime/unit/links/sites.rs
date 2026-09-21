//! Target-keyed static source discovery. Associations are weak generational
//! identities, not callable roots. They survive target withdrawal/replacement
//! and are removed when their source is unlinked, even if a compiler pins it.

use super::*;
use crate::abi::BlockKey;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::lifetime) struct StaticSite {
    pub target: BlockKey,
    pub source: UnitHandle,
    pub state_map: u32,
    pub island: usize,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct SiteHandle {
    pub source: UnitHandle,
    pub island: usize,
}

/// One per static exit, owned/accounted by its source UnitRecord. State maps
/// and target keys stay in immutable code. The array ordinal is its island.
pub(in crate::lifetime) struct SourceSite {
    pub(super) state_map: u32,
    pub(super) link: Option<H>,
    // The branch currently callable in native code. Usually equal to `link`;
    // publication may attach one newer pending record without releasing it.
    pub(super) callable: Option<H>,
    prev: Option<SiteHandle>,
    next: Option<SiteHandle>,
}

#[derive(Debug)]
pub(in crate::lifetime) struct TargetSources {
    key: BlockKey,
    head: SiteHandle,
}

pub(in crate::lifetime) struct StaticSites {
    pub entries: hashbrown::HashTable<TargetSources>,
    hash: RandomState,
}

impl StaticSites {
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: hashbrown::HashTable::with_capacity(capacity),
            hash: RandomState::new(),
        }
    }

    pub fn insert(&mut self, sources: TargetSources) {
        self.entries
            .insert_unique(self.hash.hash_one(sources.key), sources, |sources| {
                self.hash.hash_one(sources.key)
            });
    }
}

impl Input {
    pub(in crate::lifetime) fn source_sites(&self) -> Box<[SourceSite]> {
        self.states
            .iter()
            .enumerate()
            .filter_map(|(state_map, state)| {
                state.transfer.as_ref()?.static_target?;
                Some(SourceSite {
                    state_map: state_map as u32,
                    link: None,
                    callable: None,
                    prev: None,
                    next: None,
                })
            })
            .collect()
    }
}

impl Units {
    /// Includes pending/uninstalled static associations, not just callable links.
    /// Partial LCQ inclusion is decided by the logical exit instruction.
    pub(in crate::lifetime) fn has_external_static_source(
        &self,
        target: BlockKey,
        contains: impl Fn(InstructionKey) -> bool,
    ) -> bool {
        self.static_sources(target).any(|site| {
            let code = &self.records.get(site.source.0).unwrap().code;
            let exit = code.states[site.state_map as usize].exit.unwrap();
            let source = code
                .instructions
                .get(0)
                .unwrap()
                .key
                .block_key()
                .at(exit.pc)
                .unwrap();
            !contains(InstructionKey::new(source).unwrap())
        })
    }

    pub(super) fn static_source_head(&self, target: BlockKey) -> Option<SiteHandle> {
        self.static_sites
            .entries
            .find(self.static_sites.hash.hash_one(target), |sources| {
                sources.key == target
            })
            .map(|sources| sources.head)
    }

    pub(super) fn next_static_source(&self, handle: SiteHandle) -> Option<SiteHandle> {
        self.source_site(handle).next
    }

    fn source_site(&self, handle: SiteHandle) -> &SourceSite {
        &self.records.get(handle.source.0).unwrap().static_sites[handle.island]
    }

    fn source_site_mut(&mut self, handle: SiteHandle) -> &mut SourceSite {
        &mut self
            .records
            .get_mut(handle.source.0)
            .unwrap()
            .static_sites
            .value[handle.island]
    }

    fn source_target(&self, handle: SiteHandle) -> BlockKey {
        let record = self.records.get(handle.source.0).unwrap();
        record.code.states[record.static_sites[handle.island].state_map as usize]
            .transfer
            .as_ref()
            .unwrap()
            .static_target
            .unwrap()
    }

    /// Inspect only associations for this complete execution key. The caller
    /// holds JIT state; acquiring roots still requires publication/link
    /// revalidation. Invalidating sources must not acquire new link work.
    pub(in crate::lifetime) fn static_sources(
        &self,
        target: BlockKey,
    ) -> impl Iterator<Item = StaticSite> + '_ {
        let head = self.static_source_head(target);
        std::iter::successors(head, |handle| self.source_site(*handle).next).filter_map(
            move |handle| {
                let record = self.records.get(handle.source.0).unwrap();
                (matches!(
                    record.lifecycle,
                    Lifecycle::Published | Lifecycle::Superseded
                ) && record.retirement.is_none())
                .then(|| StaticSite {
                    target,
                    source: handle.source,
                    state_map: self.source_site(handle).state_map,
                    island: handle.island,
                })
            },
        )
    }

    pub(in crate::lifetime) fn insert_static_source(&mut self, source: UnitHandle) {
        for island in 0..self.records.get(source.0).unwrap().static_sites.len() {
            let handle = SiteHandle { source, island };
            let key = self.source_target(handle);
            let hash = self.static_sites.hash.hash_one(key);
            let next = match self
                .static_sites
                .entries
                .find_mut(hash, |sources| sources.key == key)
            {
                Some(sources) => Some(std::mem::replace(&mut sources.head, handle)),
                None => {
                    self.static_sites
                        .insert(TargetSources { key, head: handle });
                    None
                }
            };
            self.source_site_mut(handle).next = next;
            if let Some(next) = next {
                self.source_site_mut(next).prev = Some(handle);
            }
        }
    }

    pub(in crate::lifetime) fn remove_static_source(&mut self, source: UnitHandle) {
        // O(source exits), including high-fan-in destinations. Removing a
        // known association never searches the other sources in its bucket.
        for island in 0..self.records.get(source.0).unwrap().static_sites.len() {
            let handle = SiteHandle { source, island };
            let site = self.source_site_mut(handle);
            debug_assert!(
                site.link.is_none() && site.callable.is_none(),
                "source links must be detached before discovery"
            );
            let prev = site.prev.take();
            let next = site.next.take();
            if let Some(prev) = prev {
                self.source_site_mut(prev).next = next;
            } else {
                let key = self.source_target(handle);
                let hash = self.static_sites.hash.hash_one(key);
                if let Some(next) = next {
                    self.static_sites
                        .entries
                        .find_mut(hash, |sources| sources.key == key)
                        .unwrap()
                        .head = next;
                } else {
                    self.static_sites
                        .entries
                        .find_entry(hash, |sources| sources.key == key)
                        .unwrap()
                        .remove();
                }
            }
            if let Some(next) = next {
                self.source_site_mut(next).prev = prev;
            }
        }
    }
}
