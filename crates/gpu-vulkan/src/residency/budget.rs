//! Per-cache frame collectors share accounting for actual native allocations.
use super::*;
use std::time::{Duration, Instant};

/// Both actual cache owners keep their LRU links in their own entries.
pub(super) trait CachedEntry {
    fn contents(&self) -> &Contents;
    fn contents_mut(&mut self) -> &mut Contents;
}
impl CachedEntry for BufferEntry {
    fn contents(&self) -> &Contents {
        &self.contents
    }
    fn contents_mut(&mut self) -> &mut Contents {
        &mut self.contents
    }
}
impl CachedEntry for ImageEntry {
    fn contents(&self) -> &Contents {
        &self.contents
    }
    fn contents_mut(&mut self) -> &mut Contents {
        &mut self.contents
    }
}
#[derive(Default)]
pub(super) struct ResourceLru {
    head: Option<SlotId>,
    tail: Option<SlotId>,
    scan: Option<SlotId>,
}
impl ResourceLru {
    pub(super) fn unlink<T: CachedEntry>(&mut self, pool: &mut SlotVector<T>, id: SlotId) {
        let Some(links) = pool[id].contents_mut().lru.take() else {
            return;
        };
        if self.scan == Some(id) {
            self.scan = links.next;
        }
        if let Some(previous) = links.previous {
            pool[previous].contents_mut().lru.as_mut().unwrap().next = links.next;
        } else {
            self.head = links.next;
        }
        if let Some(next) = links.next {
            pool[next].contents_mut().lru.as_mut().unwrap().previous = links.previous;
        } else {
            self.tail = links.previous;
        }
    }
    fn touch<T: CachedEntry>(&mut self, pool: &mut SlotVector<T>, id: SlotId, epoch: u64) {
        let contents = pool[id].contents();
        if contents.lru.is_some() && contents.touched == epoch {
            return;
        }
        self.unlink(pool, id);
        let previous = self.tail;
        let contents = pool[id].contents_mut();
        contents.touched = epoch;
        contents.lru = Some(LruLinks {
            previous,
            next: None,
        });
        if let Some(previous) = previous {
            pool[previous].contents_mut().lru.as_mut().unwrap().next = Some(id);
        } else {
            self.head = Some(id);
        }
        self.tail = Some(id);
    }
}
impl Residency<'_> {
    fn cache_lru(&self, image: bool) -> &ResourceLru {
        if image {
            &self.texture_cache.lru
        } else {
            &self.buffer_cache.lru
        }
    }
    fn cache_lru_mut(&mut self, image: bool) -> &mut ResourceLru {
        if image {
            &mut self.texture_cache.lru
        } else {
            &mut self.buffer_cache.lru
        }
    }
    pub(super) fn unlink_lru(&mut self, resource: Resource) {
        match resource {
            Resource::Buffer(id) => {
                let cache = &mut *self.buffer_cache;
                cache.lru.unlink(&mut cache.buffers, id);
            }
            Resource::Image(id) => {
                let cache = &mut *self.texture_cache;
                cache.lru.unlink(&mut cache.images, id);
            }
        }
    }
    pub(super) fn touch(&mut self, resource: Resource) {
        let epoch = self.runtime.epoch;
        match resource {
            Resource::Buffer(id) => {
                let cache = &mut *self.buffer_cache;
                cache.lru.touch(&mut cache.buffers, id, epoch);
            }
            Resource::Image(id) => {
                let cache = &mut *self.texture_cache;
                cache.lru.touch(&mut cache.images, id, epoch);
            }
        }
    }
    fn evictable(&self, resource: Resource) -> bool {
        if self.runtime.constructing.contains(&resource) {
            return false;
        }
        let contents = self.contents(resource);
        if contents
            .last_use
            .is_some_and(|work| work.tick() > self.runtime.device.completed_tick())
        {
            return false;
        }
        match resource {
            Resource::Buffer(id) => Arc::strong_count(&self.buffer_cache.get(id).native) == 1,
            Resource::Image(id) => {
                let entry = self.texture_cache.get(id);
                entry
                    .views
                    .iter()
                    .all(|&id| Arc::strong_count(&self.texture_cache.views[id].native) == 1)
                    && Arc::strong_count(&entry.native) == 1 + entry.views.len()
            }
        }
    }
    fn evict_resource(&mut self, resource: Resource) -> Result<(), VulkanError> {
        // Eden's cache collectors download their modified storage before
        // removing the native entry; there is no physical producer graph.
        self.runtime.constructing.push(resource);
        let result = match resource {
            Resource::Buffer(id) => {
                let entry = self.buffer_cache.get(id);
                if let Some(address) = entry.device_address {
                    self.download_buffer_memory(address, entry.native.size())
                } else {
                    Ok(())
                }
            }
            Resource::Image(id) => self.download_image_memory(id),
        };
        self.unpin(resource);
        result?;
        self.retire_entry(resource)?;
        self.runtime.statistics.evictions += 1;
        Ok(())
    }
    /// Presentation epochs drive maintenance; headless work uses a 16 ms clock.
    /// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/buffer_cache/buffer_cache.h
    pub(crate) fn maintain_budget(&mut self, presentation: bool) -> Result<(), VulkanError> {
        let now = Instant::now();
        if !presentation
            && now.duration_since(self.runtime.maintenance_at) < Duration::from_millis(16)
        {
            return Ok(());
        }
        self.runtime.maintenance_at = now;
        self.texture_cache.retired.tick();
        self.runtime.epoch = self.runtime.epoch.saturating_add(1);
        let (available, usage) = self.runtime.device.local_memory_budget();
        let fixed = usage.saturating_sub(self.runtime.budget.used());
        self.runtime.budget.limit = (available * 8 / 10)
            .saturating_sub(fixed)
            .min(self.runtime.budget_ceiling);
        self.runtime.collection_target = (available * 7 / 10)
            .saturating_sub(fixed)
            .min(self.runtime.budget_ceiling / 8 * 7);
        if self.runtime.budget.used() <= self.runtime.collection_target {
            self.buffer_cache.lru.scan = None;
            self.texture_cache.lru.scan = None;
            return Ok(());
        }
        self.runtime.device.poll_progress()?;
        let hard = self.runtime.budget.used() > self.runtime.budget.limit;
        // Each cache owns its own frame collector and age threshold. Buffers
        // use 60/120 frames, textures 10/25 under critical/high pressure.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/texture_cache/texture_cache.h
        for image in [false, true] {
            if self.runtime.budget.used() <= self.runtime.collection_target {
                break;
            }
            let (limit, age) = match (image, hard) {
                (false, true) => (64, 60),
                (false, false) => (32, 120),
                (true, true) => (40, 10),
                (true, false) => (20, 25),
            };
            self.cache_lru_mut(image).scan =
                self.cache_lru(image).scan.or(self.cache_lru(image).head);
            for _ in 0..limit {
                let Some(id) = self.cache_lru(image).scan else {
                    break;
                };
                let resource = if image {
                    Resource::Image(id)
                } else {
                    Resource::Buffer(id)
                };
                let contents = self.contents(resource);
                let next = contents.lru.unwrap().next;
                if self.runtime.epoch.saturating_sub(contents.touched) < age {
                    self.cache_lru_mut(image).scan = None;
                    break;
                }
                self.cache_lru_mut(image).scan = next;
                if self.evictable(resource) {
                    self.evict_resource(resource)?;
                }
                if self.runtime.budget.used() <= self.runtime.collection_target {
                    break;
                }
            }
        }
        Ok(())
    }
}
