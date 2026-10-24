//! One frame-maintained LRU and budget for resident buffers and images.
use super::*;
use std::time::{Duration, Instant};

impl VulkanResidency {
    pub(super) fn unlink_lru(&mut self, resource: Resource) {
        let Some(links) = self.contents_mut(resource).lru.take() else {
            return;
        };
        if self.lru_scan == Some(resource) {
            self.lru_scan = links.next;
        }
        if let Some(previous) = links.previous {
            self.contents_mut(previous).lru.as_mut().unwrap().next = links.next;
        } else {
            self.lru_head = links.next;
        }
        if let Some(next) = links.next {
            self.contents_mut(next).lru.as_mut().unwrap().previous = links.previous;
        } else {
            self.lru_tail = links.previous;
        }
    }
    pub(super) fn touch(&mut self, resource: Resource) {
        let contents = self.contents(resource).expect("live LRU resource");
        if contents.lru.is_some() && contents.touched == self.epoch {
            return;
        }
        self.unlink_lru(resource);
        let previous = self.lru_tail;
        let epoch = self.epoch;
        let contents = self.contents_mut(resource);
        contents.touched = epoch;
        contents.lru = Some(LruLinks {
            previous,
            next: None,
        });
        if let Some(previous) = previous {
            self.contents_mut(previous).lru.as_mut().unwrap().next = Some(resource);
        } else {
            self.lru_head = Some(resource);
        }
        self.lru_tail = Some(resource);
    }
    fn evictable(&self, resource: Resource) -> bool {
        if self.constructing.contains(&resource) {
            return false;
        }
        let Ok(contents) = self.contents(resource) else {
            return false;
        };
        if contents
            .last_use
            .is_some_and(|work| work.tick() > self.device.completed_tick())
        {
            return false;
        }
        match resource {
            Resource::Buffer(id) => Arc::strong_count(&self.buffer(id).unwrap().native) == 1,
            Resource::Image(id) => {
                let entry = self.image(id).unwrap();
                entry
                    .native_views
                    .iter()
                    .all(|view| Arc::strong_count(view) == 1)
                    && Arc::strong_count(&entry.native) == 1 + entry.native_views.len()
            }
        }
    }
    fn evict_resource(&mut self, resource: Resource) -> Result<(), VulkanError> {
        // Preserve only still-observable newest bytes. Retirement notifications
        // already remove dead allocation coverage without downloading it.
        if !self.authority.can_evict(resource)? {
            self.constructing.push(resource);
            let result = self.preserve(resource);
            self.unpin(resource);
            result?;
        }
        self.authority.evict_materialized(&self.memory, resource)?;
        self.retire_entry(resource)?;
        self.statistics.evictions += 1;
        Ok(())
    }
    pub fn evict_one(&mut self) -> Result<bool, VulkanError> {
        let mut next = self.lru_head;
        while let Some(resource) = next {
            next = self.contents(resource)?.lru.unwrap().next;
            if self.evictable(resource) {
                self.evict_resource(resource)?;
                return Ok(true);
            }
        }
        Ok(false)
    }
    /// Presentation epochs drive maintenance; headless work uses a 16 ms clock.
    /// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/buffer_cache/buffer_cache.h
    pub(crate) fn maintain_budget(&mut self, presentation: bool) -> Result<(), VulkanError> {
        let now = Instant::now();
        if !presentation && now.duration_since(self.maintenance_at) < Duration::from_millis(16) {
            return Ok(());
        }
        self.maintenance_at = now;
        self.epoch = self.epoch.saturating_add(1);
        let (available, usage) = self.device.local_memory_budget();
        let fixed = usage.saturating_sub(self.budget.used());
        self.budget.limit = (available * 8 / 10)
            .saturating_sub(fixed)
            .min(self.budget_ceiling);
        self.collection_target = (available * 7 / 10)
            .saturating_sub(fixed)
            .min(self.budget_ceiling / 8 * 7);
        if self.budget.used() <= self.collection_target {
            self.lru_scan = None;
            return Ok(());
        }
        self.device.poll_progress()?;
        let hard = self.budget.used() > self.budget.limit;
        let (limit, age) = if hard { (64, 60) } else { (32, 120) };
        self.lru_scan = self.lru_scan.or(self.lru_head);
        for _ in 0..limit {
            let Some(resource) = self.lru_scan else { break };
            let contents = self.contents(resource)?;
            let next = contents.lru.unwrap().next;
            if self.epoch.saturating_sub(contents.touched) < age {
                self.lru_scan = None;
                break;
            }
            self.lru_scan = next;
            if self.evictable(resource) {
                self.evict_resource(resource)?;
            }
            if self.budget.used() <= self.collection_target {
                break;
            }
        }
        Ok(())
    }
}
