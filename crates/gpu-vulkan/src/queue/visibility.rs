//! Guest content versions are independent of command-buffer submission numbers.
//! Ordered versions share a real producer prefix; completed prefixes retire once.
use crate::VulkanError;
use nixe_memory::DeviceVisibilityPoint;
use std::collections::VecDeque;

#[derive(Default)]
pub(super) struct VisibilityTimeline {
    last: u64,
    completed: u64,
    pending: VecDeque<(u64, u64)>,
}
impl VisibilityTimeline {
    pub fn issue(
        &mut self,
        required: u64,
        completed: u64,
    ) -> Result<DeviceVisibilityPoint, VulkanError> {
        self.complete(completed);
        self.last = self
            .last
            .checked_add(1)
            .ok_or(VulkanError::InvalidResource("visibility version exhausted"))?;
        if required <= completed {
            debug_assert!(self.pending.is_empty());
            self.completed = self.last;
        } else if let Some((work, through)) = self.pending.back_mut()
            && *work == required
        {
            *through = self.last;
        } else {
            debug_assert!(self.pending.back().is_none_or(|&(work, _)| work < required));
            self.pending.push_back((required, self.last));
        }
        Ok(DeviceVisibilityPoint::new(self.last))
    }
    pub fn complete(&mut self, completed: u64) {
        while self
            .pending
            .front()
            .is_some_and(|&(work, _)| work <= completed)
        {
            self.completed = self.pending.pop_front().unwrap().1;
        }
    }
    pub fn completed(&self) -> u64 {
        self.completed
    }
    pub fn through(&self, work: u64) -> u64 {
        let index = self
            .pending
            .partition_point(|&(required, _)| required <= work);
        index
            .checked_sub(1)
            .map_or(self.completed, |i| self.pending[i].1)
    }
    pub fn required(&self, point: DeviceVisibilityPoint) -> Result<Option<u64>, VulkanError> {
        if point.get() > self.last {
            return Err(VulkanError::InvalidResource("unissued visibility version"));
        }
        if point.get() <= self.completed {
            return Ok(None);
        }
        let index = self
            .pending
            .partition_point(|&(_, through)| through < point.get());
        Ok(Some(self.pending[index].0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn versions_share_real_work_without_reusing_content_identity() {
        let mut timeline = VisibilityTimeline::default();
        let initial = timeline.issue(0, 0).unwrap();
        assert_eq!(timeline.required(initial).unwrap(), None);
        let first = timeline.issue(1, 0).unwrap();
        let later = timeline.issue(1, 0).unwrap();
        assert_ne!(first, later);
        assert_eq!(timeline.pending.len(), 1);
        assert_eq!(timeline.required(first).unwrap(), Some(1));
        assert_eq!(timeline.required(later).unwrap(), Some(1));
        let second = timeline.issue(2, 0).unwrap();
        timeline.complete(1);
        assert_eq!(timeline.required(first).unwrap(), None);
        assert_eq!(timeline.required(second).unwrap(), Some(2));
        assert_eq!(timeline.through(1), later.get());
        timeline.complete(2);
        let host_effect = timeline.issue(2, 2).unwrap();
        assert_ne!(host_effect, second);
        assert_eq!(timeline.required(host_effect).unwrap(), None);
        assert!(timeline.pending.is_empty());
        assert!(
            timeline
                .required(DeviceVisibilityPoint::new(host_effect.get() + 1))
                .is_err()
        );
    }
}
