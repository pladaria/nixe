//! Device-contiguous GPU ranges maintained under the address-space write lock.
//! Physical leases and PTE kinds remain in the mapping records.
//! https://github.com/zortos293/Lichii/blob/646b5d17135a2b2701ccc7a959afd0bbbf2f6160/src/video_core/memory_manager.cpp#L733-L846
use nixe_gpu::native::MappedGpuRange;

#[derive(Clone, Copy, Debug)]
struct Run {
    start: u64,
    end: u64,
    device: u64,
}

impl Run {
    fn device_at(self, address: u64) -> u64 {
        self.device + (address - self.start)
    }

    fn clip(self, start: u64, end: u64) -> MappedGpuRange {
        let gpu = start.max(self.start);
        MappedGpuRange {
            gpu,
            device: self.device_at(gpu),
            size: end.min(self.end) - gpu,
        }
    }
}

#[derive(Default, Debug)]
pub(super) struct MappingRuns(Vec<Run>);

impl MappingRuns {
    pub fn prefix(&self, address: u64, end: u64) -> Option<MappedGpuRange> {
        let i = self.0.partition_point(|run| run.end <= address);
        let run = self.0.get(i)?;
        (run.start <= address && address < end).then(|| run.clip(address, end))
    }

    pub fn segments(&self, start: u64, end: u64) -> impl Iterator<Item = MappedGpuRange> {
        let i = self.0.partition_point(|run| run.end <= start);
        self.0[i..]
            .iter()
            .take_while(move |run| start < end && run.start < end)
            .map(move |run| run.clip(start, end))
    }

    pub fn insert(&mut self, start: u64, end: u64, device: u64) {
        let i = self.0.partition_point(|run| run.end <= start);
        let right = self
            .0
            .get(i)
            .is_some_and(|r| r.start == end && r.device == device + (end - start));
        if i > 0 && self.0[i - 1].end == start && self.0[i - 1].device_at(start) == device {
            self.0[i - 1].end = if right { self.0.remove(i).end } else { end };
        } else if right {
            self.0[i].start = start;
            self.0[i].device = device;
        } else {
            self.0.insert(i, Run { start, end, device });
        }
    }

    /// The removed interval is part of one committed mapping, hence one run.
    pub fn remove_mapped(&mut self, start: u64, end: u64) {
        let i = self.0.partition_point(|run| run.end <= start);
        let run = self.0[i];
        debug_assert!(run.start <= start && end <= run.end && start < end);
        let right = Run {
            start: end,
            end: run.end,
            device: run.device_at(end),
        };
        if start > run.start {
            self.0[i].end = start;
            if end < run.end {
                self.0.insert(i + 1, right);
            }
        } else if end < run.end {
            self.0[i] = right;
        } else {
            self.0.remove(i);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_follow_page_map_after_insert_split_and_remap() {
        let mut runs = MappingRuns::default();
        let mut pages = [None; 32];
        let mut state = 17_u32;
        for step in 0..2048 {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let page = (state >> 27) as usize;
            let start = page as u64 * 4;
            if pages[page].take().is_some() {
                runs.remove_mapped(start, start + 4);
            }
            if step % 3 != 0 {
                let device = start + 64 * u64::from((state >> 25) & 3);
                runs.insert(start, start + 4, device);
                pages[page] = Some(device);
            }
            let mut expected = Vec::<MappedGpuRange>::new();
            for at in 1..127 {
                if let Some(base) = pages[at as usize / 4] {
                    let device = base + at % 4;
                    if let Some(last) = expected.last_mut()
                        && last.gpu + last.size == at
                        && last.device + last.size == device
                    {
                        last.size += 1;
                    } else {
                        expected.push(MappedGpuRange {
                            gpu: at,
                            device,
                            size: 1,
                        });
                    }
                }
            }
            assert!(runs.segments(1, 127).eq(expected.iter().copied()));
            let at = start + 1;
            let expected = expected
                .iter()
                .find(|r| r.gpu <= at && at < r.gpu + r.size)
                .map(|r| MappedGpuRange {
                    gpu: at,
                    device: r.device + (at - r.gpu),
                    size: r.gpu + r.size - at,
                });
            assert_eq!(runs.prefix(at, 127), expected);
        }
    }
}
