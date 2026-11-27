//! Copy the NVN visibility segments in the submission that recorded them.
//! https://github.com/scarletnx/ScarletNX/blob/6930e50/src/nx/gpu/vk/query.c
use super::*;
use std::collections::VecDeque;

const QUERIES: u32 = 256;

struct Bank {
    owner: Arc<DeviceOwner>,
    pool: vk::QueryPool,
    used: u32,
    copied: u32,
}

impl Drop for Bank {
    fn drop(&mut self) {
        unsafe { self.owner.device.destroy_query_pool(self.pool, None) };
    }
}

pub(crate) struct NvnQueryCopy {
    pub(crate) ticket: ReadbackTicket,
}

#[derive(Default)]
pub(super) struct NvnSamples {
    bank: Option<Bank>,
    retired: VecDeque<(u64, Bank)>,
    active: bool,
    enabled: bool,
    position: u64,
    copies: Vec<NvnQueryCopy>,
}

impl NvnSamples {
    pub(super) fn pause(&mut self, dispatch: &mut crate::dispatch::Dispatch) {
        if std::mem::take(&mut self.active) {
            let bank = self.bank.as_ref().unwrap();
            dispatch.end_occlusion(bank.pool, bank.used - 1);
        }
    }
}

impl VulkanDevice {
    pub(crate) fn set_nvn_samples_enabled(&mut self, enabled: bool) -> Result<(), VulkanError> {
        self.submission.nvn_samples.enabled = enabled;
        self.prepare_nvn_samples(enabled)
    }

    pub(crate) fn prepare_nvn_render_pass_samples(&mut self) -> Result<(), VulkanError> {
        self.prepare_nvn_samples(self.submission.nvn_samples.enabled)
    }

    pub(crate) fn nvn_query_boundary(&mut self) -> u64 {
        let queries = &mut self.submission.nvn_samples;
        queries.pause(&mut self.dispatch);
        queries.position
    }

    fn prepare_nvn_samples(&mut self, enabled: bool) -> Result<(), VulkanError> {
        let queries = &mut self.submission.nvn_samples;
        if !enabled {
            queries.pause(&mut self.dispatch);
            return Ok(());
        }
        if queries.active
            || queries
                .bank
                .as_ref()
                .is_some_and(|bank| bank.used < QUERIES)
        {
            return Ok(());
        }
        if queries.bank.is_some() {
            self.end_rendering()?;
            self.copy_nvn_query_results()?;
            let work = self.current_work().tick();
            let queries = &mut self.submission.nvn_samples;
            queries
                .retired
                .push_back((work, queries.bank.take().unwrap()));
        }
        let completed = self.submission.progress.completed();
        let queries = &mut self.submission.nvn_samples;
        let mut bank = if queries
            .retired
            .front()
            .is_some_and(|&(work, _)| work <= completed)
        {
            queries.retired.pop_front().unwrap().1
        } else {
            let pool = unsafe {
                self.owner.device.create_query_pool(
                    &vk::QueryPoolCreateInfo::default()
                        .query_type(vk::QueryType::OCCLUSION)
                        .query_count(QUERIES),
                    None,
                )
            }
            .map_err(|error| VulkanError::vk("vkCreateQueryPool(NVN occlusion)", error))?;
            Bank {
                owner: Arc::clone(&self.owner),
                pool,
                used: 0,
                copied: 0,
            }
        };
        bank.used = 0;
        bank.copied = 0;
        self.dispatch.reset_occlusion_pool(bank.pool, QUERIES);
        queries.bank = Some(bank);
        Ok(())
    }

    pub(crate) fn begin_nvn_samples(&mut self) {
        let queries = &mut self.submission.nvn_samples;
        if queries.active {
            return;
        }
        let bank = queries.bank.as_mut().expect("prepared NVN sample bank");
        self.dispatch.begin_occlusion(
            bank.pool,
            bank.used,
            self.profile.core.occlusion_query_precise != 0,
        );
        bank.used += 1;
        queries.position += 1;
        queries.active = true;
    }

    fn copy_nvn_query_results(&mut self) -> Result<(), VulkanError> {
        let Some(bank) = &self.submission.nvn_samples.bank else {
            return Ok(());
        };
        let count = bank.used - bank.copied;
        if count == 0 {
            return Ok(());
        }
        let (pool, first) = (bank.pool, bank.copied);
        let bytes = u64::from(count) * 8;
        let (slice, ticket) = self.submission.reserve_readback_aligned(bytes, 8)?;
        self.dispatch
            .copy_nvn_queries(pool, first, count, slice.buffer, slice.offset);
        crate::transfer::host_read_barrier(self, slice.buffer, slice.offset, bytes);
        let queries = &mut self.submission.nvn_samples;
        queries.bank.as_mut().unwrap().copied += count;
        queries.copies.push(NvnQueryCopy { ticket });
        #[cfg(feature = "execution-trace")]
        {
            self.submission.statistics.sample_report_copies += 1;
        }
        Ok(())
    }

    pub(crate) fn take_nvn_query_copies(&mut self) -> Result<Vec<NvnQueryCopy>, VulkanError> {
        let queries = &self.submission.nvn_samples;
        if queries
            .bank
            .as_ref()
            .is_some_and(|bank| bank.used != bank.copied)
        {
            self.end_rendering()?;
            self.copy_nvn_query_results()?;
        }
        Ok(std::mem::take(&mut self.submission.nvn_samples.copies))
    }
}
