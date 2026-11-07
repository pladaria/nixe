//! TextureCache's delayed destruction ring. Logical indices can be reused
//! immediately; native objects survive frame delay and their actual last use.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/delayed_destruction_ring.h
use super::*;
const TICKS_TO_DESTROY: usize = 8;
struct Image {
    _native: Arc<AllocatedImage>,
    _views: Vec<Arc<crate::AllocatedImageView>>,
    last_use: Option<WorkTicket>,
}
pub(super) struct TextureRetirement {
    entries: [Vec<Image>; TICKS_TO_DESTROY],
    framebuffers: [Vec<Arc<execution::PreparedTargets>>; TICKS_TO_DESTROY],
    index: usize,
    progress: Arc<crate::queue::progress::SubmissionProgress>,
}
impl TextureRetirement {
    pub(super) fn new(progress: Arc<crate::queue::progress::SubmissionProgress>) -> Self {
        Self {
            entries: std::array::from_fn(|_| Vec::new()),
            framebuffers: std::array::from_fn(|_| Vec::new()),
            index: 0,
            progress,
        }
    }
    pub(super) fn push(
        &mut self,
        native: Arc<AllocatedImage>,
        views: Vec<Arc<crate::AllocatedImageView>>,
        last_use: Option<WorkTicket>,
    ) {
        self.entries[self.index].push(Image {
            _native: native,
            _views: views,
            last_use,
        });
    }
    pub(super) fn push_framebuffer(&mut self, framebuffer: Arc<execution::PreparedTargets>) {
        self.framebuffers[self.index].push(framebuffer);
    }
    pub(super) fn tick(&mut self) {
        self.index = (self.index + 1) % TICKS_TO_DESTROY;
        // Native queue slots independently retain VkFramebuffer through GPU
        // completion, including work older than this cache retirement delay.
        self.framebuffers[self.index].clear();
        let completed = self.progress.completed();
        // Frame delay supplies cache retirement; the existing native timeline
        // protects genuinely pending work even if the GPU falls behind it.
        self.entries[self.index]
            .retain(|image| image.last_use.is_some_and(|work| work.tick() > completed));
    }
}
