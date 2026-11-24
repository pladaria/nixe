//! Outstanding guest compilation work, including jobs queued for Vulkan workers.
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompilationCounts {
    pub shaders: usize,
    pub pipelines: usize,
}
pub struct CompilationProgress {
    shaders: AtomicUsize,
    pipelines: AtomicUsize,
    notify: Box<dyn Fn() + Send + Sync>,
}
impl std::fmt::Debug for CompilationProgress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.counts().fmt(f)
    }
}
impl CompilationProgress {
    pub fn new(notify: impl Fn() + Send + Sync + 'static) -> Self {
        Self {
            shaders: AtomicUsize::new(0),
            pipelines: AtomicUsize::new(0),
            notify: Box::new(notify),
        }
    }
    pub fn counts(&self) -> CompilationCounts {
        CompilationCounts {
            shaders: self.shaders.load(Ordering::Acquire),
            pipelines: self.pipelines.load(Ordering::Acquire),
        }
    }
    pub(super) fn shaders(self: &Arc<Self>, count: usize) -> CompilationJob {
        self.begin(Work::Shaders(count))
    }
    pub(super) fn pipeline(self: &Arc<Self>) -> CompilationJob {
        self.begin(Work::Pipeline)
    }
    fn begin(self: &Arc<Self>, work: Work) -> CompilationJob {
        let (counter, count) = work.counter(self);
        counter.fetch_add(count, Ordering::Release);
        (self.notify)();
        CompilationJob {
            progress: self.clone(),
            work,
        }
    }
}
enum Work {
    Shaders(usize),
    Pipeline,
}
impl Work {
    fn counter<'a>(&self, p: &'a CompilationProgress) -> (&'a AtomicUsize, usize) {
        match self {
            Self::Shaders(n) => (&p.shaders, *n),
            Self::Pipeline => (&p.pipelines, 1),
        }
    }
}
pub(crate) struct CompilationJob {
    progress: Arc<CompilationProgress>,
    work: Work,
}
impl Drop for CompilationJob {
    fn drop(&mut self) {
        let (counter, count) = self.work.counter(&self.progress);
        counter.fetch_sub(count, Ordering::Release);
        (self.progress.notify)();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn independent_jobs_retire_on_drop() {
        let p = Arc::new(CompilationProgress::new(|| {}));
        let shaders = p.shaders(3);
        let pipeline = p.pipeline();
        assert_eq!(
            p.counts(),
            CompilationCounts {
                shaders: 3,
                pipelines: 1
            }
        );
        drop(shaders);
        assert_eq!(
            p.counts(),
            CompilationCounts {
                shaders: 0,
                pipelines: 1
            }
        );
        drop(pipeline);
        assert_eq!(p.counts(), CompilationCounts::default());
    }
}
