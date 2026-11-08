//! Stable renderer ownership for native pipelines, samplers and descriptor pools.
//! Queued recording borrows these objects; GPU completion does not retain a
//! second per-command set of pipeline/sampler ownership handles.
//! https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/renderer_vulkan/vk_pipeline_cache.h
use super::{
    descriptors::{DescriptorPool, SamplerData},
    pipeline::NativePipelineData,
};
use std::{
    ops::Deref,
    ptr::NonNull,
    sync::{Arc, Mutex},
};

#[derive(Default)]
pub(crate) struct NativeObjects(Mutex<Objects>);
// Objects have stable addresses while their owners grow the storage.
#[allow(clippy::vec_box)]
#[derive(Default)]
struct Objects {
    pipelines: Vec<Box<NativePipelineData>>,
    samplers: Vec<Box<SamplerData>>,
    descriptor_pools: Vec<Box<DescriptorPool>>,
}
impl NativeObjects {
    pub(super) fn pipeline(
        self: &Arc<Self>,
        data: NativePipelineData,
    ) -> NativeObject<NativePipelineData> {
        let mut object = Box::new(data);
        let pointer = NonNull::from(object.as_mut());
        self.0.lock().unwrap().pipelines.push(object);
        NativeObject {
            storage: Arc::clone(self),
            pointer,
        }
    }
    pub(super) fn sampler(self: &Arc<Self>, data: SamplerData) -> NativeObject<SamplerData> {
        let mut object = Box::new(data);
        let pointer = NonNull::from(object.as_mut());
        self.0.lock().unwrap().samplers.push(object);
        NativeObject {
            storage: Arc::clone(self),
            pointer,
        }
    }
    pub(super) fn descriptor_pool(
        self: &Arc<Self>,
        data: DescriptorPool,
    ) -> NativeObject<DescriptorPool> {
        let mut object = Box::new(data);
        let pointer = NonNull::from(object.as_mut());
        self.0.lock().unwrap().descriptor_pools.push(object);
        NativeObject {
            storage: Arc::clone(self),
            pointer,
        }
    }
}

/// An externally retained object keeps its renderer storage alive. Native jobs
/// use a borrowed reference under their recording context's storage ownership.
pub struct NativeObject<T> {
    storage: Arc<NativeObjects>,
    pointer: NonNull<T>,
}
impl<T> Clone for NativeObject<T> {
    fn clone(&self) -> Self {
        Self {
            storage: Arc::clone(&self.storage),
            pointer: self.pointer,
        }
    }
}
impl<T> Deref for NativeObject<T> {
    type Target = T;
    fn deref(&self) -> &T {
        // SAFETY: storage owns the initialized, immovable box until this handle drops.
        unsafe { self.pointer.as_ref() }
    }
}
// SAFETY: object storage is shared and pinned; data must support shared access.
unsafe impl<T: Send + Sync> Send for NativeObject<T> {}
unsafe impl<T: Send + Sync> Sync for NativeObject<T> {}
impl<T> NativeObject<T> {
    pub(crate) fn identity(&self) -> usize {
        self.pointer.as_ptr() as usize
    }
    pub(crate) fn reference(&self) -> NativeReference<T> {
        NativeReference(self.pointer)
    }
}

pub(crate) struct NativeReference<T>(NonNull<T>);
impl<T> Copy for NativeReference<T> {}
impl<T> Clone for NativeReference<T> {
    fn clone(&self) -> Self {
        *self
    }
}
// SAFETY: recording retains the same renderer storage; data permits shared access.
unsafe impl<T: Send + Sync> Send for NativeReference<T> {}
impl<T> NativeReference<T> {
    /// The storage must own this reference's object. VulkanDevice verifies the
    /// originating device before admitting a native recording operation.
    pub(crate) unsafe fn get<'a>(&self, _storage: &'a NativeObjects) -> &'a T {
        unsafe { self.0.as_ref() }
    }
}
