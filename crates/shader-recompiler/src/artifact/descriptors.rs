use crate::{CompileError, Result, diagnostics::FailureCategory};

/// The reference uses fixed semantic limits for cbufs and SSBOs, but growable
/// small vectors for image/texture descriptors. Only the former use this type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoundedDescriptors<T, const N: usize>(Vec<T>);

impl<T, const N: usize> Default for BoundedDescriptors<T, N> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl<T, const N: usize> BoundedDescriptors<T, N> {
    pub fn capacity(&self) -> usize {
        self.0.capacity()
    }

    pub fn push(&mut self, descriptor: T) -> Result<()> {
        if self.0.len() == N {
            return Err(CompileError::new(
                FailureCategory::ResourceLimit,
                format!("Descriptor count exceeds the compiler limit of {N}"),
            ));
        }
        self.0.push(descriptor);
        Ok(())
    }

    pub fn clear(&mut self) {
        self.0.clear();
    }
}

impl<T, const N: usize> std::ops::Deref for BoundedDescriptors<T, N> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        &self.0
    }
}

impl<T, const N: usize> std::ops::DerefMut for BoundedDescriptors<T, N> {
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.0
    }
}

pub trait DescriptorCount {
    fn count(&self) -> u32;
}

/// Reference unsigned arithmetic (descriptor array sizes, not vector length).
pub fn num_descriptors<T: DescriptorCount>(descriptors: &[T]) -> u32 {
    descriptors
        .iter()
        .fold(0u32, |total, desc| total.wrapping_add(desc.count()))
}
