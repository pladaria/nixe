//! Backend capability reporting and side-effect-free negotiation.

use std::fmt::{Display, Formatter};

use crate::{ImageFormat, QueryKind, SampleCount, ShaderStage};

/// Backend operation families, independent from any guest GPU profile.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct BackendFeatures(u32);

impl BackendFeatures {
    pub const COPY: Self = Self(1 << 0);
    pub const CLEAR: Self = Self(1 << 1);
    pub const DRAW: Self = Self(1 << 2);
    pub const INDEXED_DRAW: Self = Self(1 << 3);
    pub const DISPATCH: Self = Self(1 << 4);
    pub const BARRIER: Self = Self(1 << 5);
    pub const QUERY: Self = Self(1 << 6);
    pub const RENDER_PASS: Self = Self(1 << 7);

    #[must_use]
    pub const fn empty() -> Self {
        Self(0)
    }

    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }
}

/// Numeric limits which affect whether an operation is representable.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BackendLimits {
    pub max_color_attachments: u8,
    pub max_descriptor_bindings: u32,
    pub max_compute_workgroups: [u32; 3],
}

/// One capability needed to represent an immutable neutral operation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CapabilityRequirement {
    Features(BackendFeatures),
    ImageFormat(ImageFormat),
    SampleCount(SampleCount),
    ShaderStage(ShaderStage),
    QueryKind(QueryKind),
    ColorAttachments(u8),
    DescriptorBindings(u32),
    ComputeWorkgroups([u32; 3]),
}

/// Canonical, immutable requirements for one operation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CapabilityRequirements {
    requirements: Box<[CapabilityRequirement]>,
}

impl CapabilityRequirements {
    #[must_use]
    pub fn none() -> Self {
        Self {
            requirements: Box::new([]),
        }
    }

    /// Creates a set while removing only exact duplicate requirements.
    #[must_use]
    pub fn new(requirements: impl IntoIterator<Item = CapabilityRequirement>) -> Self {
        let mut unique = Vec::new();
        for requirement in requirements {
            if !unique.contains(&requirement) {
                unique.push(requirement);
            }
        }
        Self {
            requirements: unique.into_boxed_slice(),
        }
    }

    #[must_use]
    pub fn with_features(features: BackendFeatures) -> Self {
        if features.is_empty() {
            Self::none()
        } else {
            Self::new([CapabilityRequirement::Features(features)])
        }
    }

    #[must_use]
    pub fn merged(&self, other: &Self) -> Self {
        Self::new(
            self.requirements
                .iter()
                .chain(other.requirements.iter())
                .copied(),
        )
    }

    #[must_use]
    pub fn requirements(&self) -> &[CapabilityRequirement] {
        &self.requirements
    }
}

/// Immutable capabilities reported by one prospective backend.
///
/// This value never describes or modifies the emulated guest GPU profile.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BackendCapabilities {
    features: BackendFeatures,
    image_formats: Box<[ImageFormat]>,
    sample_counts: Box<[SampleCount]>,
    shader_stages: Box<[ShaderStage]>,
    query_kinds: Box<[QueryKind]>,
    limits: BackendLimits,
}

impl BackendCapabilities {
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        features: BackendFeatures,
        image_formats: impl IntoIterator<Item = ImageFormat>,
        sample_counts: impl IntoIterator<Item = SampleCount>,
        shader_stages: impl IntoIterator<Item = ShaderStage>,
        query_kinds: impl IntoIterator<Item = QueryKind>,
        limits: BackendLimits,
    ) -> Self {
        Self {
            features,
            image_formats: unique(image_formats),
            sample_counts: unique(sample_counts),
            shader_stages: unique(shader_stages),
            query_kinds: unique(query_kinds),
            limits,
        }
    }

    #[must_use]
    pub const fn features(&self) -> BackendFeatures {
        self.features
    }

    #[must_use]
    pub fn image_formats(&self) -> &[ImageFormat] {
        &self.image_formats
    }

    #[must_use]
    pub fn sample_counts(&self) -> &[SampleCount] {
        &self.sample_counts
    }

    #[must_use]
    pub fn shader_stages(&self) -> &[ShaderStage] {
        &self.shader_stages
    }

    #[must_use]
    pub fn query_kinds(&self) -> &[QueryKind] {
        &self.query_kinds
    }

    #[must_use]
    pub const fn limits(&self) -> BackendLimits {
        self.limits
    }

    /// Negotiates one operation without changing backend or guest state.
    pub fn negotiate(
        &self,
        requirements: &CapabilityRequirements,
    ) -> Result<CapabilityAgreement, BackendCapabilityError> {
        self.negotiate_all(std::slice::from_ref(requirements))
    }

    /// Validates a complete operation sequence and produces evidence only when
    /// every requirement is supported. No prefix agreement is observable.
    pub fn negotiate_all(
        &self,
        operations: &[CapabilityRequirements],
    ) -> Result<CapabilityAgreement, BackendCapabilityError> {
        for (operation_index, requirements) in operations.iter().enumerate() {
            for requirement in requirements.requirements() {
                if !self.supports(*requirement) {
                    return Err(BackendCapabilityError {
                        operation_index,
                        requirement: *requirement,
                    });
                }
            }
        }
        Ok(CapabilityAgreement {
            operation_count: operations.len(),
        })
    }

    fn supports(&self, requirement: CapabilityRequirement) -> bool {
        match requirement {
            CapabilityRequirement::Features(features) => self.features.contains(features),
            CapabilityRequirement::ImageFormat(format) => self.image_formats.contains(&format),
            CapabilityRequirement::SampleCount(samples) => self.sample_counts.contains(&samples),
            CapabilityRequirement::ShaderStage(stage) => self.shader_stages.contains(&stage),
            CapabilityRequirement::QueryKind(kind) => self.query_kinds.contains(&kind),
            CapabilityRequirement::ColorAttachments(count) => {
                count <= self.limits.max_color_attachments
            }
            CapabilityRequirement::DescriptorBindings(count) => {
                count <= self.limits.max_descriptor_bindings
            }
            CapabilityRequirement::ComputeWorkgroups(groups) => groups
                .iter()
                .zip(self.limits.max_compute_workgroups)
                .all(|(required, maximum)| *required <= maximum),
        }
    }
}

fn unique<T: Copy + PartialEq>(values: impl IntoIterator<Item = T>) -> Box<[T]> {
    let mut unique = Vec::new();
    for value in values {
        if !unique.contains(&value) {
            unique.push(value);
        }
    }
    unique.into_boxed_slice()
}

/// Evidence that every operation in a sequence fits one capability report.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CapabilityAgreement {
    operation_count: usize,
}

impl CapabilityAgreement {
    #[must_use]
    pub const fn operation_count(self) -> usize {
        self.operation_count
    }
}

/// Typed rejection of work which the selected backend cannot represent.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BackendCapabilityError {
    operation_index: usize,
    requirement: CapabilityRequirement,
}

impl BackendCapabilityError {
    #[must_use]
    pub const fn operation_index(self) -> usize {
        self.operation_index
    }

    #[must_use]
    pub const fn requirement(self) -> CapabilityRequirement {
        self.requirement
    }
}

impl Display for BackendCapabilityError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "backend cannot represent neutral GPU operation: operation-index={} requirement={:?}",
            self.operation_index, self.requirement
        )
    }
}

impl std::error::Error for BackendCapabilityError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn capabilities() -> BackendCapabilities {
        BackendCapabilities::new(
            BackendFeatures::COPY.union(BackendFeatures::DRAW),
            [ImageFormat::Rgba8Unorm],
            [SampleCount::One],
            [ShaderStage::Vertex, ShaderStage::Fragment],
            [],
            BackendLimits {
                max_color_attachments: 1,
                max_descriptor_bindings: 8,
                max_compute_workgroups: [0; 3],
            },
        )
    }

    #[test]
    fn capabilities_are_independent_and_immutable() {
        let capabilities = capabilities();
        assert!(capabilities.features().contains(BackendFeatures::COPY));
        assert_eq!(capabilities.image_formats(), &[ImageFormat::Rgba8Unorm]);
        assert_eq!(capabilities.limits().max_descriptor_bindings, 8);
    }

    #[test]
    fn complete_negotiation_rejects_the_first_unrepresentable_operation() {
        let supported = CapabilityRequirements::new([
            CapabilityRequirement::Features(BackendFeatures::DRAW),
            CapabilityRequirement::ImageFormat(ImageFormat::Rgba8Unorm),
        ]);
        let unsupported = CapabilityRequirements::new([CapabilityRequirement::Features(
            BackendFeatures::DISPATCH,
        )]);
        let later = CapabilityRequirements::with_features(BackendFeatures::COPY);
        let error = capabilities()
            .negotiate_all(&[supported, unsupported, later])
            .unwrap_err();
        assert_eq!(error.operation_index(), 1);
        assert_eq!(
            error.requirement(),
            CapabilityRequirement::Features(BackendFeatures::DISPATCH)
        );
    }

    #[test]
    fn negotiation_never_substitutes_formats_or_limits() {
        let requirements = CapabilityRequirements::new([
            CapabilityRequirement::ImageFormat(ImageFormat::Bgra8Unorm),
            CapabilityRequirement::ColorAttachments(2),
        ]);
        let error = capabilities().negotiate(&requirements).unwrap_err();
        assert_eq!(
            error.requirement(),
            CapabilityRequirement::ImageFormat(ImageFormat::Bgra8Unorm)
        );
        assert_eq!(capabilities().image_formats(), &[ImageFormat::Rgba8Unorm]);
    }
}
