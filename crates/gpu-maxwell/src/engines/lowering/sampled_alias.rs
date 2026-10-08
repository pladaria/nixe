//! Resident color views with different logical widths but identical storage.

use super::super::threed::MaxwellThreeDResolvedImage;
use super::*;
use nixe_gpu::{CopyOperation, ImageDimension, ImageKind, ImageMemoryLayout, SampleCount};

pub(super) fn copy_is_current(record: &ViewRecord, cache: &MaxwellLoweringCache) -> bool {
    let ViewMaterialization::CopiedColor { source, revision } = record.materialization else {
        return false;
    };
    if !record
        .cpu_writes
        .as_ref()
        .is_some_and(CanonicalCpuWriteDependency::remains_current)
    {
        return false;
    }
    cache.views.iter().any(|producer| {
        producer.dependency == ResourceDependency::Image(source)
            && producer.write_revision == revision
            && (producer.materialization == ViewMaterialization::Direct
                || producer
                    .cpu_writes
                    .as_ref()
                    .is_some_and(CanonicalCpuWriteDependency::remains_current))
    })
}

fn can_copy(record: &ViewRecord, image: &MaxwellThreeDResolvedImage) -> bool {
    let ViewKey::Image {
        description,
        swizzle,
        guest_pte_kind,
        bindings,
        ..
    } = &record.key
    else {
        return false;
    };
    let requested = image.description();
    let Some(bytes) = requested.format().plane_bytes_per_texel(0) else {
        return false;
    };
    // Block-linear X addressing uses the width rounded to a 64-byte GOB.
    // Different logical widths inside the same GOB row have identical storage;
    // their normalized texture coordinates still require distinct images.
    // https://github.com/devkitPro/deko3d/blob/master/source/dk_image.cpp
    if description.kind() != ImageKind::Color
        || requested.kind() != ImageKind::Color
        || description.dimension() != ImageDimension::Two
        || requested.dimension() != ImageDimension::Two
        || description.format() != requested.format()
        || description.samples() != SampleCount::One
        || requested.samples() != SampleCount::One
        || description.mip_levels() != 1
        || requested.mip_levels() != 1
        || description.array_layers() != 1
        || requested.array_layers() != 1
        || description.extent().width < requested.extent().width
        || description.extent().height != requested.extent().height
        || (u64::from(description.extent().width) * u64::from(bytes)).div_ceil(64)
            != (u64::from(requested.extent().width) * u64::from(bytes)).div_ceil(64)
        || *swizzle != image.view().swizzle()
        || *guest_pte_kind != image.guest_layout().pte_kind()
        || bindings.len() != 1
        || image.view().bindings().len() != 1
    {
        return false;
    }
    let (subresources, layout, backing) = &bindings[0];
    let current = &image.view().bindings()[0];
    matches!(layout, ImageMemoryLayout::BlockLinear(_))
        && *layout == current.layout()
        && *subresources == current.subresources()
        && same_canonical_backing(backing, current.backing())
        && (record.materialization == ViewMaterialization::Direct
            || record
                .cpu_writes
                .as_ref()
                .is_some_and(CanonicalCpuWriteDependency::remains_current))
}

fn materialization_matches(materialized: &ColorRepresentationRecord, key: &ViewKey) -> bool {
    let ViewKey::Image {
        description,
        swizzle,
        guest_format,
        guest_pte_kind,
        guest_compression_enabled,
        bindings,
        ..
    } = key
    else {
        return false;
    };
    materialized.description == *description
        && materialized.swizzle == *swizzle
        && materialized.guest_format == *guest_format
        && materialized.guest_pte_kind == *guest_pte_kind
        && materialized.guest_compression_enabled == *guest_compression_enabled
        && materialized.bindings.len() == bindings.len()
        && materialized.bindings.iter().zip(bindings).all(
            |(recorded, (subresources, layout, backing))| {
                recorded.subresources == *subresources
                    && recorded.layout == *layout
                    && same_canonical_backing(&recorded.backing, backing)
            },
        )
        && materialized
            .cpu_writes
            .as_ref()
            .is_some_and(CanonicalCpuWriteDependency::remains_current)
}

pub(super) fn prepare(
    image: &MaxwellThreeDResolvedImage,
    cache: &mut MaxwellLoweringCache,
    creations: &mut Vec<BackendResourceCreateInfo>,
) -> Result<Option<ResourceDependency>, MaxwellLoweringError> {
    let Some(producer) = cache
        .views
        .iter()
        .filter(|record| {
            can_copy(record, image)
                && (record.materialization == ViewMaterialization::Direct
                    || (record.materialization == ViewMaterialization::CompressedColor
                        && cache.color_materializations.iter().any(|materialized| {
                            materialization_matches(materialized, &record.key)
                        })))
        })
        .max_by_key(|record| record.write_revision)
    else {
        return Ok(None);
    };
    let source = image_dependency(producer.dependency)?;
    let revision = producer.write_revision;
    let key = view_key(&MaxwellThreeDResolvedResource::Image(Arc::new(
        image.clone(),
    )));
    let destination = if let Some(record) = cache.views.iter_mut().find(|record| record.key == key)
    {
        if !record
            .cpu_writes
            .as_ref()
            .is_some_and(CanonicalCpuWriteDependency::remains_current)
        {
            record.cpu_writes = Some(capture_copy_cpu_writes(image)?);
        }
        record.materialization = ViewMaterialization::CopiedColor { source, revision };
        image_dependency(record.dependency)?
    } else {
        let id = ImageId::new(take_identity(cache)?);
        creations.push(BackendResourceCreateInfo::Image {
            id,
            description: image.description(),
            view: None,
        });
        cache.views.push(ViewRecord {
            key,
            dependency: ResourceDependency::Image(id),
            materialization: ViewMaterialization::CopiedColor { source, revision },
            cpu_writes: Some(capture_copy_cpu_writes(image)?),
            write_revision: 0,
            last_used: 0,
            uninitialized_color_regions: Vec::new(),
            uninitialized_depth_stencil_regions: Default::default(),
        });
        id
    };
    let region = |image_id| ImageRegion {
        image: image_id,
        subresources: image.view().bindings()[0].subresources(),
        origin: ImageOrigin { x: 0, y: 0, z: 0 },
        extent: image.description().extent(),
    };
    // Distinct host images avoid overlapping copy storage. Copy only the
    // requested texel extent, preserving sampling at its logical dimensions.
    // https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdCopyImage.html
    cache.image_alias_copies.push(GpuOperation::new(
        GpuCommand::Copy(CopyOperation::ImageToImage {
            source: region(source),
            destination: region(destination),
        }),
        [],
        [],
        CapabilityRequirements::none(),
    ));
    Ok(Some(ResourceDependency::Image(destination)))
}

fn capture_copy_cpu_writes(
    image: &MaxwellThreeDResolvedImage,
) -> Result<CanonicalCpuWriteDependency, MaxwellLoweringError> {
    // A direct producer's canonical bytes remain importable after CPU writes;
    // the backend uploads them before the copy's read access. Track each copy
    // independently: refreshing one alias must not make another stale copy
    // current, or depend on the resolver's original observation epoch.
    CanonicalCpuWriteDependency::capture_ranges(
        image
            .view()
            .bindings()
            .iter()
            .map(|binding| binding.backing().range()),
    )
    .map_err(|error| MaxwellLoweringError::ImageBacking(error.to_string()))
}
