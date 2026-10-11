//! Render-target/TIC aliasing must retain the initialized device representation.

use super::*;
use crate::engines::tests::{program_three_d, three_d_channel};
use crate::{
    MaxwellAddressSpaceId, MaxwellAddressSpaceInitialization, MaxwellAllocationId,
    MaxwellGpuAddressSpace, MaxwellMapRequest, SWITCH_1_GM20B_PROFILE,
};
use nixe_memory::{CanonicalAllocation, MemoryPermissions};

fn map(
    space: &mut MaxwellGpuAddressSpace,
    allocation: &CanonicalAllocation,
    id: u64,
    kind: u8,
) -> u64 {
    let backing = allocation
        .backing_range(MemoryPermissions::READ_WRITE)
        .unwrap();
    space
        .map(MaxwellMapRequest {
            allocation: MaxwellAllocationId::new(id),
            size: backing.size(),
            backing,
            backing_offset: 0,
            allocation_alignment: 0x1000,
            page_size: 0,
            kind,
            cacheable: true,
            permissions: MemoryPermissions::READ_WRITE,
            fixed_offset: None,
        })
        .unwrap()
        .offset()
        .get()
}

fn tic(allocation: &CanonicalAllocation, address: u64, width: u32, format: u32) {
    for (index, word) in [
        format,
        address as u32,
        (address >> 32) as u32 | (3 << 21),
        0,
        (width - 1) | (1 << 23),
        31 | (1 << 31),
        0,
        0,
    ]
    .into_iter()
    .enumerate()
    {
        allocation.write(index * 4, &word.to_le_bytes()).unwrap();
    }
}

#[test]
fn compressed_color_sampling_reuses_only_current_matching_resident_images() {
    use super::super::threed::{
        MaxwellThreeDTextureDimension, MaxwellThreeDTextureReference,
        resolve_maxwell_three_d_resources_for_roles,
    };
    for (color_format, texture_format, kind, compression) in [
        (0xca, 0x58d7_ff83, 0xe9, 1), // RGBA16F / C64_2CRA
        (0xca, 0x58d7_ff83, 0xe9, 0),
        (0xd5, 0x58d2_4908, 0xdb, 1), // RGBA8_UNORM / C32_2CRA
        (0xd5, 0x58d2_4908, 0xdb, 0),
    ] {
        let mut space =
            MaxwellGpuAddressSpace::new(MaxwellAddressSpaceId::new(1), SWITCH_1_GM20B_PROFILE);
        space
            .initialize(MaxwellAddressSpaceInitialization::default())
            .unwrap();
        let pixels = CanonicalAllocation::zeroed(0x10000, 0x1000).unwrap();
        let metadata = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
        let address = map(&mut space, &pixels, 1, kind);
        let descriptors = map(&mut space, &metadata, 2, 0xfe);
        tic(&metadata, address, 64, texture_format);
        let mut channel = three_d_channel();
        for (method, argument) in [
            (0x0800, (address >> 32) as u32),
            (0x0804, address as u32),
            (0x0808, 64),
            (0x080c, 32),
            (0x0810, color_format),
            (0x0814, 0),
            (0x0818, 1),
            (0x081c, 0),
            (0x0820, 0),
            (0x15d0, 0),
            (0x19e0, compression),
            (0x1574, (descriptors >> 32) as u32),
            (0x1578, descriptors as u32),
            (0x157c, 0),
            (0x2380, 4),
            (0x2384, (descriptors >> 32) as u32),
            (0x2388, (descriptors + 0x100) as u32),
            (0x2490, 1),
        ] {
            program_three_d(&mut channel, method, argument);
        }
        let target = MaxwellThreeDResourceRole::ColorTarget(0);
        let sampled = MaxwellThreeDResourceRole::SampledImage {
            texture: MaxwellThreeDTextureReference::new(4, 0, 0),
            dimension: MaxwellThreeDTextureDimension::Two,
        };
        let resolve = |space: &MaxwellGpuAddressSpace, role| {
            resolve_maxwell_three_d_resources_for_roles(channel.three_d(), space, &[role]).unwrap()
        };
        let targets = resolve(&space, target);
        let textures = resolve(&space, sampled);
        let target_index = resource_index(&targets, target).unwrap();
        let texture_index = resource_index(&textures, sampled).unwrap();
        let image = resolved_image(&targets, target_index).unwrap();
        let texture = resolved_image(&textures, texture_index).unwrap();
        assert!(texture.guest_layout().requires_materialization());
        assert!(!texture.guest_layout().has_direct_canonical_representation());

        let mut cache = MaxwellLoweringCache::default();
        let mut creations = Vec::new();
        let mut invalidations = Vec::new();
        let prepare = |resources: &MaxwellThreeDResolvedResources,
                       index,
                       cache: &mut MaxwellLoweringCache,
                       creations: &mut Vec<_>,
                       invalidations: &mut Vec<_>| {
            prepare_resources(resources, &[index], cache, creations, invalidations)
        };
        assert!(matches!(
            prepare(
                &textures,
                texture_index,
                &mut cache,
                &mut creations,
                &mut invalidations
            ),
            Err(MaxwellLoweringError::CompressedSampledImageImportRequired { .. })
        ));
        assert!(creations.is_empty());
        let target_binding = prepare(
            &targets,
            target_index,
            &mut cache,
            &mut creations,
            &mut invalidations,
        )
        .unwrap()[target_index];
        assert!(creations.iter().any(|creation| matches!(
            creation,
            BackendResourceCreateInfo::Image { view: Some(view), .. } if !view.has_canonical_texels()
        )));
        creations.clear();
        // Disabling future compression does not make existing compressed
        // allocation bytes importable. Both selectors need a recorded producer.
        assert!(
            prepare(
                &textures,
                texture_index,
                &mut cache,
                &mut creations,
                &mut invalidations
            )
            .is_err()
        );
        record_color_materialization(image, &mut cache);
        for _ in 0..3 {
            assert_eq!(
                prepare(
                    &textures,
                    texture_index,
                    &mut cache,
                    &mut creations,
                    &mut invalidations
                )
                .unwrap()[texture_index],
                target_binding
            );
            assert_eq!(
                prepare(
                    &targets,
                    target_index,
                    &mut cache,
                    &mut creations,
                    &mut invalidations
                )
                .unwrap()[target_index],
                target_binding
            );
            assert!(creations.is_empty());
            assert!(invalidations.is_empty());
        }

        // Another GPU mapping of the same canonical pages is still the same image.
        let alias = map(&mut space, &pixels, 3, kind);
        tic(&metadata, alias, 64, texture_format);
        let aliased = resolve(&space, sampled);
        assert_eq!(
            prepare(
                &aliased,
                texture_index,
                &mut cache,
                &mut creations,
                &mut invalidations
            )
            .unwrap()[texture_index],
            target_binding
        );
        // A width inside the producer's last GOB uses the same storage but
        // requires different sampling coordinates. Retain both host images.
        // A partial clear initializes the visible width while leaving padding
        // undefined. Sampling the visible crop must not require that padding;
        // sampling the full producer width must still be rejected.
        cache.color_materializations.clear();
        let producer = cache
            .views
            .values_mut()
            .find(|record| Some(record.dependency) == target_binding)
            .unwrap();
        producer.uninitialized_color_regions = vec![[0, 0, 64, 32]];
        resolve::subtract_initialized_rect(
            &mut producer.uninitialized_color_regions,
            [0, 0, 63, 32],
        );
        assert!(
            prepare(
                &aliased,
                texture_index,
                &mut cache,
                &mut creations,
                &mut invalidations
            )
            .is_err()
        );
        tic(&metadata, alias, 63, texture_format);
        let cropped = resolve(&space, sampled);
        let cropped_binding = prepare(
            &cropped,
            texture_index,
            &mut cache,
            &mut creations,
            &mut invalidations,
        )
        .unwrap()[texture_index];
        assert_ne!(cropped_binding, target_binding);
        assert_eq!(creations.len(), 1);
        assert!(matches!(
            &creations[0],
            BackendResourceCreateInfo::Image { view: None, .. }
        ));
        assert_eq!(cache.image_alias_copies.len(), 1);
        let GpuCommand::Copy(nixe_gpu::CopyOperation::ImageToImage {
            source,
            destination,
        }) = cache.image_alias_copies[0].command()
        else {
            panic!("expected resident image copy");
        };
        assert_eq!(source.extent.width, 63);
        assert_eq!(destination.extent, source.extent);
        assert_ne!(source.image, destination.image);
        creations.clear();
        cache.image_alias_copies.clear();
        assert_eq!(
            prepare(
                &cropped,
                texture_index,
                &mut cache,
                &mut creations,
                &mut invalidations
            )
            .unwrap()[texture_index],
            cropped_binding
        );
        assert!(cache.image_alias_copies.is_empty());
        assert!(creations.is_empty());
        cache.revision += 1;
        record_color_materialization(image, &mut cache);
        record_image_write(image, &mut cache);
        assert_eq!(
            prepare(
                &cropped,
                texture_index,
                &mut cache,
                &mut creations,
                &mut invalidations
            )
            .unwrap()[texture_index],
            cropped_binding
        );
        assert_eq!(cache.image_alias_copies.len(), 1);
        assert!(creations.is_empty());
        assert!(invalidations.is_empty());
        cache.image_alias_copies.clear();
        // Layout/extent changes and new physical pages cannot inherit contents.
        tic(&metadata, alias, 32, texture_format);
        assert!(
            prepare(
                &resolve(&space, sampled),
                texture_index,
                &mut cache,
                &mut creations,
                &mut invalidations
            )
            .is_err()
        );
        let other_pixels = CanonicalAllocation::zeroed(0x10000, 0x1000).unwrap();
        let other = map(&mut space, &other_pixels, 4, kind);
        tic(&metadata, other, 64, texture_format);
        assert!(
            prepare(
                &resolve(&space, sampled),
                texture_index,
                &mut cache,
                &mut creations,
                &mut invalidations
            )
            .is_err()
        );
        tic(&metadata, address, 64, texture_format);
        // A historical materialization record without its resident image is not enough.
        let dependency = target_binding.unwrap();
        let resident = cache.views.remove(&dependency).unwrap();
        assert!(
            prepare(
                &textures,
                texture_index,
                &mut cache,
                &mut creations,
                &mut invalidations
            )
            .is_err()
        );
        cache.views.insert(dependency, resident);
        // CPU writes invalidate every compressed alias independently, even
        // when the producer disabled compression for subsequent writes.
        tic(&metadata, address, 62, texture_format);
        let second_cropped = resolve(&space, sampled);
        prepare(
            &second_cropped,
            texture_index,
            &mut cache,
            &mut creations,
            &mut invalidations,
        )
        .unwrap();
        creations.clear();
        cache.image_alias_copies.clear();
        pixels.write(0, &[0; 4]).unwrap();
        for textures in [&cropped, &second_cropped, &textures] {
            assert!(matches!(
                prepare(
                    textures,
                    texture_index,
                    &mut cache,
                    &mut creations,
                    &mut invalidations
                ),
                Err(MaxwellLoweringError::CompressedSampledImageImportRequired { .. })
            ));
            assert!(cache.image_alias_copies.is_empty());
        }
        assert!(creations.is_empty());
        assert!(invalidations.is_empty());

        // The same storage width is not permission to reinterpret another
        // compression family, or to confuse BC blocks with individual texels.
        let wrong_kind = if kind == 0xe9 { 0xdb } else { 0xe9 };
        let wrong = map(&mut space, &pixels, 5, wrong_kind);
        tic(&metadata, wrong, 64, texture_format);
        assert!(matches!(
            resolve_maxwell_three_d_resources_for_roles(channel.three_d(), &space, &[sampled]),
            Err(super::super::threed::MaxwellThreeDResourceError::UnsupportedKind { .. })
        ));
        tic(&metadata, address, 64, 0x78d2_4924); // BC1 has 8-byte blocks, not C64 texels.
        assert!(matches!(
            resolve_maxwell_three_d_resources_for_roles(channel.three_d(), &space, &[sampled]),
            Err(super::super::threed::MaxwellThreeDResourceError::UnsupportedKind { .. })
        ));
    }
}

#[test]
fn compressed_uploads_retain_prior_stripes_and_require_complete_coverage() {
    use super::super::threed::{
        MaxwellThreeDTextureDimension, MaxwellThreeDTextureReference,
        resolve_maxwell_three_d_resources_for_roles,
    };
    use super::super::twod::blit::{
        BlitKind, BlitSurface, BlitSurfaceLayout, MaxwellTwoDBlitOperation,
    };
    let mut space =
        MaxwellGpuAddressSpace::new(MaxwellAddressSpaceId::new(1), SWITCH_1_GM20B_PROFILE);
    space
        .initialize(MaxwellAddressSpaceInitialization::default())
        .unwrap();
    let upload = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
    let pixels = CanonicalAllocation::zeroed(0x10000, 0x1000).unwrap();
    let metadata = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
    let src = map(&mut space, &upload, 1, 0);
    let dst = map(&mut space, &pixels, 2, 0xdb);
    let descriptors = map(&mut space, &metadata, 3, 0xfe);
    tic(&metadata, dst, 8, 0x58d2_4908);
    // Match the synthetic 8x8 destination rather than the helper's 32 rows.
    metadata
        .write(5 * 4, &(7_u32 | (1 << 31)).to_le_bytes())
        .unwrap();
    let mut channel = three_d_channel();
    for (method, value) in [
        (0x1574, (descriptors >> 32) as u32),
        (0x1578, descriptors as u32),
        (0x157c, 0),
        (0x2380, 4),
        (0x2384, (descriptors >> 32) as u32),
        (0x2388, (descriptors + 0x100) as u32),
        (0x2490, 1),
    ] {
        program_three_d(&mut channel, method, value);
    }
    let sampled = MaxwellThreeDResourceRole::SampledImage {
        texture: MaxwellThreeDTextureReference::new(4, 0, 0),
        dimension: MaxwellThreeDTextureDimension::Two,
    };
    let textures =
        resolve_maxwell_three_d_resources_for_roles(channel.three_d(), &space, &[sampled]).unwrap();
    let texture_index = resource_index(&textures, sampled).unwrap();
    for (scaled, stripe_height) in [(false, 3), (false, 4), (true, 3), (true, 4)] {
        let mut cache = MaxwellLoweringCache::default();
        let mut resident = None;
        for (serial, y) in [(1, 0), (2, 4)] {
            let request = MaxwellTwoDBlitOperation {
                source: channel
                    .three_d()
                    .shader_bindings()
                    .texture_headers()
                    .maximum_index()
                    .source()
                    .unwrap(),
                images: [
                    BlitSurface {
                        address: src,
                        width: 8,
                        height: 4,
                        format: 0xd5,
                        layout: BlitSurfaceLayout::Pitch(48),
                    },
                    BlitSurface {
                        address: dst,
                        width: 8,
                        height: 8,
                        format: 0xd5,
                        layout: BlitSurfaceLayout::BlockLinear(0),
                    },
                ],
                destination_compression: true,
                kind: if scaled {
                    BlitKind::Scale {
                        filter: nixe_gpu::BlitFilter::Linear,
                        first: [1 << 31; 2],
                        step: [0xc0000000; 2],
                        destination_origin: ImageOrigin { x: 0, y, z: 0 },
                        extent: nixe_gpu::ImageExtent {
                            width: 8,
                            height: stripe_height,
                            depth: 1,
                        },
                    }
                } else {
                    BlitKind::Copy {
                        origins: [
                            ImageOrigin { x: 0, y: 0, z: 0 },
                            ImageOrigin { x: 0, y, z: 0 },
                        ],
                        extent: nixe_gpu::ImageExtent {
                            width: 8,
                            height: stripe_height,
                            depth: 1,
                        },
                    }
                },
            };
            let resources = cache
                .resolved_resources_mut()
                .resolve_color_images(&request, &space, 16)
                .unwrap();
            let source = resolved_image(&resources, 0).unwrap();
            assert!(matches!(
                source.view().bindings()[0].layout(),
                nixe_gpu::ImageMemoryLayout::PitchLinear { row_pitch: 48, .. }
            ));
            let work = cache
                .lower_color_blit(
                    &request,
                    &resources,
                    FrontendSubmissionId::new(serial),
                    vec![],
                )
                .unwrap();
            let copy = work
                .submission()
                .operations()
                .iter()
                .find_map(|op| match op.command() {
                    GpuCommand::Copy(nixe_gpu::CopyOperation::ImageToImage {
                        destination, ..
                    }) => Some(destination),
                    GpuCommand::Blit(blit) => Some(&blit.destination),
                    _ => None,
                })
                .unwrap();
            assert_eq!(copy.origin.y, y);
            if let Some(id) = resident {
                assert_eq!(copy.image, id);
                assert!(work.resource_creations().is_empty());
            } else {
                resident = Some(copy.image);
            }
            let mut creations = Vec::new();
            let mut invalidations = Vec::new();
            let result = prepare_resources(
                &textures,
                &[texture_index],
                &mut cache,
                &mut creations,
                &mut invalidations,
            );
            if y == 0 || stripe_height == 3 {
                assert!(matches!(
                    result,
                    Err(MaxwellLoweringError::CompressedSampledImageImportRequired { .. })
                ));
            } else {
                assert_eq!(
                    result.unwrap()[texture_index],
                    resident.map(ResourceDependency::Image)
                );
            }
            assert!(creations.is_empty());
            assert!(invalidations.is_empty());
        }
        if stripe_height == 3 {
            // The two transfers leave rows 3 and 7 unwritten. Inline uploads
            // must complete coverage on the existing image, without recreating it.
            for (serial, offset, row) in [(3, 80, 3), (4, 208, 7)] {
                let target = space
                    .resolve_range(
                        space.address(dst + offset).unwrap(),
                        4,
                        MemoryPermissions::WRITE,
                    )
                    .unwrap();
                let (_, mut region) = cache.inline_image_word(&target, None).unwrap().unwrap();
                assert_eq!(region.origin, ImageOrigin { x: 0, y: row, z: 0 });
                region.extent.width = 8;
                let work = cache
                    .lower_inline_images(
                        vec![(region, vec![0x55; 32])],
                        FrontendSubmissionId::new(serial),
                        vec![],
                    )
                    .unwrap();
                assert!(work.resource_creations().is_empty());
                let result = prepare_resources(
                    &textures,
                    &[texture_index],
                    &mut cache,
                    &mut Vec::new(),
                    &mut Vec::new(),
                );
                assert_eq!(result.is_ok(), row == 7);
            }
        }
        let target = space
            .resolve_range(
                space.address(dst + 152).unwrap(),
                4,
                MemoryPermissions::WRITE,
            )
            .unwrap();
        let (hint, region) = cache.inline_image_word(&target, None).unwrap().unwrap();
        assert_eq!(region.image, resident.unwrap());
        assert_eq!(region.origin, ImageOrigin { x: 2, y: 5, z: 0 });
        assert_eq!(
            cache
                .inline_image_word(&target, Some(hint))
                .unwrap()
                .unwrap()
                .1,
            region
        );
        let work = cache
            .lower_inline_images(
                vec![(region, vec![1, 2, 3, 4])],
                FrontendSubmissionId::new(5),
                vec![],
            )
            .unwrap();
        assert!(work.resource_creations().is_empty());
        assert!(work.submission().operations().iter().any(|operation| matches!(operation.command(), GpuCommand::UploadImage { destination, bytes } if *destination == region && bytes.as_ref() == [1, 2, 3, 4])));
        assert!(
            prepare_resources(
                &textures,
                &[texture_index],
                &mut cache,
                &mut Vec::new(),
                &mut Vec::new()
            )
            .is_ok()
        );
        pixels.write(0, &[1]).unwrap();
        let textures =
            resolve_maxwell_three_d_resources_for_roles(channel.three_d(), &space, &[sampled])
                .unwrap();
        assert!(
            prepare_resources(
                &textures,
                &[texture_index],
                &mut cache,
                &mut Vec::new(),
                &mut Vec::new()
            )
            .is_err()
        );
    }
}

#[test]
fn inline_image_offsets_preserve_gob_coordinates_and_block_rows() {
    for (offset, expected) in [
        (0, (0, 0)),
        (16, (0, 1)),
        (32, (16, 0)),
        (64, (0, 2)),
        (256, (32, 0)),
        (512, (64, 0)),
        (1024, (0, 8)),
    ] {
        assert_eq!(inline_block_linear_position(offset, 32, 0), expected);
    }
    assert_eq!(inline_block_linear_position(512, 32, 4), (0, 8));
    assert_eq!(inline_block_linear_position(8192, 32, 4), (64, 0));
    assert_eq!(inline_block_linear_position(16384, 32, 4), (0, 128));
}

#[test]
fn direct_overlapping_texture_views_retain_identity_until_a_device_write() {
    use super::super::threed::{
        MaxwellThreeDTextureDimension, MaxwellThreeDTextureReference,
        resolve_maxwell_three_d_resources_for_roles,
    };
    let mut space =
        MaxwellGpuAddressSpace::new(MaxwellAddressSpaceId::new(1), SWITCH_1_GM20B_PROFILE);
    space
        .initialize(MaxwellAddressSpaceInitialization::default())
        .unwrap();
    let pixels = CanonicalAllocation::zeroed(0x10000, 0x1000).unwrap();
    let metadata = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
    let address = map(&mut space, &pixels, 1, 0xfe);
    let descriptors = map(&mut space, &metadata, 2, 0xfe);
    let mut channel = three_d_channel();
    for (method, argument) in [
        (0x1574, (descriptors >> 32) as u32),
        (0x1578, descriptors as u32),
        (0x157c, 0),
        (0x2380, 4),
        (0x2384, (descriptors >> 32) as u32),
        (0x2388, (descriptors + 0x100) as u32),
        (0x2490, 1),
    ] {
        program_three_d(&mut channel, method, argument);
    }
    let role = MaxwellThreeDResourceRole::SampledImage {
        texture: MaxwellThreeDTextureReference::new(4, 0, 0),
        dimension: MaxwellThreeDTextureDimension::Two,
    };
    let mut cache = MaxwellLoweringCache::default();
    let mut invalidations = Vec::new();
    let prepare = |width, cache: &mut MaxwellLoweringCache, invalidations: &mut Vec<_>| {
        tic(&metadata, address, width, 0x58d2_4908);
        let resources =
            resolve_maxwell_three_d_resources_for_roles(channel.three_d(), &space, &[role])
                .unwrap();
        let index = resource_index(&resources, role).unwrap();
        prepare_resources(&resources, &[index], cache, &mut Vec::new(), invalidations).unwrap()
            [index]
            .unwrap()
    };
    let first = prepare(64, &mut cache, &mut invalidations);
    let overlapping = prepare(32, &mut cache, &mut invalidations);
    assert_ne!(first, overlapping);
    assert!(invalidations.is_empty());
    assert_eq!(prepare(64, &mut cache, &mut invalidations), first);
    assert_eq!(prepare(32, &mut cache, &mut invalidations), overlapping);
    // CPU contents change without changing immutable host view identities.
    pixels.write(0, &[1, 2, 3, 4]).unwrap();
    assert_eq!(prepare(64, &mut cache, &mut invalidations), first);
    cache
        .views
        .values_mut()
        .find(|r| r.dependency == first)
        .unwrap()
        .write_revision = 1;
    let third = prepare(16, &mut cache, &mut invalidations);
    assert!(invalidations.contains(&first));
    assert!(!invalidations.contains(&overlapping));
    assert!(cache.views.values().any(|r| r.dependency == third));
    // Image writes invalidate cached texels in the backend without retiring read-only identities.
    let key = cache
        .views
        .values()
        .find(|r| r.dependency == third)
        .unwrap()
        .key
        .clone();
    retire_overlapping_views(
        &key,
        true,
        std::iter::once(third),
        &mut cache,
        &mut invalidations,
    );
    assert!(!invalidations.contains(&overlapping));
    assert_eq!(cache.views.len(), 2);
}

#[test]
fn vic_producer_and_maxwell_texture_share_the_same_canonical_image() {
    use super::super::threed::{
        MaxwellThreeDTextureDimension, MaxwellThreeDTextureReference,
        resolve_maxwell_three_d_resources_for_roles,
    };
    use nixe_gpu::{ConversionImage, Nv12ColorTransform};
    let mut space =
        MaxwellGpuAddressSpace::new(MaxwellAddressSpaceId::new(1), SWITCH_1_GM20B_PROFILE);
    space
        .initialize(MaxwellAddressSpaceInitialization::default())
        .unwrap();
    let pixels = CanonicalAllocation::zeroed(0x10000, 0x1000).unwrap();
    let metadata = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
    let address = map(&mut space, &pixels, 1, 0xfe);
    let descriptors = map(&mut space, &metadata, 2, 0xfe);
    let mut channel = three_d_channel();
    for (method, argument) in [
        (0x1574, (descriptors >> 32) as u32),
        (0x1578, descriptors as u32),
        (0x157c, 0),
        (0x2380, 4),
        (0x2384, (descriptors >> 32) as u32),
        (0x2388, (descriptors + 0x100) as u32),
        (0x2490, 1),
    ] {
        program_three_d(&mut channel, method, argument);
    }
    tic(&metadata, address, 64, 0x58d2_4908);
    let role = MaxwellThreeDResourceRole::SampledImage {
        texture: MaxwellThreeDTextureReference::new(4, 0, 0),
        dimension: MaxwellThreeDTextureDimension::Two,
    };
    let resources =
        resolve_maxwell_three_d_resources_for_roles(channel.three_d(), &space, &[role]).unwrap();
    let index = resource_index(&resources, role).unwrap();
    let image = resolved_image(&resources, index).unwrap();
    let output = ConversionImage {
        description: image.description(),
        allocation: image.allocation_description(),
        view: image.view().clone(),
    };
    let plane = |id, format, width, height| {
        let storage = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
        let allocation = nixe_gpu::GpuAllocationDescription::new(0x1000, 1).unwrap();
        let description = nixe_gpu::ImageDescription::new(
            nixe_gpu::ImageDimension::Two,
            nixe_gpu::ImageExtent::new(width, height, 1).unwrap(),
            format,
            nixe_gpu::ImageKind::Color,
            1,
            1,
            nixe_gpu::SampleCount::One,
        )
        .unwrap();
        let backing = nixe_gpu::BackingView::new(
            nixe_gpu::GpuAllocationId::new(id),
            allocation,
            0,
            storage
                .backing_range(MemoryPermissions::READ_WRITE)
                .unwrap(),
        )
        .unwrap();
        ConversionImage {
            description,
            allocation,
            view: ImageView::new(
                ImageId::new(0),
                description,
                nixe_gpu::Swizzle::IDENTITY,
                vec![(
                    ImageSubresourceRange {
                        mip_level: 0,
                        base_layer: 0,
                        layer_count: 1,
                        plane: 0,
                    },
                    nixe_gpu::ImageMemoryLayout::PitchLinear {
                        row_pitch: 64,
                        layer_stride: 0x1000,
                    },
                    backing,
                )],
            )
            .unwrap(),
        }
    };
    let images = [
        plane(3, nixe_gpu::ImageFormat::R8Unorm, 64, 32),
        plane(4, nixe_gpu::ImageFormat::Rg8Unorm, 32, 16),
        output,
    ];
    let transform = Nv12ColorTransform {
        coefficients: [[0; 4]; 3],
        shift: 0,
        enabled: false,
        clamp: [0, 1023],
        alpha: 255,
        source_origin: [0, 0],
    };
    let mut cache = MaxwellLoweringCache::default();
    let upload = cache
        .lower_decoded_plane_uploads(
            [images[0].clone(), images[1].clone()],
            [vec![0; 64 * 32].into(), vec![0; 64 * 16].into()],
            [64, 32],
        )
        .unwrap();
    let uploaded = upload
        .submission()
        .operations()
        .iter()
        .filter_map(|op| match op.command() {
            GpuCommand::UploadImage { destination, .. } => Some(destination.image),
            _ => None,
        })
        .collect::<Vec<_>>();
    let work = cache
        .lower_nv12_conversion(images.clone(), transform)
        .unwrap();
    let conversion = work
        .submission()
        .operations()
        .iter()
        .find_map(|op| match op.command() {
            GpuCommand::ConvertNv12(c) => Some(c),
            _ => None,
        })
        .unwrap();
    assert_eq!(
        [conversion.luma.image, conversion.chroma.image],
        uploaded.as_slice(),
        "VIC must retain the NVDEC producer images"
    );
    let destination = work
        .submission()
        .operations()
        .iter()
        .find_map(|op| match op.command() {
            GpuCommand::ConvertNv12(c) => Some(ResourceDependency::Image(c.destination.image)),
            _ => None,
        })
        .unwrap();
    for _ in 0..3 {
        let mut creations = Vec::new();
        let mut invalidations = Vec::new();
        let resolved = prepare_resources(
            &resources,
            &[index],
            &mut cache,
            &mut creations,
            &mut invalidations,
        )
        .unwrap();
        assert_eq!(resolved[index], Some(destination));
        assert!(
            creations.is_empty(),
            "sampling a VIC output must not create an importing texture"
        );
        assert!(invalidations.is_empty());
        let work = cache
            .lower_nv12_conversion(images.clone(), transform)
            .unwrap();
        assert!(work.resource_creations().is_empty());
        assert!(work.resource_invalidations().is_empty());
    }
    // The game may consume the same bytes through an sRGB TIC. Keep that
    // interpretation resident and refresh it once per actual producer write.
    metadata
        .write(16, &((64 - 1) | (1u32 << 23) | (1 << 22)).to_le_bytes())
        .unwrap();
    let srgb_resources =
        resolve_maxwell_three_d_resources_for_roles(channel.three_d(), &space, &[role]).unwrap();
    let srgb_index = resource_index(&srgb_resources, role).unwrap();
    let mut creations = Vec::new();
    let sampled = prepare_resources(
        &srgb_resources,
        &[srgb_index],
        &mut cache,
        &mut creations,
        &mut Vec::new(),
    )
    .unwrap()[srgb_index]
        .unwrap();
    assert_ne!(sampled, destination);
    assert!(matches!(
        &creations[..],
        [BackendResourceCreateInfo::Image { view: None, .. }]
    ));
    assert_eq!(cache.image_alias_copies.len(), 1);
    finish_lowered_work(
        &mut cache,
        FrontendSubmissionId::new(1000),
        vec![],
        creations,
        vec![],
        [],
        Arc::from([]),
    )
    .unwrap();
    assert_eq!(
        prepare_resources(
            &srgb_resources,
            &[srgb_index],
            &mut cache,
            &mut Vec::new(),
            &mut Vec::new()
        )
        .unwrap()[srgb_index],
        Some(sampled)
    );
    assert!(cache.image_alias_copies.is_empty());
    cache.lower_nv12_conversion(images, transform).unwrap();
    assert_eq!(
        prepare_resources(
            &srgb_resources,
            &[srgb_index],
            &mut cache,
            &mut Vec::new(),
            &mut Vec::new()
        )
        .unwrap()[srgb_index],
        Some(sampled)
    );
    assert_eq!(cache.image_alias_copies.len(), 1);
}

#[test]
fn uncompressed_color_sampling_refreshes_resident_copies_after_gpu_and_cpu_writes() {
    use super::super::threed::{
        MaxwellThreeDTextureDimension, MaxwellThreeDTextureReference,
        resolve_maxwell_three_d_resources_for_roles,
    };
    let mut space =
        MaxwellGpuAddressSpace::new(MaxwellAddressSpaceId::new(1), SWITCH_1_GM20B_PROFILE);
    space
        .initialize(MaxwellAddressSpaceInitialization::default())
        .unwrap();
    let pixels = CanonicalAllocation::zeroed(0x10000, 0x1000).unwrap();
    let metadata = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
    let address = map(&mut space, &pixels, 1, 0xfe);
    let descriptors = map(&mut space, &metadata, 2, 0xfe);
    // R8: the attachment includes GOB padding, while the TIC excludes it.
    let format = 0x1d
        | [7, 10, 13, 16]
            .into_iter()
            .map(|shift| 2 << shift)
            .sum::<u32>()
        | (2 << 19)
        | (7 << 28);
    tic(&metadata, address, 63, format);
    let mut channel = three_d_channel();
    for (method, argument) in [
        (0x0800, (address >> 32) as u32),
        (0x0804, address as u32),
        (0x0808, 64),
        (0x080c, 32),
        (0x0810, 0xf3),
        (0x0814, 0),
        (0x0818, 1),
        (0x081c, 0),
        (0x0820, 0),
        (0x15d0, 0),
        (0x19e0, 0),
        (0x1574, (descriptors >> 32) as u32),
        (0x1578, descriptors as u32),
        (0x157c, 0),
        (0x2380, 4),
        (0x2384, (descriptors >> 32) as u32),
        (0x2388, (descriptors + 0x100) as u32),
        (0x2490, 1),
    ] {
        program_three_d(&mut channel, method, argument);
    }
    let target = MaxwellThreeDResourceRole::ColorTarget(0);
    let sampled = MaxwellThreeDResourceRole::SampledImage {
        texture: MaxwellThreeDTextureReference::new(4, 0, 0),
        dimension: MaxwellThreeDTextureDimension::Two,
    };
    let targets =
        resolve_maxwell_three_d_resources_for_roles(channel.three_d(), &space, &[target]).unwrap();
    let textures =
        resolve_maxwell_three_d_resources_for_roles(channel.three_d(), &space, &[sampled]).unwrap();
    let image = resolved_image(&targets, 0).unwrap();
    let mut cache = MaxwellLoweringCache::default();
    let mut creations = Vec::new();
    let mut invalidations = Vec::new();
    let producer = prepare_resources(
        &targets,
        &[0],
        &mut cache,
        &mut creations,
        &mut invalidations,
    )
    .unwrap()[0]
        .unwrap();
    record_image_write(image, &mut cache);
    creations.clear();
    let texture_index = resource_index(&textures, sampled).unwrap();
    let alias = prepare_resources(
        &textures,
        &[texture_index],
        &mut cache,
        &mut creations,
        &mut invalidations,
    )
    .unwrap()[texture_index]
        .unwrap();
    assert_ne!(alias, producer);
    assert!(matches!(
        creations.as_slice(),
        [BackendResourceCreateInfo::Image { view: None, .. }]
    ));
    assert_eq!(cache.image_alias_copies.len(), 1);
    creations.clear();
    cache.image_alias_copies.clear();
    // An unchanged draw reuses the copy without another transfer.
    assert_eq!(
        prepare_resources(
            &textures,
            &[texture_index],
            &mut cache,
            &mut creations,
            &mut invalidations
        )
        .unwrap()[texture_index],
        Some(alias)
    );
    assert!(cache.image_alias_copies.is_empty());
    for cpu_write in [false, true] {
        if cpu_write {
            pixels.write(0, &[7]).unwrap();
        } else {
            cache.revision += 1;
            record_image_write(image, &mut cache);
        }
        assert!(!sampled_alias::copy_is_current(
            &cache.views[&alias],
            &cache
        ));
        assert_eq!(
            prepare_resources(
                &textures,
                &[texture_index],
                &mut cache,
                &mut creations,
                &mut invalidations
            )
            .unwrap()[texture_index],
            Some(alias)
        );
        assert!(matches!(cache.image_alias_copies.as_slice(), [operation]
            if matches!(operation.command(), GpuCommand::Copy(nixe_gpu::CopyOperation::ImageToImage { source, destination })
                if ResourceDependency::Image(source.image) == producer && ResourceDependency::Image(destination.image) == alias && source.extent.width == 63)));
        assert!(creations.is_empty());
        assert!(invalidations.is_empty());
        cache.image_alias_copies.clear();
    }
    // A CPU-imported wide texture can have a non-identity component mapping.
    // An unbacked crop would lose that mapping, so this case must keep its view.
    let replicated = format | (2 << 22) | (2 << 25);
    tic(&metadata, address, 64, replicated);
    let wide =
        resolve_maxwell_three_d_resources_for_roles(channel.three_d(), &space, &[sampled]).unwrap();
    prepare_resources(
        &wide,
        &[texture_index],
        &mut cache,
        &mut creations,
        &mut invalidations,
    )
    .unwrap();
    creations.clear();
    cache.image_alias_copies.clear();
    tic(&metadata, address, 63, replicated);
    let narrow =
        resolve_maxwell_three_d_resources_for_roles(channel.three_d(), &space, &[sampled]).unwrap();
    prepare_resources(
        &narrow,
        &[texture_index],
        &mut cache,
        &mut creations,
        &mut invalidations,
    )
    .unwrap();
    assert!(cache.image_alias_copies.is_empty());
    assert!(
        matches!(creations.as_slice(), [BackendResourceCreateInfo::Image { view: Some(view), .. }]
        if view.swizzle() == nixe_gpu::Swizzle([nixe_gpu::ComponentSwizzle::Red, nixe_gpu::ComponentSwizzle::Red, nixe_gpu::ComponentSwizzle::Red, nixe_gpu::ComponentSwizzle::One]))
    );
}
