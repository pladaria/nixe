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
            BackendResourceCreateInfo::Image { view, .. } if view.is_none() == (compression != 0)
        )));
        creations.clear();
        // Opaque storage requires a recorded clear/render operation; a direct
        // producer already initializes its resident image from canonical bytes.
        assert_eq!(
            prepare(
                &textures,
                texture_index,
                &mut cache,
                &mut creations,
                &mut invalidations
            )
            .is_err(),
            compression != 0
        );
        if compression != 0 {
            record_color_materialization(image, &mut cache);
        }
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
            creations[0],
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
        let position = cache
            .views
            .iter()
            .position(|record| Some(record.dependency) == target_binding)
            .unwrap();
        let resident = cache.views.remove(position);
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
        cache.views.push(resident);
        pixels.write(0, &[0; 4]).unwrap();
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
