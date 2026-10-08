//! Color copies and resolves share the resident image identities of clear and draw.
use super::*;

impl MaxwellLoweringCache {
    pub(crate) fn lower_solid_rect(
        &mut self,
        request: &super::super::twod::MaxwellTwoDSolidOperation,
        resources: &MaxwellThreeDResolvedResources,
        submission: FrontendSubmissionId,
        predecessors: Vec<FrontendSubmissionId>,
    ) -> Result<MaxwellLoweredWork, MaxwellLoweringError> {
        let image = resolved_image(resources, 0)?;
        let [x0, y0, x1, y1] = request.rectangle;
        // The rectangle overwrites both aspects of every covered pixel. It
        // does not read compressed bytes outside the region. Keep uncovered
        // pixels unmaterialized instead of importing opaque bytes or clearing
        // alignment padding which the guest did not request.
        let mut creations = Vec::new();
        let mut invalidations = std::mem::take(&mut self.retired_resources);
        let bindings = prepare_resources(
            resources,
            &[0],
            vec![None; resources.resources().len()],
            self,
            &mut creations,
            &mut invalidations,
        )?;
        let target = ImageRegion {
            image: image_dependency(binding_at(resources, &bindings, 0)?)?,
            subresources: image.view().bindings()[0].subresources(),
            origin: ImageOrigin { x: x0, y: y0, z: 0 },
            extent: nixe_gpu::ImageExtent {
                width: x1 - x0,
                height: y1 - y0,
                depth: 1,
            },
        };
        let value = solid_clear_value(
            request.color,
            request.destination.format,
            image.guest_format(),
        )?;
        let clear = ClearOperation::image(
            target,
            image.description().kind(),
            image.description().format(),
            image.description().samples(),
            value,
        )
        .map_err(MaxwellLoweringError::Command)?;
        let record = self
            .views
            .iter_mut()
            .find(|v| v.dependency == ResourceDependency::Image(target.image))
            .expect("prepared solid rectangle view exists");
        if request.zeta {
            if let ViewMaterialization::CompressedDepthStencil { .. } = record.materialization {
                for regions in &mut record.uninitialized_depth_stencil_regions {
                    subtract_initialized_rect(regions, request.rectangle);
                }
                record.materialization = ViewMaterialization::CompressedDepthStencil {
                    depth: record.uninitialized_depth_stencil_regions[0].is_empty(),
                    stencil: record.uninitialized_depth_stencil_regions[1].is_empty(),
                };
            }
        } else {
            subtract_initialized_rect(&mut record.uninitialized_color_regions, request.rectangle);
            if record.uninitialized_color_regions.is_empty() {
                record_color_materialization(image, self);
            }
        }
        record_image_write(image, self);
        finish_lowered_work(
            self,
            submission,
            predecessors,
            creations,
            invalidations,
            [GpuOperation::new(
                GpuCommand::Clear(clear),
                [],
                [],
                CapabilityRequirements::none(),
            )],
            Arc::from([0usize]),
        )
    }

    pub(crate) fn lower_color_blit(
        &mut self,
        request: &super::super::twod::MaxwellTwoDBlitOperation,
        resources: &MaxwellThreeDResolvedResources,
        submission: FrontendSubmissionId,
        predecessors: Vec<FrontendSubmissionId>,
    ) -> Result<MaxwellLoweredWork, MaxwellLoweringError> {
        let source = resolved_image(resources, 0)?;
        let destination = resolved_image(resources, 1)?;
        if !resources.aliases().is_empty() {
            return Err(MaxwellLoweringError::AliasedDrawResources {
                first: source.role(),
                second: destination.role(),
            });
        }
        // Never substitute CPU bytes or a newly allocated texture for a source
        // initialized by 3D work. CPU writes invalidate this resident evidence.
        if (source.guest_layout().requires_materialization()
            && !source.guest_layout().has_direct_canonical_representation())
            && !self.views.iter().any(|view| {
                view.remains_current_for_image(source)
                    && (!matches!(
                        view.materialization,
                        ViewMaterialization::CopiedColor { .. }
                    ) || sampled_alias::copy_is_current(view, self))
                    && view.uninitialized_color_regions.is_empty()
            })
        {
            return Err(MaxwellLoweringError::BlitSourceNotResident);
        }
        let mut creations = Vec::new();
        let mut invalidations = std::mem::take(&mut self.retired_resources);
        let bindings = prepare_resources(
            resources,
            &[0, 1],
            vec![None; resources.resources().len()],
            self,
            &mut creations,
            &mut invalidations,
        )?;
        let region = |index,
                      image: &super::super::threed::MaxwellThreeDResolvedImage|
         -> Result<ImageRegion, MaxwellLoweringError> {
            Ok(ImageRegion {
                image: image_dependency(binding_at(resources, &bindings, index)?)?,
                subresources: image.view().bindings()[0].subresources(),
                origin: ImageOrigin { x: 0, y: 0, z: 0 },
                extent: image.description().extent(),
            })
        };
        let command = match request.kind {
            super::super::twod::blit::BlitKind::Resolve => GpuCommand::Resolve(
                nixe_gpu::ResolveOperation::new(
                    region(0, source)?,
                    region(1, destination)?,
                    source.description().format(),
                    source.description().samples(),
                )
                .map_err(MaxwellLoweringError::Command)?,
            ),
            super::super::twod::blit::BlitKind::Copy { origins, extent } => {
                let mut src = region(0, source)?;
                let mut dst = region(1, destination)?;
                src.origin = origins[0];
                src.extent = extent;
                dst.origin = origins[1];
                dst.extent = extent;
                let record = self
                    .views
                    .iter_mut()
                    .find(|record| record.dependency == ResourceDependency::Image(dst.image))
                    .expect("prepared destination view exists");
                subtract_initialized_rect(
                    &mut record.uninitialized_color_regions,
                    [
                        dst.origin.x,
                        dst.origin.y,
                        dst.origin.x + extent.width,
                        dst.origin.y + extent.height,
                    ],
                );
                GpuCommand::Copy(nixe_gpu::CopyOperation::ImageToImage {
                    source: src,
                    destination: dst,
                })
            }
        };
        record_image_write(destination, self);
        if request.kind == super::super::twod::blit::BlitKind::Resolve
            || self.views.iter().any(|record| {
                record.remains_current_for_image(destination)
                    && record.uninitialized_color_regions.is_empty()
            })
        {
            record_color_materialization(destination, self);
        }
        finish_lowered_work(
            self,
            submission,
            predecessors,
            creations,
            invalidations,
            [GpuOperation::new(
                command,
                [],
                [],
                CapabilityRequirements::none(),
            )],
            Arc::from([1usize]),
        )
    }
}

fn solid_clear_value(
    packed: u32,
    color_format: u8,
    guest_format: super::super::threed::MaxwellThreeDGuestImageFormat,
) -> Result<ClearValue, MaxwellLoweringError> {
    use super::super::threed::{
        MaxwellThreeDDepthStencilFormat as Depth, MaxwellThreeDGuestImageFormat as Guest,
    };
    match guest_format {
        // NVIDIA names are in MSB-to-LSB order: S8Z24 is Mesa Z24_UNORM_S8_UINT,
        // while Z24S8 is S8_UINT_Z24_UNORM. Preserve all packed bits in a zeta write.
        // https://github.com/chaotic-cx/mesa-mirror/blob/main/src/gallium/drivers/nouveau/nv50/nv50_formats.c#L138-L140
        Guest::DepthStencil(Depth::Stencil8Z24) => Ok(ClearValue::DepthStencil {
            depth: (packed & 0xffffff) as f32 / 16777215.0,
            stencil: (packed >> 24) as u8,
        }),
        Guest::DepthStencil(Depth::Z24Stencil8) => Ok(ClearValue::DepthStencil {
            depth: (packed >> 8) as f32 / 16777215.0,
            stencil: packed as u8,
        }),
        Guest::Color(_) => {
            let bytes = packed.to_le_bytes();
            let rgba = if color_format == 0xcf {
                [bytes[2], bytes[1], bytes[0], bytes[3]]
            } else {
                bytes
            };
            Ok(ClearValue::Color(rgba.map(|v| f32::from(v) / 255.0)))
        }
        _ => Err(MaxwellLoweringError::IncompleteClear(
            "unsupported 2D solid pixel encoding",
        )),
    }
}

/// Exact coverage of uploads, without marking untouched opaque texels as initialized.
pub(super) fn subtract_initialized_rect(regions: &mut Vec<[u32; 4]>, written: [u32; 4]) {
    let mut remaining = Vec::new();
    for [x0, y0, x1, y1] in regions.drain(..) {
        let [wx0, wy0, wx1, wy1] = written;
        let ix0 = x0.max(wx0);
        let iy0 = y0.max(wy0);
        let ix1 = x1.min(wx1);
        let iy1 = y1.min(wy1);
        if ix0 >= ix1 || iy0 >= iy1 {
            remaining.push([x0, y0, x1, y1]);
            continue;
        }
        if y0 < iy0 {
            remaining.push([x0, y0, x1, iy0]);
        }
        if iy1 < y1 {
            remaining.push([x0, iy1, x1, y1]);
        }
        if x0 < ix0 {
            remaining.push([x0, iy0, ix0, iy1]);
        }
        if ix1 < x1 {
            remaining.push([ix1, iy0, x1, iy1]);
        }
    }
    *regions = remaining;
}

#[test]
fn partial_upload_coverage_preserves_holes_and_merges_exactly() {
    let mut regions = vec![[0, 0, 16, 12]];
    subtract_initialized_rect(&mut regions, [4, 3, 12, 9]);
    assert_eq!(
        regions,
        [[0, 0, 16, 3], [0, 9, 16, 12], [0, 3, 4, 9], [12, 3, 16, 9]]
    );
    subtract_initialized_rect(&mut regions, [0, 0, 16, 6]);
    subtract_initialized_rect(&mut regions, [0, 6, 16, 11]);
    assert_eq!(regions, [[0, 11, 16, 12]]);
    subtract_initialized_rect(&mut regions, [0, 11, 8, 12]);
    assert_eq!(regions, [[8, 11, 16, 12]]);
    subtract_initialized_rect(&mut regions, [8, 11, 16, 12]);
    assert!(regions.is_empty());
}

#[cfg(test)]
mod depth_rect_tests {
    use super::*;
    use crate::engines::tests::{program_three_d, three_d_channel};
    use crate::engines::twod::{
        MaxwellTwoDSolidOperation,
        blit::{BlitSurface, BlitSurfaceLayout},
    };
    use crate::{
        MaxwellAddressSpaceId, MaxwellAddressSpaceInitialization, MaxwellAllocationId,
        MaxwellGpuAddressSpace, MaxwellMapRequest, SWITCH_1_GM20B_PROFILE,
    };
    use nixe_memory::{CanonicalAllocation, MemoryPermissions};

    #[test]
    fn partial_depth_rectangles_materialize_only_their_union_and_cpu_writes_invalidate_it() {
        let storage = CanonicalAllocation::zeroed(0x10000, 0x1000).unwrap();
        let mut space =
            MaxwellGpuAddressSpace::new(MaxwellAddressSpaceId::new(1), SWITCH_1_GM20B_PROFILE);
        space
            .initialize(MaxwellAddressSpaceInitialization::default())
            .unwrap();
        let address = space
            .map(MaxwellMapRequest {
                allocation: MaxwellAllocationId::new(1),
                backing: storage
                    .backing_range(MemoryPermissions::READ_WRITE)
                    .unwrap(),
                backing_offset: 0,
                size: 0x10000,
                allocation_alignment: 0x1000,
                page_size: 0,
                kind: 0x51,
                cacheable: true,
                permissions: MemoryPermissions::READ_WRITE,
                fixed_offset: None,
            })
            .unwrap()
            .offset()
            .get();
        let mut channel = three_d_channel();
        program_three_d(&mut channel, 0x10f8, 1);
        program_three_d(&mut channel, 0x139c, 0x7f);
        program_three_d(&mut channel, 0x19d0, 3);
        let source = channel
            .three_d()
            .render_targets()
            .clear()
            .last_surface()
            .source()
            .unwrap();
        let mut request = MaxwellTwoDSolidOperation {
            source,
            destination: BlitSurface {
                address,
                width: 32,
                height: 16,
                format: 0xcf,
                layout: BlitSurfaceLayout::BlockLinear(0),
            },
            compression: true,
            zeta: true,
            color: u32::MAX,
            rectangle: [0, 0, 16, 16],
        };
        let mut cache = MaxwellLoweringCache::default();
        let resources = cache
            .resolved_resources_mut()
            .resolve_solid_image(&request, &space, 16)
            .unwrap();
        cache
            .lower_solid_rect(&request, &resources, FrontendSubmissionId::new(1), vec![])
            .unwrap();
        let trigger = MaxwellThreeDOperationTrigger::ClearSurface { source };
        // A masked stencil write requires the old stencil values. The untouched
        // half must not be mistaken for initialized compressed contents.
        assert!(matches!(
            validate_compressed_depth_materialization(
                channel.three_d(),
                &resources,
                trigger,
                None,
                &cache
            ),
            Err(MaxwellLoweringError::CompressedDepthImportRequired { kind: 0x51 })
        ));
        request.rectangle = [16, 0, 32, 16];
        cache
            .lower_solid_rect(&request, &resources, FrontendSubmissionId::new(2), vec![])
            .unwrap();
        validate_compressed_depth_materialization(
            channel.three_d(),
            &resources,
            trigger,
            None,
            &cache,
        )
        .unwrap();
        storage.write(0, &[0; 4]).unwrap();
        assert!(matches!(
            validate_compressed_depth_materialization(
                channel.three_d(),
                &resources,
                trigger,
                None,
                &cache
            ),
            Err(MaxwellLoweringError::CompressedDepthImportRequired { kind: 0x51 })
        ));
    }
}
