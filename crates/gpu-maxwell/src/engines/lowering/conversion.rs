//! Non-raster producers share image identities, hazards and canonical ownership
//! with Maxwell. A later texture read must not round-trip through CPU memory.
use super::super::threed::MaxwellThreeDGuestImageFormat;
use super::*;
use nixe_gpu::{ConversionImage, Nv12ColorTransform, Nv12Conversion};

impl MaxwellLoweringCache {
    pub fn lower_nv12_conversion(
        &mut self,
        images: [ConversionImage; 3],
        transform: Nv12ColorTransform,
    ) -> Result<MaxwellLoweredWork, MaxwellLoweringError> {
        let format = images[2].description.format();
        let PreparedImages {
            creations,
            invalidations,
            regions,
        } = self.prepare_producer_images(
            images
                .into_iter()
                .enumerate()
                .map(|(i, image)| (image, i == 2)),
        )?;
        // Distinct from guest channel delivery IDs. There is one ordered
        // frontend owner; its revision is strictly monotonic across producers.
        let frontend = FrontendSubmissionId::new(u64::MAX - self.revision);
        finish_lowered_work(
            self,
            frontend,
            Vec::new(),
            creations,
            invalidations,
            [GpuOperation::new(
                GpuCommand::ConvertNv12(Nv12Conversion {
                    luma: regions[0],
                    chroma: regions[1],
                    destination: regions[2],
                    destination_format: format,
                    transform,
                }),
                [],
                [],
                CapabilityRequirements::none(),
            )],
            Arc::from([]),
        )
    }
    /// NVDEC planes become ordinary canonical image producers. CPU reads use
    /// the existing device writeback; VIC retains the same resident identities.
    pub fn lower_decoded_plane_uploads(
        &mut self,
        images: [ConversionImage; 2],
        bytes: [Arc<[u8]>; 2],
        visible_widths: [u32; 2],
    ) -> Result<MaxwellLoweredWork, MaxwellLoweringError> {
        let PreparedImages {
            creations,
            invalidations,
            mut regions,
        } = self.prepare_producer_images(images.into_iter().map(|image| (image, true)))?;
        let commands = regions
            .iter_mut()
            .zip(bytes)
            .zip(visible_widths)
            .map(|((region, bytes), width)| {
                region.extent.width = width;
                GpuOperation::new(
                    GpuCommand::UploadImage {
                        destination: *region,
                        bytes,
                    },
                    [],
                    [],
                    CapabilityRequirements::none(),
                )
            })
            .collect::<Vec<_>>();
        let frontend = FrontendSubmissionId::new(u64::MAX - self.revision);
        finish_lowered_work(
            self,
            frontend,
            Vec::new(),
            creations,
            invalidations,
            commands,
            Arc::from([]),
        )
    }

    pub fn lower_device_plane_uploads(
        &mut self,
        images: [ConversionImage; 2],
        frame: nixe_gpu::DeviceImageFrame,
    ) -> Result<MaxwellLoweredWork, MaxwellLoweringError> {
        let PreparedImages {
            creations,
            invalidations,
            regions,
        } = self.prepare_producer_images(images.into_iter().map(|image| (image, true)))?;
        let mut destinations = [regions[0], regions[1]];
        destinations[0].extent.width = frame.width;
        destinations[1].extent.width = frame.width.div_ceil(2);
        let frontend = FrontendSubmissionId::new(u64::MAX - self.revision);
        finish_lowered_work(
            self,
            frontend,
            Vec::new(),
            creations,
            invalidations,
            [GpuOperation::new(
                GpuCommand::ImportDecodedPlanes {
                    destinations,
                    frame,
                },
                [],
                [],
                CapabilityRequirements::none(),
            )],
            Arc::from([]),
        )
    }

    fn prepare_producer_images(
        &mut self,
        images: impl IntoIterator<Item = (ConversionImage, bool)>,
    ) -> Result<PreparedImages, MaxwellLoweringError> {
        let mut creations = Vec::new();
        let mut invalidations = std::mem::take(&mut self.retired_resources);
        let mut regions = Vec::with_capacity(3);
        for (image, written) in images {
            let [binding] = image.view.bindings() else {
                return Err(MaxwellLoweringError::ContradictoryState {
                    reason: "conversion image requires one canonical domain",
                });
            };
            let allocation = binding.backing().allocation();
            match self.allocations.iter().find(|(id, _)| *id == allocation) {
                Some((_, current)) if *current != image.allocation => {
                    return Err(MaxwellLoweringError::AllocationDescriptionChanged { allocation });
                }
                Some(_) => {}
                None => {
                    self.allocations.push((allocation, image.allocation));
                    creations.push(BackendResourceCreateInfo::Allocation {
                        id: allocation,
                        description: image.allocation,
                    });
                }
            }
            let key = ViewKey::Image {
                description: image.description,
                swizzle: image.view.swizzle(),
                guest_format: MaxwellThreeDGuestImageFormat::Canonical(image.description.format()),
                guest_pte_kind: 0,
                guest_compression_enabled: false,
                bindings: vec![(
                    binding.subresources(),
                    binding.layout(),
                    binding.backing().clone(),
                )]
                .into_boxed_slice(),
                mappings: Arc::from([]),
            };
            let id = if let Some(record) = self.image_views_mut().find(|record| record.key == key) {
                if !record
                    .cpu_writes
                    .as_ref()
                    .is_some_and(CanonicalCpuWriteDependency::remains_current)
                {
                    record.cpu_writes = Some(
                        CanonicalCpuWriteDependency::capture(binding.backing().range())
                            .map_err(|_| MaxwellLoweringError::ResourceExhausted)?,
                    );
                }
                image_dependency(record.dependency)?
            } else {
                retire_overlapping_views(
                    &key,
                    written,
                    regions
                        .iter()
                        .map(|region: &ImageRegion| ResourceDependency::Image(region.image)),
                    self,
                    &mut invalidations,
                );
                let id = ImageId::new(take_identity(self)?);
                let view = ImageView::new(
                    id,
                    image.description,
                    image.view.swizzle(),
                    vec![(
                        binding.subresources(),
                        binding.layout(),
                        binding.backing().clone(),
                    )],
                )
                .map_err(|_| MaxwellLoweringError::ResourceExhausted)?;
                creations.push(BackendResourceCreateInfo::Image {
                    id,
                    description: image.description,
                    view: Some(view),
                });
                self.views.insert(
                    ResourceDependency::Image(id),
                    ViewRecord {
                        key,
                        dependency: ResourceDependency::Image(id),
                        materialization: ViewMaterialization::Direct,
                        cpu_writes: Some(
                            nixe_memory::CanonicalCpuWriteDependency::capture(
                                binding.backing().range(),
                            )
                            .map_err(|_| MaxwellLoweringError::ResourceExhausted)?,
                        ),
                        write_revision: 0,
                        last_used: 0,
                        uninitialized_color_regions: Vec::new(),
                        uninitialized_depth_stencil_regions: Default::default(),
                    },
                );
                id
            };
            regions.push(ImageRegion {
                image: id,
                subresources: binding.subresources(),
                origin: ImageOrigin { x: 0, y: 0, z: 0 },
                extent: image.description.extent(),
            });
        }
        Ok(PreparedImages {
            creations,
            invalidations,
            regions,
        })
    }
}

struct PreparedImages {
    creations: Vec<BackendResourceCreateInfo>,
    invalidations: Vec<ResourceDependency>,
    regions: Vec<ImageRegion>,
}
