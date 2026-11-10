//! Presentation kernels from Eden's present/filters.cpp and present/fsr.cpp.
//! https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/renderer_vulkan/present/filters.cpp
use super::*;
use crate::cache::*;
use ash::vk;
use nixe_gpu::{
    ShaderStage,
    shader_module::{self as spirv, ResourceId},
};
use nixe_video::ScalingFilter;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum Program {
    Fxaa,
    SmaaEdges,
    SmaaWeights,
    SmaaBlend,
    Filter {
        filter: ScalingFilter,
        packed: bool,
        half: bool,
    },
    Easu {
        packed: bool,
        half: bool,
    },
    Rcas {
        half: bool,
    },
    Ui,
}
impl Program {
    pub fn filter(filter: ScalingFilter, packed: bool, half: bool) -> Self {
        Self::Filter {
            filter,
            packed,
            half: half && filter == ScalingFilter::ScaleForce,
        }
    }
    fn fragment(self) -> &'static [u8] {
        macro_rules! module {
            ($name:literal) => {
                include_bytes!(concat!("../../shaders/spv/", $name, ".frag.spv"))
            };
        }
        match self {
            Self::Fxaa => module!("fxaa"),
            Self::SmaaEdges => module!("smaa-edge-detection"),
            Self::SmaaWeights => module!("smaa-blending-weight-calculation"),
            Self::SmaaBlend => module!("smaa-neighborhood-blending"),
            Self::Filter {
                filter: ScalingFilter::Nearest,
                packed: false,
                ..
            } => module!("screen"),
            Self::Filter {
                filter: ScalingFilter::Nearest,
                packed: true,
                ..
            } => module!("screen-nearest-packed"),
            Self::Filter {
                filter: ScalingFilter::Bilinear,
                packed: false,
                ..
            } => module!("screen"),
            Self::Filter {
                filter: ScalingFilter::Bilinear,
                packed: true,
                ..
            } => module!("screen-packed"),
            Self::Filter {
                filter: ScalingFilter::Bicubic,
                packed: false,
                ..
            } => module!("screen-bicubic"),
            Self::Filter {
                filter: ScalingFilter::Bicubic,
                packed: true,
                ..
            } => module!("screen-bicubic-packed"),
            Self::Filter {
                filter: ScalingFilter::Lanczos,
                packed: false,
                ..
            } => module!("screen-lanczos"),
            Self::Filter {
                filter: ScalingFilter::Lanczos,
                packed: true,
                ..
            } => module!("screen-lanczos-packed"),
            Self::Filter {
                filter: ScalingFilter::Gaussian,
                packed: false,
                ..
            } => module!("screen-gaussian"),
            Self::Filter {
                filter: ScalingFilter::Gaussian,
                packed: true,
                ..
            } => module!("screen-gaussian-packed"),
            Self::Filter {
                filter: ScalingFilter::Area,
                packed: false,
                ..
            } => module!("screen-area"),
            Self::Filter {
                filter: ScalingFilter::Area,
                packed: true,
                ..
            } => module!("screen-area-packed"),
            Self::Filter {
                filter: ScalingFilter::ScaleForce,
                packed: false,
                half: false,
            } => module!("screen-scaleforce-fp32"),
            Self::Easu {
                packed: false,
                half: false,
            } => module!("screen-fsr-easu-fp32"),
            Self::Filter {
                filter: ScalingFilter::ScaleForce,
                packed: true,
                half: false,
            } => module!("screen-scaleforce-fp32-packed"),
            Self::Easu {
                packed: true,
                half: false,
            } => module!("screen-fsr-easu-fp32-packed"),
            Self::Rcas { half: false } => module!("screen-fsr-rcas-fp32"),
            Self::Filter {
                filter: ScalingFilter::ScaleForce,
                packed: false,
                half: true,
            } => module!("screen-scaleforce-fp16"),
            Self::Easu {
                packed: false,
                half: true,
            } => module!("screen-fsr-easu-fp16"),
            Self::Filter {
                filter: ScalingFilter::ScaleForce,
                packed: true,
                half: true,
            } => module!("screen-scaleforce-fp16-packed"),
            Self::Easu {
                packed: true,
                half: true,
            } => module!("screen-fsr-easu-fp16-packed"),
            Self::Rcas { half: true } => module!("screen-fsr-rcas-fp16"),
            Self::Ui => module!("ui"),
            Self::Filter {
                filter: ScalingFilter::Fsr,
                ..
            } => unreachable!("FSR has two dedicated passes"),
        }
    }
    pub fn shader(self) -> Arc<ShaderArtifact> {
        let ui = self == Self::Ui;
        let vertex: &[u8] = match self {
            Self::Fxaa => include_bytes!("../../shaders/spv/fxaa.vert.spv"),
            Self::SmaaEdges => include_bytes!("../../shaders/spv/smaa-edge-detection.vert.spv"),
            Self::SmaaWeights => {
                include_bytes!("../../shaders/spv/smaa-blending-weight-calculation.vert.spv")
            }
            Self::SmaaBlend => {
                include_bytes!("../../shaders/spv/smaa-neighborhood-blending.vert.spv")
            }
            Self::Ui => include_bytes!("../../shaders/spv/ui.vert.spv"),
            _ => include_bytes!("../../shaders/spv/screen.vert.spv"),
        };
        let aa = matches!(
            self,
            Self::Fxaa | Self::SmaaEdges | Self::SmaaWeights | Self::SmaaBlend
        );
        let push_size = match self {
            Self::Ui => 0,
            Self::Easu { .. } | Self::Rcas { .. } => 96,
            Self::Filter { .. } => 40,
            Self::Fxaa | Self::SmaaEdges | Self::SmaaWeights | Self::SmaaBlend => 16,
        };
        let images = match self {
            Self::SmaaWeights => 3,
            Self::SmaaBlend => 2,
            _ => 1,
        };
        let kinds: Vec<_> = if aa {
            vec![spirv::DescriptorKind::CombinedImageSampler; images]
        } else {
            vec![
                spirv::DescriptorKind::SampledImage,
                spirv::DescriptorKind::Sampler,
            ]
        };
        let bindings = kinds
            .into_iter()
            .enumerate()
            .map(|(i, kind)| spirv::Binding {
                resource: ResourceId::new(i as u32),
                binding: i as u32,
                kind,
                count: 1,
                sampler: None,
                readable: true,
                writable: false,
            })
            .collect();
        Arc::new(ShaderArtifact::new(Box::new([
            crate::residency::helper_stage(
                ShaderStage::Vertex,
                vertex,
                Box::new([]),
                if ui {
                    8
                } else if aa {
                    16
                } else {
                    0
                },
            ),
            crate::residency::helper_stage(
                ShaderStage::Fragment,
                self.fragment(),
                bindings,
                push_size,
            ),
        ])))
    }
}

pub(super) struct FilterTarget {
    pub view: Arc<AllocatedImageView>,
    pub framebuffer: Arc<crate::render_pass::Framebuffer>,
    pub extent: [u32; 2],
    pub format: vk::Format,
}
impl FilterTarget {
    pub fn new(
        cache: &mut crate::residency::Residency<'_>,
        extent: [u32; 2],
        format: vk::Format,
    ) -> Result<Self, VulkanError> {
        let image = cache.allocate_host_image(extent, format)?;
        let view = image.view_2d(format)?;
        let framebuffer = crate::render_pass::Framebuffer::color(
            &view,
            format,
            vk::Extent2D {
                width: extent[0],
                height: extent[1],
            },
        )?;
        Ok(Self {
            view,
            framebuffer,
            extent,
            format,
        })
    }
    pub fn layer(
        &self,
        encoding: nixe_gpu::PresentationImageFormat,
        transform: nixe_gpu::FrameTransform,
    ) -> render::LayerImage {
        render::LayerImage {
            view: Arc::clone(&self.view),
            extent: self.extent,
            format: self.format,
            encoding,
            crop: nixe_gpu::FrameCrop {
                left: 0,
                top: 0,
                width: self.extent[0],
                height: self.extent[1],
            },
            transform,
        }
    }
}

/// AMD FsrEasuConOffset and FsrRcasCon, with Eden's desktop sharpness of 0.25.
/// https://github.com/GPUOpen-Effects/FidelityFX-FSR/blob/master/ffx-fsr/ffx_fsr1.h
pub(super) fn fsr_constants(frame: &render::LayerImage, output: [u32; 2], easu: bool) -> [u8; 96] {
    let [iw, ih] = frame.extent.map(|v| v as f32);
    let [ow, oh] = output.map(|v| v as f32);
    let [cw, ch] = [frame.crop.width as f32, frame.crop.height as f32];
    let [cx, cy] = [frame.crop.left as f32, frame.crop.top as f32];
    let mut push = [0u8; 96];
    let values = if easu {
        [
            cw / ow,
            ch / oh,
            0.5 * cw / ow - 0.5 + cx,
            0.5 * ch / oh - 0.5 + cy,
            1.0 / iw,
            1.0 / ih,
            1.0 / iw,
            -1.0 / ih,
            -1.0 / iw,
            2.0 / ih,
            1.0 / iw,
            2.0 / ih,
            0.0,
            4.0 / ih,
            0.0,
            0.0,
        ]
    } else {
        [0.0; 16]
    };
    for (i, v) in values.into_iter().enumerate() {
        push[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    if !easu {
        let sharpness = (-0.25f32).exp2();
        push[..4].copy_from_slice(&sharpness.to_le_bytes());
        // AMD packs the FP16 sharpness into con0.y (round-to-zero, as Eden).
        let half = ((sharpness.to_bits() >> 13) - 0x1c000) as u16;
        let packed = u32::from(half) * 0x10001;
        push[4..8].copy_from_slice(&packed.to_le_bytes());
    }
    for (i, v) in [cx / iw, cy / ih, cw / iw, ch / ih].into_iter().enumerate() {
        push[64 + i * 4..68 + i * 4].copy_from_slice(&v.to_le_bytes());
    }
    push[88..92].copy_from_slice(
        &u32::from(frame.encoding == nixe_gpu::PresentationImageFormat::Rgbx8).to_le_bytes(),
    );
    push[92..96].copy_from_slice(
        &u32::from(frame.encoding == nixe_gpu::PresentationImageFormat::Rgb565).to_le_bytes(),
    );
    push
}
