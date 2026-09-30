//! Ordinary wgpu color output. Native Vulkan maps the same neutral contract.
use nixe_gpu::{BlendComponent, BlendFactor as F, BlendOperation as O, ColorBlendState};

pub(super) fn blend(state: ColorBlendState) -> wgpu::BlendState {
    wgpu::BlendState {
        color: component(state.color),
        alpha: component(state.alpha),
    }
}

fn component(value: BlendComponent) -> wgpu::BlendComponent {
    let operation = match value.operation {
        O::Add => wgpu::BlendOperation::Add,
        O::Subtract => wgpu::BlendOperation::Subtract,
        O::ReverseSubtract => wgpu::BlendOperation::ReverseSubtract,
        O::Min => wgpu::BlendOperation::Min,
        O::Max => wgpu::BlendOperation::Max,
    };
    // Min/max ignore factors; WebGPU additionally requires canonical One/One.
    // https://www.w3.org/TR/webgpu/#dictdef-gpublendcomponent
    let (src_factor, dst_factor) = if matches!(value.operation, O::Min | O::Max) {
        (wgpu::BlendFactor::One, wgpu::BlendFactor::One)
    } else {
        (factor(value.source), factor(value.destination))
    };
    wgpu::BlendComponent {
        src_factor,
        dst_factor,
        operation,
    }
}

fn factor(value: F) -> wgpu::BlendFactor {
    match value {
        F::Zero => wgpu::BlendFactor::Zero,
        F::One => wgpu::BlendFactor::One,
        F::SourceColor => wgpu::BlendFactor::Src,
        F::OneMinusSourceColor => wgpu::BlendFactor::OneMinusSrc,
        F::SourceAlpha => wgpu::BlendFactor::SrcAlpha,
        F::OneMinusSourceAlpha => wgpu::BlendFactor::OneMinusSrcAlpha,
        F::DestinationAlpha => wgpu::BlendFactor::DstAlpha,
        F::OneMinusDestinationAlpha => wgpu::BlendFactor::OneMinusDstAlpha,
        F::DestinationColor => wgpu::BlendFactor::Dst,
        F::OneMinusDestinationColor => wgpu::BlendFactor::OneMinusDst,
        F::SourceAlphaSaturated => wgpu::BlendFactor::SrcAlphaSaturated,
    }
}
