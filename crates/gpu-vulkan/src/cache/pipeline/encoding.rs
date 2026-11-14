//! Stable, bounded encoding of native pipeline state for guest-environment records.
use super::{BlendState, GraphicsState, StencilState, VertexAttribute, VertexBinding};
use ash::vk;

pub(crate) struct Reader<'a>(pub &'a [u8]);
impl Reader<'_> {
    pub fn u32(&mut self) -> Option<u32> {
        let (head, tail) = self.0.split_at_checked(4)?;
        self.0 = tail;
        Some(u32::from_le_bytes(head.try_into().ok()?))
    }
    pub fn u64(&mut self) -> Option<u64> {
        Some(u64::from(self.u32()?) | (u64::from(self.u32()?) << 32))
    }
    fn boolean(&mut self) -> Option<bool> {
        match self.u32()? {
            0 => Some(false),
            1 => Some(true),
            _ => None,
        }
    }
    fn list<T>(
        &mut self,
        limit: usize,
        mut read: impl FnMut(&mut Self) -> Option<T>,
    ) -> Option<Vec<T>> {
        let count = self.u32()? as usize;
        if count > limit {
            return None;
        }
        (0..count).map(|_| read(self)).collect()
    }
}
pub(crate) fn word(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}
pub(crate) fn wide(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}
impl GraphicsState {
    pub(crate) fn encode(&self, out: &mut Vec<u8>) {
        let mut w = |n| word(out, n);
        w(self.vertex_bindings.len() as u32);
        for b in &*self.vertex_bindings {
            for n in [b.binding, b.stride, b.rate.as_raw() as u32] {
                w(n);
            }
        }
        w(self.vertex_attributes.len() as u32);
        for a in &*self.vertex_attributes {
            for n in [a.location, a.binding, a.format.as_raw() as u32, a.offset] {
                w(n);
            }
        }
        w(self.vertex_divisors.len() as u32);
        for &(binding, divisor) in &*self.vertex_divisors {
            w(binding);
            w(divisor);
        }
        w(self.colors.len() as u32);
        for (format, b) in &*self.colors {
            for n in [
                format.as_raw() as u32,
                b.enabled as u32,
                b.src_color.as_raw() as u32,
                b.dst_color.as_raw() as u32,
                b.color_op.as_raw() as u32,
                b.src_alpha.as_raw() as u32,
                b.dst_alpha.as_raw() as u32,
                b.alpha_op.as_raw() as u32,
                b.write_mask.as_raw(),
            ] {
                w(n);
            }
        }
        for s in [self.front, self.back] {
            for n in [
                s.fail.as_raw(),
                s.pass.as_raw(),
                s.depth_fail.as_raw(),
                s.compare.as_raw(),
            ] {
                w(n as u32);
            }
        }
        for n in [
            self.topology.as_raw() as u32,
            self.primitive_restart as u32,
            self.patch_points,
            self.depth_format.as_raw() as u32,
            self.stencil_format.as_raw() as u32,
            self.samples.as_raw(),
            self.sample_mask as u32,
            (self.sample_mask >> 32) as u32,
            self.alpha_to_coverage as u32,
            self.polygon.as_raw() as u32,
            self.cull.as_raw(),
            self.front_face.as_raw() as u32,
            self.depth_clamp as u32,
            self.raster_discard as u32,
            self.depth_bias as u32,
            self.depth_test as u32,
            self.depth_write as u32,
            self.depth_compare.as_raw() as u32,
            self.depth_bounds as u32,
            self.stencil_test as u32,
            self.logic_enabled as u32,
            self.logic_op.as_raw() as u32,
            self.viewport_count,
            self.provoking.as_raw() as u32,
            self.line_mode.as_raw() as u32,
        ] {
            w(n);
        }
    }
    pub(crate) fn decode(r: &mut Reader<'_>) -> Option<Self> {
        let vertex_bindings = r
            .list(32, |r| {
                Some(VertexBinding {
                    binding: r.u32()?,
                    stride: r.u32()?,
                    rate: vk::VertexInputRate::from_raw(r.u32()? as i32),
                })
            })?
            .into();
        let vertex_attributes = r
            .list(32, |r| {
                Some(VertexAttribute {
                    location: r.u32()?,
                    binding: r.u32()?,
                    format: vk::Format::from_raw(r.u32()? as i32),
                    offset: r.u32()?,
                })
            })?
            .into();
        let vertex_divisors = r.list(32, |r| Some((r.u32()?, r.u32()?)))?.into();
        let colors = r
            .list(8, |r| {
                Some((
                    vk::Format::from_raw(r.u32()? as i32),
                    BlendState {
                        enabled: r.boolean()?,
                        src_color: vk::BlendFactor::from_raw(r.u32()? as i32),
                        dst_color: vk::BlendFactor::from_raw(r.u32()? as i32),
                        color_op: vk::BlendOp::from_raw(r.u32()? as i32),
                        src_alpha: vk::BlendFactor::from_raw(r.u32()? as i32),
                        dst_alpha: vk::BlendFactor::from_raw(r.u32()? as i32),
                        alpha_op: vk::BlendOp::from_raw(r.u32()? as i32),
                        write_mask: vk::ColorComponentFlags::from_raw(r.u32()?),
                    },
                ))
            })?
            .into();
        let mut stencil = || {
            Some(StencilState {
                fail: vk::StencilOp::from_raw(r.u32()? as i32),
                pass: vk::StencilOp::from_raw(r.u32()? as i32),
                depth_fail: vk::StencilOp::from_raw(r.u32()? as i32),
                compare: vk::CompareOp::from_raw(r.u32()? as i32),
            })
        };
        let front = stencil()?;
        let back = stencil()?;
        Some(Self {
            vertex_bindings,
            vertex_attributes,
            vertex_divisors,
            colors,
            front,
            back,
            topology: vk::PrimitiveTopology::from_raw(r.u32()? as i32),
            primitive_restart: r.boolean()?,
            patch_points: r.u32()?,
            depth_format: vk::Format::from_raw(r.u32()? as i32),
            stencil_format: vk::Format::from_raw(r.u32()? as i32),
            samples: vk::SampleCountFlags::from_raw(r.u32()?),
            sample_mask: r.u64()?,
            alpha_to_coverage: r.boolean()?,
            polygon: vk::PolygonMode::from_raw(r.u32()? as i32),
            cull: vk::CullModeFlags::from_raw(r.u32()?),
            front_face: vk::FrontFace::from_raw(r.u32()? as i32),
            depth_clamp: r.boolean()?,
            raster_discard: r.boolean()?,
            depth_bias: r.boolean()?,
            depth_test: r.boolean()?,
            depth_write: r.boolean()?,
            depth_compare: vk::CompareOp::from_raw(r.u32()? as i32),
            depth_bounds: r.boolean()?,
            stencil_test: r.boolean()?,
            logic_enabled: r.boolean()?,
            logic_op: vk::LogicOp::from_raw(r.u32()? as i32),
            viewport_count: r.u32()?,
            provoking: vk::ProvokingVertexModeEXT::from_raw(r.u32()? as i32),
            line_mode: vk::LineRasterizationModeEXT::from_raw(r.u32()? as i32),
        })
    }
}
