//! Decoder output pools have their own layout, separate from GPU render targets.
use super::image::Backing;
use std::sync::Arc;

#[derive(Debug, Clone, Copy)]
pub(crate) struct Layout {
    pub width: u32,
    pub height: u32,
    pub pitch: usize,
    pub chroma: usize,
    pub size: usize,
}
impl Layout {
    // @SPECULATIVE: format codes and allocation granularity follow observed
    // native SDK requests; the complete image-preference ABI is undocumented.
    // https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/core/hle/service/nx2_gsv.h
    pub fn new(preference: [u32; 16]) -> Result<Self, Box<str>> {
        if !matches!(preference[1], 1 | 4) {
            return Err(format!("unsupported decoder image preference {preference:08x?}").into());
        }
        let width = (preference[2] & 0xffff) + 1;
        let height = (preference[2] >> 16) + 1;
        let pitch = (width as usize).next_multiple_of(64);
        let chroma = pitch * (height as usize).next_multiple_of(128);
        let size = chroma + pitch * (height as usize).div_ceil(2).next_multiple_of(128);
        Ok(Self {
            width,
            height,
            pitch,
            chroma,
            size,
        })
    }
    pub fn bytes(self, count: u32) -> Result<u64, Box<str>> {
        if count == 0 {
            return Err("empty decoder image pool".into());
        }
        (self.size as u64)
            .checked_mul(u64::from(count))
            .ok_or_else(|| "decoder image pool size overflow".into())
    }
}
#[derive(Debug)]
pub(crate) struct Pool {
    pub layout: Layout,
    pub backing: Arc<Backing>,
}
#[derive(Debug, Clone)]
pub(crate) struct Image {
    pub id: u32,
    pub pool: Arc<Pool>,
    pub index: u32,
}
impl Image {
    pub fn offset(&self) -> u64 {
        self.index as u64 * self.pool.layout.size as u64
    }
    pub fn cpu_descriptor(&self) -> Result<[u8; 128], Box<str>> {
        // @SPECULATIVE: kind 10 describes pitch-linear NV12; each plane has a
        // u32 pool offset and u64 row pitch, as consumed by the reference SDK.
        let mut out = [0; 128];
        out[0] = 10;
        for (field, offset) in [
            (8, self.offset()),
            (24, self.offset() + self.pool.layout.chroma as u64),
        ] {
            let offset =
                u32::try_from(offset).map_err(|_| "decoder plane offset exceeds descriptor")?;
            out[field..field + 4].copy_from_slice(&offset.to_le_bytes());
            out[field + 8..field + 16]
                .copy_from_slice(&(self.pool.layout.pitch as u64).to_le_bytes());
        }
        Ok(out)
    }
}
