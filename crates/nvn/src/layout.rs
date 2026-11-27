//! NVN storage geometry, including packaged mip chains and array padding.
//! https://github.com/devkitPro/deko3d/blob/master/source/dk_image.cpp
use super::*;

#[derive(Clone, Copy, Debug)]
pub struct Mip {
    pub offset: u64,
    pub pitch: u32,
    pub block_height: u8,
    pub block_depth: u8,
}
#[derive(Clone, Debug)]
pub struct Layout {
    pub bytes: u64,
    pub layer_stride: u64,
    pub alignment: u32,
    pub block_height: u8,
    pub pitch: u32,
    pub mips: MipList,
}

struct StorageLayout {
    bytes: u64,
    layer_stride: u64,
    alignment: u32,
}

impl Texture {
    pub(crate) fn block(&self) -> core::result::Result<(u32, u32, u32), &'static str> {
        const ASTC: [(u32, u32); 14] = [
            (4, 4),
            (5, 4),
            (5, 5),
            (6, 5),
            (6, 6),
            (8, 5),
            (8, 6),
            (8, 8),
            (10, 5),
            (10, 6),
            (10, 8),
            (10, 10),
            (12, 10),
            (12, 12),
        ];
        if (0x79..=0x94).contains(&self.format) {
            let (w, h) = ASTC[(usize::from(self.format) - 0x79) % 14];
            return Ok((16, w, h));
        }
        if (0x41..=0x50).contains(&self.format) {
            return Ok((
                if matches!(self.format, 0x41 | 0x42 | 0x45 | 0x46 | 0x49 | 0x4a) {
                    8
                } else {
                    16
                },
                4,
                4,
            ));
        }
        let bytes = match self.format {
            1..=4 | 0x31 | 0x51..=0x52 => 1,
            5..=9 | 0xd..=0x10 | 0x32 | 0x39..=0x3c | 0x53..=0x54 | 0x57..=0x58 | 0x95..=0x98 => 2,
            0x19..=0x1c | 0x5d..=0x5e => 3,
            0xa..=0xc
            | 0x11..=0x15
            | 0x25..=0x28
            | 0x33..=0x35
            | 0x37..=0x38
            | 0x3d..=0x40
            | 0x55..=0x56
            | 0x59..=0x5a
            | 0x63..=0x64
            | 0x69..=0x70
            | 0x99..=0x9c => 4,
            0x1d..=0x21 | 0x5f..=0x60 => 6,
            0x16..=0x18 | 0x29..=0x2d | 0x36 | 0x5b..=0x5c | 0x65..=0x66 | 0x71..=0x75 => 8,
            0x22..=0x24 | 0x61..=0x62 => 12,
            0x2e..=0x30 | 0x67..=0x68 | 0x76..=0x78 => 16,
            _ => return Err("unknown NVN format"),
        };
        Ok((bytes, 1, 1))
    }
    pub fn layer_count(&self) -> u32 {
        if self.view_layers == 0 {
            self.viewable_layers(self.first_level) - u32::from(self.first_layer)
        } else {
            self.view_layers.into()
        }
    }
    pub fn level_count(&self) -> u8 {
        if self.view_levels == 0 {
            self.levels - self.first_level
        } else {
            self.view_levels
        }
    }
    pub fn target_kind(&self) -> u8 {
        self.view_target.checked_sub(1).unwrap_or(self.target)
    }
    pub fn viewable_layers(&self, level: u8) -> u32 {
        if self.target == 2 {
            (self.depth >> level).max(1)
        } else {
            self.layers()
        }
    }
    pub fn layers(&self) -> u32 {
        match self.target {
            3 => self.height,
            4 | 6 | 9 => self.depth,
            8 => 6,
            _ => 1,
        }
        .max(1)
    }
    pub fn view_extent(&self) -> [u32; 2] {
        let w = (self.width >> self.first_level).max(1);
        let h = if matches!(self.target, 0 | 3 | 10) {
            1
        } else {
            (self.height >> self.first_level).max(1)
        };
        if self.block_view == 0 {
            return [w, h];
        }
        [
            w.div_ceil(u32::from(self.block_view & 15) + 1),
            h.div_ceil(u32::from(self.block_view >> 4) + 1),
        ]
    }
    pub fn layout(&self) -> core::result::Result<Layout, &'static str> {
        let (storage, mips) = self.calculate_layout(MipList::push)?;
        Ok(Layout {
            bytes: storage.bytes,
            layer_stride: storage.layer_stride,
            alignment: storage.alignment,
            block_height: mips[0].block_height,
            pitch: mips[0].pitch,
            mips,
        })
    }

    pub fn validate_layout(&self) -> core::result::Result<(), &'static str> {
        // Texture initialization needs the storage checks without a mip table.
        self.calculate_layout(|_: &mut (), _| {}).map(|_| ())
    }

    fn calculate_layout<Mips: Default>(
        &self,
        push_mip: impl Fn(&mut Mips, Mip),
    ) -> core::result::Result<(StorageLayout, Mips), &'static str> {
        let (bpp, mut bw, mut bh) = self.block()?;
        if self.block_view != 0 {
            bw = u32::from(self.block_view & 15) + 1;
            bh = u32::from(self.block_view >> 4) + 1;
        }
        if self.width == 0
            || self.width > 1 << 27
            || self.height > 65536
            || self.depth > 65536
            || self.levels > 32
        {
            return Err("invalid texture dimensions");
        }
        let h = if matches!(self.target, 0 | 3 | 10) {
            1
        } else {
            self.height.max(1)
        };
        let z = if self.target == 2 {
            self.depth.max(1)
        } else {
            1
        };
        let (sx, sy) = match self.samples {
            0 | 1 => (1, 1),
            2 => (2, 1),
            4 => (2, 2),
            8 => (4, 2),
            16 => (4, 4),
            _ => return Err("unsupported sample count"),
        };
        let w = self.width.checked_mul(sx).ok_or("texture width overflow")?;
        let rows = h * sy;
        let row = w
            .div_ceil(bw)
            .checked_mul(bpp)
            .ok_or("texture pitch overflow")?;
        let linear = self.flags & 0x50 != 0 || self.target == 10;
        let layers = self.layers();
        let sparse = self.flags & 0x20 != 0;
        let alignment = if self.target == 10 {
            if bpp == 12 { 4 } else { bpp }
        } else if linear {
            if self.flags & 0x40 != 0 { 128 } else { 32 }
        } else if sparse {
            65536
        } else {
            512
        };
        if linear {
            let pitch = if self.target == 10 {
                row
            } else if self.stride == 0 {
                row.checked_next_multiple_of(alignment)
                    .ok_or("linear pitch overflow")?
            } else {
                self.stride
            };
            if pitch < row {
                return Err("linear stride below row size");
            }
            let bytes = u64::from(pitch) * u64::from(h);
            let mut mips = Mips::default();
            push_mip(
                &mut mips,
                Mip {
                    offset: 0,
                    pitch,
                    block_height: 0,
                    block_depth: 0,
                },
            );
            return Ok((
                StorageLayout {
                    bytes,
                    layer_stride: bytes,
                    alignment,
                },
                mips,
            ));
        }
        let pkg = self.packaged_layout();
        let pw = pkg.map_or(0, |n| n >> 8 & 15);
        let auto = |n: u32| n.clamp(1, 16).ilog2();
        let mut th = if matches!(self.target, 0 | 3 | 2) {
            0
        } else {
            auto(rows.div_ceil(bh) * 3 / 16)
        };
        let mut td = if self.target == 2 { auto(z + z / 2) } else { 0 };
        let sparse_shape = if self.target == 2 {
            match bpp {
                1 | 2 => (2, 5),
                4 => (2, 4),
                8 | 16 => (1, 4),
                _ => (th, td),
            }
        } else {
            (
                match bpp {
                    1 => 5,
                    2 | 4 => 4,
                    8 | 16 => 3,
                    _ => th,
                },
                0,
            )
        };
        let large = |x: u32, y: u32| {
            row >= (128 >> (x + y)) * 64 && rows.div_ceil(bh) >= 8 << x && z >= 1 << y
        };
        if let Some(n) = pkg {
            th = n & 15;
            td = n >> 4 & 15
        } else if sparse && large(sparse_shape.0, sparse_shape.1) {
            (th, td) = sparse_shape
        }
        if th > 5 || td > 5 || pw > 7 {
            return Err("invalid packaged texture geometry");
        }
        let adjust = |mut log: u32, unit: u32, dim: u32| {
            while log > 0 && dim <= unit << (log - 1) {
                log -= 1
            }
            log
        };
        let chain = |th: u32, td: u32| -> core::result::Result<(Mips, u64), &'static str> {
            let mut mips = Mips::default();
            let mut offset = 0u64;
            let sh = if sparse && pkg.is_none() {
                sparse_shape.0
            } else {
                th
            };
            let tw = if pw > 0 {
                1 << pw
            } else {
                (128 >> (sh + td)).max(1)
            };
            let mut tiled = sparse || pw > 0;
            for i in 0..self.levels.max(1) {
                let row = (w >> i).max(1).div_ceil(bw) * bpp;
                let height = (rows >> i).max(1).div_ceil(bh);
                let depth = (z >> i).max(1);
                let ah = adjust(th, 8, height);
                let ad = adjust(td, 1, depth);
                tiled = tiled && row >= tw * 64 && height >= 8 << sh && depth >= 1 << td;
                let pitch = row
                    .checked_next_multiple_of(if tiled { tw * 64 } else { 64 })
                    .ok_or("mip pitch overflow")?;
                push_mip(
                    &mut mips,
                    Mip {
                        offset,
                        pitch,
                        block_height: ah as u8,
                        block_depth: ad as u8,
                    },
                );
                offset += u64::from(pitch)
                    * u64::from(height.next_multiple_of(8 << ah))
                    * u64::from(depth.next_multiple_of(1 << ad));
            }
            if sparse {
                offset = offset.next_multiple_of(65536)
            }
            if matches!(self.target, 3 | 4 | 6 | 8 | 9) {
                let align = if pw > 0 {
                    pw + th + td
                } else {
                    adjust(th, 8, h.div_ceil(bh)) + adjust(td, 1, self.depth.max(1))
                };
                offset = offset.next_multiple_of(512 << align)
            }
            Ok((mips, offset))
        };
        let (mut mips, mut stride) = chain(th, td)?;
        if self.flags & 0x200 != 0 && pkg.is_none() && !sparse && (th | td) != 0 {
            let minimal = chain(0, 0)?;
            if minimal.1 != stride {
                (mips, stride) = minimal
            }
        }
        Ok((
            StorageLayout {
                bytes: stride
                    .checked_mul(layers.into())
                    .ok_or("texture size overflow")?,
                layer_stride: stride,
                alignment,
            },
            mips,
        ))
    }
}
