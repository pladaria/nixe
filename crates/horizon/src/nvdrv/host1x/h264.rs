//! NVDEC H.264 parameter-set reconstruction and progressive surface selection.
//! https://github.com/eden-emulator/mirror/blob/master/src/video_core/host1x/codec_types.h
//! https://github.com/eden-emulator/mirror/blob/master/src/video_core/host1x/codecs/h264.cpp
//! H.264 clauses 7.3 and 9.1: https://www.itu.int/rec/T-REC-H.264
use super::video::word;

pub(super) struct Picture {
    pub stream_length: usize,
    pub width: usize,
    pub height: usize,
    pub luma_pitch: usize,
    pub chroma_pitch: usize,
    pub block_height: u32,
    pub luma: u64,
    pub frame_number: u16,
}
impl Picture {
    pub(super) fn decode(bytes: &[u8], registers: &[u32]) -> Result<Self, String> {
        if bytes.len() != 0x2fc {
            return Err("invalid NVDEC H.264 context size".into());
        }
        let p = &bytes[0x58..0xb8];
        let flags = u64::from_le_bytes(p[0x58..0x60].try_into().unwrap());
        let index = (flags >> 34 & 0x7f) as usize;
        if index >= 17 {
            return Err("NVDEC H.264 current surface index exceeds its table".into());
        }
        if word(p, 8) != 1
            || word(p, 0x3c) != 0
            || word(p, 0x40) != 0
            || word(p, 0x48) != 0
            || word(p, 0x4c) != 0
            || flags & (1 << 5) != 0
        {
            return Err("interlaced NVDEC H.264 surfaces are not implemented".into());
        }
        if flags >> 63 != 0 || word(p, 0x14) & 3 != 0 {
            return Err("NVDEC H.264 requires block-linear surfaces".into());
        }
        if flags >> 12 & 3 != 1 {
            return Err("NVDEC H.264 chroma format is not 8-bit 4:2:0".into());
        }
        let width = word(p, 12) as usize * 16;
        let height = word(p, 16) as usize * 16;
        let luma_pitch = word(p, 0x34) as usize;
        let chroma_pitch = word(p, 0x38) as usize;
        let block_height = (word(p, 0x14) >> 2 & 7) + 1;
        if width == 0
            || height == 0
            || width > 16384
            || height > 16384
            || luma_pitch < width
            || chroma_pitch < width
            || !luma_pitch.is_multiple_of(64)
            || !chroma_pitch.is_multiple_of(64)
            || block_height > 5
        {
            return Err("invalid NVDEC H.264 dimensions/pitches".into());
        }
        Ok(Self {
            stream_length: word(bytes, 0x48) as usize,
            width,
            height,
            luma_pitch,
            chroma_pitch,
            block_height,
            luma: (u64::from(registers[0x10c + index]) << 8) + (u64::from(word(p, 0x44)) << 8),
            frame_number: (flags >> 46 & 0xffff) as u16,
        })
    }
}

#[derive(Default)]
struct Bits {
    bytes: Vec<u8>,
    used: u8,
}
impl Bits {
    fn bit(&mut self, value: bool) {
        if self.used == 0 {
            self.bytes.push(0);
        }
        if value {
            *self.bytes.last_mut().unwrap() |= 1 << (7 - self.used);
        }
        self.used = (self.used + 1) & 7;
    }
    fn bits(&mut self, value: u32, count: u32) {
        for i in (0..count).rev() {
            self.bit(value >> i & 1 != 0);
        }
    }
    fn ue(&mut self, value: u32) {
        let code = value + 1;
        let size = 32 - code.leading_zeros();
        for _ in 1..size {
            self.bit(false);
        }
        self.bits(code, size);
    }
    fn se(&mut self, value: i32) {
        self.ue(if value <= 0 {
            (-value as u32) * 2
        } else {
            value as u32 * 2 - 1
        });
    }
    fn end(mut self, header: u8, output: &mut Vec<u8>) {
        self.bit(true);
        output.extend_from_slice(&[0, 0, 1, header]);
        // Emulation prevention belongs to the NAL byte stream, after RBSP coding.
        let mut zeros = 0;
        for byte in self.bytes {
            if zeros == 2 && byte <= 3 {
                output.push(3);
                zeros = 0;
            }
            output.push(byte);
            zeros = if byte == 0 { zeros + 1 } else { 0 };
        }
    }
    fn scaling(&mut self, list: &[u8]) {
        const SCAN4: [usize; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];
        const SCAN8: [usize; 64] = [
            0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34,
            27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37,
            44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
        ];
        let scan = if list.len() == 16 {
            &SCAN4[..]
        } else {
            &SCAN8[..]
        };
        let mut previous = 8_i32;
        for &i in scan {
            let value = i32::from(list[i]);
            self.se(value - previous);
            previous = value;
        }
    }
}

pub(super) fn headers(context: &[u8]) -> Result<Vec<u8>, String> {
    let p = &context[0x58..0xb8];
    let flags = u64::from_le_bytes(p[0x58..0x60].try_into().unwrap());
    let signed = |shift, bits: u32| {
        ((((flags >> shift) & ((1_u64 << bits) - 1)) as i64) << (64 - bits) >> (64 - bits)) as i32
    };
    let order = flags >> 14 & 3;
    let log_lsb = word(p, 0) as i32;
    if flags >> 8 & 15 > 12
        || order > 2
        || (order == 0 && !(0..=12).contains(&log_lsb))
        || word(p, 0x20) > 31
        || word(p, 0x24) > 31
    {
        return Err("invalid NVDEC H.264 parameter-set fields".into());
    }
    let mut output = Vec::with_capacity(256);
    let mut s = Bits::default();
    s.bits(100, 8);
    s.bits(0, 8);
    s.bits(31, 8);
    s.ue(0);
    s.ue((flags >> 12 & 3) as u32);
    s.ue(0);
    s.ue(0);
    s.bit(word(context, 0x2d0) & 2 != 0);
    s.bit(false);
    s.ue((flags >> 8 & 15) as u32);
    s.ue(order as u32);
    if order == 0 {
        s.ue(log_lsb as u32);
    } else if order == 1 {
        s.bit(word(p, 4) != 0);
        s.se(0);
        s.se(0);
        s.ue(0);
    }
    s.ue((word(p, 0x20).max(word(p, 0x24)) + 1).max(4));
    s.bit(false);
    s.ue(word(p, 12) - 1);
    s.ue(word(p, 16) - 1);
    s.bit(true);
    s.bit(flags & 2 != 0);
    s.bit(false);
    s.bit(false);
    s.end(0x67, &mut output);
    let mut s = Bits::default();
    s.ue(0);
    s.ue(0);
    s.bit(word(p, 0x18) != 0);
    s.bit(word(p, 0x1c) != 0);
    s.ue(0);
    s.ue(word(p, 0x20));
    s.ue(word(p, 0x24));
    s.bit(flags & 4 != 0);
    s.bits((flags >> 32 & 3) as u32, 2);
    s.se(signed(16, 6));
    s.se(0);
    s.se(signed(22, 5));
    s.bit(word(p, 0x28) != 0);
    s.bit(flags & 8 != 0);
    s.bit(word(p, 0x2c) != 0);
    s.bit(word(p, 0x30) != 0);
    s.bit(true);
    for list in context[0x1c0..0x220].chunks_exact(16) {
        s.bit(true);
        s.scaling(list);
    }
    if word(p, 0x30) != 0 {
        for list in context[0x220..0x2a0].chunks_exact(64) {
            s.bit(true);
            s.scaling(list);
        }
    }
    s.se(signed(27, 5));
    s.end(0x68, &mut output);
    Ok(output)
}
