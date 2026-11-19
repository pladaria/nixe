use super::*;

const _: () = assert!(size_of::<Program>() <= 128);

pub fn shaders(c: &Call) -> Result<u64> {
    set_shaders(c, 16)
}
pub fn shaders_ext(c: &Call) -> Result<u64> {
    set_shaders(c, 24)
}
fn set_shaders(c: &Call, stride: u64) -> Result<u64> {
    let n = c.args[1] as u32 as usize;
    if n > 2 {
        return Err(format!("unsupported NVN program with {n} shader stages").into());
    }
    let mut p = Program {
        count: n as u64,
        ..Default::default()
    };
    for (i, out) in p.stages[..n].iter_mut().enumerate() {
        let [address, control]: [u64; 2] = c.read(c.args[2] + stride * i as u64)?;
        // GLSLC Maxwell control sections carry the extent of the code allocation
        // and the offset of its SPH. Only the code remains live after SetShaders.
        let [constants, size, header]: [u32; 3] = c.read(control + 0x700)?;
        if constants > size || header >= size || size > 16 << 20 {
            return Err(format!("invalid shader extent {size:#x} / header {header:#x}").into());
        }
        let reader = c.session.graphics()?.heap.device().reader();
        let mut magic = [0u8; 4];
        reader
            .read_block_unsafe(address, &mut magic)
            .map_err(|e| e.to_string())?;
        let [x, y, z, shared]: [u32; 4] = c.read(control + 0x748)?;
        let magic = u32::from_le_bytes(magic);
        let stage = if magic == 0x12345679 || (magic != 0x12345678 && x != 0 && y != 0 && z != 0) {
            5
        } else {
            let mut sph = [0; 4];
            reader
                .read_block_unsafe(address + u64::from(header), &mut sph)
                .map_err(|e| e.to_string())?;
            match (u32::from_le_bytes(sph) >> 10) & 15 {
                0 | 1 => 0,
                2 => 3,
                3 => 4,
                4 => 2,
                5 => 1,
                k => return Err(format!("unknown graphics shader stage {k}").into()),
            }
        };
        *out = Shader {
            address,
            bytes: size,
            header,
            constants,
            stage,
            group: [x, y, z],
            shared,
        };
    }
    c.write(c.args[0], &p)?;
    Ok(1)
}
