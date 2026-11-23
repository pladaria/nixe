use super::*;

const _: () = assert!(size_of::<Program>() <= 128);
const _: () = assert!(size_of::<AmpereProgram>() <= 160);

pub fn shaders(c: &Call) -> Result<u64> {
    set_shaders(c, 16)
}
pub fn shaders_ext(c: &Call) -> Result<u64> {
    set_shaders(c, 24)
}
pub fn shaders_ampere(c: &Call) -> Result<u64> {
    set_ampere_shaders(c, c.args[1] as u32, c.args[2], 16)
}

pub fn shaders_ext_ampere(c: &Call) -> Result<u64> {
    set_ampere_shaders(c, c.args[1] as u32, c.args[2], 24)
}

pub fn from_builder(c: &Call) -> Result<u64> {
    let builder: ProgramBuilder = c.read(c.args[1])?;
    set_ampere_shaders(c, builder.shader_count, builder.shaders, 16)
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
            group: [x as u16, y as u16, z as u16],
            shared,
            isa: 0,
            early_fragment_tests: 0,
            ..Default::default()
        };
    }
    c.write(c.args[0], &p)?;
    Ok(1)
}

fn set_ampere_shaders(c: &Call, count: u32, data: u64, stride: u64) -> Result<u64> {
    let count = count as usize;
    let mut program = AmpereProgram {
        count: count as u64,
        ..Default::default()
    };
    let stages = program
        .stages
        .get_mut(..count)
        .ok_or("unsupported stage count in NVN2 program")?;
    let reader = c.session.graphics()?.heap.device().reader();
    for (index, shader) in stages.iter_mut().enumerate() {
        let [address, control]: [u64; 2] = c.read(data + stride * index as u64)?;
        // The NVN2 control block may be freed after this call. Capture only the
        // fields consumed by translation; the shader code stays in its pool.
        // https://github.com/seedds/nxnative/blob/a366365/dm/gfx/tools/dmsh_ampere.py
        let [magic, version]: [u32; 2] = c.read(control)?;
        if (magic, version) != (0x9876_1234, 0xc9) {
            return Err(
                format!("unsupported NVN2 shader control header {magic:#x}/{version:#x}").into(),
            );
        }
        let [instructions, _, constants, bytes]: [u32; 4] = c.read(control + 0x38)?;
        let kind: u8 = c.read(control + 0x56)?;
        if instructions == 0 || instructions > constants || constants > bytes || bytes > 16 << 20 {
            return Err("invalid NVN2 shader extent".into());
        }
        let (kind, extra) = if kind == 5 {
            let [x, y, z, shared]: [u32; 4] = c.read(control + 0x58)?;
            if ![x, y, z]
                .into_iter()
                .all(|dimension| (1..=1024).contains(&dimension))
            {
                return Err("invalid NVN2 compute workgroup".into());
            }
            let words: [u8; 14] = c.read(control + 0xa9)?;
            program.compute = ComputeLayout {
                roots: words[..8].try_into().unwrap(),
                uniform: words[8],
                storage: words[9],
                textures: words[10],
                separate_textures: words[11],
                separate_samplers: words[12],
                images: words[13],
                size: c.read(control + 0xba)?,
            };
            let layout = program.compute;
            if layout.roots.iter().any(|&words| words > 64)
                || layout.uniform > 16
                || layout.storage > 16
                || layout.image_base() + u32::from(layout.images) * 8 > u32::from(layout.size)
                || layout.size > 2048
            {
                return Err(format!("invalid NVN2 compute driver-bank layout {layout:?}").into());
            }
            (
                AmpereShader::COMPUTE | (z - 1) << 20 | (y - 1) << 10 | (x - 1),
                shared,
            )
        } else {
            if kind > 5 || instructions < 0x80 {
                return Err(format!("unsupported NVN2 program stage {kind}").into());
            }
            let mut first = [0; 4];
            reader
                .read_block_unsafe(address, &mut first)
                .map_err(|e| e.to_string())?;
            let header = u32::from_le_bytes(first);
            let stage = match (header >> 10) & 15 {
                1 => 0,
                2 => 3,
                3 => 4,
                4 => 2,
                5 => 1,
                other => return Err(format!("unknown NVN2 graphics shader stage {other}").into()),
            };
            let early = if stage == 1 {
                u32::from(c.read::<u8>(control + 0x5b)?)
            } else {
                0
            };
            (stage, early)
        };
        *shader = AmpereShader {
            address,
            bytes,
            constants,
            kind,
            extra,
        };
    }
    c.write(c.args[0], &program)?;
    Ok(1)
}
