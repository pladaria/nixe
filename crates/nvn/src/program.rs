use super::*;
use bytemuck::{Pod, Zeroable};
#[derive(Clone, Copy, Default, Debug, Hash, PartialEq, Eq, Pod, Zeroable)]
#[repr(C)]
pub struct Shader {
    pub address: u64,
    pub bytes: u32,
    pub header: u32,
    pub constants: u32,
    pub padding: u32,
}
#[derive(Clone, Copy, Default, Pod, Zeroable)]
#[repr(C)]
pub(crate) struct Program {
    pub count: u64,
    pub stages: [Shader; 6],
}
pub fn initialize(c: &Call) -> Result<u64> {
    c.write(c.args[0], &0u64)?;
    Ok(1)
}
pub fn shaders(c: &Call) -> Result<u64> {
    let n = c.args[1] as u32 as usize;
    if n > 6 {
        return Err(format!("invalid stage count {n}").into());
    }
    let mut p = Program {
        count: n as u64,
        ..Default::default()
    };
    for (i, out) in p.stages[..n].iter_mut().enumerate() {
        let [address, control]: [u64; 2] = c.read(c.args[2] + 16 * i as u64)?;
        // GLSLC Maxwell control sections carry the extent of the code allocation
        // and the offset of its SPH. Only the code remains live after SetShaders.
        let [constants, size, header]: [u32; 3] = c.read(control + 0x700)?;
        if constants > size || header > size || size - header < 0x50 || size > 16 << 20 {
            return Err(format!("invalid shader extent {size:#x} / header {header:#x}").into());
        }
        *out = Shader {
            address,
            bytes: size,
            header,
            constants,
            padding: 0,
        };
    }
    c.write(c.args[0], &p)?;
    Ok(1)
}
