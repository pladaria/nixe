use super::*;
pub fn initialize(c: &Call) -> Result<u64> {
    c.write(c.args[0], &0u64)?;
    Ok(1)
}

pub fn finalize(c: &Call) -> Result<u64> {
    c.write(c.args[0], &0u64)?;
    Ok(0)
}

pub fn builder_defaults(c: &Call) -> Result<u64> {
    // Reset the shader selection, preserving the previously selected device.
    c.write(c.args[0] + 16, &[0u64; 5])?;
    Ok(0)
}

pub fn builder_device(c: &Call) -> Result<u64> {
    c.write(c.args[0] + 8, &c.args[1])?;
    Ok(0)
}

pub fn builder_shaders(c: &Call) -> Result<u64> {
    c.write(c.args[0] + 40, &[u64::from(c.args[1] as u32), c.args[2]])?;
    Ok(0)
}

impl MaxwellStage {
    pub fn expand(self, address: u64, compute_group: [u16; 3], compute_shared: u32) -> Shader {
        let stage = self.header_stage >> Self::STAGE_SHIFT;
        Shader {
            address,
            bytes: self.bytes,
            header: self.header_stage & ((1 << Self::STAGE_SHIFT) - 1),
            constants: self.constants,
            stage,
            group: if stage == 5 { compute_group } else { [0; 3] },
            shared: if stage == 5 { compute_shared } else { 0 },
            ..Default::default()
        }
    }
}

impl AmpereShader {
    pub fn expand(self, layout: ComputeLayout, local_memory_size: u32) -> Shader {
        let compute = self.kind & Self::COMPUTE != 0;
        Shader {
            address: self.address,
            bytes: self.bytes,
            header: 0,
            constants: self.constants,
            stage: if compute { 5 } else { self.kind },
            group: if compute {
                core::array::from_fn(|dimension| (((self.kind >> (10 * dimension)) & 1023) + 1) as u16)
            } else {
                [0; 3]
            },
            shared: if compute { self.extra } else { 0 },
            isa: 1,
            early_fragment_tests: if compute { 0 } else { self.extra as u8 },
            compute: if compute { layout } else { ComputeLayout::default() },
            local_memory_size: if compute { local_memory_size } else { 0 },
        }
    }
}
