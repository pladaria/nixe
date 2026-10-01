//! Maxwell QMD 1.7 geometry, resource bindings and consumed execution controls.
//!
//! https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/compute/clb1c0qmd.h#L236-L451

pub(super) const QMD_SIZE: usize = 0x100;

/// Retain the complete descriptor and validate controls at dispatch consumption.
pub(super) struct MaxwellComputeQmd {
    words: [u32; QMD_SIZE / 4],
}

impl MaxwellComputeQmd {
    /// Consume launch controls before scheduling host work. Queue execution,
    /// releases, reference counters and resumed grids need distinct semantics.
    pub(super) fn validate_execution(&self) -> Result<(), &'static str> {
        // deko3d leaves this at reset. Validate even when translation is cached.
        // https://github.com/devkitPro/deko3d/blob/master/source/maxwell/gpu_compute.cpp
        if self.sass_version() != 0 {
            return Err("nonzero compute QMD SASS_VERSION is not implemented");
        }
        let controls = self.words[0x18 / 4];
        if controls & ((1 << 8) | (1 << 9) | (1 << 13)) != 0 {
            return Err("QMD queues, linked groups and dependent launches are not implemented");
        }
        if controls & (3 << 10) != 0 {
            return Err("QMD semaphore releases are not implemented");
        }
        let execution = self.words[0x2c / 4];
        if execution & ((1 << 15) | (1 << 19)) != 0 {
            return Err("QMD reference-count updates are not implemented");
        }
        if execution & (1 << 18) != 0 {
            return Err("sequential CTA execution is not implemented");
        }
        if execution & ((3 << 24) | (1 << 31)) != 0 {
            return Err("QMD floating-point overrides are not implemented");
        }
        if (execution >> 16) & 3 == 2 {
            return Err("reserved QMD CWD memory barrier mode");
        }
        if self.words[0x38 / 4] != 0 || self.words[0x3c / 4] != 0 {
            return Err("resumed compute grids are not implemented");
        }
        if self.workgroup_size().contains(&0) {
            return Err("QMD workgroup dimensions must be nonzero");
        }
        // Cache configuration, occupancy and memory-allocation sizes do not
        // execute instructions. Unsupported local/shared/call/barrier opcodes
        // are rejected by translation when consumed. Queue/release payloads
        // are likewise irrelevant with their enables clear.
        Ok(())
    }
    pub(super) fn decode(bytes: &[u8; QMD_SIZE]) -> Result<Self, (u8, u8)> {
        let words = std::array::from_fn(|index| {
            u32::from_le_bytes(bytes[4 * index..4 * index + 4].try_into().unwrap())
        });
        let major = ((words[0x48 / 4] >> 4) & 0xf) as u8;
        let minor = (words[0x48 / 4] & 0xf) as u8;
        if (major, minor) != (1, 7) {
            return Err((major, minor));
        }
        Ok(Self { words })
    }

    pub(super) fn program_offset(&self) -> u32 {
        self.words[0x20 / 4]
    }

    pub(super) fn register_count(&self) -> u8 {
        (self.words[0xb8 / 4] >> 24) as u8
    }

    pub(super) fn sass_version(&self) -> u8 {
        (self.words[0xbc / 4] >> 24) as u8
    }

    pub(super) fn constant_buffer(&self, slot: u8) -> Result<(u64, u64), &'static str> {
        if slot >= 8 || self.words[0x50 / 4] & (1 << slot) == 0 {
            return Err("constant buffer slot is not valid in the QMD");
        }
        let index = 0x74 / 4 + usize::from(slot) * 2;
        let upper = self.words[index + 1];
        if upper & 0x3f00 != 0 {
            return Err("constant buffer address has reserved bits set");
        }
        let address = u64::from(self.words[index]) | (u64::from(upper & 0xff) << 32);
        let size = u64::from(upper >> 15);
        if size == 0 {
            return Err("consumed constant buffer has zero size");
        }
        // INVALIDATE is a cache hint; consumption resolves current backing
        // and pending writes, never a stale copy of a driver constant buffer.
        Ok((address, size))
    }

    pub(super) fn workgroups(&self) -> [u32; 3] {
        [
            self.words[0x30 / 4],
            self.words[0x34 / 4] & 0xffff,
            self.words[0x34 / 4] >> 16,
        ]
    }

    pub(super) fn workgroup_size(&self) -> [u32; 3] {
        [
            self.words[0x48 / 4] >> 16,
            self.words[0x4c / 4] & 0xffff,
            self.words[0x4c / 4] >> 16,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn launch_rejects_enabled_side_effects_but_not_inactive_payloads() {
        let mut bytes = [0; QMD_SIZE];
        bytes[0x48..0x4c].copy_from_slice(&0x0020_0017_u32.to_le_bytes());
        bytes[0x4c..0x50].copy_from_slice(&0x0001_0001_u32.to_le_bytes());
        // Payloads do not execute when queue, dependency and releases are off.
        bytes[0x14..0x18].fill(0xff);
        bytes[0x5c..0x74].fill(0xff);
        assert!(
            MaxwellComputeQmd::decode(&bytes)
                .unwrap()
                .validate_execution()
                .is_ok()
        );
        for (offset, bit) in [
            (0x18, 8),
            (0x18, 9),
            (0x18, 10),
            (0x18, 11),
            (0x18, 13),
            (0x2c, 15),
            (0x2c, 18),
            (0x2c, 19),
            (0x2c, 24),
            (0x2c, 25),
            (0x2c, 31),
            (0x38, 0),
            (0x3c, 0),
            (0xbc, 24),
        ] {
            let mut changed = bytes;
            changed[offset..offset + 4].copy_from_slice(&(1_u32 << bit).to_le_bytes());
            assert!(
                MaxwellComputeQmd::decode(&changed)
                    .unwrap()
                    .validate_execution()
                    .is_err(),
                "{offset:#x}:{bit}"
            );
        }
    }

    #[test]
    fn consumed_constant_buffers_preserve_address_size_and_ignore_only_cache_hint() {
        let mut bytes = [0; QMD_SIZE];
        bytes[0x48] = 0x17;
        bytes[0x50] = 0xff;
        for slot in 0..8 {
            let index = 0x74 + slot * 8;
            let address = 0x1234_5678_u32 + slot as u32;
            bytes[index..index + 4].copy_from_slice(&address.to_le_bytes());
            bytes[index + 4..index + 8].copy_from_slice(&0xffff_c0ff_u32.to_le_bytes());
        }
        let qmd = MaxwellComputeQmd::decode(&bytes).unwrap();
        for slot in 0..8 {
            assert_eq!(
                qmd.constant_buffer(slot),
                Ok((0xff_1234_5678 + u64::from(slot), 0x1ffff))
            );
        }
        assert!(qmd.constant_buffer(8).is_err());
        for upper in [0_u32, 1 << 8, (8 << 15) | (1 << 13)] {
            bytes[0x78..0x7c].copy_from_slice(&upper.to_le_bytes());
            assert!(
                MaxwellComputeQmd::decode(&bytes)
                    .unwrap()
                    .constant_buffer(0)
                    .is_err()
            );
        }
        bytes[0x78..0x7c].copy_from_slice(&(8_u32 << 15).to_le_bytes());
        bytes[0x50] = 0xfe;
        let qmd = MaxwellComputeQmd::decode(&bytes).unwrap();
        assert!(qmd.constant_buffer(0).is_err());
        assert!(qmd.constant_buffer(1).is_ok());
    }

    #[test]
    fn decodes_independent_qmd_dimensions_and_full_width_program_offset() {
        let mut bytes = [0; QMD_SIZE];
        for (offset, word) in [
            (0x20, 0xf123_4567_u32),
            (0x30, 0x89ab_cdef),
            (0x34, 0x7654_3210),
            (0x48, 0xabcd_0017),
            (0x4c, 0x3456_1234),
            (0xb8, 0x0b12_3456),
            (0xbc, 0x0712_3456),
        ] {
            bytes[offset..offset + 4].copy_from_slice(&word.to_le_bytes());
        }
        let qmd = MaxwellComputeQmd::decode(&bytes).unwrap();
        assert_eq!(qmd.program_offset(), 0xf123_4567);
        assert_eq!(qmd.workgroups(), [0x89ab_cdef, 0x3210, 0x7654]);
        assert_eq!(qmd.workgroup_size(), [0xabcd, 0x1234, 0x3456]);
        assert_eq!(qmd.register_count(), 11);
        assert_eq!(qmd.sass_version(), 7);
    }

    #[test]
    fn rejects_other_versions_without_interpreting_their_layout() {
        for (major, minor) in [(0, 0), (1, 6), (2, 7), (15, 15)] {
            let mut bytes = [0; QMD_SIZE];
            bytes[0x48] = (major << 4) | minor;
            assert!(
                matches!(MaxwellComputeQmd::decode(&bytes), Err(version) if version == (major, minor))
            );
        }
    }
}
