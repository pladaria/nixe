//! GA10F driver characteristics use the Switch 2 ABI, not the Maxwell layout.
//! Field prefixes: https://switchbrew.org/wiki/NV_services#NVGPU_GPU_IOCTL_GET_CHARACTERISTICS

pub(crate) fn encode() -> [u8; 0xd0] {
    let mut bytes = [0; 0xd0];
    // @SPECULATIVE: chip revision, topology, cache/page geometry, interface
    // versions and the undocumented tail follow the reference's GA10F profile.
    // Hardware classes describe Ampere, even before every engine is implemented;
    // unsupported commands are rejected at their actual execution boundary.
    // https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/core/hle/service/nvdrv/devices/nvhost_ctrl_gpu.cpp
    for (offset, value) in [
        (0x00, 0x170_u32),
        (0x04, 0xe),
        (0x08, 0xa1),
        (0x0c, 1),
        (0x20, 6),
        (0x24, 0x20),
        (0x28, 0x20000),
        (0x2c, 0x20000),
        (0x30, 0x1b),
        (0x34, 0x30000),
        (0x38, 1),
        (0x3c, 0x808),
        (0x40, 0x808),
        (0x44, 0x80),
        (0x48, 40),
        (0x58, 0x902d),
        (0x5c, 0xc797),
        (0x60, 0xc7c0),
        (0x64, 0xc56f),
        (0x68, 0xa140),
        (0x6c, 0xc7b5),
        (0x84, 1),
        (0xb4, 64),
    ] {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
    bytes[0x10..0x18].copy_from_slice(&0x40000_u64.to_le_bytes());
    let flags = 0x55_u64 | (1 << 8) | (1 << 22) | (1 << 26) | (1 << 34) | (1 << 35) | (0x1f << 40);
    bytes[0x50..0x58].copy_from_slice(&flags.to_le_bytes());
    for offset in (0x70..=0x80).step_by(2) {
        bytes[offset..offset + 2].copy_from_slice(&0x40_u16.to_le_bytes());
    }
    bytes[0x90..0x95].copy_from_slice(b"ga10f");
    bytes
}
