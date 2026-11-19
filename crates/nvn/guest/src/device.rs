use super::*;

// X1 capability values measured from the native driver.
// https://github.com/seedds/nxnative/blob/a366365/dm/gfx/nvn/hle_nvn.c
pub fn defaults(c: &Call) -> Result<u64> {
    c.write(c.args[0], &0u32)?;
    Ok(0)
}

pub fn flags(c: &Call) -> Result<u64> {
    c.write(c.args[0], &(c.args[1] as u32))?;
    Ok(0)
}

pub fn origin(c: &Call) -> Result<u64> {
    c.write(c.args[0] + 4, &(c.args[1] as u32))?;
    Ok(0)
}
pub fn depth_mode(c: &Call) -> Result<u64> {
    c.write(c.args[0] + 8, &(c.args[1] as u32))?;
    Ok(0)
}
pub fn fast_clear(_: &Call) -> Result<u64> {
    // @SIMPLIFICATION The host chooses its clear metadata when recording the clear.
    Ok(1)
}
pub fn timestamp(c: &Call) -> Result<u64> {
    let ticks: u64 = c.read(c.args[1] + 8)?;
    Ok(((u128::from(ticks) * 625) / 384) as u64)
}
pub fn texture_handle(c: &Call) -> Result<u64> {
    Ok(((c.args[1] as i32 | (c.args[2] as i32) << 20) as i64 as u64) | (1 << 32))
}
pub fn texture_handle_ampere(c: &Call) -> Result<u64> {
    Ok((c.args[1] as i32 as i64 as u64) | ((c.args[2] as i32 as i64 as u64) << 20))
}
const X1: [u32; 105] = [
    55, 15, 14, 65536, 256, 8, 16, 4, 16, 32, 16, 4, 4, 4, 16, 32, 32, 256, 256, 4, 8, 512, 256, 2,
    256, 4096, 131072, 16, 1536, 1024, 64, 1536, 65535, 65535, 65535, 8, 1048576, 4096, 16, 4096,
    1, 1, 1, 262144, 15, 2048, 1, 1, 14, 5, 1, 8, 8, 4, 32, 128, 32, 128, 65536, 1, 4, 16384,
    134217728, 2048, 16384, 16384, 1, 1, 1, 1, 5, 1, 1, 1, 1, 1, 1, 16, 1024, 32, 4096, 65536,
    65536, 256, 16384, 262144, 4096, 1, 4, 2, 0, 4096, 16384, 4096, 128, 32, 65535, 1, 32, 256, 16,
    65535, 65535, 256, 16383,
];

// The host publishes these once, before returning the SDK version resolver.
core::arch::global_asm!(
    ".pushsection .nvn_version,\"aw\",%nobits",
    ".global nixe_nvn_version",
    ".hidden nixe_nvn_version",
    "nixe_nvn_version:",
    ".zero 8",
    ".popsection"
);
fn sdk_version(property: usize) -> u32 {
    let address: *const u32;
    unsafe {
        core::arch::asm!("adr {address}, nixe_nvn_version", address = out(reg) address, options(pure, nomem, nostack));
        address.add(property).read()
    }
}
pub fn integer(c: &Call) -> Result<u64> {
    let property = c.args[1] as u32 as usize;
    let value = if property < 2 {
        sdk_version(property)
    } else {
        *X1.get(property).ok_or("unsupported NVN device property")?
    };
    c.write(c.args[2], &value)?;
    Ok(0)
}
