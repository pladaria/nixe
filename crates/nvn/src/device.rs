use super::*;

// X1 capability values measured from the native driver.
// https://github.com/seedds/nxnative/blob/a366365/dm/gfx/nvn/hle_nvn.c
pub fn defaults(c: &Call) -> Result<u64> {
    c.write(c.args[0], &0u32)?;
    Ok(0)
}
pub fn initialize(c: &Call) -> Result<u64> {
    c.session.graphics()?;
    let flags: u32 = c.read(c.args[1])?;
    c.write(c.args[0], &flags)?;
    Ok(1)
}
pub fn integer(c: &Call) -> Result<u64> {
    let value = if c.args[1] < 2 {
        c.session
            .api_version
            .get()
            .ok_or("SDK version unavailable")?[c.args[1] as usize]
    } else {
        *X1.get(c.args[1] as u32 as usize)
            .ok_or_else(|| format!("unsupported device property {:#x}", c.args[1]))?
    };
    c.write(c.args[2], &value)?;
    Ok(0)
}
pub fn texture_handle(c: &Call) -> Result<u64> {
    let texture = c.args[1] as i32;
    let sampler = c.args[2] as i32;
    Ok(match c.session.platform {
        TargetPlatform::Switch1 => ((texture | (sampler << 20)) as i64 as u64) | (1 << 32),
        TargetPlatform::Switch2 => (texture as i64 as u64) | ((sampler as i64 as u64) << 20),
    })
}
const X1: [u32; 105] = [
    55, 15, 14, 65536, 256, 8, 16, 4, 16, 32, 16, 4, 4, 4, 16, 32, 32, 256, 256, 4, 8, 512, 256, 2,
    256, 4096, 131072, 16, 1536, 1024, 64, 1536, 65535, 65535, 65535, 8, 1048576, 4096, 16, 4096,
    1, 1, 1, 262144, 15, 2048, 1, 1, 14, 5, 1, 8, 8, 4, 32, 128, 32, 128, 65536, 1, 4, 16384,
    134217728, 2048, 16384, 16384, 1, 1, 1, 1, 5, 1, 1, 1, 1, 1, 1, 16, 1024, 32, 4096, 65536,
    65536, 256, 16384, 262144, 4096, 1, 4, 2, 0, 4096, 16384, 4096, 128, 32, 65535, 1, 32, 256, 16,
    65535, 65535, 256, 16383,
];
pub fn flags(c: &Call) -> Result<u64> {
    c.write(c.args[0], &(c.args[1] as u32))?;
    Ok(0)
}
