use super::*;
use bytemuck::Zeroable;
pub fn get_flags(c: &Call) -> Result<u64> {
    Ok(u64::from(c.read::<Pool>(c.args[0])?.flags & 0x3fff_ffff))
}
pub fn defaults(c: &Call) -> Result<u64> {
    c.write_bytes(
        c.args[0] + 8,
        &bytemuck::bytes_of(&Builder {
            flags: 0x21,
            ..Builder::zeroed()
        })[8..],
    )?;
    Ok(0)
}
pub fn set_device(c: &Call) -> Result<u64> {
    c.write(c.args[0], &c.args[1])?;
    Ok(0)
}
pub fn storage(c: &Call) -> Result<u64> {
    c.write(c.args[0] + 8, &[c.args[1], c.args[2]])?;
    Ok(0)
}
pub fn flags(c: &Call) -> Result<u64> {
    c.write(c.args[0] + 24, &(c.args[1] as u32))?;
    Ok(0)
}
pub fn map(c: &Call) -> Result<u64> {
    let p: Pool = c.read(c.args[0])?;
    Ok(if p.flags & 1 != 0 { 0 } else { p.cpu })
}
pub fn address(c: &Call) -> Result<u64> {
    c.read(c.args[0] + 8)
}
pub fn flush(_: &Call) -> Result<u64> {
    // @SIMPLIFICATION CPU writes already dirty the shared physical buffer/image caches. Submission
    // drains those invalidations before consuming the pool; mapping needs no upload.
    Ok(0)
}
pub fn invalidate(_: &Call) -> Result<u64> {
    // @SIMPLIFICATION Guest CPU reads already synchronize pending GPU writes to these pages.
    Ok(0)
}
pub fn descriptor_pool(c: &Call) -> Result<u64> {
    let p: Pool = c.read(c.args[1])?;
    let off = c.args[2];
    let count = c.args[3] as u32;
    if off > p.bytes || u64::from(count) * 32 > p.bytes - off {
        return Err("descriptor pool exceeds memory pool");
    }
    c.write(
        c.args[0],
        &DescriptorPool {
            base: p.cpu + off,
            count,
            padding: 0,
        },
    )?;
    Ok(1)
}
