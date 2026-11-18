use super::*;
pub fn attrib_defaults(c: &Call) -> Result<u64> {
    c.write(c.args[0], &0u32)?;
    Ok(0)
}
pub fn stream_index(c: &Call) -> Result<u64> {
    let v: u32 = c.read(c.args[0])?;
    c.write(c.args[0], &((v & 0x07ff_ffff) | ((c.args[1] as u32) << 27)))?;
    Ok(0)
}
pub fn format(c: &Call) -> Result<u64> {
    let v: u32 = c.read(c.args[0])?;
    c.write(
        c.args[0],
        &((v & 0xf800_0000) | ((c.args[2] as u32 & 0x7ffff) << 8) | (c.args[1] as u32 & 255)),
    )?;
    Ok(0)
}
pub fn stream_defaults(c: &Call) -> Result<u64> {
    c.write(c.args[0], &0u64)?;
    Ok(0)
}
pub fn stride(c: &Call) -> Result<u64> {
    c.write(c.args[0], &(c.args[1] as u32))?;
    Ok(0)
}
pub fn divisor(c: &Call) -> Result<u64> {
    c.write(c.args[0] + 4, &(c.args[1] as u32))?;
    Ok(0)
}
