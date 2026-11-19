use super::*;
pub fn initialize(c: &Call) -> Result<u64> {
    c.session.graphics()?;
    let flags: u32 = c.read(c.args[1])?;
    c.write(c.args[0], &[flags, 0, 0])?;
    Ok(1)
}
