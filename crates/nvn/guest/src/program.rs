use super::*;
pub fn initialize(c: &Call) -> Result<u64> {
    c.write(c.args[0], &0u64)?;
    Ok(1)
}

pub fn finalize(c: &Call) -> Result<u64> {
    c.write(c.args[0], &0u64)?;
    Ok(0)
}
