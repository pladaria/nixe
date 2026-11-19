//! Buffer objects are views into a pool. Vertex and index bindings use the
//! pool's device address and share its backing pages with CPU mappings.
use super::*;

pub use crate::pool::set_device;
pub fn defaults(c: &Call) -> Result<u64> {
    c.write(c.args[0] + 8, &[0u64; 3])?;
    Ok(0)
}
pub fn storage(c: &Call) -> Result<u64> {
    c.write(c.args[0] + 8, &[c.args[1], c.args[2], c.args[3]])?;
    Ok(0)
}
pub fn initialize(c: &Call) -> Result<u64> {
    let [pool, offset, _size]: [u64; 3] = c.read(c.args[1] + 8)?;
    if pool == 0 {
        return Ok(0);
    }
    let p: Pool = c.read(pool)?;
    if p.flags & 0x100 != 0 {
        return Ok(0);
    }
    let cpu = if p.cpu == 0 || p.flags & 1 != 0 {
        0
    } else {
        p.cpu + offset
    };
    c.write(c.args[0], &[cpu, p.gpu + offset])?;
    Ok(1)
}
pub fn map(c: &Call) -> Result<u64> {
    c.read(c.args[0])
}
pub fn address(c: &Call) -> Result<u64> {
    c.read(c.args[0] + 8)
}
