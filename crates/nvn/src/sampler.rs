use super::*;
pub fn register(c: &Call) -> Result<u64> {
    let s: Sampler = c.read(c.args[2] + 8)?;
    let id = c.session.samplers.lock().unwrap().intern(s);
    c.write(pool::slot(c, c.args[0], c.args[1] as u32)?, &id)?;
    Ok(0)
}
