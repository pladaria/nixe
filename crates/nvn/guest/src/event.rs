use super::*;

pub fn defaults(c: &Call) -> Result<u64> {
    c.write(c.args[0], &[0u64; 2])?;
    Ok(0)
}

pub fn storage(c: &Call) -> Result<u64> {
    c.write(c.args[0], &[c.args[1], c.args[2]])?;
    Ok(0)
}

pub fn initialize(c: &Call) -> Result<u64> {
    let [pool, offset]: [u64; 2] = c.read(c.args[1])?;
    let storage: Pool = c.read(pool)?;
    if offset & 3 != 0 || storage.bytes.saturating_sub(offset) < 4 {
        return Err("NVN event word is outside its pool");
    }
    c.write(
        c.args[0],
        &Event {
            gpu: storage.gpu + offset,
            cpu: if storage.flags & 1 == 0 { storage.cpu + offset } else { 0 },
            pool,
            offset,
        },
    )?;
    Ok(1)
}

pub fn finalize(_: &Call) -> Result<u64> {
    Ok(0)
}

pub fn value(c: &Call) -> Result<u64> {
    use core::sync::atomic::{AtomicU32, Ordering};
    let event: Event = c.read(c.args[0])?;
    if event.cpu == 0 {
        return Err("NVN event pool is not CPU accessible");
    }
    Ok(u64::from(unsafe {
        (&*(event.cpu as *const AtomicU32)).load(Ordering::Acquire)
    }))
}

pub fn pool(c: &Call) -> Result<u64> {
    Ok(c.read::<Event>(c.args[0])?.pool)
}

pub fn offset(c: &Call) -> Result<u64> {
    Ok(c.read::<Event>(c.args[0])?.offset)
}

pub fn builder_pool(c: &Call) -> Result<u64> {
    c.read(c.args[0])
}

pub fn builder_offset(c: &Call) -> Result<u64> {
    c.read(c.args[0] + 8)
}

pub fn builder_storage(c: &Call) -> Result<u64> {
    let [pool, offset]: [u64; 2] = c.read(c.args[0])?;
    c.write(c.args[1], &offset)?;
    Ok(pool)
}
