use super::*;
pub fn builder_interval(c: &Call) -> Result<u64> {
    c.write(c.args[0] + 20, &(c.args[1] as u32))?;
    Ok(0)
}
pub fn get_builder_interval(c: &Call) -> Result<u64> {
    Ok(
        c.read::<u32>(c.args[0] + core::mem::offset_of!(WindowBuilder, interval) as u64)?
            .into(),
    )
}

pub fn primary_queue(_: &Call) -> Result<u64> {
    // @SIMPLIFICATION Vulkan records all queues on one ordered device stream.
    // Native channel sharing requested by a primary queue does not change execution.
    Ok(0)
}
pub fn active(c: &Call) -> Result<u64> {
    c.write(c.args[0] + 56, &(c.args[1] as u32))?;
    Ok(0)
}
pub fn control_bytes(c: &Call) -> Result<u64> {
    set_size(c, 48)
}
pub fn flags(c: &Call) -> Result<u64> {
    set_size(c, 56)
}
pub fn command_bytes(c: &Call) -> Result<u64> {
    set_size(c, 8)
}
pub fn compute_bytes(c: &Call) -> Result<u64> {
    set_size(c, 16)
}
pub fn queue_bytes(c: &Call) -> Result<u64> {
    let [cmd, compute]: [u64; 2] = c.read(c.args[0] + 8)?;
    // @SIMPLIFICATION The native queue owns its host submission storage. Reserve the advertised
    // control memory plus the command and compute budgets in the guest.
    let control: u64 = c.read(c.args[0] + 48)?;
    Ok((cmd + compute + control).next_multiple_of(4096))
}
pub fn queue_memory(c: &Call) -> Result<u64> {
    c.write(c.args[0] + 32, &[c.args[1], c.args[2]])?;
    Ok(0)
}
pub fn threshold(c: &Call) -> Result<u64> {
    set_size(c, 24)
}
pub fn window_defaults(c: &Call) -> Result<u64> {
    let b = WindowBuilder {
        interval: 1,
        ..Default::default()
    };
    c.write_bytes(c.args[0] + 8, &bytemuck::bytes_of(&b)[8..])?;
    Ok(0)
}
pub fn set_device(c: &Call) -> Result<u64> {
    c.write(c.args[0], &c.args[1])?;
    Ok(0)
}
pub fn native_window(c: &Call) -> Result<u64> {
    c.write(c.args[0] + 8, &c.args[1])?;
    Ok(0)
}
pub fn textures(c: &Call) -> Result<u64> {
    let n = c.args[1] as u32 as usize;
    if !(1..=4).contains(&n) {
        return Err("unsupported window texture count");
    }
    let mut tex = [0u64; 4];
    c.read_bytes(c.args[2], bytemuck::cast_slice_mut(&mut tex[..n]))?;
    c.write(c.args[0] + 24, &tex)?;
    c.write(c.args[0] + 16, &(n as u32))?;
    Ok(0)
}
pub fn defaults(c: &Call) -> Result<u64> {
    c.write(c.args[0] + 8, &[65536u64, 262144, 0, 0, 0, 16384, 0])?;
    Ok(0)
}
fn set_size(c: &Call, offset: u64) -> Result<u64> {
    c.write(c.args[0] + offset, &c.args[1])?;
    Ok(0)
}

pub fn interval(c: &Call) -> Result<u64> {
    c.write(c.args[0], &(c.args[1] as u32))?;
    Ok(0)
}
pub fn get_interval(c: &Call) -> Result<u64> {
    Ok(c.read::<u32>(c.args[0])?.into())
}
pub fn crop(c: &Call) -> Result<u64> {
    c.write(
        c.args[0] + 4,
        &core::array::from_fn::<_, 4, _>(|i| c.args[i + 1] as i32),
    )?;
    Ok(0)
}
