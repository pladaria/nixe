use super::*;
macro_rules! field {
    ($name:ident,$off:expr,$value:expr) => {
        pub fn $name(c: &Call) -> Result<u64> {
            c.write(c.args[0] + $off, &$value(c))?;
            Ok(0)
        }
    };
}
field!(blend, 0, |_: &Call| BLEND);
field!(target, 0, |c: &Call| c.args[1] as u8);
field!(factors, 1, |c: &Call| [
    c.args[1] as u8,
    c.args[2] as u8,
    c.args[3] as u8,
    c.args[4] as u8
]);
field!(equations, 5, |c: &Call| [c.args[1] as u8, c.args[2] as u8]);
field!(poly, 0, |_: &Call| POLY);
field!(front, 1, |c: &Call| c.args[1] as u8);
field!(cull, 0, |c: &Call| c.args[1] as u8);
field!(depth, 0, |_: &Call| [
    u32::from_le_bytes(DEPTH),
    0x0007_0007
]);
field!(test, 0, |c: &Call| u8::from(c.args[1] as u32 != 0));
field!(write, 1, |c: &Call| u8::from(c.args[1] as u32 != 0));
field!(compare, 2, |c: &Call| c.args[1] as u8);
field!(color, 0, |_: &Call| 0x300u32);
field!(mask, 0, |_: &Call| u32::MAX);
fn update(c: &Call, width: u32, bits: u32) -> Result<u64> {
    let target = c.args[1] as u32;
    if target >= 8 {
        return Err("invalid color target");
    }
    let shift = target * width;
    let old: u32 = c.read(c.args[0])?;
    c.write(
        c.args[0],
        &((old & !(((1 << width) - 1) << shift)) | (bits << shift)),
    )?;
    Ok(0)
}
pub fn enable(c: &Call) -> Result<u64> {
    update(c, 1, u32::from(c.args[2] as u32 != 0))
}
pub fn channels(c: &Call) -> Result<u64> {
    let bits = c.args[2..6]
        .iter()
        .enumerate()
        .fold(0, |v, (i, &a)| v | (u32::from(a as u32 != 0) << i));
    update(c, 4, bits)
}

field!(advanced, 7, |c: &Call| c.args[1] as u8);
field!(polygon_mode, 2, |c: &Call| c.args[1] as u8);
field!(polygon_offset, 3, |c: &Call| c.args[1] as u8);
field!(stencil_test, 3, |c: &Call| c.args[1] as u8);
field!(logic, 1, |c: &Call| c.args[1] as u8);
field!(multisample, 0, |_: &Call| 0u32);
field!(multisample_enable, 0, |c: &Call| c.args[1] as u8);
field!(samples, 1, |c: &Call| c.args[1] as u8);
field!(coverage, 2, |c: &Call| c.args[1] as u8);
fn stencil_fields(c: &Call, mask: u16, bits: u16) -> Result<u64> {
    let mut faces: [u16; 2] = c.read(c.args[0] + 4)?;
    for (i, s) in faces.iter_mut().enumerate() {
        if c.args[1] & (1 << i) != 0 {
            *s = (*s & !mask) | bits
        }
    }
    c.write(c.args[0] + 4, &faces)?;
    Ok(0)
}
pub fn stencil_func(c: &Call) -> Result<u64> {
    stencil_fields(c, 7, (c.args[2] as u16).wrapping_sub(1) & 7)
}
pub fn stencil_op(c: &Call) -> Result<u64> {
    let mut bits = 0;
    for i in 0..3 {
        bits |= ((c.args[2 + i] as u16).wrapping_sub(1) & 7) << (3 + i * 3)
    }
    stencil_fields(c, 0xff8, bits)
}
