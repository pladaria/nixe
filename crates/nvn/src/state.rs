//! Fixed-function NVN state objects live in application memory. A bind records
//! their value so subsequent edits to the object affect only later commands.
use super::*;

pub const BLEND: [u8; 8] = [0, 2, 1, 2, 1, 1, 1, 0];
pub const POLY: [u8; 4] = [0, 1, 0, 0];
pub const DEPTH: [u8; 4] = [0, 1, 2, 0];

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct RasterState {
    pub blend: [u8; 8],
    pub polygon: [u8; 4],
    pub depth: [u8; 4],
    pub blend_enable: u32,
    pub channel_mask: u32,
}
impl Default for RasterState {
    fn default() -> Self {
        Self {
            blend: BLEND,
            polygon: POLY,
            depth: DEPTH,
            blend_enable: 0,
            channel_mask: u32::MAX,
        }
    }
}
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
field!(depth, 0, |_: &Call| DEPTH);
field!(test, 0, |c: &Call| u8::from(c.args[1] as u32 != 0));
field!(write, 1, |c: &Call| u8::from(c.args[1] as u32 != 0));
field!(compare, 2, |c: &Call| c.args[1] as u8);
field!(color, 0, |_: &Call| 0u32);
field!(mask, 0, |_: &Call| u32::MAX);
fn update(c: &Call, width: u32, bits: u32) -> Result<u64> {
    let target = c.args[1] as u32;
    if target >= 8 {
        return Err("invalid color target".into());
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
