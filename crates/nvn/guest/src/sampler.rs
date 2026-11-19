//! Sampler objects retain the application's exact float values. The 32-byte GPU
//! descriptor slot holds a session-local ID, independent of the object's lifetime.
use super::*;

pub fn initialize(c: &Call) -> Result<u64> {
    let s: Sampler = c.read(c.args[1] + 8)?;
    c.write(c.args[0] + 8, &s)?;
    Ok(1)
}

pub fn defaults(c: &Call) -> Result<u64> {
    c.write(c.args[0] + 8, &Sampler::default())?;
    Ok(0)
}
macro_rules! set {
    ($name:ident,$field:ident,$expr:expr) => {
        pub fn $name(c: &Call) -> Result<u64> {
            let value = $expr(c);
            c.write(
                c.args[0] + 8 + core::mem::offset_of!(Sampler, $field) as u64,
                &value,
            )?;
            Ok(0)
        }
    };
}
set!(filter, min, |c: &Call| [c.args[1] as u8, c.args[2] as u8]);
set!(wrap, wrap, |c: &Call| [
    c.args[1] as u8,
    c.args[2] as u8,
    c.args[3] as u8
]);
set!(compare, comparison, |c: &Call| [
    c.args[1] as u8,
    c.args[2] as u8
]);
set!(lod, lod, |c: &Call| [c.float[0], c.float[1]]);
set!(bias, bias, |c: &Call| c.float[0]);
set!(aniso, aniso, |c: &Call| c.float[0]);
set!(reduction, reduction, |c: &Call| c.args[1] as u8);
pub fn border(c: &Call) -> Result<u64> {
    let v: [u32; 4] = c.read(c.args[1])?;
    c.write(
        c.args[0] + 8 + core::mem::offset_of!(Sampler, border) as u64,
        &v,
    )?;
    Ok(0)
}
fn read(c: &Call) -> Result<Sampler> {
    c.read(c.args[0] + 8)
}
fn outputs(c: &Call, values: &[u32]) -> Result<u64> {
    for (&addr, v) in c.args[1..].iter().zip(values) {
        c.write(addr, v)?
    }
    Ok(0)
}
pub fn get_filter(c: &Call) -> Result<u64> {
    let s = read(c)?;
    outputs(c, &[s.min.into(), s.mag.into()])
}
pub fn get_compare(c: &Call) -> Result<u64> {
    let s = read(c)?;
    outputs(c, &[s.comparison.into(), s.function.into()])
}
pub fn get_wrap(c: &Call) -> Result<u64> {
    outputs(c, &read(c)?.wrap.map(u32::from))
}
pub fn get_lod(c: &Call) -> Result<u64> {
    outputs(c, &read(c)?.lod)
}
#[unsafe(export_name = "nvnSamplerGetLodBias")]
pub unsafe extern "C" fn get_bias(object: *const u8) -> f32 {
    unsafe {
        object
            .add(8 + core::mem::offset_of!(Sampler, bias))
            .cast::<f32>()
            .read_unaligned()
    }
}
#[unsafe(export_name = "nvnSamplerGetMaxAnisotropy")]
pub unsafe extern "C" fn get_aniso(object: *const u8) -> f32 {
    unsafe {
        object
            .add(8 + core::mem::offset_of!(Sampler, aniso))
            .cast::<f32>()
            .read_unaligned()
    }
}
pub fn get_reduction(c: &Call) -> Result<u64> {
    Ok(read(c)?.reduction.into())
}
pub fn get_border(c: &Call) -> Result<u64> {
    c.write(c.args[1], &read(c)?.border)?;
    Ok(0)
}

pub fn finalize(c: &Call) -> Result<u64> {
    c.write(c.args[0], &Sampler::default())?;
    Ok(0)
}
