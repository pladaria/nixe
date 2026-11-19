use super::*;

// X1 capability values measured from the native driver.
// https://github.com/seedds/nxnative/blob/a366365/dm/gfx/nvn/hle_nvn.c
pub fn defaults(c: &Call) -> Result<u64> {
    c.write(c.args[0], &0u32)?;
    Ok(0)
}

pub fn flags(c: &Call) -> Result<u64> {
    c.write(c.args[0], &(c.args[1] as u32))?;
    Ok(0)
}

pub fn origin(c: &Call) -> Result<u64> {
    c.write(c.args[0] + 4, &(c.args[1] as u32))?;
    Ok(0)
}
pub fn depth_mode(c: &Call) -> Result<u64> {
    c.write(c.args[0] + 8, &(c.args[1] as u32))?;
    Ok(0)
}
pub fn fast_clear(_: &Call) -> Result<u64> {
    // @SIMPLIFICATION The host chooses its clear metadata when recording the clear.
    Ok(1)
}
pub fn timestamp(c: &Call) -> Result<u64> {
    let ticks: u64 = c.read(c.args[1] + 8)?;
    Ok(((u128::from(ticks) * 625) / 384) as u64)
}
pub fn texture_handle(c: &Call) -> Result<u64> {
    Ok(((c.args[1] as i32 | (c.args[2] as i32) << 20) as i64 as u64) | (1 << 32))
}
// An image handle is the signed texture pool index in a 64-bit return register.
// https://github.com/seedds/nxnative/blob/a366365/dm/gfx/nvn/hle_nvn.c#L242
pub fn image_handle(c: &Call) -> Result<u64> {
    Ok(c.args[1] as i32 as i64 as u64)
}
pub fn texel_fetch_handle(c: &Call) -> Result<u64> {
    Ok(image_handle(c)? | (1 << 32))
}
pub fn texture_handle_ampere(c: &Call) -> Result<u64> {
    Ok((c.args[1] as i32 as i64 as u64) | ((c.args[2] as i32 as i64 as u64) << 20))
}
const X1: [u32; 105] = [
    55, 15, 14, 65536, 256, 8, 16, 4, 16, 32, 16, 4, 4, 4, 16, 32, 32, 256, 256, 4, 8, 512, 256, 2,
    256, 4096, 131072, 16, 1536, 1024, 64, 1536, 65535, 65535, 65535, 8, 1048576, 4096, 16, 4096,
    1, 1, 1, 262144, 15, 2048, 1, 1, 14, 5, 1, 8, 8, 4, 32, 128, 32, 128, 65536, 1, 4, 16384,
    134217728, 2048, 16384, 16384, 1, 1, 1, 1, 5, 1, 1, 1, 1, 1, 1, 16, 1024, 32, 4096, 65536,
    65536, 256, 16384, 262144, 4096, 1, 4, 2, 0, 4096, 16384, 4096, 128, 32, 65535, 1, 32, 256, 16,
    65535, 65535, 256, 16383,
];

// The resolver fills these device-independent versions from the linked SDK.
core::arch::global_asm!(
    ".pushsection .nvn_version,\"aw\",%nobits",
    ".global nixe_nvn_version",
    ".hidden nixe_nvn_version",
    "nixe_nvn_version:",
    ".zero 24",
    ".popsection"
);
fn sdk_versions() -> *const core::sync::atomic::AtomicU32 {
    let address;
    unsafe {
        core::arch::asm!("adr {address}, nixe_nvn_version", address = out(reg) address, options(pure, nomem, nostack));
    }
    address
}

#[unsafe(no_mangle)]
pub extern "C" fn nixe_nvn_query_versions(
    replacement: u64,
    sdk_query: extern "C" fn(u64, u32, &mut u32),
) -> u64 {
    for (i, property) in [0, 1, 46, 47, 48, 49].into_iter().enumerate() {
        let mut value = 0;
        sdk_query(0, property, &mut value);
        // Concurrent first resolutions query the same SDK.
        unsafe { (*sdk_versions().add(i)).store(value, core::sync::atomic::Ordering::Relaxed) };
    }
    replacement
}

pub fn integer(c: &Call) -> Result<u64> {
    let property = c.args[1] as u32 as usize;
    let version = match property {
        0..=1 => Some(property),
        46..=49 => Some(property - 44),
        _ => None,
    };
    let value = if let Some(i) = version {
        unsafe { (*sdk_versions().add(i)).load(core::sync::atomic::Ordering::Relaxed) }
    } else {
        *X1.get(property).ok_or("unsupported NVN device property")?
    };
    c.write(c.args[2], &value)?;
    Ok(0)
}

// Command handles refer to caller-owned control and command memory.
pub fn finalize_handle(_: &Call) -> Result<u64> {
    Ok(0)
}
