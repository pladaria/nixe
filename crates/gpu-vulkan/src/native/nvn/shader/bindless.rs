//! NVN2 fixed shader operands may obtain a texture and sampler from separate
//! words in the current uniform buffers. Resolve their combined descriptor
//! against the active GPU pools without copying the uniform allocation.
use super::*;

pub(in crate::native::nvn) fn descriptor_ids(
    residency: &mut Residency<'_>,
    bindings: &nixe_nvn::Bindings,
    stage: u32,
    key: HandleKey,
) -> Result<[u32; 2], VulkanError> {
    let pools = bindings
        .bindless
        .ok_or_else(|| error("NVN2 pools are not active"))?;
    let texture = cbuf_word(residency, bindings, stage, key.index.0, key.offset.0)?;
    let sampler = if !key.has_secondary {
        0
    } else if key.index == key.sec_index && key.offset == key.sec_offset {
        texture
    } else {
        cbuf_word(
            residency,
            bindings,
            stage,
            key.sec_index.0,
            key.sec_offset.0,
        )?
    };
    let handle = texture
        .checked_shl(key.shift_left)
        .ok_or_else(|| error("invalid NVN2 handle texture shift"))?
        | sampler
            .checked_shl(key.sec_shift_left)
            .ok_or_else(|| error("invalid NVN2 handle sampler shift"))?;
    let Some([texture, sampler]) = pools.addresses(handle) else {
        return Ok([0; 2]);
    };
    Ok([
        device_word(residency, texture)?,
        if sampler == 0 {
            0
        } else {
            device_word(residency, sampler)?
        },
    ])
}

fn cbuf_word(
    residency: &mut Residency<'_>,
    bindings: &nixe_nvn::Bindings,
    stage: u32,
    bank: u32,
    offset: u32,
) -> Result<u32, VulkanError> {
    read_cbuf(residency, bindings, stage, bank, offset).map(u32::from_le_bytes)
}

pub(in crate::native::nvn) fn read_cbuf<const N: usize>(
    residency: &mut Residency<'_>,
    bindings: &nixe_nvn::Bindings,
    stage: u32,
    bank: u32,
    offset: u32,
) -> Result<[u8; N], VulkanError> {
    if (24..32).contains(&bank) {
        let Some(bytes) = bindings.buffers.push_block((bank - 24) as usize) else {
            return Ok([0; N]);
        };
        let at = offset as usize;
        return bytes
            .get(at..at + N)
            .map(|word| word.try_into().unwrap())
            .ok_or_else(|| error("NVN2 operand outside its push block"));
    }
    let slot = bank
        .checked_sub(3)
        .ok_or_else(|| error(format!("NVN2 operand in unsupported bank {bank}")))?;
    let Some(range) = bindings.buffers.get(false, stage, slot) else {
        return Ok([0; N]);
    };
    if u64::from(offset) + N as u64 > range.size {
        return Ok([0; N]);
    }
    read_device(residency, range.address + u64::from(offset))
}

fn device_word(residency: &mut Residency<'_>, address: u64) -> Result<u32, VulkanError> {
    read_device(residency, address).map(u32::from_le_bytes)
}

fn read_device<const N: usize>(
    residency: &mut Residency<'_>,
    address: u64,
) -> Result<[u8; N], VulkanError> {
    residency.flush_read_span(address, N as u64)?;
    let mut word = [0; N];
    residency
        .runtime
        .device_reader
        .read_block_unsafe(address, &mut word)
        .map_err(error)?;
    Ok(word)
}
