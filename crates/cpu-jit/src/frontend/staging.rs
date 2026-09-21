//! Append-only construction of unpublished native adapters. No allocation or
//! metadata interpretation is added to a guest edge.

use super::*;
use crate::abi::EntryContract;

pub(crate) fn append(bytes: &mut Vec<u8>, part: &[u8]) -> usize {
    bytes.resize(bytes.len().next_multiple_of(16), 0);
    let offset = bytes.len();
    bytes.extend_from_slice(part);
    offset
}

pub(crate) fn landing(abi: HostAbi) -> Vec<u8> {
    match abi {
        HostAbi::X86_64 => vec![0xf3, 0x0f, 0x1e, 0xfa],
        HostAbi::Aarch64 => 0xd503249fu32.to_le_bytes().to_vec(),
    }
}

pub(crate) fn nop(abi: HostAbi) -> Vec<u8> {
    match abi {
        HostAbi::X86_64 => vec![0x90],
        HostAbi::Aarch64 => 0xd503201fu32.to_le_bytes().to_vec(),
    }
}

pub(crate) fn jump(bytes: &mut Vec<u8>, abi: HostAbi, target: u32) -> Result<(), Error> {
    let offset = bytes.len().next_multiple_of(8);
    while bytes.len() < offset {
        bytes.extend(nop(abi));
    }
    bytes.resize(offset + 8, 0);
    StateMap {
        id: 0,
        offset: u32::try_from(offset).map_err(fail)?,
        entry: false,
        patch_bytes: if abi == HostAbi::X86_64 { 8 } else { 4 },
        fault_bytes: 0,
        poll: None,
        values: Vec::new(),
    }
    .patch_exit(bytes, 0, u64::from(target))
    .map_err(fail)
}

pub(crate) fn canonical(
    bytes: &mut Vec<u8>,
    contract: &EntryContract,
    fast: u32,
) -> Result<u32, Error> {
    let mut ingress = landing(contract.abi);
    ingress.extend(crate::native::emit_canonical_entry(contract).map_err(fail)?);
    let offset = u32::try_from(append(bytes, &ingress)).map_err(fail)?;
    jump(bytes, contract.abi, fast)?;
    Ok(offset)
}
