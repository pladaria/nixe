//! Final-address static branch emission, not publication authority. The owner
//! retains source/target roots, validates their versions and copies these bytes
//! only into unpublished storage or inside a Closed maintenance write window.

use crate::abi::HostAbi;
use cranelift_codegen::nixe::StateMap;

pub(crate) const ISLAND_BYTES: usize = 16;

/// Wrap a nonempty architectural transfer with an indirect landing and an
/// aligned terminal patch. Padding and the eventual branch preserve host flags
/// installed by the transfer. The caller supplies the final RX target later.
pub(crate) fn bridge(abi: HostAbi, transfer: &[u8]) -> (Box<[u8]>, usize) {
    let (landing, nop, width): (&[u8], &[u8], usize) = match abi {
        HostAbi::X86_64 => (&[0xf3, 0x0f, 0x1e, 0xfa], &[0x90], 8), // ENDBR64
        HostAbi::Aarch64 => (&[0x5f, 0x24, 0x03, 0xd5], &[0x1f, 0x20, 0x03, 0xd5], 4), // BTI c
    };
    let mut bytes = landing.to_vec();
    bytes.extend_from_slice(transfer);
    assert!(bytes.len().is_multiple_of(nop.len()));
    while !bytes.len().is_multiple_of(width) {
        bytes.extend_from_slice(nop);
    }
    let tail = bytes.len();
    bytes.resize(tail + width, 0);
    (bytes.into_boxed_slice(), tail)
}

/// Immutable dynamic bridge tail: near direct branch or inline absolute jump.
/// All bytes, including an Arm literal, live in the bridge's ordinary span.
/// Only reserved link scratch is changed; host flags and SP are preserved.
pub(crate) fn inline_tail(
    abi: HostAbi,
    source: u64,
    target: u64,
) -> Result<[u8; ISLAND_BYTES], &'static str> {
    let alignment = if abi == HostAbi::X86_64 { 8 } else { 4 };
    if !source.is_multiple_of(alignment)
        || source.checked_add(ISLAND_BYTES as u64).is_none()
        || (abi == HostAbi::Aarch64 && !target.is_multiple_of(4))
    {
        return Err("invalid inline bridge tail address or alignment");
    }
    if !in_range(abi, source, target) {
        return Ok(emit_island(abi, target));
    }
    let branch = emit(abi, source, target, 0)?;
    let mut bytes = [0x90; ISLAND_BYTES];
    if abi == HostAbi::Aarch64 {
        for instruction in bytes.chunks_exact_mut(4) {
            instruction.copy_from_slice(&0xd503201f_u32.to_le_bytes());
        }
    }
    bytes[..branch.patch().len()].copy_from_slice(branch.patch());
    Ok(bytes)
}

/// An optional island is initialized before the patch becomes callable. Its
/// target must be a valid indirect landing (ENDBR64 / BTI-compatible entry),
/// even if the original source-to-target edge was a direct guest branch.
#[derive(Debug)]
pub(crate) struct Branch {
    patch: [u8; 8],
    width: usize,
    pub island: Option<[u8; ISLAND_BYTES]>,
}

impl Branch {
    pub fn patch(&self) -> &[u8] {
        &self.patch[..self.width]
    }
}

/// All addresses are final RX addresses, never writable-alias addresses. Near
/// targets do not use the reserved island. This emits no bridge, budget check,
/// call, stack adjustment or flag-changing instruction.
pub(crate) fn emit(
    abi: HostAbi,
    source: u64,
    target: u64,
    reserved_island: u64,
) -> Result<Branch, &'static str> {
    let width = match abi {
        HostAbi::X86_64 => 8,
        HostAbi::Aarch64 => 4,
    };
    if !source.is_multiple_of(width) || source.checked_add(width).is_none() {
        return Err("invalid static patch address or extent");
    }
    if abi == HostAbi::Aarch64 && !target.is_multiple_of(4) {
        return Err("unaligned AArch64 static target");
    }
    let near = in_range(abi, source, target);
    let (destination, island) = if near {
        (target, None)
    } else {
        if !reserved_island.is_multiple_of(ISLAND_BYTES as u64)
            || reserved_island.checked_add(ISLAND_BYTES as u64).is_none()
        {
            return Err("invalid static island address or extent");
        }
        if !in_range(abi, source, reserved_island) {
            return Err("reserved static island is outside source branch range");
        }
        (reserved_island, Some(emit_island(abi, target)))
    };
    let mut branch = Branch {
        patch: [0; 8],
        width: width as usize,
        island,
    };
    // Reuse the fork's patch encoder; only range selection and the owner-side
    // island are new here. Empty values allocate no storage.
    StateMap {
        id: 0,
        offset: 0,
        entry: false,
        patch_bytes: width as u8,
        fault_bytes: 0,
        poll: None,
        values: Vec::new(),
    }
    .patch_exit(&mut branch.patch, source, destination)
    .map_err(|_| "backend rejected static branch encoding")?;
    Ok(branch)
}

fn in_range(abi: HostAbi, source: u64, target: u64) -> bool {
    let delta = i128::from(target) - i128::from(source);
    match abi {
        HostAbi::X86_64 => i32::try_from(delta - 5).is_ok(),
        HostAbi::Aarch64 => (-(1i128 << 27)..(1i128 << 27)).contains(&delta),
    }
}

fn emit_island(abi: HostAbi, target: u64) -> [u8; ISLAND_BYTES] {
    match abi {
        HostAbi::X86_64 => {
            // Intel SDM Vol. 2, MOV (imm64), JMP (r/m64). Neither changes
            // flags; R11 is exclusively reserved for native link scratch.
            // https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html
            let mut bytes = [0x90; ISLAND_BYTES];
            bytes[..2].copy_from_slice(&[0x49, 0xbb]); // movabs r11, target
            bytes[2..10].copy_from_slice(&target.to_le_bytes());
            bytes[10..13].copy_from_slice(&[0x41, 0xff, 0xe3]); // jmp r11
            bytes
        }
        HostAbi::Aarch64 => {
            // Arm A64 LDR (literal), BR: PC-relative literal is at +8; X16
            // is reserved link scratch. Neither modifies NZCV or the link
            // register. BR X16 permits the ABI's BTI c/jc target landings.
            // https://developer.arm.com/documentation/ddi0602/latest/Base-Instructions/LDR--literal---Load-register--literal--
            // https://developer.arm.com/documentation/ddi0602/latest/Base-Instructions/BR--Branch-to-register-
            let mut bytes = [0; ISLAND_BYTES];
            bytes[..4].copy_from_slice(&0x58000050_u32.to_le_bytes());
            bytes[4..8].copy_from_slice(&0xd61f0200_u32.to_le_bytes());
            bytes[8..].copy_from_slice(&target.to_le_bytes());
            bytes
        }
    }
}

#[cfg(test)]
mod tests;
