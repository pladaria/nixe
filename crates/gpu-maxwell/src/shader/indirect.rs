//! Bounded constant-buffer jump tables, captured with their real dependencies.
//! https://github.com/eden-emulator/mirror/blob/master/src/shader_recompiler/frontend/maxwell/control_flow.cpp (AnalyzeBRX)
use super::binary::{MaxwellShaderBinary, MaxwellShaderMemoryView};
use super::decode::decode_predicate;
use super::error::{MaxwellShaderTranslationError, malformed};
use crate::{MaxwellGpuAddressSpace, MaxwellGpuMapping, MaxwellThreeDShaderBindingState};
use nixe_gpu::{ShaderPredicate, ShaderSourceLocation};
use nixe_memory::{CanonicalBackingRange, CanonicalCpuWriteDependency, CanonicalWriteBatch};
use std::hash::{Hash, Hasher};

#[derive(Clone, Debug)]
pub(super) struct IndirectTable {
    pub offset: u32,
    pub register: u8,
    pub targets: Box<[(u32, ShaderSourceLocation)]>,
    pub load_offset: u32,
    pub read_end: u32,
    group: u8,
    slot: u8,
    buffer_address: u64,
    buffer_size: u32,
    bytes: Box<[u8]>,
    range: CanonicalBackingRange,
    cpu_writes: CanonicalCpuWriteDependency,
    mappings: Box<[MaxwellGpuMapping]>,
}
impl PartialEq for IndirectTable {
    fn eq(&self, other: &Self) -> bool {
        self.offset == other.offset
            && self.register == other.register
            && self.targets == other.targets
            && self.load_offset == other.load_offset
            && self.read_end == other.read_end
    }
}
impl Eq for IndirectTable {}
impl Hash for IndirectTable {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.offset.hash(state);
        self.register.hash(state);
        self.targets.hash(state);
        self.load_offset.hash(state);
        self.read_end.hash(state);
    }
}
impl IndirectTable {
    pub fn binding_current(&self, bindings: &MaxwellThreeDShaderBindingState) -> bool {
        bindings.groups()[self.group as usize].constant_buffers()[self.slot as usize].is_some_and(
            |binding| {
                binding.enabled()
                    && binding
                        .address()
                        .is_some_and(|address| address.get() == self.buffer_address)
                    && binding.size() == Some(self.buffer_size)
            },
        )
    }
    pub fn current(&self, space: &MaxwellGpuAddressSpace) -> bool {
        if !self
            .mappings
            .iter()
            .all(|mapping| space.retained_mapping_is_current(mapping))
        {
            return false;
        }
        if self.cpu_writes.remains_current() {
            return true;
        }
        // Tables may share pages with frequently changing constants. Compare
        // only their actual bytes rather than invalidating a shader per frame.
        let mut bytes = vec![0; self.bytes.len()];
        self.range.read(0, &mut bytes).is_ok() && bytes == *self.bytes
    }
    pub fn staged_current(&self, writes: &CanonicalWriteBatch) -> bool {
        match writes.overlaps(&self.range, 0, self.bytes.len() as u64) {
            Ok(false) => true,
            Ok(true) => {
                let mut bytes = vec![0; self.bytes.len()];
                writes.read_overlay(&self.range, 0, &mut bytes).is_ok() && bytes == *self.bytes
            }
            Err(_) => false,
        }
    }
}

pub(super) fn capture(
    binary: &mut MaxwellShaderBinary,
    bindings: &MaxwellThreeDShaderBindingState,
    group: Option<u8>,
    memory: &MaxwellShaderMemoryView<'_>,
) -> Result<(), MaxwellShaderTranslationError> {
    if !binary
        .bundles
        .iter()
        .any(|bundle| bundle.instructions.iter().any(|word| word >> 48 == 0xe250))
    {
        return Ok(());
    }
    let code: Vec<_> = binary
        .bundles
        .iter()
        .flat_map(|bundle| {
            bundle
                .instructions
                .iter()
                .enumerate()
                .map(move |(slot, &word)| (bundle.offset + 8 + slot as u32 * 8, word))
        })
        .collect();
    let mut tables = Vec::new();
    let mut proof_intervals = Vec::new();
    for (index, &(offset, word)) in code.iter().enumerate() {
        if word >> 48 != 0xe250 {
            continue;
        }
        let fail = |reason| malformed(binary.stage(), offset, word, reason);
        if word & 0x7f != 15 || decode_predicate(word) != ShaderPredicate::Always {
            return Err(fail(
                "BRX requires an unconditional relative register target",
            ));
        }
        let register = (word >> 8) as u8;
        // Track the last definition in the straight-line segment, rejecting
        // conditional writes and control edges instead of borrowing stale facts.
        let writer = |before: usize, register: u8| -> Option<usize> {
            (0..before)
                .rev()
                .take_while(|&i| {
                    !matches!(
                        code[i].1 >> 48,
                        0xe240 | 0xe250 | 0xe340 | 0xe350 | 0xf0f8 | 0xe2a0 | 0xe2b0 | 0xe290
                    )
                        // Multi-result instructions can clobber a register
                        // other than their low-byte destination. Do not carry
                        // scalar facts across them without decoding all writes.
                        && !super::texture::is_texture_access_simplified(code[i].1)
                        && (code[i].1 >> 48) as u16 & 0xfff8 != 0xeed0
                        && !super::interface::is_attribute_load(code[i].1)
                })
                .find(|&i| {
                    let word = code[i].1;
                    let destination = u16::from(word as u8);
                    let count = if super::data::is_constant_buffer_load(word) {
                        match (word >> 48) & 7 {
                            5 => 2,
                            6 => 4,
                            _ => 1,
                        }
                    } else {
                        1
                    };
                    (destination..destination + count).contains(&u16::from(register))
                })
        };
        let load_index = writer(index, register)
            .ok_or_else(|| fail("BRX register has no local constant-table load"))?;
        let load = code[load_index].1;
        if load >> 48 != 0xef94
            || (load >> 44) & 3 != 0
            || decode_predicate(load) != ShaderPredicate::Always
        {
            return Err(fail(
                "BRX requires an unconditionally loaded B32 indexed constant table",
            ));
        }
        let address_register = (load >> 8) as u8;
        let shift_index = writer(load_index, address_register)
            .ok_or_else(|| fail("BRX table byte index is unproven"))?;
        let shift = code[shift_index].1;
        if (shift >> 48) as u16 & 0xfeff != 0x3848
            || (shift >> 20) & 0x7ffff != 2
            || shift & ((1 << 47) | (1 << 43)) != 0
            || decode_predicate(shift) != ShaderPredicate::Always
        {
            return Err(fail("BRX table index requires a proven word stride"));
        }
        let index_register = (shift >> 8) as u8;
        let bound_index = writer(shift_index, index_register)
            .ok_or_else(|| fail("BRX table index bound is unproven"))?;
        let bound = code[bound_index].1;
        proof_intervals.push((code[bound_index].0, offset, word));
        if (bound >> 48) as u16 & 0xfef8 != 0x3820
            || bound & ((1 << 48) | (1 << 56) | (1 << 47) | (3 << 43)) != 0
            || super::decode::decode_predicate_fields(bound, 39, 42) != ShaderPredicate::Always
            || decode_predicate(bound) != ShaderPredicate::Always
        {
            return Err(fail(
                "BRX index requires an unsigned immediate minimum bound",
            ));
        }
        let count = ((bound >> 20) & 0x7ffff) as u32 + 1;
        let byte_count = count
            .checked_mul(4)
            .filter(|&size| size <= super::binary::MAXWELL_SHADER_READ_LIMIT as u32)
            .ok_or_else(|| fail("BRX table exceeds the bounded shader read domain"))?;
        let group = group.ok_or_else(|| fail("BRX has no effective constant-buffer group"))?;
        let slot = ((load >> 36) & 31) as u8;
        let binding = bindings.groups()[group as usize]
            .constant_buffers()
            .get(slot as usize)
            .copied()
            .flatten()
            .filter(|b| b.enabled())
            .ok_or_else(|| fail("BRX constant buffer is unbound"))?;
        let buffer_address = binding
            .address()
            .ok_or_else(|| fail("BRX constant buffer has no address"))?
            .get();
        let buffer_size = binding
            .size()
            .ok_or_else(|| fail("BRX constant buffer has no size"))?;
        let table_offset = ((load >> 20) & 0xffff) as u16 as i16 as i32;
        if table_offset < 0
            || (table_offset as u32)
                .checked_add(byte_count)
                .is_none_or(|end| end > buffer_size)
        {
            return Err(fail("BRX table exceeds its constant buffer"));
        }
        let read = memory.read(
            binary.stage(),
            buffer_address + table_offset as u64,
            byte_count as usize,
        )?;
        let raw_displacement = ((word >> 20) & 0xffffff) as i32;
        let base = offset
            .wrapping_add(8)
            .wrapping_add(((raw_displacement << 8) >> 8) as u32);
        let mut targets = Vec::new();
        for bytes in read.bytes.chunks_exact(4) {
            let value = u32::from_le_bytes(bytes.try_into().unwrap());
            let mut target = base.wrapping_add(value);
            if target % 32 == 0 {
                target = target.wrapping_add(8);
            }
            if !code.iter().any(|&(address, _)| address == target) {
                return Err(fail(
                    "BRX table entry is not an executable instruction slot",
                ));
            }
            if !targets.iter().any(|&(old, _)| old == value) {
                targets.push((value, ShaderSourceLocation::new(target)));
            }
        }
        targets.sort_by_key(|&(value, _)| value);
        log::debug!(
            "captured BRX table: stage={:?} offset={offset:#x} register={register} targets={targets:?}",
            binary.stage()
        );
        tables.push(IndirectTable {
            offset,
            register,
            targets: targets.into(),
            load_offset: code[load_index].0,
            read_end: table_offset as u32 + byte_count,
            group,
            slot,
            buffer_address,
            buffer_size,
            bytes: read.bytes.into(),
            range: read.snapshot,
            cpu_writes: read.cpu_writes,
            mappings: read.mappings.into(),
        });
    }
    // The clamp/stride/load proof must dominate BRX. Incoming branches may
    // enter at the clamp, but cannot bypass any step in the proven interval.
    // Include stack-token destinations: SYNC/BRK/CONT can enter those PCs.
    let mut incoming_targets = Vec::new();
    for &(offset, word) in &code {
        if super::control_flow::is_branch(word)
            || super::control_flow::is_set_sync_point(word)
            || super::control_flow::is_loop_push(word)
        {
            incoming_targets.push(
                super::control_flow::decode_shader_control_target(
                    binary.stage(),
                    offset,
                    word,
                    binary.bundles.last().map_or(0, |bundle| bundle.offset + 32),
                )?
                .byte_offset(),
            );
        }
    }
    incoming_targets.extend(
        tables
            .iter()
            .flat_map(|table| table.targets.iter().map(|(_, target)| target.byte_offset())),
    );
    for (start, end, word) in proof_intervals {
        if incoming_targets
            .iter()
            .any(|&target| target > start && target <= end)
        {
            return Err(malformed(
                binary.stage(),
                end,
                word,
                "BRX table bound does not dominate every incoming control path",
            ));
        }
    }
    binary.indirect_tables = tables.into();
    Ok(())
}

#[cfg(test)]
mod tests;
