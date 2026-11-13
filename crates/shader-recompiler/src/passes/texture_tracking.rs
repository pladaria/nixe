use super::fold::{arg, node};
use crate::{
    CompileError, HostTranslateInfo, Result,
    arena::{BlockId, InstId},
    diagnostics::FailureCategory,
    environment::{CbufWordKey, ConstBufferAddr, Environment, EnvironmentCaches, HandleKey},
    ir::{Ir, Opcode as O, Type, Value},
    types::{ByteOffset, CbufIndex, TextureHandle},
};
use std::collections::{HashSet, VecDeque};

pub(super) type Address = ConstBufferAddr<Value>;
pub(super) fn direct(index: CbufIndex, offset: ByteOffset) -> Address {
    Address {
        index,
        offset,
        shift_left: 0,
        secondary_index: CbufIndex(0),
        secondary_offset: ByteOffset(0),
        secondary_shift_left: 0,
        dynamic_offset: Value::None,
        handle_program: None,
        count: 1,
        has_secondary: false,
    }
}
pub(super) fn size_shift(ir: &Ir, value: Value) -> Result<u32> {
    if let Some(id) = node(ir, value, O::ShiftLeftLogical32)?
        && let Value::U32(shift @ 3..=30) = arg(ir, id, 1)?
    {
        return Ok(shift);
    }
    Ok(3)
}
pub(super) fn sampled_limit(host: &HostTranslateInfo) -> u32 {
    host.max_per_stage_descriptor_sampled_images
        .min(host.max_descriptor_set_sampled_images)
        .max(1)
}
fn descriptor_count(base: u32, shift: u32, limit: u32) -> u32 {
    let limit = limit.max(1);
    let bytes = 16u32.wrapping_mul(limit);
    if shift >= 31 || base >= bytes || bytes - base < 8 {
        return 1;
    }
    limit.min(1 + (bytes - base - 8) / (1 << shift))
}
pub(super) struct Tracker<'a> {
    pub env: &'a mut dyn Environment,
    pub caches: &'a mut EnvironmentCaches<InstId, Value>,
    pub host: &'a HostTranslateInfo,
    active: HashSet<InstId>,
    pub block: Option<BlockId>,
}
impl<'a> Tracker<'a> {
    pub fn new(
        env: &'a mut dyn Environment,
        caches: &'a mut EnvironmentCaches<InstId, Value>,
        host: &'a HostTranslateInfo,
    ) -> Self {
        caches.clear();
        Self {
            env,
            caches,
            host,
            active: HashSet::new(),
            block: None,
        }
    }
    pub fn word(&mut self, index: CbufIndex, offset: ByteOffset) -> Result<u32> {
        let key = CbufWordKey { index, offset };
        if let Some(&word) = self.caches.cbuf_word_cache.get(&key) {
            return Ok(word);
        }
        let word = self.env.read_cbuf_value(index, offset)?;
        self.caches.cbuf_word_cache.insert(key, word);
        Ok(word)
    }
    pub fn handle(&mut self, a: &Address) -> Result<TextureHandle> {
        let key = HandleKey {
            index: a.index,
            offset: a.offset,
            shift_left: a.shift_left,
            sec_index: a.secondary_index,
            sec_offset: a.secondary_offset,
            sec_shift_left: a.secondary_shift_left,
            count: a.count,
            has_secondary: a.has_secondary,
        };
        if let Some(&handle) = self.caches.handle_cache.get(&key) {
            return Ok(TextureHandle(handle));
        }
        if a.shift_left >= 32 || a.secondary_shift_left >= 32 {
            return Err(CompileError::new(
                FailureCategory::UnsupportedReferenceCase,
                "Texture handle shift exceeds the word width",
            ));
        }
        let handle = self.env.read_texture_handle(key)?;
        self.caches.handle_cache.insert(key, handle.0);
        Ok(handle)
    }
    pub fn track(&mut self, ir: &Ir, value: Value) -> Result<Option<Address>> {
        let value = super::handle_control_flow::conditional_phi(
            ir,
            value,
            self.block.expect("texture block selected"),
        )?;
        let Value::Inst(key) = ir.resolve(value)? else {
            return Ok(None);
        };
        if let Some(address) = self.caches.track_cache.get(&key) {
            return Ok(Some(address.clone()));
        }
        if !self.active.insert(key) {
            return Ok(None);
        }
        let result = self.search(ir, key);
        self.active.remove(&key);
        if let Ok(Some(address)) = &result {
            self.caches.track_cache.insert(key, address.clone());
        }
        result
    }
    fn search(&mut self, ir: &Ir, root: InstId) -> Result<Option<Address>> {
        // A loaded word is not the pointer used to address it. Track expression
        // wrappers and phis, but never descend into memory, textures or floats.
        // https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/shader_recompiler/ir_opt/texture_pass.cpp#L591
        let mut queue = VecDeque::from([root]);
        let mut seen = HashSet::from([root]);
        let floats = Type::F16.0
            | Type::F32.0
            | Type::F64.0
            | Type::F16x2.0
            | Type::F16x3.0
            | Type::F16x4.0
            | Type::F32x2.0
            | Type::F32x3.0
            | Type::F32x4.0
            | Type::F64x2.0
            | Type::F64x3.0
            | Type::F64x4.0;
        while let Some(id) = queue.pop_front() {
            if let Some(address) = self.candidate(ir, id)? {
                return Ok(Some(address));
            }
            let inst = ir.inst(id)?;
            let op = inst.opcode();
            if memory_load(op)
                || super::texture_table::indexed(op).is_some()
                || super::texture_table::sampled(op)
                || super::texture_table::storage(op)
                || inst.ty().0 & floats != 0
            {
                continue;
            }
            for index in (0..inst.num_args()).rev() {
                let value = super::handle_control_flow::conditional_phi(
                    ir,
                    inst.arg(index)?,
                    self.block.unwrap(),
                )?;
                if let Value::Inst(input) = ir.resolve(value)?
                    && seen.insert(input)
                {
                    queue.push_back(input);
                }
            }
        }
        Ok(None)
    }
    fn constant(&mut self, ir: &Ir, value: Value) -> Result<Option<u32>> {
        let Some(id) = node(ir, value, O::GetCbufU32)? else {
            return Ok(None);
        };
        if arg(ir, id, 0)? != Value::U32(1) {
            return Ok(None);
        }
        let Value::U32(offset) = arg(ir, id, 1)? else {
            return Ok(None);
        };
        self.word(CbufIndex(1), ByteOffset(offset)).map(Some)
    }
    fn candidate(&mut self, ir: &Ir, id: InstId) -> Result<Option<Address>> {
        match ir.inst(id)?.opcode() {
            O::BitwiseOr32 => {
                let left = self.track(ir, arg(ir, id, 0)?)?;
                let right = self.track(ir, arg(ir, id, 1)?)?;
                let (Some(mut left), Some(mut right)) = (left, right) else {
                    return Ok(None);
                };
                if left.has_secondary || right.has_secondary || (left.count > 1 && right.count > 1)
                {
                    return Ok(None);
                }
                if right.count > 1
                    || (left.count == 1
                        && (left.shift_left > 0
                            || left.index > right.index
                            || left.offset > right.offset))
                {
                    std::mem::swap(&mut left, &mut right);
                }
                left.secondary_index = right.index;
                left.secondary_offset = right.offset;
                left.secondary_shift_left = right.shift_left;
                left.has_secondary = true;
                return Ok(Some(left));
            }
            O::ShiftLeftLogical32 => {
                let Value::U32(shift) = arg(ir, id, 1)? else {
                    return Ok(None);
                };
                return Ok(self.track(ir, arg(ir, id, 0)?)?.map(|mut a| {
                    a.shift_left = shift;
                    a
                }));
            }
            O::BitwiseAnd32 => {
                let (mut left, mut right) = (arg(ir, id, 0)?, arg(ir, id, 1)?);
                if ir.is_immediate(left)? {
                    std::mem::swap(&mut left, &mut right);
                }
                if !ir.is_immediate(left)? && !ir.is_immediate(right)? {
                    if let Some(word) = self.constant(ir, left)? {
                        left = right;
                        right = Value::U32(word);
                    } else if let Some(word) = self.constant(ir, right)? {
                        right = Value::U32(word);
                    } else {
                        return Ok(None);
                    }
                }
                let Value::U32(mask) = right else {
                    return Ok(None);
                };
                return Ok(self.track(ir, left)?.map(|mut a| {
                    a.shift_left = mask.trailing_zeros();
                    a
                }));
            }
            O::GetCbufU32 | O::GetCbufU32x2 => (),
            _ => return Ok(None),
        }
        let Value::U32(index) = arg(ir, id, 0)? else {
            return Ok(None);
        };
        let offset = arg(ir, id, 1)?;
        if let Value::U32(offset) = offset {
            return Ok(Some(direct(CbufIndex(index), ByteOffset(offset))));
        }
        let (base, dynamic) = if let Some(add) = node(ir, offset, O::IAdd32)? {
            match (arg(ir, add, 0)?, arg(ir, add, 1)?) {
                (Value::U32(base), dynamic) | (dynamic, Value::U32(base)) => (base, dynamic),
                _ => return Ok(None),
            }
        } else if node(ir, offset, O::ShiftLeftLogical32)?.is_some() {
            (0, offset)
        } else {
            return Ok(None);
        };
        let limit = 1024
            .min(sampled_limit(self.host))
            .min(self.host.max_per_stage_resources.max(1));
        let mut address = direct(CbufIndex(index), ByteOffset(base));
        address.dynamic_offset = dynamic;
        address.count = descriptor_count(base, size_shift(ir, dynamic)?, limit);
        Ok(Some(address))
    }
}

fn memory_load(op: O) -> bool {
    matches!(
        op,
        O::LoadGlobalU8
            | O::LoadGlobalS8
            | O::LoadGlobalU16
            | O::LoadGlobalS16
            | O::LoadGlobal32
            | O::LoadGlobal64
            | O::LoadGlobal128
            | O::LoadStorageU8
            | O::LoadStorageS8
            | O::LoadStorageU16
            | O::LoadStorageS16
            | O::LoadStorage32
            | O::LoadStorage64
            | O::LoadStorage128
            | O::LoadLocal
            | O::LoadSharedU8
            | O::LoadSharedS8
            | O::LoadSharedU16
            | O::LoadSharedS16
            | O::LoadSharedU32
            | O::LoadSharedU64
            | O::LoadSharedU128
    )
}
