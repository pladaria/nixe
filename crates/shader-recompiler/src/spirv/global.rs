use super::{BufferKind as K, EmitContext, Id};
use crate::{CompileError, Result, artifact::ShaderInfo, diagnostics::FailureCategory};
use spirv::Op;
impl EmitContext<'_> {
    fn global_access(&mut self, info: &ShaderInfo, kind: K, write: bool) -> Result<Id> {
        let (element, size) = self.buffer_element(kind);
        let u = self.types.u32[0];
        let wide = self.types.u64;
        let boolean = self.types.bool;
        let zero = self.constants.u32_zero;
        let return_type = if write { self.types.void } else { element };
        let params = if write {
            vec![wide, element]
        } else {
            vec![wide]
        };
        let ft = self.builder.type_function(return_type, &params)?;
        let function = self.builder.begin_function(return_type, ft)?;
        let address = self.builder.parameter(wide)?;
        let data = if write {
            self.builder.parameter(element)?
        } else {
            Id::NONE
        };
        self.builder.new_label()?;
        for (index, descriptor) in info.storage_buffers_descriptors.iter().enumerate() {
            if index >= 16 || info.nvn_buffer_used & (1 << index) == 0 {
                continue;
            }
            let bank = descriptor.cbuf_index.0 as usize;
            let cbuf = self.cbufs.get(bank).ok_or_else(|| {
                CompileError::new(
                    FailureCategory::ResourceLimit,
                    "Global memory cbuf bank exceeds compiler limit",
                )
            })?;
            let addr_buffer = cbuf[K::U32x2 as usize];
            let size_buffer = cbuf[K::U32 as usize];
            let addr_pointer_type = self.uniform_types[K::U32x2 as usize];
            let size_pointer_type = self.uniform_types[K::U32 as usize];
            let ssbo = self.ssbos[index][kind as usize];
            let element_pointer = self.storage_types[kind as usize].element;
            if [
                addr_buffer,
                size_buffer,
                addr_pointer_type,
                size_pointer_type,
                ssbo,
                element_pointer,
            ]
            .contains(&Id::NONE)
            {
                return Err(CompileError::new(
                    FailureCategory::UnsupportedReferenceCase,
                    "Global memory helper requires undeclared typed buffer views",
                ));
            }
            let address_offset = self.const_u32(descriptor.cbuf_offset.0 / 8)?;
            let size_offset = self.const_u32(descriptor.cbuf_offset.0 / 4 + 2)?;
            let pointer = self.builder.result(
                Op::AccessChain,
                addr_pointer_type,
                &[addr_buffer.0, zero.0, address_offset.0],
            )?;
            let size_pointer = self.builder.result(
                Op::AccessChain,
                size_pointer_type,
                &[size_buffer.0, zero.0, size_offset.0],
            )?;
            let pair = self
                .builder
                .result(Op::Load, self.types.u32[1], &[pointer.0])?;
            let unaligned = self.builder.result(Op::Bitcast, wide, &[pair.0])?;
            let mask = !(self.profile.min_ssbo_alignment.wrapping_sub(1));
            let mask = self
                .builder
                .constant(wide, &[mask as u32, (mask >> 32) as u32])?;
            let base = self
                .builder
                .result(Op::BitwiseAnd, wide, &[unaligned.0, mask.0])?;
            let length = self.builder.result(Op::Load, u, &[size_pointer.0])?;
            let length = self.builder.result(Op::UConvert, wide, &[length.0])?;
            let end = self.builder.result(Op::IAdd, wide, &[base.0, length.0])?;
            let above =
                self.builder
                    .result(Op::UGreaterThanEqual, boolean, &[address.0, base.0])?;
            let below = self
                .builder
                .result(Op::ULessThan, boolean, &[address.0, end.0])?;
            let inside = self
                .builder
                .result(Op::LogicalAnd, boolean, &[above.0, below.0])?;
            let then = self.builder.reserve_id()?;
            let next = self.builder.reserve_id()?;
            self.builder.instruction(Op::SelectionMerge, &[next.0, 0])?;
            self.builder
                .instruction(Op::BranchConditional, &[inside.0, then.0, next.0])?;
            self.builder.label(then)?;
            let offset = self.builder.result(Op::ISub, wide, &[address.0, base.0])?;
            let offset = self.builder.result(Op::UConvert, u, &[offset.0])?;
            let shift = self.const_u32(size.trailing_zeros())?;
            let index = self
                .builder
                .result(Op::ShiftRightLogical, u, &[offset.0, shift.0])?;
            let pointer = self.builder.result(
                Op::AccessChain,
                element_pointer,
                &[ssbo.0, zero.0, index.0],
            )?;
            if write {
                self.builder.instruction(Op::Store, &[pointer.0, data.0])?;
                self.builder.instruction(Op::Return, &[])?;
            } else {
                let value = self.builder.result(Op::Load, element, &[pointer.0])?;
                self.builder.instruction(Op::ReturnValue, &[value.0])?;
            }
            self.builder.label(next)?;
        }
        if write {
            self.builder.instruction(Op::Return, &[])?;
        } else {
            let zero = self.builder.constant_null(element)?;
            self.builder.instruction(Op::ReturnValue, &[zero.0])?;
        }
        self.builder.end_function()?;
        Ok(function)
    }
    pub(super) fn define_global_accessors(&mut self, info: &ShaderInfo) -> Result<()> {
        if !info.uses_global_memory || !self.profile.support_int64 {
            return Ok(());
        }
        self.objects.load_global_func_u32 = self.global_access(info, K::U32, false)?;
        self.objects.write_global_func_u32 = self.global_access(info, K::U32, true)?;
        self.objects.load_global_func_u32x2 = self.global_access(info, K::U32x2, false)?;
        self.objects.write_global_func_u32x2 = self.global_access(info, K::U32x2, true)?;
        self.objects.load_global_func_u32x4 = self.global_access(info, K::U32x4, false)?;
        self.objects.write_global_func_u32x4 = self.global_access(info, K::U32x4, true)?;
        Ok(())
    }
}
