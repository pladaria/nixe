use super::*;
use crate::Result;

impl Emitter<'_> {
    pub fn condition_ref(&mut self, value: Value) -> Result<Value> {
        self.emit(Opcode::ConditionRef, &[value], 0)
    }

    pub fn reference(&mut self, value: Value) -> Result<Value> {
        self.emit(Opcode::Reference, &[value], 0)
    }

    pub fn prologue(&mut self) -> Result<Value> {
        self.emit(Opcode::Prologue, &[], 0)
    }

    pub fn epilogue(&mut self) -> Result<Value> {
        self.emit(Opcode::Epilogue, &[], 0)
    }

    pub fn demote_to_helper_invocation(&mut self) -> Result<Value> {
        self.emit(Opcode::DemoteToHelperInvocation, &[], 0)
    }

    pub fn emit_vertex(&mut self, stream: Value) -> Result<Value> {
        self.emit(Opcode::EmitVertex, &[stream], 0)
    }

    pub fn end_primitive(&mut self, stream: Value) -> Result<Value> {
        self.emit(Opcode::EndPrimitive, &[stream], 0)
    }

    pub fn barrier(&mut self) -> Result<Value> {
        self.emit(Opcode::Barrier, &[], 0)
    }

    pub fn workgroup_memory_barrier(&mut self) -> Result<Value> {
        self.emit(Opcode::WorkgroupMemoryBarrier, &[], 0)
    }

    pub fn device_memory_barrier(&mut self) -> Result<Value> {
        self.emit(Opcode::DeviceMemoryBarrier, &[], 0)
    }

    pub fn get_reg(&mut self, reg: Reg) -> Result<Value> {
        self.emit(Opcode::GetRegister, &[Value::Reg(reg)], 0)
    }

    pub fn set_reg(&mut self, reg: Reg, value: Value) -> Result<Value> {
        self.emit(Opcode::SetRegister, &[Value::Reg(reg), value], 0)
    }

    pub fn get_goto_variable(&mut self, id: u32) -> Result<Value> {
        self.emit(Opcode::GetGotoVariable, &[Value::U32(id)], 0)
    }

    pub fn set_goto_variable(&mut self, id: u32, value: Value) -> Result<Value> {
        self.emit(Opcode::SetGotoVariable, &[Value::U32(id), value], 0)
    }

    pub fn get_indirect_branch_variable(&mut self) -> Result<Value> {
        self.emit(Opcode::GetIndirectBranchVariable, &[], 0)
    }

    pub fn set_indirect_branch_variable(&mut self, value: Value) -> Result<Value> {
        self.emit(Opcode::SetIndirectBranchVariable, &[value], 0)
    }

    pub fn get_float_cbuf(&mut self, binding: Value, byte_offset: Value) -> Result<Value> {
        self.emit(Opcode::GetCbufF32, &[binding, byte_offset], 0)
    }

    pub fn get_z_flag(&mut self) -> Result<Value> {
        self.emit(Opcode::GetZFlag, &[], 0)
    }

    pub fn get_s_flag(&mut self) -> Result<Value> {
        self.emit(Opcode::GetSFlag, &[], 0)
    }

    pub fn get_c_flag(&mut self) -> Result<Value> {
        self.emit(Opcode::GetCFlag, &[], 0)
    }

    pub fn get_o_flag(&mut self) -> Result<Value> {
        self.emit(Opcode::GetOFlag, &[], 0)
    }

    pub fn set_z_flag(&mut self, value: Value) -> Result<Value> {
        self.emit(Opcode::SetZFlag, &[value], 0)
    }

    pub fn set_s_flag(&mut self, value: Value) -> Result<Value> {
        self.emit(Opcode::SetSFlag, &[value], 0)
    }

    pub fn set_c_flag(&mut self, value: Value) -> Result<Value> {
        self.emit(Opcode::SetCFlag, &[value], 0)
    }

    pub fn set_o_flag(&mut self, value: Value) -> Result<Value> {
        self.emit(Opcode::SetOFlag, &[value], 0)
    }

    pub fn get_attribute_at_vertex(
        &mut self,
        attribute: Attribute,
        vertex: Value,
    ) -> Result<Value> {
        self.emit(
            Opcode::GetAttribute,
            &[Value::Attribute(attribute), vertex],
            0,
        )
    }

    pub fn get_attribute_u32_at_vertex(
        &mut self,
        attribute: Attribute,
        vertex: Value,
    ) -> Result<Value> {
        self.emit(
            Opcode::GetAttributeU32,
            &[Value::Attribute(attribute), vertex],
            0,
        )
    }

    pub fn set_attribute(
        &mut self,
        attribute: Attribute,
        value: Value,
        vertex: Value,
    ) -> Result<Value> {
        self.emit(
            Opcode::SetAttribute,
            &[Value::Attribute(attribute), value, vertex],
            0,
        )
    }

    pub fn get_attribute_indexed_at_vertex(
        &mut self,
        phys_address: Value,
        vertex: Value,
    ) -> Result<Value> {
        self.emit(Opcode::GetAttributeIndexed, &[phys_address, vertex], 0)
    }

    pub fn set_attribute_indexed(
        &mut self,
        phys_address: Value,
        value: Value,
        vertex: Value,
    ) -> Result<Value> {
        self.emit(
            Opcode::SetAttributeIndexed,
            &[phys_address, value, vertex],
            0,
        )
    }

    pub fn get_patch(&mut self, patch: Patch) -> Result<Value> {
        self.emit(Opcode::GetPatch, &[Value::Patch(patch)], 0)
    }

    pub fn set_patch(&mut self, patch: Patch, value: Value) -> Result<Value> {
        self.emit(Opcode::SetPatch, &[Value::Patch(patch), value], 0)
    }

    pub fn set_frag_color(&mut self, index: u32, component: u32, value: Value) -> Result<Value> {
        self.emit(
            Opcode::SetFragColor,
            &[Value::U32(index), Value::U32(component), value],
            0,
        )
    }

    pub fn set_sample_mask(&mut self, value: Value) -> Result<Value> {
        self.emit(Opcode::SetSampleMask, &[value], 0)
    }

    pub fn set_frag_depth(&mut self, value: Value) -> Result<Value> {
        self.emit(Opcode::SetFragDepth, &[value], 0)
    }

    pub fn local_invocation_id(&mut self) -> Result<Value> {
        self.emit(Opcode::LocalInvocationId, &[], 0)
    }

    pub fn invocation_id(&mut self) -> Result<Value> {
        self.emit(Opcode::InvocationId, &[], 0)
    }

    pub fn invocation_info(&mut self) -> Result<Value> {
        self.emit(Opcode::InvocationInfo, &[], 0)
    }

    pub fn sample_id(&mut self) -> Result<Value> {
        self.emit(Opcode::SampleId, &[], 0)
    }

    pub fn is_helper_invocation(&mut self) -> Result<Value> {
        self.emit(Opcode::IsHelperInvocation, &[], 0)
    }

    pub fn sr_w_scale_factor_xy(&mut self) -> Result<Value> {
        self.emit(Opcode::SR_WScaleFactorXY, &[], 0)
    }

    pub fn sr_w_scale_factor_z(&mut self) -> Result<Value> {
        self.emit(Opcode::SR_WScaleFactorZ, &[], 0)
    }

    pub fn y_direction(&mut self) -> Result<Value> {
        self.emit(Opcode::YDirection, &[], 0)
    }

    pub fn resolution_down_factor(&mut self) -> Result<Value> {
        self.emit(Opcode::ResolutionDownFactor, &[], 0)
    }

    pub fn lane_id(&mut self) -> Result<Value> {
        self.emit(Opcode::LaneId, &[], 0)
    }

    pub fn load_global_u8(&mut self, address: Value) -> Result<Value> {
        self.require_type(address, Type::U64)?;
        self.emit(Opcode::LoadGlobalU8, &[address], 0)
    }

    pub fn load_global_s8(&mut self, address: Value) -> Result<Value> {
        self.require_type(address, Type::U64)?;
        self.emit(Opcode::LoadGlobalS8, &[address], 0)
    }

    pub fn load_global_u16(&mut self, address: Value) -> Result<Value> {
        self.require_type(address, Type::U64)?;
        self.emit(Opcode::LoadGlobalU16, &[address], 0)
    }

    pub fn load_global_s16(&mut self, address: Value) -> Result<Value> {
        self.require_type(address, Type::U64)?;
        self.emit(Opcode::LoadGlobalS16, &[address], 0)
    }

    pub fn load_global32(&mut self, address: Value) -> Result<Value> {
        self.require_type(address, Type::U64)?;
        self.emit(Opcode::LoadGlobal32, &[address], 0)
    }

    pub fn load_global64(&mut self, address: Value) -> Result<Value> {
        self.require_type(address, Type::U64)?;
        self.emit(Opcode::LoadGlobal64, &[address], 0)
    }

    pub fn load_global128(&mut self, address: Value) -> Result<Value> {
        self.require_type(address, Type::U64)?;
        self.emit(Opcode::LoadGlobal128, &[address], 0)
    }

    pub fn write_global_u8(&mut self, address: Value, value: Value) -> Result<Value> {
        self.require_type(address, Type::U64)?;
        self.emit(Opcode::WriteGlobalU8, &[address, value], 0)
    }

    pub fn write_global_s8(&mut self, address: Value, value: Value) -> Result<Value> {
        self.require_type(address, Type::U64)?;
        self.emit(Opcode::WriteGlobalS8, &[address, value], 0)
    }

    pub fn write_global_u16(&mut self, address: Value, value: Value) -> Result<Value> {
        self.require_type(address, Type::U64)?;
        self.emit(Opcode::WriteGlobalU16, &[address, value], 0)
    }

    pub fn write_global_s16(&mut self, address: Value, value: Value) -> Result<Value> {
        self.require_type(address, Type::U64)?;
        self.emit(Opcode::WriteGlobalS16, &[address, value], 0)
    }

    pub fn write_global32(&mut self, address: Value, value: Value) -> Result<Value> {
        self.require_type(address, Type::U64)?;
        self.emit(Opcode::WriteGlobal32, &[address, value], 0)
    }

    pub fn write_global64(&mut self, address: Value, vector: Value) -> Result<Value> {
        self.require_type(address, Type::U64)?;
        self.emit(Opcode::WriteGlobal64, &[address, vector], 0)
    }

    pub fn write_global128(&mut self, address: Value, vector: Value) -> Result<Value> {
        self.require_type(address, Type::U64)?;
        self.emit(Opcode::WriteGlobal128, &[address, vector], 0)
    }

    pub fn load_local(&mut self, word_offset: Value) -> Result<Value> {
        self.emit(Opcode::LoadLocal, &[word_offset], 0)
    }

    pub fn write_local(&mut self, word_offset: Value, value: Value) -> Result<Value> {
        self.emit(Opcode::WriteLocal, &[word_offset, value], 0)
    }

    pub fn get_zero_from_op(&mut self, op: Value) -> Result<Value> {
        self.emit(Opcode::GetZeroFromOp, &[op], 0)
    }

    pub fn get_sign_from_op(&mut self, op: Value) -> Result<Value> {
        self.emit(Opcode::GetSignFromOp, &[op], 0)
    }

    pub fn get_carry_from_op(&mut self, op: Value) -> Result<Value> {
        self.emit(Opcode::GetCarryFromOp, &[op], 0)
    }

    pub fn get_overflow_from_op(&mut self, op: Value) -> Result<Value> {
        self.emit(Opcode::GetOverflowFromOp, &[op], 0)
    }

    pub fn get_sparse_from_op(&mut self, op: Value) -> Result<Value> {
        self.emit(Opcode::GetSparseFromOp, &[op], 0)
    }

    pub fn get_in_bounds_from_op(&mut self, op: Value) -> Result<Value> {
        self.emit(Opcode::GetInBoundsFromOp, &[op], 0)
    }

    pub fn fp_add(&mut self, a: Value, b: Value, control: FpControl) -> Result<Value> {
        self.same_type(&[a, b])?;
        match self.ir.value_type(a)? {
            Type::F16 => self.emit(Opcode::FPAdd16, &[a, b], control.bits()),
            Type::F32 => self.emit(Opcode::FPAdd32, &[a, b], control.bits()),
            Type::F64 => self.emit(Opcode::FPAdd64, &[a, b], control.bits()),
            _ => Err(Self::invalid("Unsupported operand type")),
        }
    }

    pub fn select(
        &mut self,
        condition: Value,
        true_value: Value,
        false_value: Value,
    ) -> Result<Value> {
        self.same_type(&[true_value, false_value])?;
        match self.ir.value_type(true_value)? {
            Type::U1 => self.emit(Opcode::SelectU1, &[condition, true_value, false_value], 0),
            Type::U8 => self.emit(Opcode::SelectU8, &[condition, true_value, false_value], 0),
            Type::U16 => self.emit(Opcode::SelectU16, &[condition, true_value, false_value], 0),
            Type::U32 => self.emit(Opcode::SelectU32, &[condition, true_value, false_value], 0),
            Type::U64 => self.emit(Opcode::SelectU64, &[condition, true_value, false_value], 0),
            Type::F32 => self.emit(Opcode::SelectF32, &[condition, true_value, false_value], 0),
            Type::F64 => self.emit(Opcode::SelectF64, &[condition, true_value, false_value], 0),
            _ => Err(Self::invalid("Unsupported operand type")),
        }
    }

    pub fn pack_uint2x32(&mut self, vector: Value) -> Result<Value> {
        self.emit(Opcode::PackUint2x32, &[vector], 0)
    }

    pub fn unpack_uint2x32(&mut self, value: Value) -> Result<Value> {
        self.emit(Opcode::UnpackUint2x32, &[value], 0)
    }

    pub fn pack_float2x16(&mut self, vector: Value) -> Result<Value> {
        self.emit(Opcode::PackFloat2x16, &[vector], 0)
    }

    pub fn unpack_float2x16(&mut self, value: Value) -> Result<Value> {
        self.emit(Opcode::UnpackFloat2x16, &[value], 0)
    }

    pub fn pack_half2x16(&mut self, vector: Value) -> Result<Value> {
        self.emit(Opcode::PackHalf2x16, &[vector], 0)
    }

    pub fn unpack_half2x16(&mut self, value: Value) -> Result<Value> {
        self.emit(Opcode::UnpackHalf2x16, &[value], 0)
    }

    pub fn pack_double2x32(&mut self, vector: Value) -> Result<Value> {
        self.emit(Opcode::PackDouble2x32, &[vector], 0)
    }

    pub fn unpack_double2x32(&mut self, value: Value) -> Result<Value> {
        self.emit(Opcode::UnpackDouble2x32, &[value], 0)
    }

    pub fn fp_mul(&mut self, a: Value, b: Value, control: FpControl) -> Result<Value> {
        self.same_type(&[a, b])?;
        match self.ir.value_type(a)? {
            Type::F16 => self.emit(Opcode::FPMul16, &[a, b], control.bits()),
            Type::F32 => self.emit(Opcode::FPMul32, &[a, b], control.bits()),
            Type::F64 => self.emit(Opcode::FPMul64, &[a, b], control.bits()),
            _ => Err(Self::invalid("Unsupported operand type")),
        }
    }

    pub fn fp_fma(&mut self, a: Value, b: Value, c: Value, control: FpControl) -> Result<Value> {
        self.same_type(&[a, b, c])?;
        match self.ir.value_type(a)? {
            Type::F16 => self.emit(Opcode::FPFma16, &[a, b, c], control.bits()),
            Type::F32 => self.emit(Opcode::FPFma32, &[a, b, c], control.bits()),
            Type::F64 => self.emit(Opcode::FPFma64, &[a, b, c], control.bits()),
            _ => Err(Self::invalid("Unsupported operand type")),
        }
    }

    pub fn fp_abs(&mut self, value: Value) -> Result<Value> {
        match self.ir.value_type(value)? {
            Type::F16 => self.emit(Opcode::FPAbs16, &[value], 0),
            Type::F32 => self.emit(Opcode::FPAbs32, &[value], 0),
            Type::F64 => self.emit(Opcode::FPAbs64, &[value], 0),
            _ => Err(Self::invalid("Unsupported operand type")),
        }
    }

    pub fn fp_neg(&mut self, value: Value) -> Result<Value> {
        match self.ir.value_type(value)? {
            Type::F16 => self.emit(Opcode::FPNeg16, &[value], 0),
            Type::F32 => self.emit(Opcode::FPNeg32, &[value], 0),
            Type::F64 => self.emit(Opcode::FPNeg64, &[value], 0),
            _ => Err(Self::invalid("Unsupported operand type")),
        }
    }

    pub fn fp_cos(&mut self, value: Value) -> Result<Value> {
        self.emit(Opcode::FPCos, &[value], 0)
    }

    pub fn fp_sin(&mut self, value: Value) -> Result<Value> {
        self.emit(Opcode::FPSin, &[value], 0)
    }

    pub fn fp_exp2(&mut self, value: Value) -> Result<Value> {
        self.emit(Opcode::FPExp2, &[value], 0)
    }

    pub fn fp_log2(&mut self, value: Value) -> Result<Value> {
        self.emit(Opcode::FPLog2, &[value], 0)
    }

    pub fn fp_recip(&mut self, value: Value) -> Result<Value> {
        match self.ir.value_type(value)? {
            Type::F32 => self.emit(Opcode::FPRecip32, &[value], 0),
            Type::F64 => self.emit(Opcode::FPRecip64, &[value], 0),
            _ => Err(Self::invalid("Unsupported operand type")),
        }
    }

    pub fn fp_recip_sqrt(&mut self, value: Value) -> Result<Value> {
        match self.ir.value_type(value)? {
            Type::F32 => self.emit(Opcode::FPRecipSqrt32, &[value], 0),
            Type::F64 => self.emit(Opcode::FPRecipSqrt64, &[value], 0),
            _ => Err(Self::invalid("Unsupported operand type")),
        }
    }

    pub fn fp_sqrt(&mut self, value: Value) -> Result<Value> {
        self.emit(Opcode::FPSqrt, &[value], 0)
    }

    pub fn fp_saturate(&mut self, value: Value) -> Result<Value> {
        match self.ir.value_type(value)? {
            Type::F16 => self.emit(Opcode::FPSaturate16, &[value], 0),
            Type::F32 => self.emit(Opcode::FPSaturate32, &[value], 0),
            Type::F64 => self.emit(Opcode::FPSaturate64, &[value], 0),
            _ => Err(Self::invalid("Unsupported operand type")),
        }
    }

    pub fn fp_clamp(&mut self, value: Value, min_value: Value, max_value: Value) -> Result<Value> {
        self.same_type(&[value, min_value, max_value])?;
        match self.ir.value_type(value)? {
            Type::F16 => self.emit(Opcode::FPClamp16, &[value, min_value, max_value], 0),
            Type::F32 => self.emit(Opcode::FPClamp32, &[value, min_value, max_value], 0),
            Type::F64 => self.emit(Opcode::FPClamp64, &[value, min_value, max_value], 0),
            _ => Err(Self::invalid("Unsupported operand type")),
        }
    }

    pub fn fp_round_even(&mut self, value: Value, control: FpControl) -> Result<Value> {
        match self.ir.value_type(value)? {
            Type::F16 => self.emit(Opcode::FPRoundEven16, &[value], control.bits()),
            Type::F32 => self.emit(Opcode::FPRoundEven32, &[value], control.bits()),
            Type::F64 => self.emit(Opcode::FPRoundEven64, &[value], control.bits()),
            _ => Err(Self::invalid("Unsupported operand type")),
        }
    }

    pub fn fp_floor(&mut self, value: Value, control: FpControl) -> Result<Value> {
        match self.ir.value_type(value)? {
            Type::F16 => self.emit(Opcode::FPFloor16, &[value], control.bits()),
            Type::F32 => self.emit(Opcode::FPFloor32, &[value], control.bits()),
            Type::F64 => self.emit(Opcode::FPFloor64, &[value], control.bits()),
            _ => Err(Self::invalid("Unsupported operand type")),
        }
    }

    pub fn fp_ceil(&mut self, value: Value, control: FpControl) -> Result<Value> {
        match self.ir.value_type(value)? {
            Type::F16 => self.emit(Opcode::FPCeil16, &[value], control.bits()),
            Type::F32 => self.emit(Opcode::FPCeil32, &[value], control.bits()),
            Type::F64 => self.emit(Opcode::FPCeil64, &[value], control.bits()),
            _ => Err(Self::invalid("Unsupported operand type")),
        }
    }

    pub fn fp_trunc(&mut self, value: Value, control: FpControl) -> Result<Value> {
        match self.ir.value_type(value)? {
            Type::F16 => self.emit(Opcode::FPTrunc16, &[value], control.bits()),
            Type::F32 => self.emit(Opcode::FPTrunc32, &[value], control.bits()),
            Type::F64 => self.emit(Opcode::FPTrunc64, &[value], control.bits()),
            _ => Err(Self::invalid("Unsupported operand type")),
        }
    }

    pub fn fp_equal(
        &mut self,
        lhs: Value,
        rhs: Value,
        control: FpControl,
        ordered: bool,
    ) -> Result<Value> {
        self.same_type(&[lhs, rhs])?;
        match self.ir.value_type(lhs)? {
            Type::F16 => self.emit(
                if ordered {
                    Opcode::FPOrdEqual16
                } else {
                    Opcode::FPUnordEqual16
                },
                &[lhs, rhs],
                control.bits(),
            ),
            Type::F32 => self.emit(
                if ordered {
                    Opcode::FPOrdEqual32
                } else {
                    Opcode::FPUnordEqual32
                },
                &[lhs, rhs],
                control.bits(),
            ),
            Type::F64 => self.emit(
                if ordered {
                    Opcode::FPOrdEqual64
                } else {
                    Opcode::FPUnordEqual64
                },
                &[lhs, rhs],
                control.bits(),
            ),
            _ => Err(Self::invalid("Unsupported operand type")),
        }
    }

    pub fn fp_not_equal(
        &mut self,
        lhs: Value,
        rhs: Value,
        control: FpControl,
        ordered: bool,
    ) -> Result<Value> {
        self.same_type(&[lhs, rhs])?;
        match self.ir.value_type(lhs)? {
            Type::F16 => self.emit(
                if ordered {
                    Opcode::FPOrdNotEqual16
                } else {
                    Opcode::FPUnordNotEqual16
                },
                &[lhs, rhs],
                control.bits(),
            ),
            Type::F32 => self.emit(
                if ordered {
                    Opcode::FPOrdNotEqual32
                } else {
                    Opcode::FPUnordNotEqual32
                },
                &[lhs, rhs],
                control.bits(),
            ),
            Type::F64 => self.emit(
                if ordered {
                    Opcode::FPOrdNotEqual64
                } else {
                    Opcode::FPUnordNotEqual64
                },
                &[lhs, rhs],
                control.bits(),
            ),
            _ => Err(Self::invalid("Unsupported operand type")),
        }
    }

    pub fn fp_less_than(
        &mut self,
        lhs: Value,
        rhs: Value,
        control: FpControl,
        ordered: bool,
    ) -> Result<Value> {
        self.same_type(&[lhs, rhs])?;
        match self.ir.value_type(lhs)? {
            Type::F16 => self.emit(
                if ordered {
                    Opcode::FPOrdLessThan16
                } else {
                    Opcode::FPUnordLessThan16
                },
                &[lhs, rhs],
                control.bits(),
            ),
            Type::F32 => self.emit(
                if ordered {
                    Opcode::FPOrdLessThan32
                } else {
                    Opcode::FPUnordLessThan32
                },
                &[lhs, rhs],
                control.bits(),
            ),
            Type::F64 => self.emit(
                if ordered {
                    Opcode::FPOrdLessThan64
                } else {
                    Opcode::FPUnordLessThan64
                },
                &[lhs, rhs],
                control.bits(),
            ),
            _ => Err(Self::invalid("Unsupported operand type")),
        }
    }

    pub fn fp_greater_than(
        &mut self,
        lhs: Value,
        rhs: Value,
        control: FpControl,
        ordered: bool,
    ) -> Result<Value> {
        self.same_type(&[lhs, rhs])?;
        match self.ir.value_type(lhs)? {
            Type::F16 => self.emit(
                if ordered {
                    Opcode::FPOrdGreaterThan16
                } else {
                    Opcode::FPUnordGreaterThan16
                },
                &[lhs, rhs],
                control.bits(),
            ),
            Type::F32 => self.emit(
                if ordered {
                    Opcode::FPOrdGreaterThan32
                } else {
                    Opcode::FPUnordGreaterThan32
                },
                &[lhs, rhs],
                control.bits(),
            ),
            Type::F64 => self.emit(
                if ordered {
                    Opcode::FPOrdGreaterThan64
                } else {
                    Opcode::FPUnordGreaterThan64
                },
                &[lhs, rhs],
                control.bits(),
            ),
            _ => Err(Self::invalid("Unsupported operand type")),
        }
    }

    pub fn fp_less_than_equal(
        &mut self,
        lhs: Value,
        rhs: Value,
        control: FpControl,
        ordered: bool,
    ) -> Result<Value> {
        self.same_type(&[lhs, rhs])?;
        match self.ir.value_type(lhs)? {
            Type::F16 => self.emit(
                if ordered {
                    Opcode::FPOrdLessThanEqual16
                } else {
                    Opcode::FPUnordLessThanEqual16
                },
                &[lhs, rhs],
                control.bits(),
            ),
            Type::F32 => self.emit(
                if ordered {
                    Opcode::FPOrdLessThanEqual32
                } else {
                    Opcode::FPUnordLessThanEqual32
                },
                &[lhs, rhs],
                control.bits(),
            ),
            Type::F64 => self.emit(
                if ordered {
                    Opcode::FPOrdLessThanEqual64
                } else {
                    Opcode::FPUnordLessThanEqual64
                },
                &[lhs, rhs],
                control.bits(),
            ),
            _ => Err(Self::invalid("Unsupported operand type")),
        }
    }

    pub fn fp_greater_than_equal(
        &mut self,
        lhs: Value,
        rhs: Value,
        control: FpControl,
        ordered: bool,
    ) -> Result<Value> {
        self.same_type(&[lhs, rhs])?;
        match self.ir.value_type(lhs)? {
            Type::F16 => self.emit(
                if ordered {
                    Opcode::FPOrdGreaterThanEqual16
                } else {
                    Opcode::FPUnordGreaterThanEqual16
                },
                &[lhs, rhs],
                control.bits(),
            ),
            Type::F32 => self.emit(
                if ordered {
                    Opcode::FPOrdGreaterThanEqual32
                } else {
                    Opcode::FPUnordGreaterThanEqual32
                },
                &[lhs, rhs],
                control.bits(),
            ),
            Type::F64 => self.emit(
                if ordered {
                    Opcode::FPOrdGreaterThanEqual64
                } else {
                    Opcode::FPUnordGreaterThanEqual64
                },
                &[lhs, rhs],
                control.bits(),
            ),
            _ => Err(Self::invalid("Unsupported operand type")),
        }
    }

    pub fn fp_is_nan(&mut self, value: Value) -> Result<Value> {
        match self.ir.value_type(value)? {
            Type::F16 => self.emit(Opcode::FPIsNan16, &[value], 0),
            Type::F32 => self.emit(Opcode::FPIsNan32, &[value], 0),
            Type::F64 => self.emit(Opcode::FPIsNan64, &[value], 0),
            _ => Err(Self::invalid("Unsupported operand type")),
        }
    }

    pub fn fp_max(&mut self, lhs: Value, rhs: Value, control: FpControl) -> Result<Value> {
        self.same_type(&[lhs, rhs])?;
        match self.ir.value_type(lhs)? {
            Type::F32 => self.emit(Opcode::FPMax32, &[lhs, rhs], control.bits()),
            Type::F64 => self.emit(Opcode::FPMax64, &[lhs, rhs], control.bits()),
            _ => Err(Self::invalid("Unsupported operand type")),
        }
    }

    pub fn fp_min(&mut self, lhs: Value, rhs: Value, control: FpControl) -> Result<Value> {
        self.same_type(&[lhs, rhs])?;
        match self.ir.value_type(lhs)? {
            Type::F32 => self.emit(Opcode::FPMin32, &[lhs, rhs], control.bits()),
            Type::F64 => self.emit(Opcode::FPMin64, &[lhs, rhs], control.bits()),
            _ => Err(Self::invalid("Unsupported operand type")),
        }
    }

    pub fn i_add(&mut self, a: Value, b: Value) -> Result<Value> {
        self.same_type(&[a, b])?;
        match self.ir.value_type(a)? {
            Type::U32 => self.emit(Opcode::IAdd32, &[a, b], 0),
            Type::U64 => self.emit(Opcode::IAdd64, &[a, b], 0),
            _ => Err(Self::invalid("Unsupported operand type")),
        }
    }

    pub fn i_sub(&mut self, a: Value, b: Value) -> Result<Value> {
        self.same_type(&[a, b])?;
        match self.ir.value_type(a)? {
            Type::U32 => self.emit(Opcode::ISub32, &[a, b], 0),
            Type::U64 => self.emit(Opcode::ISub64, &[a, b], 0),
            _ => Err(Self::invalid("Unsupported operand type")),
        }
    }

    pub fn i_mul(&mut self, a: Value, b: Value) -> Result<Value> {
        self.emit(Opcode::IMul32, &[a, b], 0)
    }

    pub fn i_div(&mut self, a: Value, b: Value, is_signed: bool) -> Result<Value> {
        self.emit(
            if is_signed {
                Opcode::SDiv32
            } else {
                Opcode::UDiv32
            },
            &[a, b],
            0,
        )
    }

    pub fn i_neg(&mut self, value: Value) -> Result<Value> {
        match self.ir.value_type(value)? {
            Type::U32 => self.emit(Opcode::INeg32, &[value], 0),
            Type::U64 => self.emit(Opcode::INeg64, &[value], 0),
            _ => Err(Self::invalid("Unsupported operand type")),
        }
    }

    pub fn i_abs(&mut self, value: Value) -> Result<Value> {
        match self.ir.value_type(value)? {
            Type::U32 => self.emit(Opcode::IAbs32, &[value], 0),
            Type::U64 => self.emit(Opcode::IAbs64, &[value], 0),
            _ => Err(Self::invalid("Unsupported operand type")),
        }
    }

    pub fn shift_left_logical(&mut self, base: Value, shift: Value) -> Result<Value> {
        match self.ir.value_type(base)? {
            Type::U32 => self.emit(Opcode::ShiftLeftLogical32, &[base, shift], 0),
            Type::U64 => self.emit(Opcode::ShiftLeftLogical64, &[base, shift], 0),
            _ => Err(Self::invalid("Unsupported operand type")),
        }
    }

    pub fn shift_right_logical(&mut self, base: Value, shift: Value) -> Result<Value> {
        match self.ir.value_type(base)? {
            Type::U32 => self.emit(Opcode::ShiftRightLogical32, &[base, shift], 0),
            Type::U64 => self.emit(Opcode::ShiftRightLogical64, &[base, shift], 0),
            _ => Err(Self::invalid("Unsupported operand type")),
        }
    }

    pub fn shift_right_arithmetic(&mut self, base: Value, shift: Value) -> Result<Value> {
        match self.ir.value_type(base)? {
            Type::U32 => self.emit(Opcode::ShiftRightArithmetic32, &[base, shift], 0),
            Type::U64 => self.emit(Opcode::ShiftRightArithmetic64, &[base, shift], 0),
            _ => Err(Self::invalid("Unsupported operand type")),
        }
    }

    pub fn bitwise_and(&mut self, a: Value, b: Value) -> Result<Value> {
        self.emit(Opcode::BitwiseAnd32, &[a, b], 0)
    }

    pub fn bitwise_or(&mut self, a: Value, b: Value) -> Result<Value> {
        self.emit(Opcode::BitwiseOr32, &[a, b], 0)
    }

    pub fn bitwise_xor(&mut self, a: Value, b: Value) -> Result<Value> {
        self.emit(Opcode::BitwiseXor32, &[a, b], 0)
    }

    pub fn bit_field_insert(
        &mut self,
        base: Value,
        insert: Value,
        offset: Value,
        count: Value,
    ) -> Result<Value> {
        self.emit(Opcode::BitFieldInsert, &[base, insert, offset, count], 0)
    }

    pub fn bit_field_extract(
        &mut self,
        base: Value,
        offset: Value,
        count: Value,
        is_signed: bool,
    ) -> Result<Value> {
        self.emit(
            if is_signed {
                Opcode::BitFieldSExtract
            } else {
                Opcode::BitFieldUExtract
            },
            &[base, offset, count],
            0,
        )
    }

    pub fn bit_reverse(&mut self, value: Value) -> Result<Value> {
        self.emit(Opcode::BitReverse32, &[value], 0)
    }

    pub fn bit_count(&mut self, value: Value) -> Result<Value> {
        self.emit(Opcode::BitCount32, &[value], 0)
    }

    pub fn bitwise_not(&mut self, value: Value) -> Result<Value> {
        self.emit(Opcode::BitwiseNot32, &[value], 0)
    }

    pub fn find_s_msb(&mut self, value: Value) -> Result<Value> {
        self.emit(Opcode::FindSMsb32, &[value], 0)
    }

    pub fn find_u_msb(&mut self, value: Value) -> Result<Value> {
        self.emit(Opcode::FindUMsb32, &[value], 0)
    }

    pub fn s_min(&mut self, a: Value, b: Value) -> Result<Value> {
        self.emit(Opcode::SMin32, &[a, b], 0)
    }

    pub fn u_min(&mut self, a: Value, b: Value) -> Result<Value> {
        self.emit(Opcode::UMin32, &[a, b], 0)
    }

    pub fn i_min(&mut self, a: Value, b: Value, is_signed: bool) -> Result<Value> {
        if is_signed {
            self.s_min(a, b)
        } else {
            self.u_min(a, b)
        }
    }

    pub fn s_max(&mut self, a: Value, b: Value) -> Result<Value> {
        self.emit(Opcode::SMax32, &[a, b], 0)
    }

    pub fn u_max(&mut self, a: Value, b: Value) -> Result<Value> {
        self.emit(Opcode::UMax32, &[a, b], 0)
    }

    pub fn i_max(&mut self, a: Value, b: Value, is_signed: bool) -> Result<Value> {
        if is_signed {
            self.s_max(a, b)
        } else {
            self.u_max(a, b)
        }
    }

    pub fn s_clamp(&mut self, value: Value, min: Value, max: Value) -> Result<Value> {
        self.emit(Opcode::SClamp32, &[value, min, max], 0)
    }

    pub fn u_clamp(&mut self, value: Value, min: Value, max: Value) -> Result<Value> {
        self.emit(Opcode::UClamp32, &[value, min, max], 0)
    }

    pub fn i_less_than(&mut self, lhs: Value, rhs: Value, is_signed: bool) -> Result<Value> {
        self.emit(
            if is_signed {
                Opcode::SLessThan
            } else {
                Opcode::ULessThan
            },
            &[lhs, rhs],
            0,
        )
    }

    pub fn i_less_than_equal(&mut self, lhs: Value, rhs: Value, is_signed: bool) -> Result<Value> {
        self.emit(
            if is_signed {
                Opcode::SLessThanEqual
            } else {
                Opcode::ULessThanEqual
            },
            &[lhs, rhs],
            0,
        )
    }

    pub fn i_greater_than(&mut self, lhs: Value, rhs: Value, is_signed: bool) -> Result<Value> {
        self.emit(
            if is_signed {
                Opcode::SGreaterThan
            } else {
                Opcode::UGreaterThan
            },
            &[lhs, rhs],
            0,
        )
    }

    pub fn i_not_equal(&mut self, lhs: Value, rhs: Value) -> Result<Value> {
        self.emit(Opcode::INotEqual, &[lhs, rhs], 0)
    }

    pub fn i_greater_than_equal(
        &mut self,
        lhs: Value,
        rhs: Value,
        is_signed: bool,
    ) -> Result<Value> {
        self.emit(
            if is_signed {
                Opcode::SGreaterThanEqual
            } else {
                Opcode::UGreaterThanEqual
            },
            &[lhs, rhs],
            0,
        )
    }

    pub fn shared_atomic_i_add(&mut self, pointer_offset: Value, value: Value) -> Result<Value> {
        self.emit(Opcode::SharedAtomicIAdd32, &[pointer_offset, value], 0)
    }

    pub fn shared_atomic_s_min(&mut self, pointer_offset: Value, value: Value) -> Result<Value> {
        self.emit(Opcode::SharedAtomicSMin32, &[pointer_offset, value], 0)
    }

    pub fn shared_atomic_u_min(&mut self, pointer_offset: Value, value: Value) -> Result<Value> {
        self.emit(Opcode::SharedAtomicUMin32, &[pointer_offset, value], 0)
    }

    pub fn shared_atomic_i_min(
        &mut self,
        pointer_offset: Value,
        value: Value,
        is_signed: bool,
    ) -> Result<Value> {
        if is_signed {
            self.shared_atomic_s_min(pointer_offset, value)
        } else {
            self.shared_atomic_u_min(pointer_offset, value)
        }
    }

    pub fn shared_atomic_s_max(&mut self, pointer_offset: Value, value: Value) -> Result<Value> {
        self.emit(Opcode::SharedAtomicSMax32, &[pointer_offset, value], 0)
    }

    pub fn shared_atomic_u_max(&mut self, pointer_offset: Value, value: Value) -> Result<Value> {
        self.emit(Opcode::SharedAtomicUMax32, &[pointer_offset, value], 0)
    }

    pub fn shared_atomic_i_max(
        &mut self,
        pointer_offset: Value,
        value: Value,
        is_signed: bool,
    ) -> Result<Value> {
        if is_signed {
            self.shared_atomic_s_max(pointer_offset, value)
        } else {
            self.shared_atomic_u_max(pointer_offset, value)
        }
    }

    pub fn shared_atomic_inc(&mut self, pointer_offset: Value, value: Value) -> Result<Value> {
        self.emit(Opcode::SharedAtomicInc32, &[pointer_offset, value], 0)
    }

    pub fn shared_atomic_dec(&mut self, pointer_offset: Value, value: Value) -> Result<Value> {
        self.emit(Opcode::SharedAtomicDec32, &[pointer_offset, value], 0)
    }

    pub fn shared_atomic_and(&mut self, pointer_offset: Value, value: Value) -> Result<Value> {
        self.emit(Opcode::SharedAtomicAnd32, &[pointer_offset, value], 0)
    }

    pub fn shared_atomic_or(&mut self, pointer_offset: Value, value: Value) -> Result<Value> {
        self.emit(Opcode::SharedAtomicOr32, &[pointer_offset, value], 0)
    }

    pub fn shared_atomic_xor(&mut self, pointer_offset: Value, value: Value) -> Result<Value> {
        self.emit(Opcode::SharedAtomicXor32, &[pointer_offset, value], 0)
    }

    pub fn shared_atomic_exchange(&mut self, pointer_offset: Value, value: Value) -> Result<Value> {
        match self.ir.value_type(value)? {
            Type::U32 => self.emit(Opcode::SharedAtomicExchange32, &[pointer_offset, value], 0),
            Type::U64 => self.emit(Opcode::SharedAtomicExchange64, &[pointer_offset, value], 0),
            _ => Err(Self::invalid("Unsupported operand type")),
        }
    }

    pub fn global_atomic_i_add(&mut self, pointer_offset: Value, value: Value) -> Result<Value> {
        self.require_type(pointer_offset, Type::U64)?;
        match self.ir.value_type(value)? {
            Type::U32 => self.emit(Opcode::GlobalAtomicIAdd32, &[pointer_offset, value], 0),
            Type::U64 => self.emit(Opcode::GlobalAtomicIAdd64, &[pointer_offset, value], 0),
            _ => Err(Self::invalid("Unsupported operand type")),
        }
    }

    pub fn global_atomic_s_min(&mut self, pointer_offset: Value, value: Value) -> Result<Value> {
        self.require_type(pointer_offset, Type::U64)?;
        match self.ir.value_type(value)? {
            Type::U32 => self.emit(Opcode::GlobalAtomicSMin32, &[pointer_offset, value], 0),
            Type::U64 => self.emit(Opcode::GlobalAtomicSMin64, &[pointer_offset, value], 0),
            _ => Err(Self::invalid("Unsupported operand type")),
        }
    }

    pub fn global_atomic_u_min(&mut self, pointer_offset: Value, value: Value) -> Result<Value> {
        self.require_type(pointer_offset, Type::U64)?;
        match self.ir.value_type(value)? {
            Type::U32 => self.emit(Opcode::GlobalAtomicUMin32, &[pointer_offset, value], 0),
            Type::U64 => self.emit(Opcode::GlobalAtomicUMin64, &[pointer_offset, value], 0),
            _ => Err(Self::invalid("Unsupported operand type")),
        }
    }

    pub fn global_atomic_i_min(
        &mut self,
        pointer_offset: Value,
        value: Value,
        is_signed: bool,
    ) -> Result<Value> {
        self.require_type(pointer_offset, Type::U64)?;
        if is_signed {
            self.global_atomic_s_min(pointer_offset, value)
        } else {
            self.global_atomic_u_min(pointer_offset, value)
        }
    }

    pub fn global_atomic_s_max(&mut self, pointer_offset: Value, value: Value) -> Result<Value> {
        self.require_type(pointer_offset, Type::U64)?;
        match self.ir.value_type(value)? {
            Type::U32 => self.emit(Opcode::GlobalAtomicSMax32, &[pointer_offset, value], 0),
            Type::U64 => self.emit(Opcode::GlobalAtomicSMax64, &[pointer_offset, value], 0),
            _ => Err(Self::invalid("Unsupported operand type")),
        }
    }

    pub fn global_atomic_u_max(&mut self, pointer_offset: Value, value: Value) -> Result<Value> {
        self.require_type(pointer_offset, Type::U64)?;
        match self.ir.value_type(value)? {
            Type::U32 => self.emit(Opcode::GlobalAtomicUMax32, &[pointer_offset, value], 0),
            Type::U64 => self.emit(Opcode::GlobalAtomicUMax64, &[pointer_offset, value], 0),
            _ => Err(Self::invalid("Unsupported operand type")),
        }
    }

    pub fn global_atomic_i_max(
        &mut self,
        pointer_offset: Value,
        value: Value,
        is_signed: bool,
    ) -> Result<Value> {
        self.require_type(pointer_offset, Type::U64)?;
        if is_signed {
            self.global_atomic_s_max(pointer_offset, value)
        } else {
            self.global_atomic_u_max(pointer_offset, value)
        }
    }

    pub fn global_atomic_inc(&mut self, pointer_offset: Value, value: Value) -> Result<Value> {
        self.require_type(pointer_offset, Type::U64)?;
        self.emit(Opcode::GlobalAtomicInc32, &[pointer_offset, value], 0)
    }

    pub fn global_atomic_dec(&mut self, pointer_offset: Value, value: Value) -> Result<Value> {
        self.require_type(pointer_offset, Type::U64)?;
        self.emit(Opcode::GlobalAtomicDec32, &[pointer_offset, value], 0)
    }

    pub fn global_atomic_and(&mut self, pointer_offset: Value, value: Value) -> Result<Value> {
        self.require_type(pointer_offset, Type::U64)?;
        match self.ir.value_type(value)? {
            Type::U32 => self.emit(Opcode::GlobalAtomicAnd32, &[pointer_offset, value], 0),
            Type::U64 => self.emit(Opcode::GlobalAtomicAnd64, &[pointer_offset, value], 0),
            _ => Err(Self::invalid("Unsupported operand type")),
        }
    }

    pub fn global_atomic_or(&mut self, pointer_offset: Value, value: Value) -> Result<Value> {
        self.require_type(pointer_offset, Type::U64)?;
        match self.ir.value_type(value)? {
            Type::U32 => self.emit(Opcode::GlobalAtomicOr32, &[pointer_offset, value], 0),
            Type::U64 => self.emit(Opcode::GlobalAtomicOr64, &[pointer_offset, value], 0),
            _ => Err(Self::invalid("Unsupported operand type")),
        }
    }

    pub fn global_atomic_xor(&mut self, pointer_offset: Value, value: Value) -> Result<Value> {
        self.require_type(pointer_offset, Type::U64)?;
        match self.ir.value_type(value)? {
            Type::U32 => self.emit(Opcode::GlobalAtomicXor32, &[pointer_offset, value], 0),
            Type::U64 => self.emit(Opcode::GlobalAtomicXor64, &[pointer_offset, value], 0),
            _ => Err(Self::invalid("Unsupported operand type")),
        }
    }

    pub fn global_atomic_exchange(&mut self, pointer_offset: Value, value: Value) -> Result<Value> {
        self.require_type(pointer_offset, Type::U64)?;
        match self.ir.value_type(value)? {
            Type::U32 => self.emit(Opcode::GlobalAtomicExchange32, &[pointer_offset, value], 0),
            Type::U64 => self.emit(Opcode::GlobalAtomicExchange64, &[pointer_offset, value], 0),
            _ => Err(Self::invalid("Unsupported operand type")),
        }
    }

    pub fn global_atomic_f32_add(
        &mut self,
        pointer_offset: Value,
        value: Value,
        control: FpControl,
    ) -> Result<Value> {
        self.require_type(pointer_offset, Type::U64)?;
        self.emit(
            Opcode::GlobalAtomicAddF32,
            &[pointer_offset, value],
            control.bits(),
        )
    }

    pub fn global_atomic_f16x2_add(
        &mut self,
        pointer_offset: Value,
        value: Value,
        control: FpControl,
    ) -> Result<Value> {
        self.require_type(pointer_offset, Type::U64)?;
        self.emit(
            Opcode::GlobalAtomicAddF16x2,
            &[pointer_offset, value],
            control.bits(),
        )
    }

    pub fn global_atomic_f16x2_min(
        &mut self,
        pointer_offset: Value,
        value: Value,
        control: FpControl,
    ) -> Result<Value> {
        self.require_type(pointer_offset, Type::U64)?;
        self.emit(
            Opcode::GlobalAtomicMinF16x2,
            &[pointer_offset, value],
            control.bits(),
        )
    }

    pub fn global_atomic_f16x2_max(
        &mut self,
        pointer_offset: Value,
        value: Value,
        control: FpControl,
    ) -> Result<Value> {
        self.require_type(pointer_offset, Type::U64)?;
        self.emit(
            Opcode::GlobalAtomicMaxF16x2,
            &[pointer_offset, value],
            control.bits(),
        )
    }

    pub fn logical_or(&mut self, a: Value, b: Value) -> Result<Value> {
        self.emit(Opcode::LogicalOr, &[a, b], 0)
    }

    pub fn logical_and(&mut self, a: Value, b: Value) -> Result<Value> {
        self.emit(Opcode::LogicalAnd, &[a, b], 0)
    }

    pub fn logical_xor(&mut self, a: Value, b: Value) -> Result<Value> {
        self.emit(Opcode::LogicalXor, &[a, b], 0)
    }

    pub fn logical_not(&mut self, value: Value) -> Result<Value> {
        self.emit(Opcode::LogicalNot, &[value], 0)
    }

    pub fn image_sample_implicit_lod(
        &mut self,
        handle: Value,
        coords: Value,
        bias: Value,
        offset: Value,
        lod_clamp: Value,
        info: TextureInstInfo,
    ) -> Result<Value> {
        let bias_lc = self.make_lod_clamp_pair(bias, lod_clamp)?;
        let op = if self.ir.is_immediate(handle)? {
            Opcode::BoundImageSampleImplicitLod
        } else {
            Opcode::BindlessImageSampleImplicitLod
        };
        self.emit(op, &[handle, coords, bias_lc, offset], info.0)
    }

    pub fn image_sample_explicit_lod(
        &mut self,
        handle: Value,
        coords: Value,
        lod: Value,
        offset: Value,
        info: TextureInstInfo,
    ) -> Result<Value> {
        let op = if self.ir.is_immediate(handle)? {
            Opcode::BoundImageSampleExplicitLod
        } else {
            Opcode::BindlessImageSampleExplicitLod
        };
        self.emit(op, &[handle, coords, lod, offset], info.0)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn image_sample_dref_implicit_lod(
        &mut self,
        handle: Value,
        coords: Value,
        dref: Value,
        bias: Value,
        offset: Value,
        lod_clamp: Value,
        info: TextureInstInfo,
    ) -> Result<Value> {
        let bias_lc = self.make_lod_clamp_pair(bias, lod_clamp)?;
        let op = if self.ir.is_immediate(handle)? {
            Opcode::BoundImageSampleDrefImplicitLod
        } else {
            Opcode::BindlessImageSampleDrefImplicitLod
        };
        self.emit(op, &[handle, coords, dref, bias_lc, offset], info.0)
    }

    pub fn image_sample_dref_explicit_lod(
        &mut self,
        handle: Value,
        coords: Value,
        dref: Value,
        lod: Value,
        offset: Value,
        info: TextureInstInfo,
    ) -> Result<Value> {
        let op = if self.ir.is_immediate(handle)? {
            Opcode::BoundImageSampleDrefExplicitLod
        } else {
            Opcode::BindlessImageSampleDrefExplicitLod
        };
        self.emit(op, &[handle, coords, dref, lod, offset], info.0)
    }

    pub fn image_gather(
        &mut self,
        handle: Value,
        coords: Value,
        offset: Value,
        offset2: Value,
        info: TextureInstInfo,
    ) -> Result<Value> {
        let op = if self.ir.is_immediate(handle)? {
            Opcode::BoundImageGather
        } else {
            Opcode::BindlessImageGather
        };
        self.emit(op, &[handle, coords, offset, offset2], info.0)
    }

    pub fn image_gather_dref(
        &mut self,
        handle: Value,
        coords: Value,
        offset: Value,
        offset2: Value,
        dref: Value,
        info: TextureInstInfo,
    ) -> Result<Value> {
        let op = if self.ir.is_immediate(handle)? {
            Opcode::BoundImageGatherDref
        } else {
            Opcode::BindlessImageGatherDref
        };
        self.emit(op, &[handle, coords, offset, offset2, dref], info.0)
    }

    pub fn image_fetch(
        &mut self,
        handle: Value,
        coords: Value,
        offset: Value,
        lod: Value,
        multisampling: Value,
        info: TextureInstInfo,
    ) -> Result<Value> {
        let op = if self.ir.is_immediate(handle)? {
            Opcode::BoundImageFetch
        } else {
            Opcode::BindlessImageFetch
        };
        self.emit(op, &[handle, coords, offset, lod, multisampling], info.0)
    }

    pub fn image_query_dimension(
        &mut self,
        handle: Value,
        lod: Value,
        skip_mips: Value,
    ) -> Result<Value> {
        let op = if self.ir.is_immediate(handle)? {
            Opcode::BoundImageQueryDimensions
        } else {
            Opcode::BindlessImageQueryDimensions
        };
        self.emit(op, &[handle, lod, skip_mips], 0)
    }

    pub fn image_query_dimension_indexed(
        &mut self,
        handle: Value,
        lod: Value,
        skip_mips: Value,
        info: TextureInstInfo,
    ) -> Result<Value> {
        self.emit(
            Opcode::ImageQueryDimensions,
            &[handle, lod, skip_mips],
            info.0,
        )
    }

    pub fn image_query_lod(
        &mut self,
        handle: Value,
        coords: Value,
        info: TextureInstInfo,
    ) -> Result<Value> {
        let op = if self.ir.is_immediate(handle)? {
            Opcode::BoundImageQueryLod
        } else {
            Opcode::BindlessImageQueryLod
        };
        self.emit(op, &[handle, coords], info.0)
    }

    pub fn image_gradient(
        &mut self,
        handle: Value,
        coords: Value,
        derivatives: Value,
        offset: Value,
        lod_clamp: Value,
        info: TextureInstInfo,
    ) -> Result<Value> {
        let op = if self.ir.is_immediate(handle)? {
            Opcode::BoundImageGradient
        } else {
            Opcode::BindlessImageGradient
        };
        self.emit(
            op,
            &[handle, coords, derivatives, offset, lod_clamp],
            info.0,
        )
    }

    pub fn image_read(
        &mut self,
        handle: Value,
        coords: Value,
        info: TextureInstInfo,
    ) -> Result<Value> {
        let op = if self.ir.is_immediate(handle)? {
            Opcode::BoundImageRead
        } else {
            Opcode::BindlessImageRead
        };
        self.emit(op, &[handle, coords], info.0)
    }

    pub fn image_write(
        &mut self,
        handle: Value,
        coords: Value,
        color: Value,
        info: TextureInstInfo,
    ) -> Result<Value> {
        let op = if self.ir.is_immediate(handle)? {
            Opcode::BoundImageWrite
        } else {
            Opcode::BindlessImageWrite
        };
        self.emit(op, &[handle, coords, color], info.0)
    }

    pub fn image_atomic_i_add(
        &mut self,
        handle: Value,
        coords: Value,
        value: Value,
        info: TextureInstInfo,
    ) -> Result<Value> {
        let op = if self.ir.is_immediate(handle)? {
            Opcode::BoundImageAtomicIAdd32
        } else {
            Opcode::BindlessImageAtomicIAdd32
        };
        self.emit(op, &[handle, coords, value], info.0)
    }

    pub fn image_atomic_s_min(
        &mut self,
        handle: Value,
        coords: Value,
        value: Value,
        info: TextureInstInfo,
    ) -> Result<Value> {
        let op = if self.ir.is_immediate(handle)? {
            Opcode::BoundImageAtomicSMin32
        } else {
            Opcode::BindlessImageAtomicSMin32
        };
        self.emit(op, &[handle, coords, value], info.0)
    }

    pub fn image_atomic_u_min(
        &mut self,
        handle: Value,
        coords: Value,
        value: Value,
        info: TextureInstInfo,
    ) -> Result<Value> {
        let op = if self.ir.is_immediate(handle)? {
            Opcode::BoundImageAtomicUMin32
        } else {
            Opcode::BindlessImageAtomicUMin32
        };
        self.emit(op, &[handle, coords, value], info.0)
    }

    pub fn image_atomic_i_min(
        &mut self,
        handle: Value,
        coords: Value,
        value: Value,
        is_signed: bool,
        info: TextureInstInfo,
    ) -> Result<Value> {
        if is_signed {
            self.image_atomic_s_min(handle, coords, value, info)
        } else {
            self.image_atomic_u_min(handle, coords, value, info)
        }
    }

    pub fn image_atomic_s_max(
        &mut self,
        handle: Value,
        coords: Value,
        value: Value,
        info: TextureInstInfo,
    ) -> Result<Value> {
        let op = if self.ir.is_immediate(handle)? {
            Opcode::BoundImageAtomicSMax32
        } else {
            Opcode::BindlessImageAtomicSMax32
        };
        self.emit(op, &[handle, coords, value], info.0)
    }

    pub fn image_atomic_u_max(
        &mut self,
        handle: Value,
        coords: Value,
        value: Value,
        info: TextureInstInfo,
    ) -> Result<Value> {
        let op = if self.ir.is_immediate(handle)? {
            Opcode::BoundImageAtomicUMax32
        } else {
            Opcode::BindlessImageAtomicUMax32
        };
        self.emit(op, &[handle, coords, value], info.0)
    }

    pub fn image_atomic_i_max(
        &mut self,
        handle: Value,
        coords: Value,
        value: Value,
        is_signed: bool,
        info: TextureInstInfo,
    ) -> Result<Value> {
        if is_signed {
            self.image_atomic_s_max(handle, coords, value, info)
        } else {
            self.image_atomic_u_max(handle, coords, value, info)
        }
    }

    pub fn image_atomic_inc(
        &mut self,
        handle: Value,
        coords: Value,
        value: Value,
        info: TextureInstInfo,
    ) -> Result<Value> {
        let op = if self.ir.is_immediate(handle)? {
            Opcode::BoundImageAtomicInc32
        } else {
            Opcode::BindlessImageAtomicInc32
        };
        self.emit(op, &[handle, coords, value], info.0)
    }

    pub fn image_atomic_dec(
        &mut self,
        handle: Value,
        coords: Value,
        value: Value,
        info: TextureInstInfo,
    ) -> Result<Value> {
        let op = if self.ir.is_immediate(handle)? {
            Opcode::BoundImageAtomicDec32
        } else {
            Opcode::BindlessImageAtomicDec32
        };
        self.emit(op, &[handle, coords, value], info.0)
    }

    pub fn image_atomic_and(
        &mut self,
        handle: Value,
        coords: Value,
        value: Value,
        info: TextureInstInfo,
    ) -> Result<Value> {
        let op = if self.ir.is_immediate(handle)? {
            Opcode::BoundImageAtomicAnd32
        } else {
            Opcode::BindlessImageAtomicAnd32
        };
        self.emit(op, &[handle, coords, value], info.0)
    }

    pub fn image_atomic_or(
        &mut self,
        handle: Value,
        coords: Value,
        value: Value,
        info: TextureInstInfo,
    ) -> Result<Value> {
        let op = if self.ir.is_immediate(handle)? {
            Opcode::BoundImageAtomicOr32
        } else {
            Opcode::BindlessImageAtomicOr32
        };
        self.emit(op, &[handle, coords, value], info.0)
    }

    pub fn image_atomic_xor(
        &mut self,
        handle: Value,
        coords: Value,
        value: Value,
        info: TextureInstInfo,
    ) -> Result<Value> {
        let op = if self.ir.is_immediate(handle)? {
            Opcode::BoundImageAtomicXor32
        } else {
            Opcode::BindlessImageAtomicXor32
        };
        self.emit(op, &[handle, coords, value], info.0)
    }

    pub fn image_atomic_exchange(
        &mut self,
        handle: Value,
        coords: Value,
        value: Value,
        info: TextureInstInfo,
    ) -> Result<Value> {
        let op = if self.ir.is_immediate(handle)? {
            Opcode::BoundImageAtomicExchange32
        } else {
            Opcode::BindlessImageAtomicExchange32
        };
        self.emit(op, &[handle, coords, value], info.0)
    }

    pub fn is_texture_scaled(&mut self, index: Value) -> Result<Value> {
        self.emit(Opcode::IsTextureScaled, &[index], 0)
    }

    pub fn is_image_scaled(&mut self, index: Value) -> Result<Value> {
        self.emit(Opcode::IsImageScaled, &[index], 0)
    }

    pub fn vote_all(&mut self, value: Value) -> Result<Value> {
        self.emit(Opcode::VoteAll, &[value], 0)
    }

    pub fn vote_any(&mut self, value: Value) -> Result<Value> {
        self.emit(Opcode::VoteAny, &[value], 0)
    }

    pub fn vote_equal(&mut self, value: Value) -> Result<Value> {
        self.emit(Opcode::VoteEqual, &[value], 0)
    }

    pub fn subgroup_ballot(&mut self, value: Value) -> Result<Value> {
        self.emit(Opcode::SubgroupBallot, &[value], 0)
    }

    pub fn subgroup_eq_mask(&mut self) -> Result<Value> {
        self.emit(Opcode::SubgroupEqMask, &[], 0)
    }

    pub fn subgroup_lt_mask(&mut self) -> Result<Value> {
        self.emit(Opcode::SubgroupLtMask, &[], 0)
    }

    pub fn subgroup_le_mask(&mut self) -> Result<Value> {
        self.emit(Opcode::SubgroupLeMask, &[], 0)
    }

    pub fn subgroup_gt_mask(&mut self) -> Result<Value> {
        self.emit(Opcode::SubgroupGtMask, &[], 0)
    }

    pub fn subgroup_ge_mask(&mut self) -> Result<Value> {
        self.emit(Opcode::SubgroupGeMask, &[], 0)
    }

    pub fn shuffle_index(
        &mut self,
        value: Value,
        index: Value,
        clamp: Value,
        seg_mask: Value,
    ) -> Result<Value> {
        self.emit(Opcode::ShuffleIndex, &[value, index, clamp, seg_mask], 0)
    }

    pub fn shuffle_up(
        &mut self,
        value: Value,
        index: Value,
        clamp: Value,
        seg_mask: Value,
    ) -> Result<Value> {
        self.emit(Opcode::ShuffleUp, &[value, index, clamp, seg_mask], 0)
    }

    pub fn shuffle_down(
        &mut self,
        value: Value,
        index: Value,
        clamp: Value,
        seg_mask: Value,
    ) -> Result<Value> {
        self.emit(Opcode::ShuffleDown, &[value, index, clamp, seg_mask], 0)
    }

    pub fn shuffle_butterfly(
        &mut self,
        value: Value,
        index: Value,
        clamp: Value,
        seg_mask: Value,
    ) -> Result<Value> {
        self.emit(
            Opcode::ShuffleButterfly,
            &[value, index, clamp, seg_mask],
            0,
        )
    }

    pub fn quad_broadcast(&mut self, value: Value, lane: Value) -> Result<Value> {
        self.emit(Opcode::QuadBroadcast, &[value, lane], 0)
    }

    pub fn quad_swap(&mut self, value: Value, direction: Value) -> Result<Value> {
        self.emit(Opcode::QuadSwap, &[value, direction], 0)
    }

    pub fn f_swizzle_add(
        &mut self,
        a: Value,
        b: Value,
        swizzle: Value,
        control: FpControl,
    ) -> Result<Value> {
        self.emit(Opcode::FSwizzleAdd, &[a, b, swizzle], control.bits())
    }

    pub fn d_pdx_fine(&mut self, a: Value) -> Result<Value> {
        self.emit(Opcode::DPdxFine, &[a], 0)
    }

    pub fn d_pdy_fine(&mut self, a: Value) -> Result<Value> {
        self.emit(Opcode::DPdyFine, &[a], 0)
    }

    pub fn d_pdx_coarse(&mut self, a: Value) -> Result<Value> {
        self.emit(Opcode::DPdxCoarse, &[a], 0)
    }

    pub fn d_pdy_coarse(&mut self, a: Value) -> Result<Value> {
        self.emit(Opcode::DPdyCoarse, &[a], 0)
    }
}
