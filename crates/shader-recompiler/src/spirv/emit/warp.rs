use super::*;
use spirv::Scope;
impl EmitContext<'_> {
    fn thread_id(&mut self) -> Result<Id> {
        if self.profile.supports_subgroup_stage(self.stage) {
            self.op(
                Op::Load,
                self.types.u32[0],
                &[self.objects.subgroup_local_invocation_id],
            )
        } else {
            Ok(self.constants.u32_zero)
        }
    }
    fn guest_lane(&mut self, value: Id) -> Result<Id> {
        let mask = self.const_u32(31)?;
        self.op(Op::BitwiseAnd, self.types.u32[0], &[value, mask])
    }
    fn lane_id(&mut self) -> Result<Id> {
        let id = self.thread_id()?;
        if self.profile.warp_size_potentially_larger_than_guest {
            self.guest_lane(id)
        } else {
            Ok(id)
        }
    }
    fn partition_index(&mut self, arithmetic: bool) -> Result<Id> {
        let thread = self.thread_id()?;
        let shift = self.const_u32(5)?;
        self.op(
            if arithmetic {
                Op::ShiftRightArithmetic
            } else {
                Op::ShiftRightLogical
            },
            self.types.u32[0],
            &[thread, shift],
        )
    }
    fn host_thread(&mut self, guest: Id) -> Result<Id> {
        if !self.profile.warp_size_potentially_larger_than_guest {
            return Ok(guest);
        }
        let partition = self.partition_index(false)?;
        let shift = self.const_u32(5)?;
        let base = self.op(Op::ShiftLeftLogical, self.types.u32[0], &[partition, shift])?;
        self.op(Op::IAdd, self.types.u32[0], &[guest, base])
    }
    fn warp_extract(&mut self, vector: Id) -> Result<Id> {
        let index = self.partition_index(true)?;
        let u = self.types.u32[0];
        if !self
            .profile
            .has_broken_spirv_subgroup_mask_vector_extract_dynamic
        {
            return self.op(Op::VectorExtractDynamic, u, &[vector, index]);
        }
        let mut parts = [Id::NONE; 4];
        for (i, part) in parts.iter_mut().enumerate() {
            let lane = self.const_u32(i as u32)?;
            let matches = self.op(Op::IEqual, self.types.bool, &[index, lane])?;
            let component = self.extract(u, vector, i as u32)?;
            *part = self.op(
                Op::Select,
                u,
                &[matches, component, self.constants.u32_zero],
            )?;
        }
        let lo = self.op(Op::BitwiseOr, u, &parts[..2])?;
        let hi = self.op(Op::BitwiseOr, u, &parts[2..])?;
        self.op(Op::BitwiseOr, u, &[lo, hi])
    }
    fn shuffle_absolute(&mut self, value: Id, source: Id) -> Result<Id> {
        let scope = self.const_u32(Scope::Subgroup as u32)?;
        let u = self.types.u32[0];
        if !self.profile.has_broken_spirv_subgroup_shuffle {
            return self.op(Op::GroupNonUniformShuffle, u, &[scope, value, source]);
        }
        let mut result = self.constants.u32_zero;
        for lane in 0..self.profile.max_subgroup_size {
            let lane = self.const_u32(lane)?;
            let read = self.op(Op::GroupNonUniformBroadcast, u, &[scope, value, lane])?;
            let matches = self.op(Op::IEqual, self.types.bool, &[source, lane])?;
            result = self.op(Op::Select, u, &[matches, read, result])?;
        }
        Ok(result)
    }
    fn broadcast_lane(&mut self, value: Id, lane: u32) -> Result<Id> {
        let scope = self.const_u32(Scope::Subgroup as u32)?;
        let u = self.types.u32[0];
        let lane_id = self.const_u32(lane)?;
        let mut result = self.op(Op::GroupNonUniformBroadcast, u, &[scope, value, lane_id])?;
        if self.profile.warp_size_potentially_larger_than_guest {
            let partition = self.partition_index(false)?;
            for base in (32..self.profile.max_subgroup_size).step_by(32) {
                let index = self.const_u32(base + lane)?;
                let read = self.op(Op::GroupNonUniformBroadcast, u, &[scope, value, index])?;
                let index = self.const_u32(base >> 5)?;
                let matches = self.op(Op::IEqual, self.types.bool, &[partition, index])?;
                result = self.op(Op::Select, u, &[matches, read, result])?;
            }
        }
        Ok(result)
    }
    fn ballot(&mut self, predicate: Id) -> Result<Id> {
        let scope = self.const_u32(Scope::Subgroup as u32)?;
        let vector = self.op(
            Op::GroupNonUniformBallot,
            self.types.u32[3],
            &[scope, predicate],
        )?;
        if self.profile.warp_size_potentially_larger_than_guest {
            self.warp_extract(vector)
        } else {
            self.extract(self.types.u32[0], vector, 0)
        }
    }
    pub(super) fn warp_instruction(&mut self, ir: &mut Ir, id: InstId) -> Result<Option<Id>> {
        let opcode = ir.inst(id)?.opcode();
        let supported = self.profile.supports_subgroup_stage(self.stage);
        let u = self.types.u32[0];
        let b = self.types.bool;
        let result = match opcode {
            O::LaneId => self.lane_id()?,
            O::VoteAll | O::VoteAny | O::VoteEqual | O::SubgroupBallot => {
                let pred = self.def(ir, ir.inst(id)?.arg(0)?)?;
                if !supported {
                    match opcode {
                        O::VoteEqual => self.constants.true_value,
                        O::SubgroupBallot => {
                            let one = self.const_u32(1)?;
                            self.op(Op::Select, u, &[pred, one, self.constants.u32_zero])?
                        }
                        _ => pred,
                    }
                } else if opcode == O::SubgroupBallot {
                    self.ballot(pred)?
                } else if !self.profile.warp_size_potentially_larger_than_guest {
                    let scope = self.const_u32(Scope::Subgroup as u32)?;
                    let op = match opcode {
                        O::VoteAll => Op::GroupNonUniformAll,
                        O::VoteAny => Op::GroupNonUniformAny,
                        _ => Op::GroupNonUniformAllEqual,
                    };
                    self.op(op, b, &[scope, pred])?
                } else {
                    let active = self.ballot(self.constants.true_value)?;
                    let ballot = self.ballot(pred)?;
                    if opcode == O::VoteEqual {
                        let xor = self.op(Op::BitwiseXor, u, &[ballot, active])?;
                        let all = self.op(Op::IEqual, b, &[xor, self.constants.u32_zero])?;
                        let none = self.op(Op::IEqual, b, &[xor, active])?;
                        self.op(Op::LogicalOr, b, &[all, none])?
                    } else {
                        let mask = self.op(Op::BitwiseAnd, u, &[ballot, active])?;
                        self.op(
                            if opcode == O::VoteAll {
                                Op::IEqual
                            } else {
                                Op::INotEqual
                            },
                            b,
                            &[
                                mask,
                                if opcode == O::VoteAll {
                                    active
                                } else {
                                    self.constants.u32_zero
                                },
                            ],
                        )?
                    }
                }
            }
            O::SubgroupEqMask
            | O::SubgroupLtMask
            | O::SubgroupLeMask
            | O::SubgroupGtMask
            | O::SubgroupGeMask => {
                if !supported {
                    self.const_u32(u32::from(!matches!(
                        opcode,
                        O::SubgroupLtMask | O::SubgroupGtMask
                    )))?
                } else {
                    let object = match opcode {
                        O::SubgroupEqMask => self.objects.subgroup_mask_eq,
                        O::SubgroupLtMask => self.objects.subgroup_mask_lt,
                        O::SubgroupLeMask => self.objects.subgroup_mask_le,
                        O::SubgroupGtMask => self.objects.subgroup_mask_gt,
                        _ => self.objects.subgroup_mask_ge,
                    };
                    let vector = self.op(Op::Load, self.types.u32[3], &[object])?;
                    if self.profile.warp_size_potentially_larger_than_guest {
                        self.warp_extract(vector)?
                    } else {
                        self.extract(u, vector, 0)?
                    }
                }
            }
            O::ShuffleIndex | O::ShuffleUp | O::ShuffleDown | O::ShuffleButterfly => {
                let value = self.def(ir, ir.inst(id)?.arg(0)?)?;
                if !supported && opcode != O::ShuffleIndex {
                    self.pseudo(ir, id, O::GetInBoundsFromOp, self.constants.false_value)?;
                    return Ok(Some(value));
                }
                let index = self.def(ir, ir.inst(id)?.arg(1)?)?;
                let clamp = self.def(ir, ir.inst(id)?.arg(2)?)?;
                let segment = self.def(ir, ir.inst(id)?.arg(3)?)?;
                let not_segment = self.op(Op::Not, u, &[segment])?;
                let thread = self.lane_id()?;
                let min = self.op(Op::BitwiseAnd, u, &[thread, segment])?;
                let clamp = self.op(Op::BitwiseAnd, u, &[clamp, not_segment])?;
                let max = self.op(Op::BitwiseOr, u, &[min, clamp])?;
                let delta = self.guest_lane(index)?;
                let source = match opcode {
                    O::ShuffleIndex => {
                        let low = self.op(Op::BitwiseAnd, u, &[delta, not_segment])?;
                        self.op(Op::BitwiseOr, u, &[low, min])?
                    }
                    O::ShuffleUp => self.op(Op::ISub, u, &[thread, delta])?,
                    O::ShuffleDown => self.op(Op::IAdd, u, &[thread, delta])?,
                    _ => self.op(Op::BitwiseXor, u, &[thread, delta])?,
                };
                let in_range = self.op(
                    if opcode == O::ShuffleUp {
                        Op::SGreaterThanEqual
                    } else {
                        Op::SLessThanEqual
                    },
                    b,
                    &[source, max],
                )?;
                self.pseudo(ir, id, O::GetInBoundsFromOp, in_range)?;
                if !supported {
                    return Ok(Some(value));
                }
                let scope = self.const_u32(Scope::Subgroup as u32)?;
                let shuffled = if opcode == O::ShuffleIndex {
                    match (
                        ir.resolve(ir.inst(id)?.arg(1)?)?,
                        ir.resolve(ir.inst(id)?.arg(3)?)?,
                    ) {
                        (Value::U32(lane), Value::U32(0)) => {
                            self.broadcast_lane(value, lane & 31)?
                        }
                        _ => {
                            let host = self.host_thread(source)?;
                            self.shuffle_absolute(value, host)?
                        }
                    }
                } else if opcode == O::ShuffleButterfly || self.profile.support_shuffle_relative {
                    let result =
                        self.op(Op::GroupNonUniformShuffleXor, u, &[scope, value, delta])?;
                    if opcode != O::ShuffleButterfly {
                        self.opcode_rewrites.push((
                            result,
                            if opcode == O::ShuffleUp {
                                Op::GroupNonUniformShuffleUp
                            } else {
                                Op::GroupNonUniformShuffleDown
                            },
                        ));
                    }
                    result
                } else {
                    let host = self.host_thread(source)?;
                    self.shuffle_absolute(value, host)?
                };
                self.op(Op::Select, u, &[in_range, shuffled, value])?
            }
            O::QuadBroadcast | O::QuadSwap => {
                let value = self.def(ir, ir.inst(id)?.arg(0)?)?;
                let lane = self.def(ir, ir.inst(id)?.arg(1)?)?;
                let scope = self.const_u32(Scope::Subgroup as u32)?;
                if self.profile.support_quad_shuffles {
                    let result =
                        self.op(Op::GroupNonUniformQuadBroadcast, u, &[scope, value, lane])?;
                    if opcode == O::QuadSwap {
                        self.opcode_rewrites
                            .push((result, Op::GroupNonUniformQuadSwap));
                    }
                    result
                } else if opcode == O::QuadSwap {
                    let one = self.const_u32(1)?;
                    let mask = self.op(Op::IAdd, u, &[lane, one])?;
                    self.op(Op::GroupNonUniformShuffleXor, u, &[scope, value, mask])?
                } else {
                    let thread = self.thread_id()?;
                    let mask = self.const_u32(!3)?;
                    let base = self.op(Op::BitwiseAnd, u, &[thread, mask])?;
                    let mask = self.const_u32(3)?;
                    let lane = self.op(Op::BitwiseAnd, u, &[lane, mask])?;
                    let source = self.op(Op::BitwiseOr, u, &[base, lane])?;
                    self.shuffle_absolute(value, source)?
                }
            }
            O::FSwizzleAdd => {
                let a = self.def(ir, ir.inst(id)?.arg(0)?)?;
                let c = self.def(ir, ir.inst(id)?.arg(1)?)?;
                let swizzle = self.def(ir, ir.inst(id)?.arg(2)?)?;
                let thread = self.thread_id()?;
                let three = self.const_u32(3)?;
                let one = self.const_u32(1)?;
                let mask = self.op(Op::BitwiseAnd, u, &[thread, three])?;
                let mask = self.op(Op::ShiftLeftLogical, u, &[mask, one])?;
                let mask = self.op(Op::ShiftRightLogical, u, &[swizzle, mask])?;
                let mask = self.op(Op::BitwiseAnd, u, &[mask, three])?;
                let f = self.types.f32[0];
                let ma = self.op(
                    Op::VectorExtractDynamic,
                    f,
                    &[self.objects.fswzadd_lut_a, mask],
                )?;
                let mb = self.op(
                    Op::VectorExtractDynamic,
                    f,
                    &[self.objects.fswzadd_lut_b, mask],
                )?;
                let a = self.op(Op::FMul, f, &[a, ma])?;
                let c = self.op(Op::FMul, f, &[c, mb])?;
                self.op(Op::FAdd, f, &[a, c])?
            }
            _ => return Ok(None),
        };
        Ok(Some(result))
    }
}
