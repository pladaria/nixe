use super::*;

impl Visitor<'_> {
    pub(super) fn match_all(&mut self) -> Result<()> {
        let i = self.instruction;
        let source = i.bits(24, 8);
        let words = if i.bits(73, 1) != 0 { 2 } else { 1 };
        if i.bits(12, 4) != 7
            || i.bits(16, 8) != 255
            || source >= 254
            || (words == 2 && source & 1 != 0)
            || i.bits(32, 32) != 0
            || i.bits(64, 32) & !0xe0200 != 0
        {
            return Err(unsupported("Native warp match mode"));
        }
        // MATCH.ALL compares the complete value of every participating lane.
        // https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-match-sync
        let mask = self.emit.subgroup_ballot(Value::U1(true))?;
        let negative = self.emit.i_neg(mask)?;
        let first_bit = self.emit.bitwise_and(mask, negative)?;
        let lane = self.emit.find_u_msb(first_bit)?;
        let mut equal = Value::U1(true);
        for word in 0..words {
            let value = self.reg(source + word)?;
            let first = self
                .emit
                .shuffle_index(value, lane, Value::U32(31), Value::U32(0))?;
            let matches = self.emit.i_equal(value, first)?;
            equal = self.emit.logical_and(equal, matches)?;
        }
        let all = self.emit.vote_all(equal)?;
        self.emit.set_pred(Pred::new(i.bits(81, 3) as u8)?, all)
    }

    pub(super) fn shuffle(&mut self) -> Result<()> {
        let i = self.instruction;
        let immediate_lane = matches!(i.opcode(), 0x989 | 0xf89);
        let immediate_clamp = matches!(i.opcode(), 0x589 | 0xf89);
        if i.bits(64, 32) & !if immediate_clamp { 0xe0000 } else { 0xe00ff } != 0
            || (!immediate_lane && i.bits(53, 5) != 0)
            || (immediate_lane && i.bits(32, 8) != 0)
            || (immediate_clamp && i.bits(45, 3) != 0)
            || (!immediate_clamp && i.bits(40, 13) != 0)
            || i.bits(60, 4) != 0
            || i.bits(96, 9) != 0
            || i.bits(126, 2) != 0
        {
            return Err(unsupported("Native shuffle mode"));
        }
        let input = self.reg(i.bits(24, 8))?;
        let lane = if immediate_lane {
            Value::U32(i.bits(53, 5))
        } else {
            self.reg(i.bits(32, 8))?
        };
        let mask = if immediate_clamp {
            Value::U32(i.bits(40, 13))
        } else {
            self.reg(i.bits(64, 8))?
        };
        let (value, in_bounds) = crate::frontend::shuffle(
            &mut self.emit,
            self.metadata.stage,
            i.bits(58, 2),
            input,
            lane,
            mask,
        )?;
        self.emit
            .set_pred(Pred::new(i.bits(81, 3) as u8)?, in_bounds)?;
        self.write(value)
    }
    pub(super) fn special_register(&mut self) -> Result<()> {
        let i = self.instruction;
        let selector = i.bits(72, 8);
        if i.bits(32, 32) != 0 || i.bits(64, 32) != selector << 8 {
            return Err(unsupported("Native special-register read mode"));
        }
        let value = match selector {
            0 => self.emit.lane_id()?,
            17 => self.emit.invocation_id()?,
            18 => {
                let direction = self.emit.y_direction()?;
                self.emit.bit_cast(Type::U32, direction)?
            }
            29 => self.emit.invocation_info()?,
            19 => {
                let helper = self.emit.is_helper_invocation()?;
                self.emit
                    .select(helper, Value::U32(u32::MAX), Value::U32(0))?
            }
            30 => self.emit.sr_w_scale_factor_xy()?,
            31 => self.emit.sr_w_scale_factor_z()?,
            32 => {
                let tid = self.emit.local_invocation_id()?;
                let x = self.emit.composite_extract(tid, 0)?;
                let y = self.emit.composite_extract(tid, 1)?;
                let z = self.emit.composite_extract(tid, 2)?;
                let xy = self
                    .emit
                    .bit_field_insert(x, y, Value::U32(16), Value::U32(8))?;
                self.emit
                    .bit_field_insert(xy, z, Value::U32(26), Value::U32(6))?
            }
            33 => self.emit.local_invocation_id_x()?,
            34 => self.emit.local_invocation_id_y()?,
            35 => self.emit.local_invocation_id_z()?,
            37 => self.emit.workgroup_id_x()?,
            38 => self.emit.workgroup_id_y()?,
            39 => self.emit.workgroup_id_z()?,
            56 => self.emit.subgroup_eq_mask()?,
            57 => self.emit.subgroup_lt_mask()?,
            58 => self.emit.subgroup_le_mask()?,
            59 => self.emit.subgroup_gt_mask()?,
            60 => self.emit.subgroup_ge_mask()?,
            _ => return Err(unsupported(format!("Native special register {selector}"))),
        };
        self.write_register(value, i.opcode() == 0x9c3)
    }
}
