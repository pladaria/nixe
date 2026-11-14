//! Warp-distributed matrix operations expressed directly in semantic IR.
use super::*;

impl Visitor<'_> {
    pub(super) fn matrix_multiply(&mut self) -> Result<()> {
        let i = self.instruction;
        let mode = i.bits(64, 32) & !0xff;
        let depth = match mode {
            0 => 8,
            0x800 => 16,
            _ => return Err(unsupported("Native matrix multiply mode")),
        };
        let a = i.bits(24, 8);
        let b = i.bits(32, 8);
        let c = i.bits(64, 8);
        let destination = i.bits(16, 8);
        let a_count = depth / 4;
        let b_count = depth / 8;
        if i.bits(12, 4) != 7
            || i.bits(32, 32) & !0xff != 0
            || i.bits(96, 32) & 0xc00001ff != 0
            || !destination.is_multiple_of(2)
            || destination + 2 > 255
            || !a.is_multiple_of(a_count)
            || a + a_count > 255
            || !b.is_multiple_of(b_count)
            || b + b_count > 255
            || (c != 255 && (!c.is_multiple_of(2) || c + 2 > 255))
        {
            return Err(unsupported("Native matrix multiply operands"));
        }
        let mut operands = [Value::None; 3];
        for (operand, (base, count, width)) in
            operands
                .iter_mut()
                .zip([(a, a_count, 4), (b, b_count, 2), (c, 2, 2)])
        {
            let mut registers = [Value::U32(0); 4];
            if base != 255 {
                for index in 0..count {
                    registers[index as usize] = self.reg(base + index)?;
                }
            }
            *operand = self.emit.composite_construct(&registers[..width])?;
        }
        let result = self
            .emit
            .emit(crate::ir::Opcode::MatrixMultiplyF16, &operands, depth)?;
        for index in 0..2 {
            let value = self.emit.composite_extract(result, index)?;
            self.emit
                .set_reg(Reg((destination + index as u32) as u8), value)?;
        }
        Ok(())
    }

    pub(super) fn matrix_transpose(&mut self) -> Result<()> {
        let i = self.instruction;
        if i.bits(32, 32) != 0 || i.bits(64, 32) != 0 {
            return Err(unsupported("Native matrix transpose mode"));
        }
        // Transpose an 8x8 matrix of 16-bit elements distributed like LDSM:
        // lane t owns row t/4 and columns 2(t%4), 2(t%4)+1.
        // https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-movmatrix
        let source = self.reg(i.bits(24, 8))?;
        let lane = self.emit.lane_id()?;
        let quarter = self.emit.bitwise_and(lane, Value::U32(3))?;
        let row = self.emit.shift_left_logical(quarter, Value::U32(3))?;
        let column = self.emit.shift_right_logical(lane, Value::U32(3))?;
        let first = self.emit.i_add(row, column)?;
        let second = self.emit.i_add(first, Value::U32(4))?;
        let half = self.emit.bitwise_and(lane, Value::U32(4))?;
        let offset = self.emit.shift_left_logical(half, Value::U32(2))?;
        let a = self
            .emit
            .shuffle_index(source, first, Value::U32(31), Value::U32(0))?;
        let b = self
            .emit
            .shuffle_index(source, second, Value::U32(31), Value::U32(0))?;
        let a = self
            .emit
            .bit_field_extract(a, offset, Value::U32(16), false)?;
        let b = self
            .emit
            .bit_field_extract(b, offset, Value::U32(16), false)?;
        let value = self
            .emit
            .bit_field_insert(a, b, Value::U32(16), Value::U32(16))?;
        self.write(value)
    }

    pub(super) fn matrix_load(&mut self) -> Result<()> {
        let i = self.instruction;
        let count = 1 << i.bits(72, 2);
        let transpose = i.bits(78, 1) != 0;
        let uniform = i.bits(91, 1) != 0;
        let destination = i.bits(16, 8);
        if i.bits(12, 4) != 7
            || count > 4
            || i.bits(64, 32) & !0x08004300 != 0
            || (!uniform && i.bits(32, 8) != 0)
            || !destination.is_multiple_of(count)
            || destination + count > 255
        {
            return Err(unsupported("Native matrix load mode"));
        }
        let mut base = self.reg(i.bits(24, 8))?;
        if uniform {
            let offset = self.ureg(i.bits(32, 8))?;
            base = self.emit.i_add(base, offset)?;
        }
        let offset = ((i.bits(40, 24) as i32) << 8 >> 8) as u32;
        base = self.emit.i_add(base, Value::U32(offset))?;
        // Each group of eight lanes supplies row addresses for one 8x8 matrix.
        // Four consecutive lanes receive a row, two 16-bit elements per lane.
        // https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-ldmatrix
        let lane = self.emit.lane_id()?;
        let group = self.emit.shift_right_logical(lane, Value::U32(2))?;
        let column = self.emit.bitwise_and(lane, Value::U32(3))?;
        let (row, byte) = if transpose {
            (
                self.emit.shift_left_logical(column, Value::U32(1))?,
                self.emit.shift_left_logical(group, Value::U32(1))?,
            )
        } else {
            (group, self.emit.shift_left_logical(column, Value::U32(2))?)
        };
        for matrix in 0..count {
            let mut parts = [Value::None; 2];
            for (half, value) in parts
                .iter_mut()
                .enumerate()
                .take(if transpose { 2 } else { 1 })
            {
                let lane = self.emit.i_add(row, Value::U32(matrix * 8 + half as u32))?;
                let address = self
                    .emit
                    .shuffle_index(base, lane, Value::U32(31), Value::U32(0))?;
                let address = self.emit.i_add(address, byte)?;
                *value = self
                    .emit
                    .load_shared(if transpose { 16 } else { 32 }, false, address)?;
            }
            let value = if transpose {
                self.emit
                    .bit_field_insert(parts[0], parts[1], Value::U32(16), Value::U32(16))?
            } else {
                parts[0]
            };
            self.emit
                .set_reg(Reg((destination + matrix) as u8), value)?;
        }
        Ok(())
    }
}
