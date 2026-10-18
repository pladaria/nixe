use super::{Opcode, Result, Source, Value, Visitor, unsupported};

impl Visitor<'_> {
    fn video_extract(
        &mut self,
        value: Value,
        width: u32,
        selector: u32,
        signed: bool,
    ) -> Result<Value> {
        match width {
            0 | 1 => {
                self.emit
                    .bit_field_extract(value, Value::U32(selector * 8), Value::U32(8), signed)
            }
            2 => self.emit.bit_field_extract(
                value,
                Value::U32(selector * 16),
                Value::U32(16),
                signed,
            ),
            3 => Ok(value),
            _ => unreachable!(),
        }
    }
    pub(super) fn video(&mut self, opcode: Opcode) -> Result<()> {
        if opcode != Opcode::VSETP {
            if self.bit::<47>() {
                return Err(unsupported("Video instruction CC"));
            }
            if self.bit::<55>() {
                return Err(unsupported("Video instruction SAT"));
            }
        }
        if opcode == Opcode::VMAD {
            if self.field::<51, 2>() != 0 {
                return Err(unsupported("VMAD SCALE"));
            }
            if self.bit::<53>() && self.bit::<54>() {
                return Err(unsupported("VMAD PO"));
            }
            if self.bit::<53>() || self.bit::<54>() {
                return Err(unsupported("VMAD NEG"));
            }
        }
        let immediate = !self.bit::<50>();
        let a = self.a()?;
        let b = self.source(if immediate {
            Source::Imm16
        } else {
            Source::Reg20
        })?;
        let b_width = if immediate { 2 } else { self.field::<29, 2>() };
        // VSETP retains the encoded selector even for immediate sources.
        let b_selector = if immediate && opcode != Opcode::VSETP {
            0
        } else {
            self.field::<28, 2>()
        };
        let a = self.video_extract(
            a,
            self.field::<37, 2>(),
            self.field::<36, 2>(),
            self.bit::<48>(),
        )?;
        let b = self.video_extract(b, b_width, b_selector, self.bit::<49>())?;
        if opcode == Opcode::VSETP {
            let comparison = match self.field::<43, 5>() {
                op @ 0..=3 => op,
                op @ 16..=19 => op - 12,
                _ => return Err(unsupported("Invalid VSETP comparison")),
            };
            let comparison = self.compare(a, b, comparison, self.bit::<49>())?;
            return self.predicate_pair(comparison);
        }
        let c = self.source(Source::Reg39)?;
        let result = if opcode == Opcode::VMAD {
            let product = self.emit.i_mul(a, b)?;
            self.emit.i_add(product, c)?
        } else {
            let lhs = if self.bit::<56>() {
                self.emit.i_max(a, b, self.bit::<49>())?
            } else {
                self.emit.i_min(a, b, self.bit::<49>())?
            };
            match self.field::<51, 3>() {
                5 => self.emit.i_min(lhs, c, self.bit::<54>())?,
                6 => self.emit.i_max(lhs, c, self.bit::<54>())?,
                _ => return Err(unsupported("VMNMX operation")),
            }
        };
        self.write(result)
    }
}
