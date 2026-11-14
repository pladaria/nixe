use super::*;
use crate::types::Stage;

impl Visitor<'_> {
    // The input map holds vertex indices, while host geometry inputs are already
    // indexed by vertex. Output handles likewise have no host storage identity.
    // https://docs.mesa3d.org/drivers/nvk/isbe_layout.html
    // https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/shader_recompiler/frontend/ampere/lower_to_maxwell.h
    pub(super) fn geometry_vertex_index(&mut self) -> Result<()> {
        let i = self.instruction;
        if self.metadata.stage != Stage::Geometry
            || i.bits(32, 32) != 0
            || i.bits(64, 32) != 0
            || i.bits(24, 8) == 255
        {
            return Err(unsupported("Native geometry input map mode"));
        }
        let vertex = self.reg(i.bits(24, 8))?;
        self.write(vertex)
    }

    pub(super) fn geometry_output(&mut self) -> Result<()> {
        let i = self.instruction;
        let operation = i.bits(78, 2);
        if self.metadata.stage != Stage::Geometry || i.bits(64, 32) & !0xc000 != 0 {
            return Err(unsupported("Native geometry output stage/mode"));
        }
        if operation == 0 && (i.opcode() != 0x324 || i.bits(16, 8) != 255 || i.bits(32, 32) != 255)
        {
            return Err(unsupported("Native geometry finalization mode"));
        }
        let stream = if i.opcode() == 0x324 && i.bits(40, 24) == 0 {
            self.reg(i.bits(32, 8))?
        } else if i.opcode() == 0x924 && i.bits(51, 13) == 0 {
            Value::U32(i.bits(32, 19))
        } else {
            return Err(unsupported("Native geometry output stream"));
        };
        let stream = self.emit.bitwise_and(stream, Value::U32(3))?;
        if operation & 1 != 0 {
            self.emit.emit_vertex(stream)?;
        }
        if operation & 2 != 0 {
            self.emit.end_primitive(stream)?;
        }
        self.write(Value::U32(0))
    }
}
