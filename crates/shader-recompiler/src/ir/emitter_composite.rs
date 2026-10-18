use super::*;
use crate::Result;

impl Emitter<'_> {
    pub fn composite_construct(&mut self, elements: &[Value]) -> Result<Value> {
        let ty = self.same_type(elements)?;
        let opcode = match (ty, elements.len()) {
            (Type::U32, 2) => Opcode::CompositeConstructU32x2,
            (Type::U32, 3) => Opcode::CompositeConstructU32x3,
            (Type::U32, 4) => Opcode::CompositeConstructU32x4,
            (Type::F16, 2) => Opcode::CompositeConstructF16x2,
            (Type::F16, 3) => Opcode::CompositeConstructF16x3,
            (Type::F16, 4) => Opcode::CompositeConstructF16x4,
            (Type::F32, 2) => Opcode::CompositeConstructF32x2,
            (Type::F32, 3) => Opcode::CompositeConstructF32x3,
            (Type::F32, 4) => Opcode::CompositeConstructF32x4,
            (Type::F64, 2) => Opcode::CompositeConstructF64x2,
            (Type::F64, 3) => Opcode::CompositeConstructF64x3,
            (Type::F64, 4) => Opcode::CompositeConstructF64x4,
            _ => return Err(Self::invalid("Invalid composite element type or count")),
        };
        self.emit(opcode, elements, 0)
    }
    pub fn composite_extract(&mut self, vector: Value, element: usize) -> Result<Value> {
        let (opcode, count) = match self.ir.value_type(vector)? {
            Type::U32x2 => (Opcode::CompositeExtractU32x2, 2),
            Type::U32x3 => (Opcode::CompositeExtractU32x3, 3),
            Type::U32x4 => (Opcode::CompositeExtractU32x4, 4),
            Type::F16x2 => (Opcode::CompositeExtractF16x2, 2),
            Type::F16x3 => (Opcode::CompositeExtractF16x3, 3),
            Type::F16x4 => (Opcode::CompositeExtractF16x4, 4),
            Type::F32x2 => (Opcode::CompositeExtractF32x2, 2),
            Type::F32x3 => (Opcode::CompositeExtractF32x3, 3),
            Type::F32x4 => (Opcode::CompositeExtractF32x4, 4),
            Type::F64x2 => (Opcode::CompositeExtractF64x2, 2),
            Type::F64x3 => (Opcode::CompositeExtractF64x3, 3),
            Type::F64x4 => (Opcode::CompositeExtractF64x4, 4),
            _ => return Err(Self::invalid("Value is not a composite")),
        };
        if element >= count {
            return Err(Self::invalid("Composite element index out of range"));
        }
        self.emit(opcode, &[vector, Value::U32(element as u32)], 0)
    }
    pub fn composite_insert(
        &mut self,
        vector: Value,
        object: Value,
        element: usize,
    ) -> Result<Value> {
        let (opcode, count) = match self.ir.value_type(vector)? {
            Type::U32x2 => (Opcode::CompositeInsertU32x2, 2),
            Type::U32x3 => (Opcode::CompositeInsertU32x3, 3),
            Type::U32x4 => (Opcode::CompositeInsertU32x4, 4),
            Type::F16x2 => (Opcode::CompositeInsertF16x2, 2),
            Type::F16x3 => (Opcode::CompositeInsertF16x3, 3),
            Type::F16x4 => (Opcode::CompositeInsertF16x4, 4),
            Type::F32x2 => (Opcode::CompositeInsertF32x2, 2),
            Type::F32x3 => (Opcode::CompositeInsertF32x3, 3),
            Type::F32x4 => (Opcode::CompositeInsertF32x4, 4),
            Type::F64x2 => (Opcode::CompositeInsertF64x2, 2),
            Type::F64x3 => (Opcode::CompositeInsertF64x3, 3),
            Type::F64x4 => (Opcode::CompositeInsertF64x4, 4),
            _ => return Err(Self::invalid("Value is not a composite")),
        };
        if element >= count {
            return Err(Self::invalid("Composite element index out of range"));
        }
        self.emit(opcode, &[vector, object, Value::U32(element as u32)], 0)
    }
    pub fn bit_cast(&mut self, result: Type, value: Value) -> Result<Value> {
        let opcode = match (result, self.ir.value_type(value)?) {
            (Type::U16, Type::F16) => Opcode::BitCastU16F16,
            (Type::F16, Type::U16) => Opcode::BitCastF16U16,
            (Type::U32, Type::F32) => Opcode::BitCastU32F32,
            (Type::F32, Type::U32) => Opcode::BitCastF32U32,
            (Type::U64, Type::F64) => Opcode::BitCastU64F64,
            (Type::F64, Type::U64) => Opcode::BitCastF64U64,
            _ => return Err(Self::invalid("Invalid bitcast types")),
        };
        self.emit(opcode, &[value], 0)
    }
    pub fn get_attribute(&mut self, attribute: Attribute) -> Result<Value> {
        self.emit(
            Opcode::GetAttribute,
            &[Value::Attribute(attribute), Value::U32(0)],
            0,
        )
    }
    pub fn get_attribute_u32(&mut self, attribute: Attribute) -> Result<Value> {
        self.emit(
            Opcode::GetAttributeU32,
            &[Value::Attribute(attribute), Value::U32(0)],
            0,
        )
    }
    pub fn get_attribute_indexed(&mut self, attribute: Value) -> Result<Value> {
        self.emit(Opcode::GetAttributeIndexed, &[attribute, Value::U32(0)], 0)
    }
    pub fn workgroup_id_x(&mut self) -> Result<Value> {
        let vector = self.emit(Opcode::WorkgroupId, &[], 0)?;
        self.composite_extract(vector, 0)
    }
    pub fn workgroup_id_y(&mut self) -> Result<Value> {
        let vector = self.emit(Opcode::WorkgroupId, &[], 0)?;
        self.composite_extract(vector, 1)
    }
    pub fn workgroup_id_z(&mut self) -> Result<Value> {
        let vector = self.emit(Opcode::WorkgroupId, &[], 0)?;
        self.composite_extract(vector, 2)
    }
    pub fn local_invocation_id_x(&mut self) -> Result<Value> {
        let vector = self.emit(Opcode::LocalInvocationId, &[], 0)?;
        self.composite_extract(vector, 0)
    }
    pub fn local_invocation_id_y(&mut self) -> Result<Value> {
        let vector = self.emit(Opcode::LocalInvocationId, &[], 0)?;
        self.composite_extract(vector, 1)
    }
    pub fn local_invocation_id_z(&mut self) -> Result<Value> {
        let vector = self.emit(Opcode::LocalInvocationId, &[], 0)?;
        self.composite_extract(vector, 2)
    }
    pub fn render_area_width(&mut self) -> Result<Value> {
        let vector = self.emit(Opcode::RenderArea, &[], 0)?;
        self.composite_extract(vector, 0)
    }
    pub fn render_area_height(&mut self) -> Result<Value> {
        let vector = self.emit(Opcode::RenderArea, &[], 0)?;
        self.composite_extract(vector, 1)
    }
}
