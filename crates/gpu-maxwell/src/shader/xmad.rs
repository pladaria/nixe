//! XMAD multiplies independently selected 16-bit halves with a 32-bit addend.
//! https://github.com/devkitPro/uam/blob/master/mesa-imported/codegen/nv50_ir_emit_gm107.cpp (emitXMAD)
//! https://github.com/eden-emulator/mirror/blob/master/src/shader_recompiler/frontend/maxwell/translate/impl/integer_short_multiply_add.cpp
use super::{
    decode::{allocate_shader_temporary, validate_register_range},
    error::MaxwellShaderTranslationError,
    integer::DecodedIntegerOperation,
};
use crate::MaxwellShaderStage;
use nixe_gpu::{
    ShaderBitwiseOperation, ShaderFloatControl, ShaderOperation, ShaderRegister, ShaderScalarType,
};

pub(super) const fn is_xmad(encoding: u64) -> bool {
    let opcode = encoding >> 48;
    opcode & 0xffc0 == 0x5b00
        || opcode & 0xfe00 == 0x4e00
        || opcode & 0xffc0 == 0x5100
        || opcode & 0xffc0 == 0x3600
}

struct Expansion<'a> {
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    registers: u8,
    next: &'a mut u16,
    ops: Vec<ShaderOperation>,
}
impl Expansion<'_> {
    fn temp(&mut self) -> Result<ShaderRegister, MaxwellShaderTranslationError> {
        allocate_shader_temporary(
            self.stage,
            self.offset,
            self.encoding,
            "XMAD temporary overflow",
            self.next,
        )
    }
    fn immediate(&mut self, bits: u32) -> Result<ShaderRegister, MaxwellShaderTranslationError> {
        let destination = self.temp()?;
        self.ops.push(ShaderOperation::MoveImmediate32 {
            destination,
            bits,
            scalar_type: ShaderScalarType::Unsigned32,
        });
        Ok(destination)
    }
    fn reg(&mut self, raw: u8) -> Result<ShaderRegister, MaxwellShaderTranslationError> {
        if raw == 255 {
            return self.immediate(0);
        }
        validate_register_range(
            self.stage,
            self.offset,
            self.encoding,
            raw,
            1,
            self.registers,
        )?;
        Ok(ShaderRegister::new(u16::from(raw)))
    }
    fn bits(
        &mut self,
        source: ShaderRegister,
        high: bool,
        signed: bool,
    ) -> Result<ShaderRegister, MaxwellShaderTranslationError> {
        let mask = self.immediate(0xffff)?;
        let mut value = source;
        if high {
            value = self.shift(source, false)?;
        }
        let destination = self.temp()?;
        self.ops.push(ShaderOperation::Bitwise32 {
            destination,
            left: value,
            right: mask,
            operation: ShaderBitwiseOperation::And,
        });
        if !signed {
            return Ok(destination);
        }
        let bias = self.immediate(0x8000)?;
        let flipped = self.temp()?;
        self.ops.push(ShaderOperation::Bitwise32 {
            destination: flipped,
            left: destination,
            right: bias,
            operation: ShaderBitwiseOperation::Xor,
        });
        let negative_bias = self.immediate(0xffff8000)?;
        self.add(flipped, negative_bias)
    }
    fn shift(
        &mut self,
        value: ShaderRegister,
        left: bool,
    ) -> Result<ShaderRegister, MaxwellShaderTranslationError> {
        let amount = self.immediate(16)?;
        let destination = self.temp()?;
        self.ops.push(if left {
            ShaderOperation::ShiftLeft32 {
                destination,
                value,
                amount,
                wrap: false,
            }
        } else {
            ShaderOperation::ShiftRight32 {
                destination,
                value,
                amount,
                wrap: false,
                signed: false,
                reverse: false,
            }
        });
        Ok(destination)
    }
    fn add(
        &mut self,
        left: ShaderRegister,
        right: ShaderRegister,
    ) -> Result<ShaderRegister, MaxwellShaderTranslationError> {
        let destination = self.temp()?;
        self.ops.push(ShaderOperation::Add32 {
            destination,
            left,
            right,
            scalar_type: ShaderScalarType::Unsigned32,
            float_control: ShaderFloatControl::PRECISE,
        });
        Ok(destination)
    }
}

pub(super) fn decode_xmad(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
    next: &mut u16,
) -> Result<DecodedIntegerOperation, MaxwellShaderTranslationError> {
    let opcode = encoding >> 48;
    let cb_b = opcode & 0xfe00 == 0x4e00;
    let cb_c = opcode & 0xffc0 == 0x5100;
    let constant = cb_b || cb_c;
    let immediate = opcode & 0xffc0 == 0x3600;
    let mode = (encoding >> 50) & if constant { 3 } else { 7 };
    if encoding & (1 << 47) != 0
        || encoding & (1 << if constant { 54 } else { 38 }) != 0
        || mode == 3
        || mode > 4
    {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "XMAD CC, extended carry or unsupported addend selection",
        });
    }
    let mut e = Expansion {
        stage,
        offset,
        encoding,
        registers: register_count,
        next,
        ops: Vec::new(),
    };
    let destination = encoding as u8;
    let destination = if destination == 255 {
        e.temp()?
    } else {
        validate_register_range(stage, offset, encoding, destination, 1, register_count)?;
        ShaderRegister::new(u16::from(destination))
    };
    let a = e.reg((encoding >> 8) as u8)?;
    let mut constant_buffer_binding = None;
    let cb = if constant {
        let binding = ((encoding >> 34) & 31) as u8;
        constant_buffer_binding = Some(binding);
        let destination = e.temp()?;
        e.ops.push(ShaderOperation::LoadConstantBuffer32 {
            destination,
            binding,
            byte_offset: ((encoding >> 20) as u32 & 0x3fff) * 4,
            scalar_type: ShaderScalarType::Unsigned32,
        });
        Some(destination)
    } else {
        None
    };
    let (b, c) = if cb_b {
        (cb.unwrap(), e.reg((encoding >> 39) as u8)?)
    } else if cb_c {
        (e.reg((encoding >> 39) as u8)?, cb.unwrap())
    } else {
        (
            if immediate {
                e.immediate((encoding >> 20) as u32 & 0xffff)?
            } else {
                e.reg((encoding >> 20) as u8)?
            },
            e.reg((encoding >> 39) as u8)?,
        )
    };
    let a = e.bits(a, encoding & (1 << 53) != 0, encoding & (1 << 48) != 0)?;
    let selected_b = e.bits(
        b,
        !immediate && encoding & (1 << if constant { 52 } else { 35 }) != 0,
        encoding & (1 << 49) != 0,
    )?;
    let mut product = e.temp()?;
    e.ops.push(ShaderOperation::Multiply32 {
        destination: product,
        left: a,
        right: selected_b,
        scalar_type: ShaderScalarType::Unsigned32,
        float_control: ShaderFloatControl::PRECISE,
    });
    if !cb_c && encoding & (1 << if cb_b { 55 } else { 36 }) != 0 {
        product = e.shift(product, true)?;
    }
    let c = match mode {
        0 => c,
        1 => e.bits(c, false, false)?,
        2 => e.bits(c, true, false)?,
        4 => {
            let shifted_b = e.shift(b, true)?;
            e.add(shifted_b, c)?
        }
        _ => unreachable!(),
    };
    let mut result = e.add(product, c)?;
    if !cb_c && encoding & (1 << if cb_b { 56 } else { 37 }) != 0 {
        let low = e.bits(result, false, false)?;
        let high = e.shift(b, true)?;
        result = e.temp()?;
        e.ops.push(ShaderOperation::Bitwise32 {
            destination: result,
            left: low,
            right: high,
            operation: ShaderBitwiseOperation::Or,
        });
    }
    e.ops.push(ShaderOperation::Move32 {
        destination,
        source: result,
        scalar_type: ShaderScalarType::Unsigned32,
    });
    Ok(DecodedIntegerOperation {
        operations: e.ops,
        constant_buffer_binding,
    })
}
