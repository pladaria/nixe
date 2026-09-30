//! Native descriptors use set zero and the neutral pipeline-wide binding index,
//! already shared across stages by the frontend linker. Constant buffers are
//! read-only storage buffers of raw 32-bit words, without repacking guest data.
//! https://docs.vulkan.org/spec/latest/chapters/interfaces.html#interfaces-resources
use super::*;
use rspirv::dr::Operand;

impl Emitter {
    pub(super) fn emit_resources(
        &mut self,
        ir: &ShaderIr,
        live: &[bool],
    ) -> Result<Box<[ShaderResourceAccess]>> {
        let mut block_type = None;
        let mut bindings = Vec::new();
        let mut declarations = [None; 256];
        for resource in &ir.resources {
            declarations[usize::from(resource.binding)] = Some(resource);
        }
        for (instruction, live) in ir.instructions.iter().zip(live) {
            if !live {
                continue;
            }
            let binding = match instruction.operation {
                ShaderOperation::LoadConstantBuffer32 { binding, .. }
                | ShaderOperation::LoadConstantBufferIndexed32 { binding, .. } => binding,
                _ => continue,
            };
            if self.constant_buffers[usize::from(binding)].is_some() {
                continue;
            }
            let resource = declarations[usize::from(binding)];
            if !resource.is_some_and(|r| {
                r.kind == ShaderResourceKind::ConstantBuffer && r.readable && !r.writable
            }) {
                return Err(SpirvShaderError::Instruction {
                    source: instruction.source,
                    reason: "native constant-buffer descriptor must be declared read-only",
                });
            }
            let block = *block_type.get_or_insert_with(|| {
                self.constant_word_pointer = Some(self.b.type_pointer(
                    None,
                    spv::StorageClass::StorageBuffer,
                    self.uint,
                ));
                let array = self.b.type_runtime_array(self.uint);
                self.b.decorate(
                    array,
                    spv::Decoration::ArrayStride,
                    [Operand::LiteralBit32(4)],
                );
                let block = self.b.type_struct([array]);
                self.b.decorate(block, spv::Decoration::Block, []);
                self.b.member_decorate(
                    block,
                    0,
                    spv::Decoration::Offset,
                    [Operand::LiteralBit32(0)],
                );
                self.b
                    .member_decorate(block, 0, spv::Decoration::NonWritable, []);
                block
            });
            let pointer = self
                .b
                .type_pointer(None, spv::StorageClass::StorageBuffer, block);
            let variable = self
                .b
                .variable(pointer, None, spv::StorageClass::StorageBuffer, None);
            self.b.decorate(
                variable,
                spv::Decoration::DescriptorSet,
                [Operand::LiteralBit32(0)],
            );
            self.b.decorate(
                variable,
                spv::Decoration::Binding,
                [Operand::LiteralBit32(u32::from(binding))],
            );
            self.constant_buffers[usize::from(binding)] = Some(variable);
            bindings.push(*resource.expect("validated read-only descriptor"));
        }
        bindings.sort_unstable_by_key(|resource| resource.binding);
        Ok(bindings.into_boxed_slice())
    }

    pub(super) fn load_constant_buffer(&mut self, binding: u8, word: u32) -> Result<u32> {
        let buffer = self.constant_buffers[usize::from(binding)]
            .ok_or_else(|| self.unsupported("missing native constant-buffer descriptor"))?;
        let pointer = self
            .constant_word_pointer
            .expect("declared buffer has a scalar pointer type");
        let zero = self.constant(0);
        let address = self.b.access_chain(pointer, None, buffer, [zero, word])?;
        Ok(self.b.load(self.uint, None, address, None, [])?)
    }
}
