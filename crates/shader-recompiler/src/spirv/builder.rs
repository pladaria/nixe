use crate::{CompileError, Result, diagnostics::FailureCategory};
use spirv::{Capability, Decoration, ExecutionMode, ExecutionModel, Op, StorageClass};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Id(pub u32);
impl Id {
    pub const NONE: Self = Self(0);
}
#[derive(Clone, Copy, Debug)]
#[repr(usize)]
pub enum Section {
    Capabilities,
    Extensions,
    Imports,
    MemoryModel,
    EntryPoints,
    ExecutionModes,
    Debug,
    Annotations,
    Declarations,
    Functions,
}
#[derive(Debug)]
pub struct DeferredPhi {
    pub id: Id,
    offset: usize,
    count: usize,
}

pub struct Builder {
    version: u32,
    next_id: u32,
    sections: [Vec<u32>; 10],
    declarations: HashMap<Vec<u32>, Id>,
    capabilities: HashSet<Capability>,
    extensions: HashSet<String>,
    glsl: Id,
    pending_phis: HashSet<usize>,
}
impl Builder {
    pub fn new(version: u32) -> Self {
        Self {
            version,
            next_id: 1,
            sections: std::array::from_fn(|_| Vec::new()),
            declarations: HashMap::new(),
            capabilities: HashSet::new(),
            extensions: HashSet::new(),
            glsl: Id::NONE,
            pending_phis: HashSet::new(),
        }
    }
    pub fn reserve_id(&mut self) -> Result<Id> {
        let id = Id(self.next_id);
        self.next_id = self.next_id.checked_add(1).ok_or_else(|| {
            CompileError::new(FailureCategory::ResourceLimit, "SPIR-V ID space exhausted")
        })?;
        Ok(id)
    }
    pub fn emit(&mut self, section: Section, op: Op, operands: &[u32]) -> Result<()> {
        let count = u16::try_from(operands.len() + 1).map_err(|_| {
            CompileError::new(
                FailureCategory::ResourceLimit,
                "SPIR-V instruction exceeds word count limit",
            )
        })?;
        let words = &mut self.sections[section as usize];
        words.push((u32::from(count) << 16) | op as u32);
        words.extend_from_slice(operands);
        Ok(())
    }
    pub fn result(&mut self, op: Op, ty: Id, args: &[u32]) -> Result<Id> {
        if ty == Id::NONE {
            return Err(CompileError::new(
                FailureCategory::Internal,
                "Missing SPIR-V result type",
            ));
        }
        let id = self.reserve_id()?;
        let mut words = Vec::with_capacity(args.len() + 2);
        words.extend([ty.0, id.0]);
        words.extend_from_slice(args);
        self.emit(Section::Functions, op, &words)?;
        Ok(id)
    }
    pub fn instruction(&mut self, op: Op, args: &[u32]) -> Result<()> {
        self.emit(Section::Functions, op, args)
    }
    fn intern(&mut self, op: Op, ty: Option<Id>, args: &[u32]) -> Result<Id> {
        if ty == Some(Id::NONE) {
            return Err(CompileError::new(
                FailureCategory::Internal,
                "Missing SPIR-V constant type",
            ));
        }
        let mut key = Vec::with_capacity(args.len() + 2);
        key.push(op as u32);
        if let Some(ty) = ty {
            key.push(ty.0);
        }
        key.extend_from_slice(args);
        if let Some(&id) = self.declarations.get(&key) {
            return Ok(id);
        }
        let id = self.reserve_id()?;
        let mut operands = Vec::with_capacity(args.len() + 2);
        if let Some(ty) = ty {
            operands.push(ty.0);
        }
        operands.push(id.0);
        operands.extend_from_slice(args);
        self.emit(Section::Declarations, op, &operands)?;
        self.declarations.insert(key, id);
        Ok(id)
    }
    pub fn type_void(&mut self) -> Result<Id> {
        self.intern(Op::TypeVoid, None, &[])
    }
    pub fn type_bool(&mut self) -> Result<Id> {
        self.intern(Op::TypeBool, None, &[])
    }
    pub fn type_int(&mut self, width: u32, signed: bool) -> Result<Id> {
        self.intern(Op::TypeInt, None, &[width, u32::from(signed)])
    }
    pub fn type_float(&mut self, width: u32) -> Result<Id> {
        self.intern(Op::TypeFloat, None, &[width])
    }
    pub fn type_vector(&mut self, element: Id, count: u32) -> Result<Id> {
        self.intern(Op::TypeVector, None, &[element.0, count])
    }
    pub fn type_pointer(&mut self, class: StorageClass, ty: Id) -> Result<Id> {
        self.intern(Op::TypePointer, None, &[class as u32, ty.0])
    }
    pub fn type_array(&mut self, element: Id, count: Id) -> Result<Id> {
        self.intern(Op::TypeArray, None, &[element.0, count.0])
    }
    pub fn type_runtime_array(&mut self, element: Id) -> Result<Id> {
        self.intern(Op::TypeRuntimeArray, None, &[element.0])
    }
    pub fn type_struct(&mut self, members: &[Id]) -> Result<Id> {
        // Decorations are part of a structure's meaning, so equal member lists
        // alone do not establish type identity.
        let id = self.reserve_id()?;
        let mut words = vec![id.0];
        words.extend(members.iter().map(|member| member.0));
        self.emit(Section::Declarations, Op::TypeStruct, &words)?;
        Ok(id)
    }
    pub fn type_function(&mut self, result: Id, parameters: &[Id]) -> Result<Id> {
        let mut args = vec![result.0];
        args.extend(parameters.iter().map(|i| i.0));
        self.intern(Op::TypeFunction, None, &args)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn type_image(
        &mut self,
        sampled: Id,
        dim: spirv::Dim,
        depth: bool,
        array: bool,
        multisample: bool,
        usage: u32,
        format: spirv::ImageFormat,
    ) -> Result<Id> {
        self.intern(
            Op::TypeImage,
            None,
            &[
                sampled.0,
                dim as u32,
                u32::from(depth),
                u32::from(array),
                u32::from(multisample),
                usage,
                format as u32,
            ],
        )
    }
    pub fn type_sampled_image(&mut self, image: Id) -> Result<Id> {
        self.intern(Op::TypeSampledImage, None, &[image.0])
    }
    pub fn constant(&mut self, ty: Id, words: &[u32]) -> Result<Id> {
        self.intern(Op::Constant, Some(ty), words)
    }
    pub fn constant_bool(&mut self, ty: Id, value: bool) -> Result<Id> {
        self.intern(
            if value {
                Op::ConstantTrue
            } else {
                Op::ConstantFalse
            },
            Some(ty),
            &[],
        )
    }
    pub fn constant_composite(&mut self, ty: Id, values: &[Id]) -> Result<Id> {
        self.intern(
            Op::ConstantComposite,
            Some(ty),
            &values.iter().map(|i| i.0).collect::<Vec<_>>(),
        )
    }
    pub fn constant_null(&mut self, ty: Id) -> Result<Id> {
        self.intern(Op::ConstantNull, Some(ty), &[])
    }
    pub fn undef(&mut self, ty: Id) -> Result<Id> {
        self.intern(Op::Undef, Some(ty), &[])
    }
    pub fn variable(
        &mut self,
        pointer: Id,
        class: StorageClass,
        initializer: Option<Id>,
    ) -> Result<Id> {
        let id = self.reserve_id()?;
        let mut args = vec![pointer.0, id.0, class as u32];
        if let Some(initializer) = initializer {
            args.push(initializer.0);
        }
        self.emit(
            if class == StorageClass::Function {
                Section::Functions
            } else {
                Section::Declarations
            },
            Op::Variable,
            &args,
        )?;
        Ok(id)
    }
    pub fn capability(&mut self, capability: Capability) -> Result<()> {
        if self.capabilities.insert(capability) {
            self.emit(Section::Capabilities, Op::Capability, &[capability as u32])?;
        }
        Ok(())
    }
    pub fn extension(&mut self, extension: &str) -> Result<()> {
        if self.extensions.insert(extension.to_owned()) {
            self.emit(Section::Extensions, Op::Extension, &string_words(extension))?;
        }
        Ok(())
    }
    pub fn memory_model(
        &mut self,
        addressing: spirv::AddressingModel,
        model: spirv::MemoryModel,
    ) -> Result<()> {
        self.sections[Section::MemoryModel as usize].clear();
        self.emit(
            Section::MemoryModel,
            Op::MemoryModel,
            &[addressing as u32, model as u32],
        )
    }
    pub fn decorate(&mut self, id: Id, decoration: Decoration, args: &[u32]) -> Result<()> {
        let mut words = vec![id.0, decoration as u32];
        words.extend_from_slice(args);
        self.emit(Section::Annotations, Op::Decorate, &words)
    }
    pub fn member_decorate(
        &mut self,
        id: Id,
        member: u32,
        decoration: Decoration,
        args: &[u32],
    ) -> Result<()> {
        let mut words = vec![id.0, member, decoration as u32];
        words.extend_from_slice(args);
        self.emit(Section::Annotations, Op::MemberDecorate, &words)
    }
    pub fn name(&mut self, id: Id, name: &str) -> Result<()> {
        let mut words = vec![id.0];
        words.extend(string_words(name));
        self.emit(Section::Debug, Op::Name, &words)
    }
    pub fn entry_point(
        &mut self,
        model: ExecutionModel,
        function: Id,
        name: &str,
        interfaces: &[Id],
    ) -> Result<()> {
        let mut words = vec![model as u32, function.0];
        words.extend(string_words(name));
        words.extend(interfaces.iter().map(|i| i.0));
        self.emit(Section::EntryPoints, Op::EntryPoint, &words)
    }
    pub fn execution_mode(
        &mut self,
        function: Id,
        mode: ExecutionMode,
        args: &[u32],
    ) -> Result<()> {
        let mut words = vec![function.0, mode as u32];
        words.extend_from_slice(args);
        self.emit(Section::ExecutionModes, Op::ExecutionMode, &words)
    }
    pub fn glsl(&mut self, ty: Id, instruction: spirv::GlslStd450Op, args: &[Id]) -> Result<Id> {
        if self.glsl == Id::NONE {
            self.glsl = self.reserve_id()?;
            let mut words = vec![self.glsl.0];
            words.extend(string_words("GLSL.std.450"));
            self.emit(Section::Imports, Op::ExtInstImport, &words)?;
        }
        let mut words = vec![self.glsl.0, instruction as u32];
        words.extend(args.iter().map(|i| i.0));
        self.result(Op::ExtInst, ty, &words)
    }
    pub fn begin_function(&mut self, result: Id, function_type: Id) -> Result<Id> {
        self.result(Op::Function, result, &[0, function_type.0])
    }
    pub fn parameter(&mut self, ty: Id) -> Result<Id> {
        self.result(Op::FunctionParameter, ty, &[])
    }
    pub fn label(&mut self, id: Id) -> Result<()> {
        self.instruction(Op::Label, &[id.0])
    }
    pub fn new_label(&mut self) -> Result<Id> {
        let id = self.reserve_id()?;
        self.label(id)?;
        Ok(id)
    }
    pub fn end_function(&mut self) -> Result<()> {
        self.instruction(Op::FunctionEnd, &[])
    }
    pub fn deferred_phi(&mut self, ty: Id, predecessors: &[Id]) -> Result<DeferredPhi> {
        let offset = self.sections[Section::Functions as usize].len();
        let args = predecessors
            .iter()
            .flat_map(|p| [0, p.0])
            .collect::<Vec<_>>();
        let id = self.result(Op::Phi, ty, &args)?;
        self.pending_phis.insert(offset);
        Ok(DeferredPhi {
            id,
            offset,
            count: predecessors.len(),
        })
    }
    pub fn patch_phi(&mut self, phi: DeferredPhi, values: &[Id]) -> Result<()> {
        if phi.count != values.len()
            || !self.pending_phis.contains(&phi.offset)
            || values.contains(&Id::NONE)
        {
            return Err(CompileError::new(
                FailureCategory::Internal,
                "Invalid deferred SPIR-V phi patch",
            ));
        }
        let words = &mut self.sections[Section::Functions as usize];
        for (i, value) in values.iter().enumerate() {
            words[phi.offset + 3 + i * 2] = value.0;
        }
        self.pending_phis.remove(&phi.offset);
        Ok(())
    }
    pub fn assemble(&self) -> Result<Vec<u32>> {
        if !self.pending_phis.is_empty() {
            return Err(CompileError::new(
                FailureCategory::Internal,
                "Unresolved deferred SPIR-V phis",
            ));
        }
        let mut words = Vec::with_capacity(5 + self.sections.iter().map(Vec::len).sum::<usize>());
        words.extend([spirv::MAGIC_NUMBER, self.version, 0, self.next_id, 0]);
        for section in &self.sections {
            words.extend_from_slice(section);
        }
        Ok(words)
    }
}
fn string_words(text: &str) -> Vec<u32> {
    let mut words = vec![0; (text.len() + 1).div_ceil(4)];
    for (i, &byte) in text.as_bytes().iter().enumerate() {
        words[i / 4] |= u32::from(byte) << ((i % 4) * 8);
    }
    words
}
