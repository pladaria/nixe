use nixe_shader_recompiler::spirv::{Builder, Id};
use spirv::{
    AddressingModel, Capability, ExecutionMode, ExecutionModel, GlslStd450Op, MemoryModel, Op,
};
fn instructions(words: &[u32]) -> Vec<(Op, &[u32])> {
    let mut result = Vec::new();
    let mut pos = 5;
    while pos < words.len() {
        let count = (words[pos] >> 16) as usize;
        assert!(count > 0 && pos + count <= words.len());
        result.push((
            Op::from_u32(words[pos] & 65535).unwrap(),
            &words[pos + 1..pos + count],
        ));
        pos += count;
    }
    result
}
#[test]
fn module_sections_constants_ids_and_glsl_import_are_stable() {
    let mut b = Builder::new(0x00010500);
    let float = b.type_float(32).unwrap();
    let uint = b.type_int(32, false).unwrap();
    let sint = b.type_int(32, true).unwrap();
    let zero = b.constant(float, &[0]).unwrap();
    let negative_zero = b.constant(float, &[0x80000000]).unwrap();
    assert_ne!(zero, negative_zero);
    assert_ne!(zero, b.constant(uint, &[0]).unwrap());
    assert_ne!(zero, b.constant(sint, &[0]).unwrap());
    assert_eq!(float, b.type_float(32).unwrap());
    assert_eq!(negative_zero, b.constant(float, &[0x80000000]).unwrap());
    let void = b.type_void().unwrap();
    let ft = b.type_function(void, &[]).unwrap();
    let main = b.begin_function(void, ft).unwrap();
    b.new_label().unwrap();
    b.glsl(float, GlslStd450Op::FAbs, &[negative_zero]).unwrap();
    b.glsl(float, GlslStd450Op::Floor, &[zero]).unwrap();
    b.instruction(Op::Return, &[]).unwrap();
    b.end_function().unwrap();
    b.capability(Capability::Shader).unwrap();
    b.capability(Capability::Shader).unwrap();
    b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450)
        .unwrap();
    b.entry_point(ExecutionModel::GLCompute, main, "main", &[])
        .unwrap();
    b.execution_mode(main, ExecutionMode::LocalSize, &[1, 1, 1])
        .unwrap();
    let words = b.assemble().unwrap();
    assert_eq!(&words[..3], &[spirv::MAGIC_NUMBER, 0x00010500, 0]);
    assert_eq!(words[4], 0);
    let inst = instructions(&words);
    assert_eq!(inst[0].0, Op::Capability);
    assert_eq!(inst[1].0, Op::ExtInstImport);
    assert_eq!(
        inst.iter()
            .filter(|(op, _)| *op == Op::ExtInstImport)
            .count(),
        1
    );
    assert_eq!(
        inst.iter().filter(|(op, _)| *op == Op::Capability).count(),
        1
    );
    let function = inst.iter().position(|(op, _)| *op == Op::Function).unwrap();
    let declarations = inst
        .iter()
        .rposition(|(op, _)| matches!(op, Op::TypeFloat | Op::TypeInt | Op::Constant))
        .unwrap();
    assert!(declarations < function);
    assert!(words[3] > main.0);
    assert_eq!(inst.last().unwrap().0, Op::FunctionEnd);
}
#[test]
fn deferred_phi_preserves_predecessor_order_without_placeholder_values() {
    let mut b = Builder::new(0x00010300);
    let uint = b.type_int(32, false).unwrap();
    let first = b.reserve_id().unwrap();
    let second = b.reserve_id().unwrap();
    let phi = b.deferred_phi(uint, &[first, second]).unwrap();
    let result = phi.id;
    assert!(b.assemble().is_err());
    let a = b.constant(uint, &[17]).unwrap();
    let c = b.constant(uint, &[42]).unwrap();
    b.patch_phi(phi, &[a, c]).unwrap();
    let words = b.assemble().unwrap();
    let inst = instructions(&words);
    let operands = inst.iter().find(|(op, _)| *op == Op::Phi).unwrap().1;
    assert_eq!(operands, &[uint.0, result.0, a.0, first.0, c.0, second.0]);
    assert!(!operands.contains(&Id::NONE.0));
}
