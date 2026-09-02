use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::mem::{align_of, offset_of, size_of};
use std::sync::Mutex;

use cranelift_codegen::ir::{
    self, AbiParam, Block, BlockArg, InstBuilder, MemFlagsData, Signature, SourceLoc, TrapCode,
    UserFuncName, condcodes::IntCC, types,
};
use cranelift_codegen::isa::CallConv;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module, default_libcall_names};
use nixe_cpu::decode::a64::{A64Instruction, control, fp_simd, integer, memory, system};
use nixe_cpu::execution::CpuControl;
use nixe_cpu::location::DecodedInstruction;
use nixe_cpu::semantics::a64::{
    SimdMemoryMode, simd_multiple_structure_shape, simd_single_structure_shape,
};
use nixe_cpu::semantics::conditions::Condition;
use nixe_memory::CpuMemoryBackend;
use nixe_memory::GuestVirtualAddress;

use nixe_cpu_direct_memory::{NativeMemoryAccess, NativeMemoryAccessKind};

use super::lookup::{
    DIRECT_LOOKUP_MASK, NATIVE_LOOKUP_HEAD_OFFSET, NATIVE_LOOKUP_NODE_ENTRY_OFFSET,
    NATIVE_LOOKUP_NODE_NEXT_OFFSET, NATIVE_LOOKUP_NODE_PC_OFFSET, NativeLookupSlot, lookup_salt,
};
use super::region::{BlockTerminator, NativeRegion};
use super::slow;
use super::{
    DirectJitError, EXIT_ARCHITECTURAL, EXIT_CONTROL, EXIT_DATA_FAULT, EXIT_DISPATCH,
    EXIT_INTERNAL, EXIT_RECONCILE, EXIT_UNSUPPORTED, NativeContext,
};

const GENERAL_REGISTER_COUNT: usize = 31;
const DIRECT_MEMORY_TRAP: TrapCode = TrapCode::unwrap_user(1);

mod a64;
mod a64_fp_simd;
mod a64_memory;
mod a64_system;

pub(super) type NativeGateway = unsafe extern "C" fn(*mut NativeContext, usize);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct CompilerPolicy {
    opt_level: &'static str,
    regalloc_algorithm: &'static str,
}

pub(super) const LCQ_COMPILER_POLICY: CompilerPolicy = CompilerPolicy {
    opt_level: "none",
    regalloc_algorithm: "single_pass",
};

pub(super) const HCQ_COMPILER_POLICY: CompilerPolicy = CompilerPolicy {
    opt_level: "speed",
    regalloc_algorithm: "backtracking",
};

#[derive(Clone, Copy)]
pub(super) struct CompilerRuntimeAddresses {
    gateway: NativeGateway,
    cold_calls: ColdCallBoundaries,
}

pub(super) struct CompilerRuntime {
    _module: Mutex<JITModule>,
    addresses: CompilerRuntimeAddresses,
}

impl CompilerRuntime {
    pub(super) fn new() -> Result<Self, DirectJitError> {
        let mut module = create_module(HCQ_COMPILER_POLICY)?;
        let mut context = module.make_context();
        let mut function_builder = FunctionBuilderContext::new();
        let gateway = compile_gateway(&mut module, &mut context, &mut function_builder)?;
        let cold_calls =
            compile_cold_call_boundaries(&mut module, &mut context, &mut function_builder)?;
        Ok(Self {
            _module: Mutex::new(module),
            addresses: CompilerRuntimeAddresses {
                gateway,
                cold_calls,
            },
        })
    }

    pub(super) const fn addresses(&self) -> CompilerRuntimeAddresses {
        self.addresses
    }

    pub(super) const fn gateway(&self) -> NativeGateway {
        self.addresses.gateway
    }
}

pub(super) struct CompiledRegion {
    pub(super) entry: usize,
    pub(super) native_bytes: usize,
    #[cfg(test)]
    pub(super) clif_instructions: usize,
    pub(super) fault_sites: Box<[CompiledFaultSite]>,
    #[cfg(test)]
    pub(super) deferred_register_loads: usize,
    #[cfg(test)]
    pub(super) exit_tail_count: usize,
}

#[derive(Clone, Copy)]
pub(super) struct Promotion {
    pub(super) hotness_address: usize,
    pub(super) node_address: usize,
}

pub(super) struct CompiledFaultSite {
    pub(super) native_start: u32,
    pub(super) native_end: u32,
    pub(super) access: NativeMemoryAccess,
}

struct PendingFaultSite {
    access: NativeMemoryAccess,
    source_location: SourceLoc,
}

struct PendingDirectMetadata {
    fault_sites: Vec<PendingFaultSite>,
    #[cfg(test)]
    exit_tail_count: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct DirtyState {
    integer: IntegerRegisterSet,
    vector: [bool; 32],
    fpsr: bool,
}

#[derive(Clone, Copy, Debug)]
struct RegisterLoadBlocks {
    integer: [Option<GuestVirtualAddress>; GENERAL_REGISTER_COUNT],
    sp: Option<GuestVirtualAddress>,
    vector: [Option<GuestVirtualAddress>; 32],
}

impl DirtyState {
    const fn all() -> Self {
        Self {
            integer: IntegerRegisterSet {
                x: [true; GENERAL_REGISTER_COUNT],
                sp: true,
            },
            vector: [true; 32],
            fpsr: true,
        }
    }

    fn merge(&mut self, other: Self) -> bool {
        let previous = *self;
        for (dirty, incoming) in self.integer.x.iter_mut().zip(other.integer.x) {
            *dirty |= incoming;
        }
        self.integer.sp |= other.integer.sp;
        for (dirty, incoming) in self.vector.iter_mut().zip(other.vector) {
            *dirty |= incoming;
        }
        self.fpsr |= other.fpsr;
        *self != previous
    }

    fn intersect(&mut self, other: Self) {
        for (dirty, incoming) in self.integer.x.iter_mut().zip(other.integer.x) {
            *dirty &= incoming;
        }
        self.integer.sp &= other.integer.sp;
        for (dirty, incoming) in self.vector.iter_mut().zip(other.vector) {
            *dirty &= incoming;
        }
        self.fpsr &= other.fpsr;
    }
}

pub(super) struct DirectCompiler {
    module: JITModule,
    context: cranelift_codegen::Context,
    function_builder: FunctionBuilderContext,
    runtime: CompilerRuntimeAddresses,
    next_function: u32,
    memory_backend: CpuMemoryBackend,
}

impl DirectCompiler {
    pub(super) fn new(
        policy: CompilerPolicy,
        runtime: CompilerRuntimeAddresses,
    ) -> Result<Self, DirectJitError> {
        let module = create_module(policy)?;
        let context = module.make_context();
        let function_builder = FunctionBuilderContext::new();
        Ok(Self {
            module,
            context,
            function_builder,
            runtime,
            next_function: 1,
            memory_backend: CpuMemoryBackend::Checked,
        })
    }
}

fn create_module(policy: CompilerPolicy) -> Result<JITModule, DirectJitError> {
    let isa_builder = cranelift_native::builder().map_err(|detail| {
        DirectJitError::unsupported(format!("direct JIT host ISA is unavailable: {detail}"))
    })?;
    let mut flags = settings::builder();
    for (name, value) in [
        ("preserve_frame_pointers", "true"),
        ("use_colocated_libcalls", "false"),
        ("is_pic", "false"),
        ("opt_level", policy.opt_level),
        ("regalloc_algorithm", policy.regalloc_algorithm),
        (
            "enable_verifier",
            if cfg!(any(debug_assertions, test)) {
                "true"
            } else {
                "false"
            },
        ),
    ] {
        flags.set(name, value).map_err(|error| {
            DirectJitError::internal(format!(
                "direct JIT Cranelift setting {name}={value} failed: {error}"
            ))
        })?;
    }
    let isa = isa_builder
        .finish(settings::Flags::new(flags))
        .map_err(|error| {
            DirectJitError::unsupported(format!(
                "direct JIT host ISA configuration failed: {error}"
            ))
        })?;
    Ok(JITModule::new(JITBuilder::with_isa(
        isa,
        default_libcall_names(),
    )))
}

impl DirectCompiler {
    pub(super) fn bind_memory_backend(
        &mut self,
        backend: CpuMemoryBackend,
    ) -> Result<(), DirectJitError> {
        if self.next_function != 1 && self.memory_backend != backend {
            return Err(DirectJitError::internal(
                "JIT memory backend changed after native code publication",
            ));
        }
        self.memory_backend = backend;
        Ok(())
    }

    pub(super) fn compile(
        &mut self,
        region: &NativeRegion,
        static_entry_cells: &[usize],
        promotion: Option<Promotion>,
    ) -> Result<CompiledRegion, DirectJitError> {
        self.context.func.signature = tail_signature();
        self.context.func.name = UserFuncName::user(1, self.next_function);
        let function_name = format!(
            "direct_{:08x}_{:016x}_{:016x}",
            self.next_function,
            region.key.address_space.get(),
            region.key.start.get()
        );
        let function = self
            .module
            .declare_function(&function_name, Linkage::Local, &self.context.func.signature)
            .map_err(module_error)?;
        self.context.func.name = UserFuncName::user(1, function.as_u32());
        self.next_function = self.next_function.checked_add(1).ok_or_else(|| {
            DirectJitError::capacity("direct JIT function identity space exhausted")
        })?;

        let pending_direct = {
            let builder = FunctionBuilder::new(&mut self.context.func, &mut self.function_builder);
            CraneliftTranslator::new(
                builder,
                region,
                static_entry_cells,
                self.module.target_config().default_call_conv,
                self.memory_backend == CpuMemoryBackend::LinuxDirect,
                self.runtime.cold_calls,
                promotion,
            )?
            .translate(self.module.target_config())?
        };
        #[cfg(test)]
        let clif_instructions = self
            .context
            .func
            .layout
            .blocks()
            .map(|block| self.context.func.layout.block_insts(block).count())
            .sum();
        self.module
            .define_function(function, &mut self.context)
            .map_err(module_error)?;
        let fault_sites = compile_fault_sites(
            self.context
                .compiled_code()
                .expect("defined direct JIT function retains compiled code"),
            &pending_direct.fault_sites,
        )?;
        let native_bytes = self
            .context
            .compiled_code()
            .expect("defined direct JIT function retains compiled code")
            .code_buffer()
            .len();
        self.module.finalize_definitions().map_err(module_error)?;
        let entry = self.module.get_finalized_function(function).addr();
        self.module.clear_context(&mut self.context);
        #[cfg(test)]
        let dirty_at_entry = dirty_states_at_entry(region);
        #[cfg(test)]
        let load_blocks = register_load_blocks(region, &dirty_at_entry);
        #[cfg(test)]
        let deferred_register_loads = load_blocks
            .integer
            .into_iter()
            .chain([load_blocks.sp])
            .chain(load_blocks.vector)
            .flatten()
            .filter(|block| *block != region.key.start)
            .count();
        Ok(CompiledRegion {
            entry,
            native_bytes,
            #[cfg(test)]
            clif_instructions,
            fault_sites: fault_sites.into_boxed_slice(),
            #[cfg(test)]
            deferred_register_loads,
            #[cfg(test)]
            exit_tail_count: pending_direct.exit_tail_count,
        })
    }
}

fn compile_gateway(
    module: &mut JITModule,
    context: &mut cranelift_codegen::Context,
    function_builder: &mut FunctionBuilderContext,
) -> Result<NativeGateway, DirectJitError> {
    let pointer = module.target_config().pointer_type();
    let mut signature = module.make_signature();
    signature.params.push(AbiParam::new(pointer));
    signature.params.push(AbiParam::new(pointer));
    let function = module
        .declare_function("direct_jit_gateway", Linkage::Local, &signature)
        .map_err(module_error)?;
    context.func.signature = signature;
    context.func.name = UserFuncName::user(0, function.as_u32());
    {
        let mut builder = FunctionBuilder::new(&mut context.func, function_builder);
        let entry = builder.create_block();
        builder.append_block_params_for_function_params(entry);
        builder.switch_to_block(entry);
        let native_context = builder.block_params(entry)[0];
        let target = builder.block_params(entry)[1];
        let tail = builder.import_signature(tail_signature());
        builder.ins().call_indirect(tail, target, &[native_context]);
        builder.ins().return_(&[]);
        builder.seal_all_blocks();
        builder.finalize(module.target_config());
    }
    module
        .define_function(function, context)
        .map_err(module_error)?;
    module.finalize_definitions().map_err(module_error)?;
    let entry = module.get_finalized_function(function);
    module.clear_context(context);
    Ok(unsafe { std::mem::transmute::<*const u8, NativeGateway>(entry) })
}

#[derive(Clone, Copy)]
struct ColdCallBoundaries {
    general_0: usize,
    general_1: usize,
    general_2: usize,
    general_3: usize,
    general_5: usize,
    fp_scalar: usize,
    fp_unary: usize,
    fp_binary: usize,
    fp_fused: usize,
    fp_compare: usize,
    materialize_fp: usize,
}

impl ColdCallBoundaries {
    fn general(self, arguments: usize) -> Result<usize, DirectJitError> {
        match arguments {
            0 => Ok(self.general_0),
            1 => Ok(self.general_1),
            2 => Ok(self.general_2),
            3 => Ok(self.general_3),
            5 => Ok(self.general_5),
            _ => Err(DirectJitError::internal(
                "direct JIT slow call has no typed cold boundary",
            )),
        }
    }

    fn fp(self, operands: usize, extra: usize) -> Result<usize, DirectJitError> {
        match (operands, extra) {
            (1, 0) => Ok(self.fp_scalar),
            (2, 0) => Ok(self.fp_unary),
            (4, 0) => Ok(self.fp_binary),
            (3, 0) => Ok(self.fp_fused),
            (2, 1) => Ok(self.fp_compare),
            _ => Err(DirectJitError::internal(
                "direct JIT exact FP call has no typed cold boundary",
            )),
        }
    }
}

fn compile_cold_call_boundaries(
    module: &mut JITModule,
    context: &mut cranelift_codegen::Context,
    function_builder: &mut FunctionBuilderContext,
) -> Result<ColdCallBoundaries, DirectJitError> {
    // These functions are compiled once with the runtime, not copied into each
    // guest region. Their active check also covers FP state inherited through a
    // native tail link from a preceding region.
    let general_0 = compile_cold_call_boundary(
        module,
        context,
        function_builder,
        "direct_cold_call_0",
        0,
        None,
    )?;
    let general_1 = compile_cold_call_boundary(
        module,
        context,
        function_builder,
        "direct_cold_call_1",
        1,
        None,
    )?;
    let general_2 = compile_cold_call_boundary(
        module,
        context,
        function_builder,
        "direct_cold_call_2",
        2,
        None,
    )?;
    let general_3 = compile_cold_call_boundary(
        module,
        context,
        function_builder,
        "direct_cold_call_3",
        3,
        None,
    )?;
    let general_5 = compile_cold_call_boundary(
        module,
        context,
        function_builder,
        "direct_cold_call_5",
        5,
        None,
    )?;
    let fp_scalar = compile_cold_call_boundary(
        module,
        context,
        function_builder,
        "direct_cold_fp_scalar",
        2,
        Some(1),
    )?;
    let fp_unary = compile_cold_call_boundary(
        module,
        context,
        function_builder,
        "direct_cold_fp_unary",
        3,
        Some(2),
    )?;
    let fp_binary = compile_cold_call_boundary(
        module,
        context,
        function_builder,
        "direct_cold_fp_binary",
        5,
        Some(4),
    )?;
    let fp_fused = compile_cold_call_boundary(
        module,
        context,
        function_builder,
        "direct_cold_fp_fused",
        4,
        Some(3),
    )?;
    let fp_compare = compile_cold_call_boundary(
        module,
        context,
        function_builder,
        "direct_cold_fp_compare",
        4,
        Some(2),
    )?;
    let materialize_fp = compile_fp_materialize_boundary(module, context, function_builder)?;
    module.finalize_definitions().map_err(module_error)?;
    Ok(ColdCallBoundaries {
        general_0: module.get_finalized_function(general_0) as usize,
        general_1: module.get_finalized_function(general_1) as usize,
        general_2: module.get_finalized_function(general_2) as usize,
        general_3: module.get_finalized_function(general_3) as usize,
        general_5: module.get_finalized_function(general_5) as usize,
        fp_scalar: module.get_finalized_function(fp_scalar) as usize,
        fp_unary: module.get_finalized_function(fp_unary) as usize,
        fp_binary: module.get_finalized_function(fp_binary) as usize,
        fp_fused: module.get_finalized_function(fp_fused) as usize,
        fp_compare: module.get_finalized_function(fp_compare) as usize,
        materialize_fp: module.get_finalized_function(materialize_fp) as usize,
    })
}

fn compile_cold_call_boundary(
    module: &mut JITModule,
    context: &mut cranelift_codegen::Context,
    function_builder: &mut FunctionBuilderContext,
    name: &str,
    argument_count: usize,
    fp_insert_at: Option<usize>,
) -> Result<cranelift_module::FuncId, DirectJitError> {
    let call_conv = module.target_config().default_call_conv;
    let mut boundary_signature = Signature::new(call_conv);
    boundary_signature.params.push(AbiParam::new(types::I64));
    boundary_signature.params.push(AbiParam::new(types::I64));
    boundary_signature
        .params
        .extend((0..argument_count).map(|_| AbiParam::new(types::I64)));
    boundary_signature.returns.push(AbiParam::new(types::I32));
    let function = module
        .declare_function(name, Linkage::Local, &boundary_signature)
        .map_err(module_error)?;
    context.func.signature = boundary_signature;
    context.func.name = UserFuncName::user(0, function.as_u32());
    {
        let mut builder = FunctionBuilder::new(&mut context.func, function_builder);
        let entry = builder.create_block();
        let suspend = builder.create_block();
        let invoke = builder.create_block();
        let resume = builder.create_block();
        let done = builder.create_block();
        builder.set_cold_block(suspend);
        builder.set_cold_block(resume);
        builder.append_block_params_for_function_params(entry);
        builder.switch_to_block(entry);
        let parameters = builder.block_params(entry).to_vec();
        let native_context = parameters[0];
        let target = parameters[1];
        let arguments = &parameters[2..];
        let active = builder.ins().load(
            types::I32,
            trusted_flags(),
            native_context,
            context_offset(
                offset_of!(NativeContext, host_fp) + offset_of!(super::fp_env::HostFpState, active),
            )?,
        );
        let active = builder.ins().icmp_imm_s(IntCC::NotEqual, active, 0);
        builder.ins().brif(active, suspend, &[], invoke, &[]);

        builder.switch_to_block(suspend);
        emit_context_leaf_call(
            &mut builder,
            call_conv,
            native_context,
            super::fp_env::suspend as *const () as usize,
        );
        builder.ins().jump(invoke, &[]);

        builder.switch_to_block(invoke);
        let mut semantic_arguments = Vec::with_capacity(argument_count + 3);
        semantic_arguments.push(native_context);
        if let Some(index) = fp_insert_at {
            semantic_arguments.extend_from_slice(&arguments[..index]);
            let fpcr = builder.ins().load(
                types::I32,
                trusted_flags(),
                native_context,
                context_offset(offset_of!(NativeContext, guest_fpcr))?,
            );
            let fpsr_pointer = builder.ins().load(
                types::I64,
                trusted_flags(),
                native_context,
                context_offset(offset_of!(NativeContext, fpsr))?,
            );
            let fpsr = builder
                .ins()
                .load(types::I32, trusted_flags(), fpsr_pointer, 0);
            semantic_arguments.push(builder.ins().uextend(types::I64, fpcr));
            semantic_arguments.push(builder.ins().uextend(types::I64, fpsr));
            semantic_arguments.extend_from_slice(&arguments[index..]);
        } else {
            semantic_arguments.extend_from_slice(arguments);
        }
        let mut semantic_signature = Signature::new(call_conv);
        semantic_signature
            .params
            .extend((0..semantic_arguments.len()).map(|_| AbiParam::new(types::I64)));
        let semantic_signature = builder.import_signature(semantic_signature);
        builder
            .ins()
            .call_indirect(semantic_signature, target, &semantic_arguments);
        let status = builder.ins().load(
            types::I32,
            trusted_flags(),
            native_context,
            context_offset(offset_of!(NativeContext, slow_status))?,
        );
        let succeeded = builder.ins().icmp_imm_s(IntCC::Equal, status, 0);
        let should_resume = builder.ins().band(active, succeeded);
        builder.ins().brif(should_resume, resume, &[], done, &[]);

        builder.switch_to_block(resume);
        emit_context_leaf_call(
            &mut builder,
            call_conv,
            native_context,
            super::fp_env::resume as *const () as usize,
        );
        builder.ins().jump(done, &[]);
        builder.switch_to_block(done);
        builder.ins().return_(&[status]);
        builder.seal_all_blocks();
        builder.finalize(module.target_config());
    }
    module
        .define_function(function, context)
        .map_err(module_error)?;
    module.clear_context(context);
    Ok(function)
}

fn compile_fp_materialize_boundary(
    module: &mut JITModule,
    context: &mut cranelift_codegen::Context,
    function_builder: &mut FunctionBuilderContext,
) -> Result<cranelift_module::FuncId, DirectJitError> {
    let call_conv = module.target_config().default_call_conv;
    let mut signature = Signature::new(call_conv);
    signature.params.push(AbiParam::new(types::I64));
    let function = module
        .declare_function("direct_materialize_fp", Linkage::Local, &signature)
        .map_err(module_error)?;
    context.func.signature = signature;
    context.func.name = UserFuncName::user(0, function.as_u32());
    {
        let mut builder = FunctionBuilder::new(&mut context.func, function_builder);
        let entry = builder.create_block();
        let materialize = builder.create_block();
        let done = builder.create_block();
        builder.set_cold_block(materialize);
        builder.append_block_params_for_function_params(entry);
        builder.switch_to_block(entry);
        let native_context = builder.block_params(entry)[0];
        let active = builder.ins().load(
            types::I32,
            trusted_flags(),
            native_context,
            context_offset(
                offset_of!(NativeContext, host_fp) + offset_of!(super::fp_env::HostFpState, active),
            )?,
        );
        let active = builder.ins().icmp_imm_s(IntCC::NotEqual, active, 0);
        builder.ins().brif(active, materialize, &[], done, &[]);
        builder.switch_to_block(materialize);
        emit_context_leaf_call(
            &mut builder,
            call_conv,
            native_context,
            super::fp_env::suspend as *const () as usize,
        );
        emit_context_leaf_call(
            &mut builder,
            call_conv,
            native_context,
            super::fp_env::resume as *const () as usize,
        );
        builder.ins().jump(done, &[]);
        builder.switch_to_block(done);
        builder.ins().return_(&[]);
        builder.seal_all_blocks();
        builder.finalize(module.target_config());
    }
    module
        .define_function(function, context)
        .map_err(module_error)?;
    module.clear_context(context);
    Ok(function)
}

fn emit_context_leaf_call(
    builder: &mut FunctionBuilder<'_>,
    call_conv: CallConv,
    native_context: ir::Value,
    function: usize,
) {
    let callee = builder.ins().iconst(types::I64, function as i64);
    let mut signature = Signature::new(call_conv);
    signature.params.push(AbiParam::new(types::I64));
    let signature = builder.import_signature(signature);
    builder
        .ins()
        .call_indirect(signature, callee, &[native_context]);
}

fn tail_signature() -> Signature {
    let mut signature = Signature::new(CallConv::Tail);
    signature.params.push(AbiParam::new(types::I64));
    signature
}

fn module_error(error: cranelift_module::ModuleError) -> DirectJitError {
    DirectJitError::internal(format!("direct JIT compilation failed: {error}"))
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum LazyFlags {
    Canonical(ir::Value),
    Packed(ir::Value),
    Add {
        lhs: ir::Value,
        rhs: ir::Value,
        result: ir::Value,
        width: u8,
    },
    Subtract {
        lhs: ir::Value,
        rhs: ir::Value,
        result: ir::Value,
        width: u8,
    },
    AddCarry {
        lhs: ir::Value,
        rhs: ir::Value,
        carry: ir::Value,
        result: ir::Value,
        width: u8,
    },
    SubtractCarry {
        lhs: ir::Value,
        rhs: ir::Value,
        carry: ir::Value,
        result: ir::Value,
        width: u8,
    },
    Logical {
        result: ir::Value,
        width: u8,
    },
    Conditional {
        predicate: ir::Value,
        when_true: Box<LazyFlags>,
        when_false: u32,
    },
}

impl LazyFlags {
    const fn dirty(&self) -> bool {
        !matches!(self, Self::Canonical(_))
    }
}

struct CraneliftTranslator<'a, 'region> {
    builder: FunctionBuilder<'a>,
    region: &'region NativeRegion,
    blocks: HashMap<GuestVirtualAddress, Block>,
    flags_at_entry: HashMap<GuestVirtualAddress, LazyFlags>,
    merged_flag_entries: HashSet<GuestVirtualAddress>,
    dirty_at_entry: HashMap<GuestVirtualAddress, DirtyState>,
    register_load_blocks: RegisterLoadBlocks,
    static_entry_cells: VecDeque<(GuestVirtualAddress, usize)>,
    exit_tails: BTreeMap<(u32, u32), Block>,
    slow_failure_tail: Option<Block>,
    fp_failure_tail: Option<Block>,
    prologue: Block,
    context: ir::Value,
    x: ir::Value,
    vector: ir::Value,
    sp: ir::Value,
    pc: ir::Value,
    nzcv: ir::Value,
    fpsr: ir::Value,
    native_fp_enabled: ir::Value,
    direct_base: ir::Value,
    direct_size: ir::Value,
    loader_return: ir::Value,
    control_pending: ir::Value,
    synchronization_counter: ir::Value,
    invalidation_signal: ir::Value,
    process_pending: ir::Value,
    initial_flags: ir::Value,
    packed_flags: Variable,
    fpsr_state: Variable,
    registers: [Option<Variable>; GENERAL_REGISTER_COUNT],
    stack_pointer: Option<Variable>,
    vector_registers: [Option<Variable>; 32],
    block_dirty_registers: [bool; GENERAL_REGISTER_COUNT],
    block_dirty_stack_pointer: bool,
    block_dirty_vector_registers: [bool; 32],
    block_dirty_fpsr: bool,
    fp_status_accessed: bool,
    uses_native_fp: bool,
    call_conv: CallConv,
    direct_memory: bool,
    cold_calls: ColdCallBoundaries,
    promotion: Option<Promotion>,
    fault_sites: Vec<PendingFaultSite>,
}

impl<'a, 'region> CraneliftTranslator<'a, 'region> {
    fn new(
        mut builder: FunctionBuilder<'a>,
        region: &'region NativeRegion,
        static_entry_cells: &[usize],
        call_conv: CallConv,
        direct_memory: bool,
        cold_calls: ColdCallBoundaries,
        promotion: Option<Promotion>,
    ) -> Result<Self, DirectJitError> {
        let expected_cells = region
            .external_exits
            .iter()
            .filter(|exit| exit.target.is_some())
            .count();
        if static_entry_cells.len() != expected_cells {
            return Err(DirectJitError::internal(
                "direct JIT static entry cells disagree with region exits",
            ));
        }
        let static_entry_cells = region
            .external_exits
            .iter()
            .filter_map(|exit| exit.target)
            .zip(static_entry_cells.iter().copied())
            .collect();
        let blocks = region
            .blocks
            .iter()
            .map(|block| (block.start.pc, builder.create_block()))
            .collect();
        let prologue = builder.create_block();
        builder.append_block_params_for_function_params(prologue);
        builder.switch_to_block(prologue);
        let placeholder = builder.ins().iconst(types::I64, 0);
        let packed_flags = builder.declare_var(types::I32);
        let fpsr_state = builder.declare_var(types::I32);
        let dirty_at_entry = dirty_states_at_entry(region);
        let register_load_blocks = register_load_blocks(region, &dirty_at_entry);
        let fp_status_accessed = fp_status_access(region).0;
        let uses_native_fp = region_uses_native_fp(region);
        Ok(Self {
            builder,
            region,
            blocks,
            flags_at_entry: HashMap::new(),
            merged_flag_entries: merged_flag_entries(region),
            dirty_at_entry,
            register_load_blocks,
            static_entry_cells,
            exit_tails: BTreeMap::new(),
            slow_failure_tail: None,
            fp_failure_tail: None,
            prologue,
            context: placeholder,
            x: placeholder,
            vector: placeholder,
            sp: placeholder,
            pc: placeholder,
            nzcv: placeholder,
            fpsr: placeholder,
            native_fp_enabled: placeholder,
            direct_base: placeholder,
            direct_size: placeholder,
            loader_return: placeholder,
            control_pending: placeholder,
            synchronization_counter: placeholder,
            invalidation_signal: placeholder,
            process_pending: placeholder,
            initial_flags: placeholder,
            packed_flags,
            fpsr_state,
            registers: [None; GENERAL_REGISTER_COUNT],
            stack_pointer: None,
            vector_registers: [None; 32],
            block_dirty_registers: [false; GENERAL_REGISTER_COUNT],
            block_dirty_stack_pointer: false,
            block_dirty_vector_registers: [false; 32],
            block_dirty_fpsr: false,
            fp_status_accessed,
            uses_native_fp,
            call_conv,
            direct_memory,
            cold_calls,
            promotion,
            fault_sites: Vec::new(),
        })
    }

    fn translate(
        mut self,
        target_config: cranelift_codegen::isa::TargetFrontendConfig,
    ) -> Result<PendingDirectMetadata, DirectJitError> {
        self.context = self.builder.block_params(self.prologue)[0];
        self.x = self.load_context_pointer(offset_of!(NativeContext, x))?;
        self.vector = self.load_context_pointer(offset_of!(NativeContext, vector))?;
        self.sp = self.load_context_pointer(offset_of!(NativeContext, sp))?;
        self.pc = self.load_context_pointer(offset_of!(NativeContext, pc))?;
        self.nzcv = self.load_context_pointer(offset_of!(NativeContext, nzcv))?;
        if self.uses_native_fp {
            let enabled =
                self.load_context(types::I32, offset_of!(NativeContext, native_fp_enabled))?;
            self.native_fp_enabled = self.builder.ins().icmp_imm_s(IntCC::NotEqual, enabled, 0);
        }
        if self.direct_memory {
            self.direct_base =
                self.load_context(types::I64, offset_of!(NativeContext, direct_base))?;
            self.direct_size =
                self.load_context(types::I64, offset_of!(NativeContext, direct_size))?;
        }
        self.loader_return =
            self.load_context(types::I64, offset_of!(NativeContext, loader_return))?;
        self.control_pending =
            self.load_context_pointer(offset_of!(NativeContext, control_pending))?;
        self.synchronization_counter =
            self.load_context_pointer(offset_of!(NativeContext, synchronization_counter))?;
        self.invalidation_signal =
            self.load_context_pointer(offset_of!(NativeContext, invalidation_signal))?;
        self.process_pending =
            self.load_context_pointer(offset_of!(NativeContext, process_pending))?;
        self.initial_flags = self
            .builder
            .ins()
            .load(types::I32, trusted_flags(), self.nzcv, 0);
        self.builder.def_var(self.packed_flags, self.initial_flags);
        if self.fp_status_accessed {
            self.fpsr = self.load_context_pointer(offset_of!(NativeContext, fpsr))?;
            let initial_fpsr = self
                .builder
                .ins()
                .load(types::I32, trusted_flags(), self.fpsr, 0);
            self.builder.def_var(self.fpsr_state, initial_fpsr);
        }
        let (read, written) = register_access(self.region);
        for index in 0..GENERAL_REGISTER_COUNT {
            if !read.x[index] && !written.x[index] {
                continue;
            }
            let variable = self.builder.declare_var(types::I64);
            if self.register_load_blocks.integer[index] == Some(self.region.key.start) {
                let value = self.builder.ins().load(
                    types::I64,
                    trusted_flags(),
                    self.x,
                    i32::try_from(index * size_of::<u64>())
                        .expect("architectural GPR offset fits i32"),
                );
                self.builder.def_var(variable, value);
            }
            self.registers[index] = Some(variable);
        }
        if read.sp || written.sp {
            let variable = self.builder.declare_var(types::I64);
            if self.register_load_blocks.sp == Some(self.region.key.start) {
                let value = self
                    .builder
                    .ins()
                    .load(types::I64, trusted_flags(), self.sp, 0);
                self.builder.def_var(variable, value);
            }
            self.stack_pointer = Some(variable);
        }

        let (read_vectors, written_vectors) = vector_register_access(self.region);
        for index in 0..read_vectors.len() {
            if !read_vectors[index] && !written_vectors[index] {
                continue;
            }
            let variable = self.builder.declare_var(types::I8X16);
            if self.register_load_blocks.vector[index] == Some(self.region.key.start) {
                let value = self.builder.ins().load(
                    types::I8X16,
                    trusted_flags(),
                    self.vector,
                    i32::try_from(index * size_of::<u128>())
                        .expect("architectural vector offset fits i32"),
                );
                self.builder.def_var(variable, value);
            }
            self.vector_registers[index] = Some(variable);
        }

        let entry_pc = self
            .builder
            .ins()
            .iconst(types::I64, self.region.key.start.get() as i64);
        let current =
            self.builder
                .ins()
                .atomic_load(types::I64, plain_flags(), self.invalidation_signal);
        let observed =
            self.load_context(types::I64, offset_of!(NativeContext, invalidation_cursor))?;
        let invalidated = self.builder.ins().icmp(IntCC::NotEqual, current, observed);
        let process =
            self.builder
                .ins()
                .atomic_load(types::I32, plain_flags(), self.process_pending);
        let shutting_down = self.builder.ins().icmp_imm_s(IntCC::NotEqual, process, 0);
        let reconcile = self.builder.ins().bor(invalidated, shutting_down);
        let reconcile_exit = self.cold_block();
        let control_check = self.builder.create_block();
        self.builder
            .ins()
            .brif(reconcile, reconcile_exit, &[], control_check, &[]);
        self.builder.switch_to_block(reconcile_exit);
        self.finish_exit_value(EXIT_RECONCILE, 0, entry_pc)?;
        self.builder.switch_to_block(control_check);
        let pending =
            self.builder
                .ins()
                .atomic_load(types::I32, plain_flags(), self.control_pending);
        let has_control = self.builder.ins().icmp_imm_s(IntCC::NotEqual, pending, 0);
        let control_exit = self.cold_block();
        let dispatch = self.builder.create_block();
        self.builder
            .ins()
            .brif(has_control, control_exit, &[], dispatch, &[]);
        self.builder.switch_to_block(control_exit);
        self.finish_exit_value(EXIT_CONTROL, 0, entry_pc)?;
        self.builder.switch_to_block(dispatch);
        if self.uses_native_fp {
            let active = self.load_context(
                types::I32,
                offset_of!(NativeContext, host_fp) + offset_of!(super::fp_env::HostFpState, active),
            )?;
            let active = self.builder.ins().icmp_imm_s(IntCC::NotEqual, active, 0);
            let uncommon_mode = self.invert_bit(self.native_fp_enabled);
            let skip_activation = self.builder.ins().bor(active, uncommon_mode);
            let ready = self.builder.create_block();
            let activate = self.cold_block();
            self.builder
                .ins()
                .brif(skip_activation, ready, &[], activate, &[]);
            self.builder.switch_to_block(activate);
            self.call_context_leaf(super::fp_env::ensure as *const () as usize);
            self.builder.ins().jump(ready, &[]);
            self.builder.switch_to_block(ready);
        }
        let primary = self.block(self.region.key.start)?;
        if let Some(promotion) = self.promotion {
            self.emit_promotion_check(primary, promotion)?;
        } else {
            self.builder.ins().jump(primary, &[]);
        }

        for block_index in 0..self.region.blocks.len() {
            self.translate_block(block_index)?;
        }
        self.emit_helper_failure_tails()?;
        self.emit_exit_tails()?;
        if !self.static_entry_cells.is_empty() {
            return Err(DirectJitError::internal(
                "direct JIT region left static entry cells unused",
            ));
        }
        self.builder.seal_all_blocks();
        self.builder.finalize(target_config);
        Ok(PendingDirectMetadata {
            fault_sites: self.fault_sites,
            #[cfg(test)]
            exit_tail_count: self.exit_tails.len(),
        })
    }

    fn emit_promotion_check(
        &mut self,
        primary: Block,
        promotion: Promotion,
    ) -> Result<(), DirectJitError> {
        if !promotion.hotness_address.is_multiple_of(align_of::<u32>()) {
            return Err(DirectJitError::internal(
                "direct JIT hotness counter is not naturally aligned",
            ));
        }
        let pointer = self.builder.func.dfg.value_type(self.context);
        let hotness = self
            .builder
            .ins()
            .iconst(pointer, promotion.hotness_address as i64);

        // This is the relaxed external access promised by
        // `NativeLookupNode::hotness_address`. The memory operations cannot
        // move and these aligned I32 accesses lower directly to the indivisible
        // host word load/store on x86-64 and AArch64. Cranelift's `atomic_load`
        // and `atomic_store` are sequentially consistent in 0.134; using them
        // here would add ordering, including an x86 `mfence`, which this
        // approximate counter neither needs nor permits on its hot path.
        let count = self
            .builder
            .ins()
            .load(types::I32, trusted_flags(), hotness, 0);
        let expires = self.builder.ins().icmp_imm_u(IntCC::Equal, count, 1);
        let decremented = self.builder.ins().iadd_imm_s(count, -1);
        let zero = self.builder.ins().iconst(types::I32, 0);
        let remaining = self.builder.ins().select(expires, zero, decremented);
        self.builder
            .ins()
            .store(trusted_flags(), remaining, hotness, 0);
        let promote = self.cold_block();
        self.builder.ins().brif(expires, promote, &[], primary, &[]);

        self.builder.switch_to_block(promote);
        let scheduler = self.load_context_pointer(offset_of!(NativeContext, hcq_scheduler))?;
        let invalidation_cursor =
            self.load_context(types::I64, offset_of!(NativeContext, invalidation_cursor))?;
        let node = self
            .builder
            .ins()
            .iconst(pointer, promotion.node_address as i64);
        let function = self
            .builder
            .ins()
            .iconst(pointer, super::request_hcq as *const () as usize as i64);
        let mut signature = Signature::new(self.call_conv);
        signature.params.push(AbiParam::new(pointer));
        signature.params.push(AbiParam::new(pointer));
        signature.params.push(AbiParam::new(types::I64));
        let signature = self.builder.import_signature(signature);
        self.builder.ins().call_indirect(
            signature,
            function,
            &[scheduler, node, invalidation_cursor],
        );
        self.builder.ins().jump(primary, &[]);
        Ok(())
    }

    fn record_direct_fault_state(
        &mut self,
        source: GuestVirtualAddress,
        size: u8,
        kind: NativeMemoryAccessKind,
        element_index: u8,
    ) {
        let access = NativeMemoryAccess {
            address_space: self.region.key.address_space,
            guest_pc: source,
            kind,
            size,
            element_index,
        };
        let source_location = direct_fault_source_location(self.fault_sites.len());
        self.builder.set_srcloc(source_location);
        self.fault_sites.push(PendingFaultSite {
            access,
            source_location,
        });
    }

    fn translate_block(&mut self, block_index: usize) -> Result<(), DirectJitError> {
        let record = &self.region.blocks[block_index];
        let block = self.block(record.start.pc)?;
        self.builder.switch_to_block(block);
        self.load_block_registers(record.start.pc)?;
        let mut flags = if self.merged_flag_entries.contains(&record.start.pc) {
            LazyFlags::Packed(self.builder.use_var(self.packed_flags))
        } else {
            self.flags_at_entry
                .get(&record.start.pc)
                .cloned()
                .unwrap_or(LazyFlags::Canonical(self.initial_flags))
        };
        let dirty = self
            .dirty_at_entry
            .get(&record.start.pc)
            .copied()
            .unwrap_or_default();
        self.block_dirty_registers = dirty.integer.x;
        self.block_dirty_stack_pointer = dirty.integer.sp;
        self.block_dirty_vector_registers = dirty.vector;
        self.block_dirty_fpsr = dirty.fpsr;
        for (index, decoded) in record.instructions.iter().enumerate() {
            self.builder
                .set_srcloc(source_location(self.region, decoded));
            let instruction =
                nixe_cpu::decode::a64::normalize(&decoded.instruction, decoded.encoding);
            let terminal = index + 1 == record.instructions.len();
            if terminal && matches!(record.terminator, BlockTerminator::Unsupported) {
                break;
            }
            match instruction {
                A64Instruction::Control(control::Instruction::Nop(_)) => {}
                A64Instruction::Integer(instruction) => {
                    if let Some(updated) =
                        self.emit_integer(decoded.location.pc, instruction, &flags)?
                    {
                        flags = updated;
                    }
                }
                A64Instruction::Memory(instruction) => {
                    self.emit_memory(decoded.location.pc, instruction, &flags)?;
                }
                A64Instruction::System(instruction) => {
                    self.emit_system(decoded.location.pc, instruction, &mut flags)?;
                }
                A64Instruction::FpSimd(instruction) => {
                    self.emit_fp_simd(decoded.location.pc, instruction, &mut flags)?;
                }
                A64Instruction::Control(
                    control::Instruction::BranchImmediate(_)
                    | control::Instruction::ConditionalBranch(_)
                    | control::Instruction::CompareBranch(_)
                    | control::Instruction::TestBranch(_)
                    | control::Instruction::BranchLinkImmediate(_)
                    | control::Instruction::BranchRegister(_)
                    | control::Instruction::SupervisorCall(_)
                    | control::Instruction::Breakpoint(_),
                ) => {}
                _ => {
                    return Err(unsupported_instruction(decoded));
                }
            }
        }
        self.emit_terminator(record, flags)
    }

    fn load_block_registers(&mut self, block: GuestVirtualAddress) -> Result<(), DirectJitError> {
        if block == self.region.key.start {
            return Ok(());
        }
        for index in 0..GENERAL_REGISTER_COUNT {
            if self.register_load_blocks.integer[index] != Some(block) {
                continue;
            }
            let variable = self.registers[index].ok_or_else(|| {
                DirectJitError::internal("lazy-loaded direct JIT GPR has no SSA variable")
            })?;
            let value = self.builder.ins().load(
                types::I64,
                trusted_flags(),
                self.x,
                i32::try_from(index * size_of::<u64>()).expect("architectural GPR offset fits i32"),
            );
            self.builder.def_var(variable, value);
        }
        if self.register_load_blocks.sp == Some(block) {
            let variable = self.stack_pointer.ok_or_else(|| {
                DirectJitError::internal("lazy-loaded direct JIT SP has no SSA variable")
            })?;
            let value = self
                .builder
                .ins()
                .load(types::I64, trusted_flags(), self.sp, 0);
            self.builder.def_var(variable, value);
        }
        for index in 0..self.vector_registers.len() {
            if self.register_load_blocks.vector[index] != Some(block) {
                continue;
            }
            let variable = self.vector_registers[index].ok_or_else(|| {
                DirectJitError::internal("lazy-loaded direct JIT vector has no SSA variable")
            })?;
            let value = self.builder.ins().load(
                types::I8X16,
                trusted_flags(),
                self.vector,
                i32::try_from(index * size_of::<u128>())
                    .expect("architectural vector offset fits i32"),
            );
            self.builder.def_var(variable, value);
        }
        Ok(())
    }

    fn emit_terminator(
        &mut self,
        record: &super::region::BasicBlockRecord,
        flags: LazyFlags,
    ) -> Result<(), DirectJitError> {
        // A64 branch targets, link updates, conditions, and register-target
        // alignment follow Arm DDI 0602 (2025-12), Base Instructions:
        // https://developer.arm.com/documentation/ddi0602/2025-12/Base-Instructions/B-cond--Branch-conditionally-
        // https://developer.arm.com/documentation/ddi0602/2025-12/Base-Instructions/BLR--Branch-with-link-to-register-
        let source = record
            .instructions
            .last()
            .map_or(record.start.pc, |instruction| instruction.location.pc);
        match record.terminator {
            BlockTerminator::Direct { target } => self.emit_edge(source, target, &flags),
            BlockTerminator::Conditional { taken, not_taken } => {
                let decoded = record
                    .instructions
                    .last()
                    .expect("conditional block contains its terminator");
                let condition = match nixe_cpu::decode::a64::normalize(
                    &decoded.instruction,
                    decoded.encoding,
                ) {
                    A64Instruction::Control(control::Instruction::ConditionalBranch(fields)) => {
                        self.emit_condition(Condition::from_encoding(fields.condition), &flags)
                    }
                    A64Instruction::Control(control::Instruction::CompareBranch(fields)) => {
                        let value = self.read_register(fields.rd, false)?;
                        let value = self.integer_value(value, fields.width_64);
                        let zero = self.builder.ins().icmp_imm_s(IntCC::Equal, value, 0);
                        if fields.nonzero {
                            self.invert_bit(zero)
                        } else {
                            zero
                        }
                    }
                    A64Instruction::Control(control::Instruction::TestBranch(fields)) => {
                        let value = self.read_register(fields.rd, false)?;
                        let shifted = self
                            .builder
                            .ins()
                            .ushr_imm_u(value, i64::from(fields.bit_index));
                        let bit = self.builder.ins().band_imm_u(shifted, 1);
                        let set = self.builder.ins().ireduce(types::I8, bit);
                        if fields.nonzero {
                            set
                        } else {
                            self.invert_bit(set)
                        }
                    }
                    _ => {
                        return Err(DirectJitError::internal(
                            "conditional region terminator lacks a conditional instruction",
                        ));
                    }
                };
                let taken_edge = self.builder.create_block();
                let not_taken_edge = self.builder.create_block();
                self.builder
                    .ins()
                    .brif(condition, taken_edge, &[], not_taken_edge, &[]);
                self.builder.switch_to_block(taken_edge);
                self.emit_edge(source, taken, &flags)?;
                self.builder.switch_to_block(not_taken_edge);
                self.emit_edge(source, not_taken, &flags)
            }
            BlockTerminator::Call {
                target,
                return_address,
            } => {
                let value = self
                    .builder
                    .ins()
                    .iconst(types::I64, return_address.get() as i64);
                self.write_register(30, value)?;
                self.emit_static_exit(target, &flags)
            }
            BlockTerminator::Indirect => {
                let decoded = record
                    .instructions
                    .last()
                    .expect("indirect block contains its terminator");
                let A64Instruction::Control(control::Instruction::BranchRegister(fields)) =
                    nixe_cpu::decode::a64::normalize(&decoded.instruction, decoded.encoding)
                else {
                    return Err(DirectJitError::internal(
                        "indirect region terminator lacks a register branch",
                    ));
                };
                let target = self.read_register(fields.rn, false)?;
                if fields.branch_register_key == 0xd63f_0000 {
                    let return_address = source.get().wrapping_add(4);
                    let value = self.builder.ins().iconst(types::I64, return_address as i64);
                    self.write_register(30, value)?;
                }
                self.emit_dynamic_exit(target, &flags)
            }
            BlockTerminator::Architectural { kind, syndrome } => {
                let class = match kind {
                    nixe_cpu::exception::ExceptionKind::SupervisorCall => 1_u32,
                    nixe_cpu::exception::ExceptionKind::Breakpoint => 2_u32,
                    _ => 0,
                };
                let immediate = syndrome.unwrap_or(0) as u32 & 0x00ff_ffff;
                let detail = (class << 24) | immediate;
                self.emit_exit(EXIT_ARCHITECTURAL, detail, source, &flags)
            }
            BlockTerminator::Unsupported => self.emit_exit(EXIT_UNSUPPORTED, 0, source, &flags),
            BlockTerminator::FpModeChange { continuation } => {
                self.commit_state(continuation, &flags)?;
                self.finish_exit(EXIT_DISPATCH, 0, continuation)
            }
            BlockTerminator::Limit { continuation } => self.emit_static_exit(continuation, &flags),
        }
    }

    fn emit_edge(
        &mut self,
        source: GuestVirtualAddress,
        target: GuestVirtualAddress,
        flags: &LazyFlags,
    ) -> Result<(), DirectJitError> {
        if let Some(block) = self.blocks.get(&target).copied() {
            self.propagate_flags(target, flags)?;
            if target.get() <= source.get() {
                self.emit_backedge_synchronization(target, flags, block)
            } else {
                self.builder.ins().jump(block, &[]);
                Ok(())
            }
        } else {
            self.emit_static_exit(target, flags)
        }
    }

    fn emit_backedge_synchronization(
        &mut self,
        target: GuestVirtualAddress,
        flags: &LazyFlags,
        destination: Block,
    ) -> Result<(), DirectJitError> {
        let count =
            self.builder
                .ins()
                .atomic_load(types::I32, plain_flags(), self.synchronization_counter);
        let expired = self.builder.ins().icmp_imm_s(IntCC::Equal, count, 0);
        let poll = self.cold_block();
        let proceed = self.builder.create_block();
        self.builder.ins().brif(expired, poll, &[], proceed, &[]);

        self.builder.switch_to_block(proceed);
        let next = self.builder.ins().iadd_imm_s(count, -1);
        self.builder
            .ins()
            .atomic_store(plain_flags(), next, self.synchronization_counter);
        self.builder.ins().jump(destination, &[]);

        self.builder.switch_to_block(poll);
        self.guard_reconciliation(target, flags)?;
        let reset = self
            .builder
            .ins()
            .iconst(types::I32, i64::from(CpuControl::SYNCHRONIZATION_INTERVAL));
        self.builder
            .ins()
            .atomic_store(plain_flags(), reset, self.synchronization_counter);
        self.emit_exit(EXIT_CONTROL, 0, target, flags)
    }

    fn emit_static_exit(
        &mut self,
        target_pc: GuestVirtualAddress,
        flags: &LazyFlags,
    ) -> Result<(), DirectJitError> {
        let Some((expected, entry_cell)) = self.static_entry_cells.pop_front() else {
            return Err(DirectJitError::internal(
                "direct JIT region is missing a static target entry cell",
            ));
        };
        if expected != target_pc {
            return Err(DirectJitError::internal(format!(
                "direct JIT static target order mismatch: expected={expected} actual={target_pc}"
            )));
        }
        self.commit_state(target_pc, flags)?;
        let cell = self.builder.ins().iconst(types::I64, entry_cell as i64);
        let target = self
            .builder
            .ins()
            .atomic_load(types::I64, plain_flags(), cell);
        let linked = self.builder.ins().icmp_imm_s(IntCC::NotEqual, target, 0);
        let target_pc_value = self
            .builder
            .ins()
            .iconst(types::I64, target_pc.get() as i64);
        let not_loader_return =
            self.builder
                .ins()
                .icmp(IntCC::NotEqual, target_pc_value, self.loader_return);
        let linked = self.builder.ins().band(linked, not_loader_return);
        let chain = self.builder.create_block();
        let miss = self.cold_block();
        self.builder.ins().brif(linked, chain, &[], miss, &[]);
        self.builder.switch_to_block(chain);
        let signature = self.builder.import_signature(tail_signature());
        self.builder
            .ins()
            .return_call_indirect(signature, target, &[self.context]);
        self.builder.switch_to_block(miss);
        self.finish_exit(EXIT_DISPATCH, 0, target_pc)
    }

    fn emit_dynamic_exit(
        &mut self,
        target: ir::Value,
        flags: &LazyFlags,
    ) -> Result<(), DirectJitError> {
        self.commit_registers()?;
        self.commit_flags(flags)?;
        self.builder
            .ins()
            .store(trusted_flags(), target, self.pc, 0);
        self.store_context(target, offset_of!(NativeContext, exit_pc))?;
        let lookup = self.load_context_pointer(offset_of!(NativeContext, native_lookup))?;
        let words = self.builder.ins().ushr_imm_u(target, 2);
        let middle = self.builder.ins().ushr_imm_u(words, 16);
        let high = self.builder.ins().ushr_imm_u(words, 32);
        let index = self.builder.ins().bxor(words, middle);
        let index = self.builder.ins().bxor(index, high);
        let salt = self
            .builder
            .ins()
            .iconst(types::I64, lookup_salt(self.region.key) as i64);
        let index = self.builder.ins().bxor(index, salt);
        let index = self
            .builder
            .ins()
            .band_imm_u(index, DIRECT_LOOKUP_MASK as i64);
        let slot_offset = self
            .builder
            .ins()
            .imul_imm_u(index, size_of::<NativeLookupSlot>() as i64);
        let slot = self.builder.ins().iadd(lookup, slot_offset);
        let head_address = self.builder.ins().iadd_imm_s(
            slot,
            i64::try_from(NATIVE_LOOKUP_HEAD_OFFSET).expect("native lookup head offset fits i64"),
        );
        let head = self
            .builder
            .ins()
            .atomic_load(types::I64, plain_flags(), head_address);
        let search = self.builder.create_block();
        self.builder.append_block_param(search, types::I64);
        let inspect = self.builder.create_block();
        let chain = self.builder.create_block();
        let miss = self.cold_block();
        self.builder.ins().jump(search, &[BlockArg::from(head)]);
        self.builder.switch_to_block(search);
        let node = self.builder.block_params(search)[0];
        let empty = self.builder.ins().icmp_imm_s(IntCC::Equal, node, 0);
        self.builder.ins().brif(empty, miss, &[], inspect, &[]);
        self.builder.switch_to_block(inspect);
        let entry_address = self.builder.ins().iadd_imm_s(
            node,
            i64::try_from(NATIVE_LOOKUP_NODE_ENTRY_OFFSET)
                .expect("native lookup node entry offset fits i64"),
        );
        let entry = self
            .builder
            .ins()
            .atomic_load(types::I64, plain_flags(), entry_address);
        let published_pc = self.builder.ins().load(
            types::I64,
            trusted_flags(),
            node,
            i32::try_from(NATIVE_LOOKUP_NODE_PC_OFFSET)
                .expect("native lookup node PC offset fits i32"),
        );
        let has_entry = self.builder.ins().icmp_imm_s(IntCC::NotEqual, entry, 0);
        let same_pc = self.builder.ins().icmp(IntCC::Equal, published_pc, target);
        let linked = self.builder.ins().band(has_entry, same_pc);
        let not_loader_return =
            self.builder
                .ins()
                .icmp(IntCC::NotEqual, target, self.loader_return);
        let linked = self.builder.ins().band(linked, not_loader_return);
        let next = self.builder.create_block();
        self.builder.ins().brif(linked, chain, &[], next, &[]);
        self.builder.switch_to_block(next);
        let next_node = self.builder.ins().load(
            types::I64,
            trusted_flags(),
            node,
            i32::try_from(NATIVE_LOOKUP_NODE_NEXT_OFFSET)
                .expect("native lookup node next offset fits i32"),
        );
        self.builder
            .ins()
            .jump(search, &[BlockArg::from(next_node)]);
        self.builder.switch_to_block(chain);
        let signature = self.builder.import_signature(tail_signature());
        self.builder
            .ins()
            .return_call_indirect(signature, entry, &[self.context]);
        self.builder.switch_to_block(miss);
        let kind = self
            .builder
            .ins()
            .iconst(types::I32, i64::from(EXIT_DISPATCH));
        self.store_context(kind, offset_of!(NativeContext, exit_kind))?;
        let detail = self.builder.ins().iconst(types::I32, 0);
        self.store_context(detail, offset_of!(NativeContext, exit_detail))?;
        self.builder.ins().return_(&[]);
        Ok(())
    }

    fn guard_reconciliation(
        &mut self,
        source: GuestVirtualAddress,
        flags: &LazyFlags,
    ) -> Result<(), DirectJitError> {
        let current =
            self.builder
                .ins()
                .atomic_load(types::I64, plain_flags(), self.invalidation_signal);
        let observed =
            self.load_context(types::I64, offset_of!(NativeContext, invalidation_cursor))?;
        let invalidated = self.builder.ins().icmp(IntCC::NotEqual, current, observed);
        let process =
            self.builder
                .ins()
                .atomic_load(types::I32, plain_flags(), self.process_pending);
        let shutting_down = self.builder.ins().icmp_imm_s(IntCC::NotEqual, process, 0);
        let pending = self.builder.ins().bor(invalidated, shutting_down);
        let exit = self.cold_block();
        let execute = self.builder.create_block();
        self.builder.ins().brif(pending, exit, &[], execute, &[]);
        self.builder.switch_to_block(exit);
        self.emit_exit(EXIT_RECONCILE, 0, source, flags)?;
        self.builder.switch_to_block(execute);
        Ok(())
    }

    fn emit_condition(&mut self, condition: Condition, flags: &LazyFlags) -> ir::Value {
        match condition {
            Condition::Eq => self.flag_z(flags),
            Condition::Ne => {
                let z = self.flag_z(flags);
                self.invert_bit(z)
            }
            Condition::Cs => self.flag_c(flags),
            Condition::Cc => {
                let c = self.flag_c(flags);
                self.invert_bit(c)
            }
            Condition::Mi => self.flag_n(flags),
            Condition::Pl => {
                let n = self.flag_n(flags);
                self.invert_bit(n)
            }
            Condition::Vs => self.flag_v(flags),
            Condition::Vc => {
                let v = self.flag_v(flags);
                self.invert_bit(v)
            }
            Condition::Hi => {
                let c = self.flag_c(flags);
                let z = self.flag_z(flags);
                let not_z = self.invert_bit(z);
                self.builder.ins().band(c, not_z)
            }
            Condition::Ls => {
                let c = self.flag_c(flags);
                let z = self.flag_z(flags);
                let not_c = self.invert_bit(c);
                self.builder.ins().bor(not_c, z)
            }
            Condition::Ge => {
                let n = self.flag_n(flags);
                let v = self.flag_v(flags);
                self.builder.ins().icmp(IntCC::Equal, n, v)
            }
            Condition::Lt => {
                let n = self.flag_n(flags);
                let v = self.flag_v(flags);
                self.builder.ins().icmp(IntCC::NotEqual, n, v)
            }
            Condition::Gt => {
                let z = self.flag_z(flags);
                let n = self.flag_n(flags);
                let v = self.flag_v(flags);
                let not_z = self.invert_bit(z);
                let equal = self.builder.ins().icmp(IntCC::Equal, n, v);
                self.builder.ins().band(not_z, equal)
            }
            Condition::Le => {
                let z = self.flag_z(flags);
                let n = self.flag_n(flags);
                let v = self.flag_v(flags);
                let different = self.builder.ins().icmp(IntCC::NotEqual, n, v);
                self.builder.ins().bor(z, different)
            }
            Condition::Al | Condition::Nv => self.builder.ins().iconst(types::I8, 1),
        }
    }

    fn invert_bit(&mut self, value: ir::Value) -> ir::Value {
        self.builder.ins().bxor_imm_u(value, 1)
    }

    fn flag_n(&mut self, flags: &LazyFlags) -> ir::Value {
        match flags {
            LazyFlags::Canonical(packed) | LazyFlags::Packed(packed) => {
                let shifted = self.builder.ins().ushr_imm_u(*packed, 31);
                self.builder.ins().ireduce(types::I8, shifted)
            }
            LazyFlags::Add { result, width, .. }
            | LazyFlags::Subtract { result, width, .. }
            | LazyFlags::AddCarry { result, width, .. }
            | LazyFlags::SubtractCarry { result, width, .. }
            | LazyFlags::Logical { result, width } => {
                let shifted = self
                    .builder
                    .ins()
                    .ushr_imm_u(*result, i64::from(*width - 1));
                self.builder.ins().ireduce(types::I8, shifted)
            }
            LazyFlags::Conditional {
                predicate,
                when_true,
                when_false,
            } => {
                let when_true = self.flag_n(when_true);
                let when_false = self
                    .builder
                    .ins()
                    .iconst(types::I8, i64::from((when_false >> 3) & 1));
                self.builder.ins().select(*predicate, when_true, when_false)
            }
        }
    }

    fn flag_z(&mut self, flags: &LazyFlags) -> ir::Value {
        match flags {
            LazyFlags::Canonical(packed) | LazyFlags::Packed(packed) => {
                let shifted = self.builder.ins().ushr_imm_u(*packed, 30);
                let bit = self.builder.ins().band_imm_u(shifted, 1);
                self.builder.ins().ireduce(types::I8, bit)
            }
            LazyFlags::Add { result, .. }
            | LazyFlags::Subtract { result, .. }
            | LazyFlags::AddCarry { result, .. }
            | LazyFlags::SubtractCarry { result, .. }
            | LazyFlags::Logical { result, .. } => {
                self.builder.ins().icmp_imm_s(IntCC::Equal, *result, 0)
            }
            LazyFlags::Conditional {
                predicate,
                when_true,
                when_false,
            } => {
                let when_true = self.flag_z(when_true);
                let when_false = self
                    .builder
                    .ins()
                    .iconst(types::I8, i64::from((when_false >> 2) & 1));
                self.builder.ins().select(*predicate, when_true, when_false)
            }
        }
    }

    fn flag_c(&mut self, flags: &LazyFlags) -> ir::Value {
        match flags {
            LazyFlags::Canonical(packed) | LazyFlags::Packed(packed) => {
                let shifted = self.builder.ins().ushr_imm_u(*packed, 29);
                let bit = self.builder.ins().band_imm_u(shifted, 1);
                self.builder.ins().ireduce(types::I8, bit)
            }
            LazyFlags::Add { lhs, result, .. } => {
                self.builder
                    .ins()
                    .icmp(IntCC::UnsignedLessThan, *result, *lhs)
            }
            LazyFlags::Subtract { lhs, rhs, .. } => {
                self.builder
                    .ins()
                    .icmp(IntCC::UnsignedGreaterThanOrEqual, *lhs, *rhs)
            }
            LazyFlags::AddCarry {
                lhs, carry, result, ..
            } => {
                let wrapped = self
                    .builder
                    .ins()
                    .icmp(IntCC::UnsignedLessThan, *result, *lhs);
                let equal = self.builder.ins().icmp(IntCC::Equal, *result, *lhs);
                let equal_with_carry = self.builder.ins().band(equal, *carry);
                self.builder.ins().bor(wrapped, equal_with_carry)
            }
            LazyFlags::SubtractCarry {
                lhs, rhs, carry, ..
            } => {
                let greater = self
                    .builder
                    .ins()
                    .icmp(IntCC::UnsignedGreaterThan, *lhs, *rhs);
                let equal = self.builder.ins().icmp(IntCC::Equal, *lhs, *rhs);
                let equal_with_carry = self.builder.ins().band(equal, *carry);
                self.builder.ins().bor(greater, equal_with_carry)
            }
            LazyFlags::Logical { .. } => self.builder.ins().iconst(types::I8, 0),
            LazyFlags::Conditional {
                predicate,
                when_true,
                when_false,
            } => {
                let when_true = self.flag_c(when_true);
                let when_false = self
                    .builder
                    .ins()
                    .iconst(types::I8, i64::from((when_false >> 1) & 1));
                self.builder.ins().select(*predicate, when_true, when_false)
            }
        }
    }

    fn flag_v(&mut self, flags: &LazyFlags) -> ir::Value {
        match flags {
            LazyFlags::Canonical(packed) | LazyFlags::Packed(packed) => {
                let shifted = self.builder.ins().ushr_imm_u(*packed, 28);
                let bit = self.builder.ins().band_imm_u(shifted, 1);
                self.builder.ins().ireduce(types::I8, bit)
            }
            LazyFlags::Add {
                lhs,
                rhs,
                result,
                width,
            } => {
                let xor_operands = self.builder.ins().bxor(*lhs, *rhs);
                let same_sign = self.builder.ins().bnot(xor_operands);
                let changed = self.builder.ins().bxor(*lhs, *result);
                let overflow = self.builder.ins().band(same_sign, changed);
                let shifted = self
                    .builder
                    .ins()
                    .ushr_imm_u(overflow, i64::from(*width - 1));
                self.builder.ins().ireduce(types::I8, shifted)
            }
            LazyFlags::Subtract {
                lhs,
                rhs,
                result,
                width,
            } => {
                let different = self.builder.ins().bxor(*lhs, *rhs);
                let changed = self.builder.ins().bxor(*lhs, *result);
                let overflow = self.builder.ins().band(different, changed);
                let shifted = self
                    .builder
                    .ins()
                    .ushr_imm_u(overflow, i64::from(*width - 1));
                self.builder.ins().ireduce(types::I8, shifted)
            }
            LazyFlags::AddCarry {
                lhs,
                rhs,
                result,
                width,
                ..
            } => {
                let xor_operands = self.builder.ins().bxor(*lhs, *rhs);
                let same_sign = self.builder.ins().bnot(xor_operands);
                let changed = self.builder.ins().bxor(*lhs, *result);
                let overflow = self.builder.ins().band(same_sign, changed);
                let shifted = self
                    .builder
                    .ins()
                    .ushr_imm_u(overflow, i64::from(*width - 1));
                self.builder.ins().ireduce(types::I8, shifted)
            }
            LazyFlags::SubtractCarry {
                lhs,
                rhs,
                result,
                width,
                ..
            } => {
                let different = self.builder.ins().bxor(*lhs, *rhs);
                let changed = self.builder.ins().bxor(*lhs, *result);
                let overflow = self.builder.ins().band(different, changed);
                let shifted = self
                    .builder
                    .ins()
                    .ushr_imm_u(overflow, i64::from(*width - 1));
                self.builder.ins().ireduce(types::I8, shifted)
            }
            LazyFlags::Logical { .. } => self.builder.ins().iconst(types::I8, 0),
            LazyFlags::Conditional {
                predicate,
                when_true,
                when_false,
            } => {
                let when_true = self.flag_v(when_true);
                let when_false = self
                    .builder
                    .ins()
                    .iconst(types::I8, i64::from(*when_false & 1));
                self.builder.ins().select(*predicate, when_true, when_false)
            }
        }
    }

    fn packed_flags(&mut self, flags: &LazyFlags) -> ir::Value {
        if let LazyFlags::Canonical(value) | LazyFlags::Packed(value) = flags {
            return *value;
        }
        let n = self.flag_n(flags);
        let z = self.flag_z(flags);
        let c = self.flag_c(flags);
        let v = self.flag_v(flags);
        let n = self.builder.ins().uextend(types::I32, n);
        let z = self.builder.ins().uextend(types::I32, z);
        let c = self.builder.ins().uextend(types::I32, c);
        let v = self.builder.ins().uextend(types::I32, v);
        let n = self.builder.ins().ishl_imm_u(n, 31);
        let z = self.builder.ins().ishl_imm_u(z, 30);
        let c = self.builder.ins().ishl_imm_u(c, 29);
        let v = self.builder.ins().ishl_imm_u(v, 28);
        let nz = self.builder.ins().bor(n, z);
        let cv = self.builder.ins().bor(c, v);
        self.builder.ins().bor(nz, cv)
    }

    fn propagate_flags(
        &mut self,
        target: GuestVirtualAddress,
        flags: &LazyFlags,
    ) -> Result<(), DirectJitError> {
        if self.merged_flag_entries.contains(&target) {
            let packed = self.packed_flags(flags);
            self.builder.def_var(self.packed_flags, packed);
        } else if let Some(existing) = self.flags_at_entry.get(&target) {
            if existing != flags {
                return Err(DirectJitError::internal(format!(
                    "direct JIT flag predecessor analysis missed a merge at {target}"
                )));
            }
        } else {
            self.flags_at_entry.insert(target, flags.clone());
        }
        Ok(())
    }

    fn read_register(
        &mut self,
        index: u8,
        register31_is_sp: bool,
    ) -> Result<ir::Value, DirectJitError> {
        if index == 31 && !register31_is_sp {
            return Ok(self.builder.ins().iconst(types::I64, 0));
        }
        let variable = if index == 31 {
            self.stack_pointer
        } else {
            self.registers[usize::from(index)]
        }
        .ok_or_else(|| {
            DirectJitError::internal(format!("direct JIT register X{index} was not planned"))
        })?;
        Ok(self.builder.use_var(variable))
    }

    fn write_register(&mut self, index: u8, value: ir::Value) -> Result<(), DirectJitError> {
        self.write_register_with_sp(index, false, value)
    }

    fn write_register_with_sp(
        &mut self,
        index: u8,
        register31_is_sp: bool,
        value: ir::Value,
    ) -> Result<(), DirectJitError> {
        if index == 31 && !register31_is_sp {
            return Ok(());
        }
        let variable = if index == 31 {
            self.stack_pointer
        } else {
            self.registers[usize::from(index)]
        }
        .ok_or_else(|| {
            DirectJitError::internal(format!("direct JIT register X{index} was not planned"))
        })?;
        self.builder.def_var(variable, value);
        if index == 31 {
            self.block_dirty_stack_pointer = true;
        } else {
            self.block_dirty_registers[usize::from(index)] = true;
        }
        Ok(())
    }

    fn commit_state(
        &mut self,
        target_pc: GuestVirtualAddress,
        flags: &LazyFlags,
    ) -> Result<(), DirectJitError> {
        self.commit_registers()?;
        self.commit_flags(flags)?;
        let target = self
            .builder
            .ins()
            .iconst(types::I64, target_pc.get() as i64);
        self.builder
            .ins()
            .store(trusted_flags(), target, self.pc, 0);
        Ok(())
    }

    fn commit_registers(&mut self) -> Result<(), DirectJitError> {
        for index in 0..self.block_dirty_registers.len() {
            if !self.block_dirty_registers[index] {
                continue;
            }
            let variable = self.registers[index].ok_or_else(|| {
                DirectJitError::internal("dirty direct JIT register has no SSA variable")
            })?;
            let value = self.builder.use_var(variable);
            self.builder.ins().store(
                trusted_flags(),
                value,
                self.x,
                i32::try_from(index * size_of::<u64>()).expect("architectural GPR offset fits i32"),
            );
        }
        if self.block_dirty_stack_pointer {
            let variable = self.stack_pointer.ok_or_else(|| {
                DirectJitError::internal("dirty direct JIT stack pointer has no SSA variable")
            })?;
            let value = self.builder.use_var(variable);
            self.builder.ins().store(trusted_flags(), value, self.sp, 0);
        }
        for index in 0..self.block_dirty_vector_registers.len() {
            if !self.block_dirty_vector_registers[index] {
                continue;
            }
            let variable = self.vector_registers[index].ok_or_else(|| {
                DirectJitError::internal("dirty direct JIT vector register has no SSA variable")
            })?;
            let value = self.builder.use_var(variable);
            self.builder.ins().store(
                trusted_flags(),
                value,
                self.vector,
                i32::try_from(index * size_of::<u128>())
                    .expect("architectural vector offset fits i32"),
            );
        }
        if self.block_dirty_fpsr {
            let fpsr = self.builder.use_var(self.fpsr_state);
            self.builder
                .ins()
                .store(trusted_flags(), fpsr, self.fpsr, 0);
        }
        Ok(())
    }

    fn commit_flags(&mut self, flags: &LazyFlags) -> Result<(), DirectJitError> {
        if flags.dirty() {
            let packed = self.packed_flags(flags);
            self.builder
                .ins()
                .store(trusted_flags(), packed, self.nzcv, 0);
        }
        Ok(())
    }

    fn emit_exit(
        &mut self,
        kind: u32,
        detail: u32,
        pc: GuestVirtualAddress,
        flags: &LazyFlags,
    ) -> Result<(), DirectJitError> {
        self.commit_state(pc, flags)?;
        self.finish_exit(kind, detail, pc)
    }

    fn finish_exit(
        &mut self,
        kind: u32,
        detail: u32,
        pc: GuestVirtualAddress,
    ) -> Result<(), DirectJitError> {
        let pc = self.builder.ins().iconst(types::I64, pc.get() as i64);
        self.finish_exit_value(kind, detail, pc)
    }

    fn finish_exit_value(
        &mut self,
        kind: u32,
        detail: u32,
        pc: ir::Value,
    ) -> Result<(), DirectJitError> {
        let tail = if let Some(tail) = self.exit_tails.get(&(kind, detail)).copied() {
            tail
        } else {
            let tail = self.builder.create_block();
            self.builder.append_block_param(tail, types::I64);
            self.builder.set_cold_block(tail);
            self.exit_tails.insert((kind, detail), tail);
            tail
        };
        self.builder.ins().jump(tail, &[BlockArg::from(pc)]);
        Ok(())
    }

    fn dispatch_slow_failure(&mut self, status: ir::Value, pc: GuestVirtualAddress) {
        let tail = if let Some(tail) = self.slow_failure_tail {
            tail
        } else {
            let tail = self.cold_block();
            self.builder.append_block_param(tail, types::I32);
            self.builder.append_block_param(tail, types::I64);
            self.slow_failure_tail = Some(tail);
            tail
        };
        let pc = self.builder.ins().iconst(types::I64, pc.get() as i64);
        self.builder
            .ins()
            .jump(tail, &[BlockArg::from(status), BlockArg::from(pc)]);
    }

    fn dispatch_fp_failure(&mut self, status: ir::Value, pc: GuestVirtualAddress) {
        let tail = if let Some(tail) = self.fp_failure_tail {
            tail
        } else {
            let tail = self.cold_block();
            self.builder.append_block_param(tail, types::I32);
            self.builder.append_block_param(tail, types::I64);
            self.fp_failure_tail = Some(tail);
            tail
        };
        let pc = self.builder.ins().iconst(types::I64, pc.get() as i64);
        self.builder
            .ins()
            .jump(tail, &[BlockArg::from(status), BlockArg::from(pc)]);
    }

    fn emit_helper_failure_tails(&mut self) -> Result<(), DirectJitError> {
        if let Some(tail) = self.slow_failure_tail {
            self.builder.switch_to_block(tail);
            let status = self.builder.block_params(tail)[0];
            let pc = self.builder.block_params(tail)[1];
            let data_fault = self.builder.ins().icmp_imm_s(IntCC::Equal, status, 1);
            let fault = self.cold_block();
            let internal = self.cold_block();
            self.builder
                .ins()
                .brif(data_fault, fault, &[], internal, &[]);
            self.builder.switch_to_block(fault);
            self.finish_exit_value(EXIT_DATA_FAULT, 0, pc)?;
            self.builder.switch_to_block(internal);
            self.finish_exit_value(EXIT_INTERNAL, 0, pc)?;
        }
        if let Some(tail) = self.fp_failure_tail {
            self.builder.switch_to_block(tail);
            let status = self.builder.block_params(tail)[0];
            let pc = self.builder.block_params(tail)[1];
            let trapped = self.builder.ins().icmp_imm_s(
                IntCC::Equal,
                status,
                i64::from(slow::STATUS_FP_TRAP),
            );
            let trap = self.cold_block();
            let internal = self.cold_block();
            self.builder.ins().brif(trapped, trap, &[], internal, &[]);
            self.builder.switch_to_block(trap);
            self.finish_exit_value(EXIT_ARCHITECTURAL, 6 << 24, pc)?;
            self.builder.switch_to_block(internal);
            self.finish_exit_value(EXIT_INTERNAL, 0, pc)?;
        }
        Ok(())
    }

    fn emit_exit_tails(&mut self) -> Result<(), DirectJitError> {
        let tails: Vec<_> = self
            .exit_tails
            .iter()
            .map(|(&(kind, detail), &block)| (kind, detail, block))
            .collect();
        for (kind, detail, block) in tails {
            self.builder.switch_to_block(block);
            let pc = self.builder.block_params(block)[0];
            self.store_context(pc, offset_of!(NativeContext, exit_pc))?;
            let kind = self.builder.ins().iconst(types::I32, i64::from(kind));
            self.store_context(kind, offset_of!(NativeContext, exit_kind))?;
            let detail = self.builder.ins().iconst(types::I32, i64::from(detail));
            self.store_context(detail, offset_of!(NativeContext, exit_detail))?;
            self.builder.ins().return_(&[]);
        }
        Ok(())
    }

    fn load_context_pointer(&mut self, offset: usize) -> Result<ir::Value, DirectJitError> {
        self.load_context(types::I64, offset)
    }

    fn load_context(&mut self, ty: ir::Type, offset: usize) -> Result<ir::Value, DirectJitError> {
        Ok(self
            .builder
            .ins()
            .load(ty, trusted_flags(), self.context, context_offset(offset)?))
    }

    fn store_context(&mut self, value: ir::Value, offset: usize) -> Result<(), DirectJitError> {
        self.builder.ins().store(
            trusted_flags(),
            value,
            self.context,
            context_offset(offset)?,
        );
        Ok(())
    }

    fn block(&self, pc: GuestVirtualAddress) -> Result<Block, DirectJitError> {
        self.blocks.get(&pc).copied().ok_or_else(|| {
            DirectJitError::internal(format!("direct JIT region has no block for {pc}"))
        })
    }

    fn cold_block(&mut self) -> Block {
        let block = self.builder.create_block();
        self.builder.set_cold_block(block);
        block
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct IntegerRegisterSet {
    x: [bool; GENERAL_REGISTER_COUNT],
    sp: bool,
}

fn register_access(region: &NativeRegion) -> (IntegerRegisterSet, IntegerRegisterSet) {
    let mut accessed = IntegerRegisterSet::default();
    let mut dirty = IntegerRegisterSet::default();
    for block in &region.blocks {
        for decoded in &block.instructions {
            match nixe_cpu::decode::a64::normalize(&decoded.instruction, decoded.encoding) {
                A64Instruction::Integer(instruction) => {
                    register_access_integer(instruction, &mut accessed, &mut dirty)
                }
                A64Instruction::Memory(instruction) => {
                    register_access_memory(instruction, &mut accessed, &mut dirty)
                }
                A64Instruction::System(instruction) => {
                    register_access_system(instruction, &mut accessed, &mut dirty)
                }
                A64Instruction::FpSimd(instruction) => {
                    register_access_fp_simd_general(instruction, &mut accessed, &mut dirty)
                }
                A64Instruction::Control(control::Instruction::BranchLinkImmediate(_)) => {
                    mark_write(&mut accessed, &mut dirty, 30, false);
                }
                A64Instruction::Control(control::Instruction::BranchRegister(fields)) => {
                    mark_read(&mut accessed, fields.rn, false);
                    if fields.branch_register_key == 0xd63f_0000 {
                        mark_write(&mut accessed, &mut dirty, 30, false);
                    }
                }
                A64Instruction::Control(
                    control::Instruction::CompareBranch(fields)
                    | control::Instruction::TestBranch(fields),
                ) => mark_read(&mut accessed, fields.rd, false),
                _ => {}
            }
        }
    }
    (accessed, dirty)
}

fn vector_register_access(region: &NativeRegion) -> ([bool; 32], [bool; 32]) {
    let mut accessed = [false; 32];
    let mut dirty = [false; 32];
    for block in &region.blocks {
        for decoded in &block.instructions {
            if let A64Instruction::FpSimd(instruction) =
                nixe_cpu::decode::a64::normalize(&decoded.instruction, decoded.encoding)
            {
                register_access_fp_simd_vector(instruction, &mut accessed, &mut dirty);
            }
        }
    }
    (accessed, dirty)
}

fn fp_status_access(region: &NativeRegion) -> (bool, bool) {
    let mut accessed = false;
    let mut dirty = false;
    for block in &region.blocks {
        for decoded in &block.instructions {
            match nixe_cpu::decode::a64::normalize(&decoded.instruction, decoded.encoding) {
                A64Instruction::FpSimd(instruction)
                    if a64_fp_simd::fp_lowering_disposition(instruction).accesses_status() =>
                {
                    accessed = true;
                    dirty = true;
                }
                A64Instruction::System(system::Instruction::ReadRegister(fields))
                    if fields.system_key == 0xd53b_4420 =>
                {
                    accessed = true;
                }
                A64Instruction::System(system::Instruction::WriteRegister(fields))
                    if fields.system_key == 0xd51b_4420 =>
                {
                    accessed = true;
                    dirty = true;
                }
                _ => {}
            }
        }
    }
    (accessed, dirty)
}

fn region_uses_native_fp(region: &NativeRegion) -> bool {
    region.blocks.iter().any(|block| {
        block.instructions.iter().any(|decoded| {
            let A64Instruction::FpSimd(instruction) =
                nixe_cpu::decode::a64::normalize(&decoded.instruction, decoded.encoding)
            else {
                return false;
            };
            a64_fp_simd::fp_lowering_disposition(instruction).uses_native_status()
        })
    })
}

fn register_access_memory(
    instruction: memory::Instruction,
    accessed: &mut IntegerRegisterSet,
    dirty: &mut IntegerRegisterSet,
) {
    use nixe_cpu::semantics::a64::{ScalarTransfer, pair_transfer, scalar_transfer};

    let fields = instruction.operands();
    if !matches!(instruction, memory::Instruction::Literal(_)) {
        mark_read(accessed, fields.rn, true);
    }
    if matches!(instruction, memory::Instruction::Register(_)) {
        mark_read(accessed, fields.rm, false);
    }
    match instruction {
        memory::Instruction::Literal(_) => mark_write(accessed, dirty, fields.rt, false),
        memory::Instruction::Unsigned(_)
        | memory::Instruction::Unscaled(_)
        | memory::Instruction::PostIndex(_)
        | memory::Instruction::PreIndex(_)
        | memory::Instruction::Register(_) => {
            match scalar_transfer(
                fields.opc,
                nixe_cpu::semantics::a64::memory_size(fields.size),
            ) {
                Some(ScalarTransfer::Store) => mark_read(accessed, fields.rt, false),
                Some(ScalarTransfer::Load(_)) => mark_write(accessed, dirty, fields.rt, false),
                None => {}
            }
            if matches!(
                instruction,
                memory::Instruction::PostIndex(_) | memory::Instruction::PreIndex(_)
            ) {
                mark_write(accessed, dirty, fields.rn, true);
            }
        }
        memory::Instruction::Pair(_) => {
            if pair_transfer(fields.size, fields.load).is_some() {
                if fields.load {
                    mark_write(accessed, dirty, fields.rt, false);
                    mark_write(accessed, dirty, fields.rt2, false);
                } else {
                    mark_read(accessed, fields.rt, false);
                    mark_read(accessed, fields.rt2, false);
                }
                if matches!(fields.mode, 1 | 3) {
                    mark_write(accessed, dirty, fields.rn, true);
                }
            }
        }
        memory::Instruction::LoadAcquire(_) | memory::Instruction::LoadExclusive(_) => {
            mark_write(accessed, dirty, fields.rt, false);
        }
        memory::Instruction::LoadExclusivePair(_) => {
            mark_write(accessed, dirty, fields.rt, false);
            mark_write(accessed, dirty, fields.rt2, false);
        }
        memory::Instruction::StoreRelease(_) => mark_read(accessed, fields.rt, false),
        memory::Instruction::StoreExclusive(_) => {
            mark_read(accessed, fields.rt, false);
            mark_write(accessed, dirty, fields.rm, false);
        }
        memory::Instruction::StoreExclusivePair(_) => {
            mark_read(accessed, fields.rt, false);
            mark_read(accessed, fields.rt2, false);
            mark_write(accessed, dirty, fields.rm, false);
        }
        memory::Instruction::AtomicReadModifyWrite(_) => {
            mark_read(accessed, fields.rm, false);
            mark_write(accessed, dirty, fields.rt, false);
        }
        memory::Instruction::CompareAndSwap(_) => {
            mark_read(accessed, fields.rm, false);
            mark_read(accessed, fields.rt, false);
            mark_write(accessed, dirty, fields.rm, false);
        }
        memory::Instruction::CompareAndSwapPair(_) => {
            mark_read(accessed, fields.rm, false);
            mark_read(accessed, fields.rm.wrapping_add(1), false);
            mark_read(accessed, fields.rt, false);
            mark_read(accessed, fields.rt.wrapping_add(1), false);
            mark_write(accessed, dirty, fields.rm, false);
            mark_write(accessed, dirty, fields.rm.wrapping_add(1), false);
        }
    }
}

fn register_access_system(
    instruction: system::Instruction,
    accessed: &mut IntegerRegisterSet,
    dirty: &mut IntegerRegisterSet,
) {
    let fields = instruction.operands();
    match instruction {
        system::Instruction::ReadRegister(_) => mark_write(accessed, dirty, fields.rt, false),
        system::Instruction::WriteRegister(_) => mark_read(accessed, fields.rt, false),
        system::Instruction::System(_) if fields.system_key != 0xd508_7500 => {
            mark_read(accessed, fields.rt, false);
        }
        system::Instruction::Hint(_)
        | system::Instruction::Barrier(_)
        | system::Instruction::ClearExclusive(_)
        | system::Instruction::System(_) => {}
    }
}

fn register_access_fp_simd_general(
    instruction: fp_simd::Instruction,
    accessed: &mut IntegerRegisterSet,
    dirty: &mut IntegerRegisterSet,
) {
    let fields = instruction.operands();
    match instruction {
        fp_simd::Instruction::DuplicateGeneral(_)
        | fp_simd::Instruction::InsertGeneral(_)
        | fp_simd::Instruction::MoveFromGeneral(_)
        | fp_simd::Instruction::SignedIntToFloat(_)
        | fp_simd::Instruction::UnsignedIntToFloat(_) => {
            mark_read(accessed, fields.rn, false);
        }
        fp_simd::Instruction::UnsignedMoveToGeneral(_)
        | fp_simd::Instruction::MoveToGeneral(_)
        | fp_simd::Instruction::FloatToSignedInt(_)
        | fp_simd::Instruction::FloatToUnsignedInt(_) => {
            mark_write(accessed, dirty, fields.rd, false);
        }
        fp_simd::Instruction::MemoryUnsigned(_)
        | fp_simd::Instruction::MemoryUnscaled(_)
        | fp_simd::Instruction::MemoryPostIndex(_)
        | fp_simd::Instruction::MemoryPreIndex(_)
        | fp_simd::Instruction::MemoryRegister(_)
        | fp_simd::Instruction::MemoryPair(_)
        | fp_simd::Instruction::MemoryMultipleStructures(_)
        | fp_simd::Instruction::MemoryMultipleStructuresPostIndex(_)
        | fp_simd::Instruction::MemorySingleStructure(_)
        | fp_simd::Instruction::MemorySingleStructurePostIndex(_) => {
            mark_read(accessed, fields.rn, true);
            if matches!(instruction, fp_simd::Instruction::MemoryRegister(_))
                || (matches!(
                    instruction,
                    fp_simd::Instruction::MemoryMultipleStructuresPostIndex(_)
                        | fp_simd::Instruction::MemorySingleStructurePostIndex(_)
                ) && fields.rm != 31)
            {
                mark_read(accessed, fields.rm, false);
            }
            if matches!(
                instruction,
                fp_simd::Instruction::MemoryPostIndex(_)
                    | fp_simd::Instruction::MemoryPreIndex(_)
                    | fp_simd::Instruction::MemoryMultipleStructuresPostIndex(_)
                    | fp_simd::Instruction::MemorySingleStructurePostIndex(_)
            ) || (matches!(instruction, fp_simd::Instruction::MemoryPair(_))
                && matches!(fields.mode, 1 | 3))
            {
                mark_write(accessed, dirty, fields.rn, true);
            }
        }
        _ => {}
    }
}

fn register_access_fp_simd_vector(
    instruction: fp_simd::Instruction,
    accessed: &mut [bool; 32],
    dirty: &mut [bool; 32],
) {
    let fields = instruction.operands();
    let read = |accessed: &mut [bool; 32], index: u8| {
        accessed[usize::from(index)] = true;
    };
    let write = |_accessed: &mut [bool; 32], dirty: &mut [bool; 32], index: u8| {
        dirty[usize::from(index)] = true;
    };
    match instruction {
        fp_simd::Instruction::UnsignedMoveToGeneral(_)
        | fp_simd::Instruction::MoveToGeneral(_)
        | fp_simd::Instruction::FloatToSignedInt(_)
        | fp_simd::Instruction::FloatToUnsignedInt(_)
        | fp_simd::Instruction::CompareZero(_) => read(accessed, fields.rn),
        fp_simd::Instruction::CompareRegister(_) | fp_simd::Instruction::ConditionalCompare(_) => {
            read(accessed, fields.rn);
            read(accessed, fields.rm);
        }
        fp_simd::Instruction::MemoryUnsigned(_)
        | fp_simd::Instruction::MemoryUnscaled(_)
        | fp_simd::Instruction::MemoryPostIndex(_)
        | fp_simd::Instruction::MemoryPreIndex(_)
        | fp_simd::Instruction::MemoryRegister(_) => {
            if fields.load {
                write(accessed, dirty, fields.rd);
            } else {
                read(accessed, fields.rd);
            }
        }
        fp_simd::Instruction::MemoryPair(_) => {
            for register in [fields.rd, fields.rt2] {
                if fields.load {
                    write(accessed, dirty, register);
                } else {
                    read(accessed, register);
                }
            }
        }
        fp_simd::Instruction::MemoryMultipleStructures(_)
        | fp_simd::Instruction::MemoryMultipleStructuresPostIndex(_) => {
            let shape = simd_multiple_structure_shape(fields)
                .expect("allocation validated the SIMD multiple-structure shape");
            for index in 0..shape.register_count() {
                let register = fields.rd.wrapping_add(index) & 31;
                if fields.load {
                    if shape.structure_registers > 1 {
                        read(accessed, register);
                    }
                    write(accessed, dirty, register);
                } else {
                    read(accessed, register);
                }
            }
        }
        fp_simd::Instruction::MemorySingleStructure(_)
        | fp_simd::Instruction::MemorySingleStructurePostIndex(_) => {
            let shape = simd_single_structure_shape(fields)
                .expect("allocation validated the SIMD single-structure shape");
            for index in 0..shape.register_count() {
                let register = fields.rd.wrapping_add(index) & 31;
                if !fields.load || matches!(shape.mode, SimdMemoryMode::Lane(_)) {
                    read(accessed, register);
                }
                if fields.load {
                    write(accessed, dirty, register);
                }
            }
        }
        fp_simd::Instruction::MoveFromGeneral(_)
        | fp_simd::Instruction::DuplicateGeneral(_)
        | fp_simd::Instruction::ModifiedImmediate(_)
        | fp_simd::Instruction::ScalarFloatImmediate(_)
        | fp_simd::Instruction::VectorFloatImmediate(_) => {
            if matches!(instruction, fp_simd::Instruction::ModifiedImmediate(_))
                && fields.cmode <= 11
                && fields.cmode & 1 != 0
                || matches!(instruction, fp_simd::Instruction::MoveFromGeneral(_))
                    && fields.size & 2 != 0
                    && fields.opc == 2
            {
                read(accessed, fields.rd);
            }
            write(accessed, dirty, fields.rd);
        }
        fp_simd::Instruction::InsertElement(_) => {
            read(accessed, fields.rn);
            read(accessed, fields.rd);
            write(accessed, dirty, fields.rd);
        }
        fp_simd::Instruction::InsertGeneral(_)
        | fp_simd::Instruction::ShiftRightNarrow(_)
        | fp_simd::Instruction::ExtractNarrow(_) => {
            read(accessed, fields.rn);
            if !matches!(instruction, fp_simd::Instruction::InsertGeneral(_)) && fields.vector_128 {
                read(accessed, fields.rd);
            }
            if matches!(instruction, fp_simd::Instruction::InsertGeneral(_)) {
                read(accessed, fields.rd);
            }
            write(accessed, dirty, fields.rd);
        }
        fp_simd::Instruction::Bitwise(_) => {
            read(accessed, fields.rn);
            read(accessed, fields.rm);
            read(accessed, fields.rd);
            write(accessed, dirty, fields.rd);
        }
        fp_simd::Instruction::Integer(_)
        | fp_simd::Instruction::IntegerCompare(_)
        | fp_simd::Instruction::IntegerPairwise(_)
        | fp_simd::Instruction::IntegerMinMax(_)
        | fp_simd::Instruction::PermuteTwoSource(_)
        | fp_simd::Instruction::Extract(_)
        | fp_simd::Instruction::VectorSignedShiftRegister(_)
        | fp_simd::Instruction::VectorUnsignedShiftRegister(_)
        | fp_simd::Instruction::VectorFloatDivide(_)
        | fp_simd::Instruction::VectorFloatMultiplyElement(_)
        | fp_simd::Instruction::ScalarFloatDivide(_)
        | fp_simd::Instruction::ScalarFloatAdd(_)
        | fp_simd::Instruction::ScalarFloatMultiply(_)
        | fp_simd::Instruction::ScalarFloatConditionalSelect(_) => {
            read(accessed, fields.rn);
            read(accessed, fields.rm);
            write(accessed, dirty, fields.rd);
        }
        fp_simd::Instruction::ScalarFloatFusedMultiplyAdd(_) => {
            read(accessed, fields.rn);
            read(accessed, fields.rm);
            read(accessed, fields.ra);
            write(accessed, dirty, fields.rd);
        }
        fp_simd::Instruction::ScalarMove(_)
        | fp_simd::Instruction::ScalarAbsolute(_)
        | fp_simd::Instruction::ScalarNegate(_)
        | fp_simd::Instruction::VectorFloatAbsolute(_)
        | fp_simd::Instruction::VectorFloatNegate(_)
        | fp_simd::Instruction::DuplicateElement(_)
        | fp_simd::Instruction::ScalarShiftRightImmediate(_)
        | fp_simd::Instruction::VectorShiftRightImmediate(_)
        | fp_simd::Instruction::ScalarShiftLeftImmediate(_)
        | fp_simd::Instruction::VectorShiftLeftImmediate(_)
        | fp_simd::Instruction::ShiftLeftLong(_)
        | fp_simd::Instruction::CountBits(_)
        | fp_simd::Instruction::AddAcrossVector(_)
        | fp_simd::Instruction::VectorSignedIntToFloat(_)
        | fp_simd::Instruction::VectorUnsignedIntToFloat(_)
        | fp_simd::Instruction::ScalarVectorSignedIntToFloat(_)
        | fp_simd::Instruction::ScalarVectorUnsignedIntToFloat(_)
        | fp_simd::Instruction::ScalarFloatConvert(_)
        | fp_simd::Instruction::ScalarFloatRound(_)
        | fp_simd::Instruction::ScalarFloatSquareRoot(_) => {
            read(accessed, fields.rn);
            write(accessed, dirty, fields.rd);
        }
        fp_simd::Instruction::SignedIntToFloat(_) | fp_simd::Instruction::UnsignedIntToFloat(_) => {
            write(accessed, dirty, fields.rd);
        }
    }
}

fn merged_flag_entries(region: &NativeRegion) -> HashSet<GuestVirtualAddress> {
    let starts: HashSet<_> = region.blocks.iter().map(|block| block.start.pc).collect();
    let mut predecessors: HashMap<_, _> = starts
        .iter()
        .copied()
        .map(|start| (start, 1_usize))
        .collect();
    for block in &region.blocks {
        let mut record = |target| {
            if starts.contains(&target) {
                *predecessors.entry(target).or_default() += 1;
            }
        };
        match block.terminator {
            BlockTerminator::Direct { target } => record(target),
            BlockTerminator::Conditional { taken, not_taken } => {
                record(taken);
                record(not_taken);
            }
            BlockTerminator::Call { .. }
            | BlockTerminator::Indirect
            | BlockTerminator::Architectural { .. }
            | BlockTerminator::Unsupported
            | BlockTerminator::FpModeChange { .. }
            | BlockTerminator::Limit { .. } => {}
        }
    }
    predecessors
        .into_iter()
        .filter_map(|(target, count)| (count > 1).then_some(target))
        .collect()
}

fn register_load_blocks(
    region: &NativeRegion,
    dirty_at_entry: &HashMap<GuestVirtualAddress, DirtyState>,
) -> RegisterLoadBlocks {
    let primary = region.key.start;
    let dominators = block_dominators(region);
    let (_, integer_written) = register_access(region);
    let (_, vector_written) = vector_register_access(region);
    let conditionally_dirty = conditionally_dirty_at_entry(region, dirty_at_entry);
    let mut integer_uses: [Vec<GuestVirtualAddress>; GENERAL_REGISTER_COUNT] =
        std::array::from_fn(|_| Vec::new());
    let mut sp_uses = Vec::new();
    let mut vector_uses: [Vec<GuestVirtualAddress>; 32] = std::array::from_fn(|_| Vec::new());
    for block in &region.blocks {
        let (integer_read, _, vector_read, _, _, _) = block_register_access(block);
        for (index, read) in integer_read.x.into_iter().enumerate() {
            if read {
                integer_uses[index].push(block.start.pc);
            }
        }
        if integer_read.sp {
            sp_uses.push(block.start.pc);
        }
        for (index, read) in vector_read.into_iter().enumerate() {
            if read {
                vector_uses[index].push(block.start.pc);
            }
        }
    }
    let integer = std::array::from_fn(|index| {
        if integer_uses[index].is_empty() && !conditionally_dirty.integer.x[index] {
            None
        } else if integer_written.x[index] {
            Some(primary)
        } else {
            common_dominator(&integer_uses[index], &dominators)
        }
    });
    let sp = if sp_uses.is_empty() && !conditionally_dirty.integer.sp {
        None
    } else if integer_written.sp {
        Some(primary)
    } else {
        common_dominator(&sp_uses, &dominators)
    };
    let vector = std::array::from_fn(|index| {
        if vector_uses[index].is_empty() && !conditionally_dirty.vector[index] {
            None
        } else if vector_written[index] {
            Some(primary)
        } else {
            common_dominator(&vector_uses[index], &dominators)
        }
    });
    RegisterLoadBlocks {
        integer,
        sp,
        vector,
    }
}

fn block_dominators(
    region: &NativeRegion,
) -> HashMap<GuestVirtualAddress, HashSet<GuestVirtualAddress>> {
    let starts: HashSet<_> = region.blocks.iter().map(|block| block.start.pc).collect();
    let mut predecessors: HashMap<_, Vec<_>> = starts
        .iter()
        .copied()
        .map(|start| (start, Vec::new()))
        .collect();
    for block in &region.blocks {
        let mut record = |target| {
            if let Some(incoming) = predecessors.get_mut(&target) {
                incoming.push(block.start.pc);
            }
        };
        match block.terminator {
            BlockTerminator::Direct { target } => record(target),
            BlockTerminator::Conditional { taken, not_taken } => {
                record(taken);
                record(not_taken);
            }
            BlockTerminator::Call { .. }
            | BlockTerminator::Indirect
            | BlockTerminator::Architectural { .. }
            | BlockTerminator::Unsupported
            | BlockTerminator::FpModeChange { .. }
            | BlockTerminator::Limit { .. } => {}
        }
    }
    let primary = region.key.start;
    let mut dominators: HashMap<_, _> = starts
        .iter()
        .copied()
        .map(|start| {
            let set = if start == primary {
                HashSet::from([primary])
            } else {
                starts.clone()
            };
            (start, set)
        })
        .collect();
    loop {
        let previous = dominators.clone();
        let mut changed = false;
        for &start in &starts {
            if start == primary {
                continue;
            }
            let incoming = &predecessors[&start];
            let mut next = incoming
                .first()
                .map_or_else(HashSet::new, |first| previous[first].clone());
            for predecessor in incoming.iter().skip(1) {
                next.retain(|candidate| previous[predecessor].contains(candidate));
            }
            next.insert(start);
            if next != dominators[&start] {
                dominators.insert(start, next);
                changed = true;
            }
        }
        if !changed {
            return dominators;
        }
    }
}

fn common_dominator(
    uses: &[GuestVirtualAddress],
    dominators: &HashMap<GuestVirtualAddress, HashSet<GuestVirtualAddress>>,
) -> Option<GuestVirtualAddress> {
    let first = *uses.first()?;
    let mut common = dominators.get(&first)?.clone();
    for used in &uses[1..] {
        common.retain(|candidate| dominators[used].contains(candidate));
    }
    common
        .into_iter()
        .max_by_key(|candidate| dominators[candidate].len())
}

fn dirty_states_at_entry(region: &NativeRegion) -> HashMap<GuestVirtualAddress, DirtyState> {
    let starts: HashSet<_> = region.blocks.iter().map(|block| block.start.pc).collect();
    let mut states: HashMap<_, _> = starts
        .iter()
        .copied()
        .map(|start| (start, DirtyState::default()))
        .collect();
    let mut changed = true;
    while changed {
        changed = false;
        for block in &region.blocks {
            let mut outgoing = states[&block.start.pc];
            outgoing.merge(block_dirty_state(block));
            let mut propagate = |target| {
                if let Some(state) = states.get_mut(&target) {
                    changed |= state.merge(outgoing);
                }
            };
            match block.terminator {
                BlockTerminator::Direct { target } => propagate(target),
                BlockTerminator::Conditional { taken, not_taken } => {
                    propagate(taken);
                    propagate(not_taken);
                }
                BlockTerminator::Call { .. }
                | BlockTerminator::Indirect
                | BlockTerminator::Architectural { .. }
                | BlockTerminator::Unsupported
                | BlockTerminator::FpModeChange { .. }
                | BlockTerminator::Limit { .. } => {}
            }
        }
    }
    states
}

fn conditionally_dirty_at_entry(
    region: &NativeRegion,
    may: &HashMap<GuestVirtualAddress, DirtyState>,
) -> DirtyState {
    let starts: HashSet<_> = region.blocks.iter().map(|block| block.start.pc).collect();
    let block_dirty: HashMap<_, _> = region
        .blocks
        .iter()
        .map(|block| (block.start.pc, block_dirty_state(block)))
        .collect();
    let primary = region.key.start;
    let mut predecessors: HashMap<_, Vec<_>> = starts
        .iter()
        .copied()
        .map(|start| (start, Vec::new()))
        .collect();
    for block in &region.blocks {
        let mut record = |target| {
            if let Some(incoming) = predecessors.get_mut(&target) {
                incoming.push(block.start.pc);
            }
        };
        match block.terminator {
            BlockTerminator::Direct { target } => record(target),
            BlockTerminator::Conditional { taken, not_taken } => {
                record(taken);
                record(not_taken);
            }
            BlockTerminator::Call { .. }
            | BlockTerminator::Indirect
            | BlockTerminator::Architectural { .. }
            | BlockTerminator::Unsupported
            | BlockTerminator::FpModeChange { .. }
            | BlockTerminator::Limit { .. } => {}
        }
    }

    let mut must: HashMap<_, _> = starts
        .iter()
        .copied()
        .map(|start| {
            (
                start,
                if start == primary {
                    DirtyState::default()
                } else {
                    DirtyState::all()
                },
            )
        })
        .collect();
    loop {
        let previous = must.clone();
        let mut changed = false;
        for &start in &starts {
            if start == primary {
                continue;
            }
            let mut incoming = predecessors[&start].iter();
            let mut next = incoming
                .next()
                .map_or_else(DirtyState::default, |predecessor| {
                    let mut state = previous[predecessor];
                    state.merge(block_dirty[predecessor]);
                    state
                });
            for predecessor in incoming {
                let mut state = previous[predecessor];
                state.merge(block_dirty[predecessor]);
                next.intersect(state);
            }
            if next != must[&start] {
                must.insert(start, next);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    let mut conditional = DirtyState::default();
    for start in starts {
        let mut partial = may[&start];
        let definitely = must[&start];
        for (value, definite) in partial.integer.x.iter_mut().zip(definitely.integer.x) {
            *value &= !definite;
        }
        partial.integer.sp &= !definitely.integer.sp;
        for (value, definite) in partial.vector.iter_mut().zip(definitely.vector) {
            *value &= !definite;
        }
        partial.fpsr &= !definitely.fpsr;
        conditional.merge(partial);
    }
    conditional
}

fn block_dirty_state(block: &super::region::BasicBlockRecord) -> DirtyState {
    let (_, integer_dirty, _, vector_dirty, _, fpsr_dirty) = block_register_access(block);
    DirtyState {
        integer: integer_dirty,
        vector: vector_dirty,
        fpsr: fpsr_dirty,
    }
}

fn block_register_access(
    block: &super::region::BasicBlockRecord,
) -> (
    IntegerRegisterSet,
    IntegerRegisterSet,
    [bool; 32],
    [bool; 32],
    bool,
    bool,
) {
    let mut integer_accessed = IntegerRegisterSet::default();
    let mut integer_dirty = IntegerRegisterSet::default();
    let mut vector_accessed = [false; 32];
    let mut vector_dirty = [false; 32];
    let mut fpsr_read = false;
    let mut fpsr_dirty = false;
    for decoded in &block.instructions {
        match nixe_cpu::decode::a64::normalize(&decoded.instruction, decoded.encoding) {
            A64Instruction::Integer(instruction) => {
                register_access_integer(instruction, &mut integer_accessed, &mut integer_dirty)
            }
            A64Instruction::Memory(instruction) => {
                register_access_memory(instruction, &mut integer_accessed, &mut integer_dirty)
            }
            A64Instruction::System(instruction) => {
                register_access_system(instruction, &mut integer_accessed, &mut integer_dirty);
                let fields = instruction.operands();
                fpsr_read |= matches!(instruction, system::Instruction::ReadRegister(_))
                    && fields.system_key == 0xd53b_4420;
                if matches!(instruction, system::Instruction::WriteRegister(_))
                    && fields.system_key == 0xd51b_4420
                {
                    fpsr_dirty = true;
                }
            }
            A64Instruction::FpSimd(instruction) => {
                register_access_fp_simd_general(
                    instruction,
                    &mut integer_accessed,
                    &mut integer_dirty,
                );
                register_access_fp_simd_vector(
                    instruction,
                    &mut vector_accessed,
                    &mut vector_dirty,
                );
                if a64_fp_simd::fp_lowering_disposition(instruction).accesses_status() {
                    fpsr_read = true;
                    fpsr_dirty = true;
                }
            }
            A64Instruction::Control(control::Instruction::BranchLinkImmediate(_)) => {
                mark_write(&mut integer_accessed, &mut integer_dirty, 30, false);
            }
            A64Instruction::Control(control::Instruction::BranchRegister(fields)) => {
                mark_read(&mut integer_accessed, fields.rn, false);
                if fields.branch_register_key == 0xd63f_0000 {
                    mark_write(&mut integer_accessed, &mut integer_dirty, 30, false);
                }
            }
            A64Instruction::Control(
                control::Instruction::CompareBranch(fields)
                | control::Instruction::TestBranch(fields),
            ) => mark_read(&mut integer_accessed, fields.rd, false),
            _ => {}
        }
    }
    (
        integer_accessed,
        integer_dirty,
        vector_accessed,
        vector_dirty,
        fpsr_read,
        fpsr_dirty,
    )
}

fn register_access_integer(
    instruction: integer::Instruction,
    accessed: &mut IntegerRegisterSet,
    dirty: &mut IntegerRegisterSet,
) {
    let fields = instruction.operands();
    match instruction {
        integer::Instruction::MoveWide(_) => {
            let opcode = u8::from(fields.subtract) * 2 + u8::from(fields.set_flags);
            if opcode == 3 {
                mark_read(accessed, fields.rd, false);
            }
            mark_write(accessed, dirty, fields.rd, false);
        }
        integer::Instruction::AddSubImmediate(_) => {
            mark_read(accessed, fields.rn, true);
            mark_write(accessed, dirty, fields.rd, !fields.set_flags);
        }
        integer::Instruction::AddSubExtended(_) => {
            mark_read(accessed, fields.rn, true);
            mark_read(accessed, fields.rm, false);
            mark_write(accessed, dirty, fields.rd, !fields.set_flags);
        }
        integer::Instruction::AddSubShifted(_) | integer::Instruction::AddSubCarry(_) => {
            mark_read(accessed, fields.rn, false);
            mark_read(accessed, fields.rm, false);
            mark_write(accessed, dirty, fields.rd, false);
        }
        integer::Instruction::LogicalImmediate(_) => {
            mark_read(accessed, fields.rn, false);
            mark_write(accessed, dirty, fields.rd, false);
        }
        integer::Instruction::LogicalShifted(_)
        | integer::Instruction::Extract(_)
        | integer::Instruction::TwoSource(_)
        | integer::Instruction::ConditionalSelect(_) => {
            mark_read(accessed, fields.rn, false);
            mark_read(accessed, fields.rm, false);
            mark_write(accessed, dirty, fields.rd, false);
        }
        integer::Instruction::Bitfield(_) => {
            mark_read(accessed, fields.rn, false);
            if u8::from(fields.subtract) * 2 + u8::from(fields.set_flags) == 1 {
                mark_read(accessed, fields.rd, false);
            }
            mark_write(accessed, dirty, fields.rd, false);
        }
        integer::Instruction::ConditionalCompareRegister(_) => {
            mark_read(accessed, fields.rn, false);
            mark_read(accessed, fields.rm, false);
        }
        integer::Instruction::ConditionalCompareImmediate(_) => {
            mark_read(accessed, fields.rn, false);
        }
        integer::Instruction::ThreeSource(_) => {
            mark_read(accessed, fields.rn, false);
            mark_read(accessed, fields.rm, false);
            if !matches!(fields.opcode_3, 2 | 6) {
                mark_read(accessed, fields.ra, false);
            }
            mark_write(accessed, dirty, fields.rd, false);
        }
        integer::Instruction::OneSource(_) => {
            mark_read(accessed, fields.rn, false);
            mark_write(accessed, dirty, fields.rd, false);
        }
        integer::Instruction::Adr(_) | integer::Instruction::Adrp(_) => {
            mark_write(accessed, dirty, fields.rd, false);
        }
    }
}

fn mark_read(accessed: &mut IntegerRegisterSet, index: u8, register31_is_sp: bool) {
    if index == 31 {
        accessed.sp |= register31_is_sp;
    } else {
        accessed.x[usize::from(index)] = true;
    }
}

fn mark_write(
    _accessed: &mut IntegerRegisterSet,
    dirty: &mut IntegerRegisterSet,
    index: u8,
    register31_is_sp: bool,
) {
    if index == 31 {
        dirty.sp |= register31_is_sp;
    } else {
        let slot = usize::from(index);
        dirty.x[slot] = true;
    }
}

fn source_location(
    region: &NativeRegion,
    instruction: &DecodedInstruction<nixe_cpu::decode::DecodedOpcode>,
) -> SourceLoc {
    source_location_for_pc(region, instruction.location.pc)
}

fn source_location_for_pc(region: &NativeRegion, pc: GuestVirtualAddress) -> SourceLoc {
    let offset = pc.get().wrapping_sub(region.key.start.get());
    SourceLoc::new(
        u32::try_from(offset)
            .unwrap_or(u32::MAX - 1)
            .saturating_add(1),
    )
}

fn direct_fault_source_location(index: usize) -> SourceLoc {
    let index = u32::try_from(index).expect("direct fault-site identity space exhausted");
    SourceLoc::new(
        u32::MAX
            .checked_sub(index)
            .and_then(|value| value.checked_sub(1))
            .expect("direct fault-site identity space exhausted"),
    )
}

fn compile_fault_sites(
    compiled: &cranelift_codegen::CompiledCode,
    pending_sites: &[PendingFaultSite],
) -> Result<Vec<CompiledFaultSite>, DirectJitError> {
    let pending_by_source: HashMap<_, _> = pending_sites
        .iter()
        .map(|pending| (pending.source_location, pending))
        .collect();
    if pending_by_source.len() != pending_sites.len() {
        return Err(DirectJitError::internal(
            "direct fault-site source locations are not unique",
        ));
    }
    let source_ranges = compiled.buffer.get_srclocs_sorted();
    let mut output = Vec::new();
    for trap in compiled
        .buffer
        .traps()
        .iter()
        .filter(|trap| trap.code == DIRECT_MEMORY_TRAP)
    {
        let source_index = source_ranges
            .partition_point(|range| range.start <= trap.offset)
            .checked_sub(1)
            .ok_or_else(|| {
                DirectJitError::internal("direct memory trap has no source-location range")
            })?;
        let source = &source_ranges[source_index];
        if trap.offset >= source.end {
            return Err(DirectJitError::internal(
                "direct memory trap falls outside its source-location range",
            ));
        }
        let pending = pending_by_source.get(&source.loc).ok_or_else(|| {
            DirectJitError::internal("direct memory trap has no pending fault site")
        })?;
        let native_start = trap.offset;
        output.push(CompiledFaultSite {
            native_start,
            native_end: native_start.saturating_add(1),
            access: pending.access,
        });
    }
    output.sort_unstable_by_key(|site| site.native_start);
    if output
        .windows(2)
        .any(|pair| pair[0].native_end > pair[1].native_start)
    {
        return Err(DirectJitError::internal(
            "Cranelift direct fault-site intervals overlap",
        ));
    }
    Ok(output)
}

fn unsupported_instruction(
    decoded: &DecodedInstruction<nixe_cpu::decode::DecodedOpcode>,
) -> DirectJitError {
    DirectJitError::unsupported(format!(
        "direct JIT instruction is not implemented: {} encoding={} disassembly={}",
        decoded.location,
        decoded.encoding,
        nixe_cpu::decode::disassemble(&decoded.instruction)
    ))
}

fn context_offset(offset: usize) -> Result<i32, DirectJitError> {
    i32::try_from(offset)
        .map_err(|_| DirectJitError::internal("direct JIT native-context offset exceeds i32"))
}

fn trusted_flags() -> MemFlagsData {
    MemFlagsData::trusted()
}

fn plain_flags() -> MemFlagsData {
    MemFlagsData::new()
}

const _: () = assert!(size_of::<NativeContext>() <= i32::MAX as usize);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier_policies_select_the_intended_cranelift_cost() {
        let runtime = CompilerRuntime::new().unwrap();
        let lcq = DirectCompiler::new(LCQ_COMPILER_POLICY, runtime.addresses()).unwrap();
        let lcq_flags = lcq.module.isa().flags();
        assert_eq!(lcq_flags.opt_level(), settings::OptLevel::None);
        assert_eq!(
            lcq_flags.regalloc_algorithm(),
            settings::RegallocAlgorithm::SinglePass
        );
        assert!(lcq_flags.preserve_frame_pointers());

        let hcq = DirectCompiler::new(HCQ_COMPILER_POLICY, runtime.addresses()).unwrap();
        let hcq_flags = hcq.module.isa().flags();
        assert_eq!(hcq_flags.opt_level(), settings::OptLevel::Speed);
        assert_eq!(
            hcq_flags.regalloc_algorithm(),
            settings::RegallocAlgorithm::Backtracking
        );
        assert!(hcq_flags.preserve_frame_pointers());
    }
}
