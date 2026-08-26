//! Verified Nixe IR to Cranelift lowering and native publication.

use std::collections::{BTreeMap, HashMap};
use std::fmt;

use cranelift_codegen::{
    Context,
    control::ControlPlane,
    ir::{
        self, AbiParam, AtomicRmwOp, BlockArg, InstBuilder, MemFlagsData, Signature, SourceLoc,
        UserFuncName, condcodes::IntCC, types,
    },
    isa::{CallConv, OwnedTargetIsa, TargetFrontendConfig},
};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use nixe_cpu::{
    decode::a64::fp_simd::{
        BitwiseOperation, Instruction as A64FpSimdInstruction, IntegerComparison,
        PairwiseOperation, PermuteOperation,
    },
    exception::ExceptionKind,
    ir::{
        block::{BlockId, IrBlock},
        op::{
            AddressOperation, ArithmeticFlagOutput, AtomicOperation, Condition, ExclusiveOperation,
            FlagOperation, GuestAddressWidth, IntegerBinaryKind, IntegerPredicate, IrOperation,
            MemoryOperation, OperationKind, ScalarOperation, ShiftKind, StateRegister,
            VectorOperation,
        },
        region::{IrRegion, RegionSafepointKind},
        terminator::{ControlTarget, StopReason, Terminator},
        types::IrType,
        value::{Immediate, Operand, ValueId},
        verify::verify_region,
    },
    location::{ExecutionState, InstructionEncoding, LocationDescriptor},
    memory::CacheMaintenanceKind,
    semantics::{
        a64::HintOperation, a64_fp_simd::semantic_instruction, immediate::decode_a64_bit_masks,
    },
    state::{a32::A32GeneralRegister, a64::A64GeneralRegister},
};
use nixe_memory::GuestVirtualAddress;

use crate::{
    abi::{
        EXECUTION_STATE_A32, EXECUTION_STATE_A64, EXECUTION_STATE_T32, EXIT_ARCHITECTURAL,
        EXIT_BUDGET_EXHAUSTED, EXIT_DATA_FAULT, EXIT_DISPATCH, EXIT_INTERNAL, EXIT_INTERPRET_ONE,
        EXIT_LOADER_RETURN, EXIT_PENDING_EVENT, EXIT_SAFEPOINT, EXIT_SCHEDULED, EXIT_UNSUPPORTED,
        FRAME_OFFSETS, HELPER_OFFSETS, NativeEntryAddress, NativeGateway, SCHEDULE_SEND_EVENT,
        SCHEDULE_WAIT_FOR_EVENT, SCHEDULE_WAIT_FOR_INTERRUPT, SCHEDULE_YIELD,
        SYSTEM_CACHE_DATA_CLEAN, SYSTEM_CACHE_DATA_CLEAN_INVALIDATE, SYSTEM_CACHE_DATA_INVALIDATE,
        SYSTEM_CACHE_INSTRUCTION_INVALIDATE, SYSTEM_CACHE_INSTRUCTION_PREFETCH, SYSTEM_POLL,
        SYSTEM_READ_RUNTIME_REGISTER, SYSTEM_SEND_EVENT_LOCAL, SYSTEM_WAIT_FOR_EVENT,
        SYSTEM_WAIT_FOR_INTERRUPT,
    },
    cache::{CompilationCancellation, CompilationCancellationReason},
    executable_memory::{
        ExecutableMemoryError, PublishedCode, SharedExecutableMemory, publish_code,
    },
    helpers::encode_access,
    links::{LINK_OFFSETS, LinkKind, LinkSiteMetadata},
    tlb::{
        FLAG_CPU_VISIBLE, FLAG_ORDINARY, FLAG_READ, FLAG_WRITE, FLAG_WRITE_ARMED, PAGE_BITS,
        PAGE_SIZE, TLB_ENTRY_SIZE, TLB_OFFSETS,
    },
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum SideExit {
    Architectural {
        source: u32,
        kind: ExceptionKind,
        syndrome: Option<u64>,
    },
    Unsupported {
        source: u32,
        encoding: InstructionEncoding,
        coverage_id: u32,
        disassembly: Box<str>,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SemanticCall {
    pub(crate) helper: Box<str>,
    pub(crate) result_types: Box<[IrType]>,
}

/// Immutable source and slow-path data kept beside, never inside, RX code.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CompiledRegionMetadata {
    pub(crate) start: LocationDescriptor,
    pub(crate) sources: Box<[LocationDescriptor]>,
    pub(crate) side_exits: Box<[SideExit]>,
    pub(crate) semantic_calls: Box<[SemanticCall]>,
    pub(crate) native_named_operations: u64,
    pub(crate) link_sites: Box<[LinkSiteMetadata]>,
}

pub(crate) struct CompiledRegion {
    entry: NativeEntryAddress,
    pub(crate) metadata: CompiledRegionMetadata,
    mapped_len: usize,
    _publication: Option<PublishedCode>,
}

impl CompiledRegion {
    pub(crate) unsafe fn execute(
        &self,
        gateway: NativeGateway,
        frame: *mut crate::abi::ExecutionFrame,
    ) {
        // SAFETY: publication produced a tail-convention entry and the
        // Cranelift-generated C gateway owns the only ABI transition to it.
        unsafe { gateway(frame, self.entry) }
    }

    pub(crate) const fn entry_address(&self) -> NativeEntryAddress {
        self.entry
    }

    pub(crate) const fn mapped_len(&self) -> usize {
        self.mapped_len
    }

    #[cfg(test)]
    pub(crate) fn for_cache_test(metadata: CompiledRegionMetadata, mapped_len: usize) -> Self {
        Self {
            entry: 1,
            metadata,
            mapped_len,
            _publication: None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CompilerError {
    detail: Box<str>,
    cancellation: Option<CompilationCancellationReason>,
}

impl CompilerError {
    fn new(detail: impl Into<Box<str>>) -> Self {
        Self {
            detail: detail.into(),
            cancellation: None,
        }
    }

    fn cancelled(reason: CompilationCancellationReason) -> Self {
        Self {
            detail: "JIT compilation was cancelled".into(),
            cancellation: Some(reason),
        }
    }

    pub(crate) fn detail(&self) -> &str {
        &self.detail
    }

    pub(crate) const fn cancellation_reason(&self) -> Option<CompilationCancellationReason> {
        self.cancellation
    }
}

impl fmt::Display for CompilerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.detail)
    }
}

impl std::error::Error for CompilerError {}

impl From<ExecutableMemoryError> for CompilerError {
    fn from(error: ExecutableMemoryError) -> Self {
        Self::new(error.to_string())
    }
}

struct StateSlot {
    register: StateRegister,
    variable: Variable,
    ty: ir::Type,
    offset: i32,
}

struct LoweringState {
    frame: ir::Value,
    retired: Variable,
    execution_state: Variable,
    state: Vec<StateSlot>,
    blocks: Vec<ir::Block>,
    exit: ir::Block,
    sources: Vec<LocationDescriptor>,
    source_indices: HashMap<LocationDescriptor, u32>,
    side_exits: Vec<SideExit>,
    semantic_calls: Vec<SemanticCall>,
    native_named_operations: u64,
    link_sites: Vec<LinkSiteMetadata>,
    helper_call_conv: CallConv,
}

/// Executor-local Cranelift state reused across bounded compilations.
pub(crate) struct CompilerContext {
    isa: OwnedTargetIsa,
    context: Context,
    builder: FunctionBuilderContext,
}

impl CompilerContext {
    pub(crate) fn new(isa: OwnedTargetIsa) -> Self {
        debug_assert!(
            FRAME_OFFSETS
                .all()
                .into_iter()
                .chain(HELPER_OFFSETS.all())
                .all(|offset| i32::try_from(offset).is_ok()),
            "native frame offsets must fit Cranelift immediates"
        );
        Self {
            isa,
            context: Context::new(),
            builder: FunctionBuilderContext::new(),
        }
    }

    pub(crate) fn compile(
        &mut self,
        region: &IrRegion,
        executable_memory: &SharedExecutableMemory,
        cancellation: CompilationCancellation<'_>,
    ) -> Result<CompiledRegion, CompilerError> {
        check_cancellation(cancellation)?;
        verify_region(region)
            .map_err(|error| CompilerError::new(format!("Nixe IR verification failed: {error}")))?;
        validate_region_state(region)?;
        check_cancellation(cancellation)?;
        self.context.clear();
        self.context.func.name = UserFuncName::user(0, 0);
        self.context.func.signature = Signature::new(CallConv::Tail);
        self.context
            .func
            .signature
            .params
            .push(AbiParam::new(self.isa.pointer_type()));

        let builder = FunctionBuilder::new(&mut self.context.func, &mut self.builder);
        let metadata = lower_region(
            builder,
            region,
            self.isa.frontend_config(),
            self.isa.default_call_conv(),
            cancellation,
        )?;
        #[cfg(debug_assertions)]
        cranelift_codegen::verifier::verify_function(&self.context.func, self.isa.as_ref())
            .map_err(|errors| {
                CompilerError::new(format!(
                    "generated Cranelift IR verification failed: {errors}"
                ))
            })?;
        check_cancellation(cancellation)?;
        let mut control_plane = ControlPlane::default();
        let compiled = self
            .context
            .compile(self.isa.as_ref(), &mut control_plane)
            .map_err(|error| {
                CompilerError::new(format!("Cranelift compilation failed: {error:?}"))
            })?;
        check_cancellation(cancellation)?;
        if !compiled.buffer.relocs().is_empty() {
            return Err(CompilerError::new(
                "lowered region unexpectedly requires native relocations",
            ));
        }
        let code = compiled.code_buffer();
        let alignment = self.isa.function_alignment().preferred as usize;
        check_cancellation(cancellation)?;
        let published = publish_code(executable_memory, code, alignment)?;
        let address: *mut u8 = std::ptr::with_exposed_provenance_mut(published.address);
        let entry = address.addr();
        Ok(CompiledRegion {
            entry,
            metadata,
            mapped_len: published.mapped_len,
            _publication: Some(published),
        })
    }
}

pub(crate) fn compile_gateway(
    isa: &OwnedTargetIsa,
    executable_memory: &SharedExecutableMemory,
) -> Result<(NativeGateway, PublishedCode), CompilerError> {
    let mut context = Context::new();
    context.func.name = UserFuncName::user(0, 1);
    context.func.signature = Signature::new(isa.default_call_conv());
    let pointer_type = isa.pointer_type();
    context
        .func
        .signature
        .params
        .push(AbiParam::new(pointer_type));
    context
        .func
        .signature
        .params
        .push(AbiParam::new(pointer_type));
    let mut builder_context = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut context.func, &mut builder_context);
    let entry = builder.create_block();
    builder.append_block_params_for_function_params(entry);
    builder.switch_to_block(entry);
    builder.seal_block(entry);
    let frame = builder.block_params(entry)[0];
    let callee = builder.block_params(entry)[1];
    let mut tail_signature = Signature::new(CallConv::Tail);
    tail_signature.params.push(AbiParam::new(pointer_type));
    let tail_signature = builder.import_signature(tail_signature);
    builder
        .ins()
        .call_indirect(tail_signature, callee, &[frame]);
    builder.ins().return_(&[]);
    builder.finalize(isa.frontend_config());
    #[cfg(debug_assertions)]
    cranelift_codegen::verifier::verify_function(&context.func, isa.as_ref()).map_err(
        |errors| {
            CompilerError::new(format!(
                "generated native-gateway verification failed: {errors}"
            ))
        },
    )?;
    let mut control_plane = ControlPlane::default();
    let compiled = context
        .compile(isa.as_ref(), &mut control_plane)
        .map_err(|error| CompilerError::new(format!("Cranelift gateway failed: {error:?}")))?;
    if !compiled.buffer.relocs().is_empty() {
        return Err(CompilerError::new(
            "native gateway unexpectedly requires relocations",
        ));
    }
    let published = publish_code(
        executable_memory,
        compiled.code_buffer(),
        isa.function_alignment().preferred as usize,
    )?;
    // SAFETY: Cranelift emitted the exact C signature represented by
    // NativeGateway and publication made it immutable and executable.
    let gateway = unsafe {
        std::mem::transmute::<*mut u8, NativeGateway>(std::ptr::with_exposed_provenance_mut(
            published.address,
        ))
    };
    Ok((gateway, published))
}

fn validate_region_state(region: &IrRegion) -> Result<(), CompilerError> {
    let root_a64 = region.metadata.start.execution_state == ExecutionState::A64;
    if region
        .blocks
        .iter()
        .any(|block| (block.metadata.start.execution_state == ExecutionState::A64) != root_a64)
    {
        return Err(CompilerError::new(
            "a region cannot change between A64 and AArch32 state representations",
        ));
    }
    Ok(())
}

fn lower_region(
    mut builder: FunctionBuilder<'_>,
    region: &IrRegion,
    frontend_config: TargetFrontendConfig,
    helper_call_conv: CallConv,
    cancellation: CompilationCancellation<'_>,
) -> Result<CompiledRegionMetadata, CompilerError> {
    let metadata = lower_region_body(&mut builder, region, helper_call_conv, cancellation)?;
    builder.finalize(frontend_config);
    Ok(metadata)
}

fn lower_region_body(
    builder: &mut FunctionBuilder<'_>,
    region: &IrRegion,
    helper_call_conv: CallConv,
    cancellation: CompilationCancellation<'_>,
) -> Result<CompiledRegionMetadata, CompilerError> {
    check_cancellation(cancellation)?;
    let entry = builder.create_block();
    builder.append_block_params_for_function_params(entry);
    let blocks: Vec<_> = region
        .blocks
        .iter()
        .map(|_| builder.create_block())
        .collect();
    let invalid_entry = builder.create_block();
    let exit = create_exit_block(builder);
    builder.switch_to_block(entry);
    builder.seal_block(entry);
    let frame = builder.block_params(entry)[0];
    let retired = builder.declare_var(types::I64);
    let zero = builder.ins().iconst(types::I64, 0);
    builder.def_var(retired, zero);
    let execution_state = builder.declare_var(types::I32);
    let imported_state = load(builder, types::I32, frame, FRAME_OFFSETS.execution_state)?;
    builder.def_var(execution_state, imported_state);
    let state = declare_state(builder, region.metadata.start.execution_state, frame)?;
    let mut lowering = LoweringState {
        frame,
        retired,
        execution_state,
        state,
        blocks,
        exit,
        sources: Vec::new(),
        source_indices: HashMap::new(),
        side_exits: Vec::new(),
        semantic_calls: Vec::new(),
        native_named_operations: 0,
        link_sites: Vec::new(),
        helper_call_conv,
    };

    emit_entry_dispatch(builder, region, &lowering, invalid_entry)?;
    builder.switch_to_block(invalid_entry);
    emit_exit(
        builder,
        &lowering,
        EXIT_INTERNAL,
        0,
        region.metadata.start.pc.get(),
        0,
        0,
    );

    for (index, block) in region.blocks.iter().enumerate() {
        check_cancellation(cancellation)?;
        builder.switch_to_block(lowering.blocks[index]);
        lower_block(
            builder,
            region,
            BlockId::new(index as u32),
            block,
            &mut lowering,
            cancellation,
        )?;
    }
    lower_exit_block(builder, &lowering)?;
    builder.seal_all_blocks();
    let metadata = CompiledRegionMetadata {
        start: region.metadata.start,
        sources: lowering.sources.into_boxed_slice(),
        side_exits: lowering.side_exits.into_boxed_slice(),
        semantic_calls: lowering.semantic_calls.into_boxed_slice(),
        native_named_operations: lowering.native_named_operations,
        link_sites: lowering.link_sites.into_boxed_slice(),
    };
    Ok(metadata)
}

fn create_exit_block(builder: &mut FunctionBuilder<'_>) -> ir::Block {
    let block = builder.create_block();
    for ty in [
        types::I32,
        types::I32,
        types::I64,
        types::I64,
        types::I64,
        types::I64,
    ] {
        builder.append_block_param(block, ty);
    }
    block
}

fn emit_entry_dispatch(
    builder: &mut FunctionBuilder<'_>,
    region: &IrRegion,
    lowering: &LoweringState,
    invalid_entry: ir::Block,
) -> Result<(), CompilerError> {
    let (pc_register, expected_state) = match region.metadata.start.execution_state {
        ExecutionState::A64 => (StateRegister::A64Pc, EXECUTION_STATE_A64),
        ExecutionState::A32 => (StateRegister::A32Pc, EXECUTION_STATE_A32),
        ExecutionState::T32 => (StateRegister::A32Pc, EXECUTION_STATE_T32),
    };
    let state = builder.use_var(lowering.execution_state);
    let matches = builder
        .ins()
        .icmp_imm_s(IntCC::Equal, state, expected_state as i64);
    let state_valid = builder.create_block();
    builder
        .ins()
        .brif(matches, state_valid, &[], invalid_entry, &[]);
    builder.switch_to_block(state_valid);
    let pc = state_value_for(builder, lowering, pc_register)?;
    for (index, entry) in region.metadata.entries.iter().enumerate() {
        let target = lowering.blocks[entry.block.index() as usize];
        let matches = builder
            .ins()
            .icmp_imm_s(IntCC::Equal, pc, entry.location.pc.get() as i64);
        let next = if index + 1 == region.metadata.entries.len() {
            invalid_entry
        } else {
            builder.create_block()
        };
        builder.ins().brif(matches, target, &[], next, &[]);
        if next != invalid_entry {
            builder.switch_to_block(next);
        }
    }
    Ok(())
}

fn lower_block(
    builder: &mut FunctionBuilder<'_>,
    region: &IrRegion,
    id: BlockId,
    block: &IrBlock,
    lowering: &mut LoweringState,
    cancellation: CompilationCancellation<'_>,
) -> Result<(), CompilerError> {
    let mut values = BTreeMap::new();
    let mut operation_index = 0;
    if region
        .metadata
        .safepoints
        .iter()
        .any(|safepoint| safepoint.block == id && safepoint.kind == RegionSafepointKind::Entry)
    {
        // Internal edges can reach any published entry, so an entry poll must
        // export that entry rather than the last instruction in its predecessor.
        set_current_location(builder, lowering, block.metadata.start)?;
        emit_control_poll(builder, lowering, block.metadata.start)?;
    }
    for source in &block.metadata.sources {
        check_cancellation(cancellation)?;
        set_source_location(builder, lowering, source.location)?;
        set_current_location(builder, lowering, source.location)?;
        emit_instruction_boundary(builder, lowering, source.location)?;
        while let Some(operation) = block.operations.get(operation_index) {
            if operation.source != source.location {
                break;
            }
            lower_operation(builder, lowering, operation, &mut values)?;
            operation_index += 1;
        }
        increment_retired(builder, lowering);
    }
    if operation_index != block.operations.len() {
        return Err(CompilerError::new(format!(
            "block {} operations are not ordered by instruction source",
            id.index()
        )));
    }
    lower_terminator(builder, region, id, block, lowering, &values)
}

fn check_cancellation(cancellation: CompilationCancellation<'_>) -> Result<(), CompilerError> {
    match cancellation.reason() {
        Some(reason) => Err(CompilerError::cancelled(reason)),
        None => Ok(()),
    }
}

fn set_source_location(
    builder: &mut FunctionBuilder<'_>,
    lowering: &mut LoweringState,
    source: LocationDescriptor,
) -> Result<u32, CompilerError> {
    let index = if let Some(index) = lowering.source_indices.get(&source).copied() {
        index
    } else {
        let index = u32::try_from(lowering.sources.len())
            .map_err(|_| CompilerError::new("source metadata index overflow"))?;
        lowering.sources.push(source);
        lowering.source_indices.insert(source, index);
        index
    };
    builder.set_srcloc(SourceLoc::new(index));
    Ok(index)
}

fn emit_instruction_boundary(
    builder: &mut FunctionBuilder<'_>,
    lowering: &LoweringState,
    source: LocationDescriptor,
) -> Result<(), CompilerError> {
    if source.execution_state == ExecutionState::A64 {
        let valid = load(
            builder,
            types::I32,
            lowering.frame,
            FRAME_OFFSETS.control_loader_return_valid,
        )?;
        let valid = builder.ins().icmp_imm_s(IntCC::NotEqual, valid, 0);
        let loader_return = load(
            builder,
            types::I64,
            lowering.frame,
            FRAME_OFFSETS.control_loader_return,
        )?;
        let matches = builder
            .ins()
            .icmp_imm_s(IntCC::Equal, loader_return, source.pc.get() as i64);
        let stop = builder.ins().band(valid, matches);
        let loader = builder.create_block();
        let execute = builder.create_block();
        builder.ins().brif(stop, loader, &[], execute, &[]);
        builder.switch_to_block(loader);
        let result_code = state_value_for(
            builder,
            lowering,
            StateRegister::A64X(A64GeneralRegister::new(0).unwrap()),
        )?;
        emit_exit_with_payload(
            builder,
            lowering,
            EXIT_LOADER_RETURN,
            source.pc.get(),
            result_code,
        );
        builder.switch_to_block(execute);
    }
    let retired = builder.use_var(lowering.retired);
    let carried = load(
        builder,
        types::I64,
        lowering.frame,
        FRAME_OFFSETS.dispatch_retired,
    )?;
    let retired = builder.ins().iadd(carried, retired);
    let budget = load(
        builder,
        types::I64,
        lowering.frame,
        FRAME_OFFSETS.control_instruction_budget,
    )?;
    let available = builder.ins().icmp(IntCC::UnsignedLessThan, retired, budget);
    let execute = builder.create_block();
    let exhausted = builder.create_block();
    builder.ins().brif(available, execute, &[], exhausted, &[]);
    builder.switch_to_block(exhausted);
    emit_exit(
        builder,
        lowering,
        EXIT_BUDGET_EXHAUSTED,
        0,
        source.pc.get(),
        0,
        0,
    );
    builder.switch_to_block(execute);
    Ok(())
}

fn emit_control_poll(
    builder: &mut FunctionBuilder<'_>,
    lowering: &LoweringState,
    source: LocationDescriptor,
) -> Result<(), CompilerError> {
    let pointer_type = builder.func.dfg.value_type(lowering.frame);
    let flags = trusted_mem_flags(builder);
    let control_pending_address = load(
        builder,
        pointer_type,
        lowering.frame,
        FRAME_OFFSETS.control_pending_address,
    )?;
    let control_pending = builder
        .ins()
        .atomic_load(types::I32, flags, control_pending_address);
    let interrupt_pending_address = load(
        builder,
        pointer_type,
        lowering.frame,
        FRAME_OFFSETS.interrupt_pending_address,
    )?;
    let interrupt_pending = builder
        .ins()
        .atomic_load(types::I32, flags, interrupt_pending_address);
    let local_requests = load(
        builder,
        types::I32,
        lowering.frame,
        FRAME_OFFSETS.control_request_flags,
    )?;
    let local_events = load(
        builder,
        types::I32,
        lowering.frame,
        FRAME_OFFSETS.control_event_mask,
    )?;
    let shared_pending = builder.ins().bor(control_pending, interrupt_pending);
    let local_pending = builder.ins().bor(local_requests, local_events);
    let pending = builder.ins().bor(shared_pending, local_pending);
    let needs_slow_path = builder.ins().icmp_imm_s(IntCC::NotEqual, pending, 0);
    let slow = builder.create_block();
    let resume = builder.create_block();
    builder.ins().brif(needs_slow_path, slow, &[], resume, &[]);

    builder.switch_to_block(slow);
    let argument = builder.ins().iconst(types::I64, 0);
    let status = call_system_helper(builder, lowering, SYSTEM_POLL, argument)?;
    let pending = builder.ins().icmp_imm_s(IntCC::Equal, status, 1);
    let handle_pending = builder.create_block();
    let classify_error = builder.create_block();
    builder
        .ins()
        .brif(pending, handle_pending, &[], classify_error, &[]);

    builder.switch_to_block(classify_error);
    let failed = builder.ins().icmp_imm_s(IntCC::NotEqual, status, 0);
    let failure = builder.create_block();
    builder.ins().brif(failed, failure, &[], resume, &[]);
    builder.switch_to_block(failure);
    emit_exit(builder, lowering, EXIT_INTERNAL, 0, source.pc.get(), 0, 0);

    builder.switch_to_block(handle_pending);
    let event_mask = load(
        builder,
        types::I32,
        lowering.frame,
        FRAME_OFFSETS.control_event_mask,
    )?;
    let has_event = builder.ins().icmp_imm_s(IntCC::NotEqual, event_mask, 0);
    let event = builder.create_block();
    let safepoint = builder.create_block();
    builder.ins().brif(has_event, event, &[], safepoint, &[]);
    builder.switch_to_block(event);
    emit_exit_dynamic(
        builder,
        lowering,
        EXIT_PENDING_EVENT,
        event_mask,
        source.pc.get(),
    );
    builder.switch_to_block(safepoint);
    emit_exit(builder, lowering, EXIT_SAFEPOINT, 0, source.pc.get(), 0, 0);

    builder.switch_to_block(resume);
    Ok(())
}

fn call_system_helper(
    builder: &mut FunctionBuilder<'_>,
    lowering: &LoweringState,
    operation: u32,
    argument: ir::Value,
) -> Result<ir::Value, CompilerError> {
    let pointer_type = builder.func.dfg.value_type(lowering.frame);
    let table = load(builder, pointer_type, lowering.frame, FRAME_OFFSETS.helpers)?;
    let flags = trusted_mem_flags(builder);
    let callee = builder
        .ins()
        .load(pointer_type, flags, table, offset(HELPER_OFFSETS.system)?);
    let operation = builder.ins().iconst(types::I32, i64::from(operation));
    let mut signature = Signature::new(lowering.helper_call_conv);
    signature.params.push(AbiParam::new(pointer_type));
    signature.params.push(AbiParam::new(types::I32));
    signature.params.push(AbiParam::new(types::I64));
    signature.returns.push(AbiParam::new(types::I32));
    let signature = builder.import_signature(signature);
    let call =
        builder
            .ins()
            .call_indirect(signature, callee, &[lowering.frame, operation, argument]);
    Ok(builder.inst_results(call)[0])
}

fn increment_retired(builder: &mut FunctionBuilder<'_>, lowering: &LoweringState) {
    let retired = builder.use_var(lowering.retired);
    let retired = builder.ins().iadd_imm_s(retired, 1);
    builder.def_var(lowering.retired, retired);
}

fn decrement_retired(builder: &mut FunctionBuilder<'_>, lowering: &LoweringState) {
    let retired = builder.use_var(lowering.retired);
    let retired = builder.ins().iadd_imm_s(retired, -1);
    builder.def_var(lowering.retired, retired);
}

fn set_current_location(
    builder: &mut FunctionBuilder<'_>,
    lowering: &LoweringState,
    location: LocationDescriptor,
) -> Result<(), CompilerError> {
    match location.execution_state {
        ExecutionState::A64 => {
            let pc = builder.ins().iconst(types::I64, location.pc.get() as i64);
            define_state(builder, lowering, StateRegister::A64Pc, pc)?;
            let state = builder.ins().iconst(types::I32, EXECUTION_STATE_A64 as i64);
            builder.def_var(lowering.execution_state, state);
        }
        ExecutionState::A32 | ExecutionState::T32 => {
            let pc = builder.ins().iconst(types::I32, location.pc.get() as i64);
            define_state(builder, lowering, StateRegister::A32Pc, pc)?;
            let state = if location.execution_state == ExecutionState::A32 {
                EXECUTION_STATE_A32
            } else {
                EXECUTION_STATE_T32
            };
            let state = builder.ins().iconst(types::I32, state as i64);
            builder.def_var(lowering.execution_state, state);
        }
    }
    Ok(())
}

fn lower_operation(
    builder: &mut FunctionBuilder<'_>,
    lowering: &mut LoweringState,
    operation: &IrOperation,
    values: &mut BTreeMap<ValueId, ir::Value>,
) -> Result<(), CompilerError> {
    let source = operation.source;
    let results = match &operation.kind {
        OperationKind::Constant(value) => vec![immediate(builder, *value)],
        OperationKind::Scalar(operation) => lower_scalar(builder, *operation, values)?,
        OperationKind::Address(operation) => vec![lower_address(builder, *operation, values)?],
        OperationKind::ReadState(register) => {
            vec![state_value_for(builder, lowering, *register)?]
        }
        OperationKind::WriteState { register, value } => {
            let value = operand(builder, values, *value)?;
            define_state(builder, lowering, *register, value)?;
            Vec::new()
        }
        OperationKind::Flags(operation) => vec![lower_flags(builder, *operation, values)?],
        OperationKind::Vector(operation) => lower_vector(builder, *operation, values)?,
        OperationKind::Memory(operation) => {
            lower_memory(builder, lowering, *operation, source, values)?
        }
        OperationKind::Barrier(_) => {
            builder.ins().fence();
            Vec::new()
        }
        OperationKind::ProcessorHint(operation) => {
            lower_processor_hint(builder, lowering, *operation, source)?
        }
        OperationKind::RuntimeRegisterRead(key) => {
            let key = builder.ins().iconst(types::I64, i64::from(*key));
            let status = call_system_helper(builder, lowering, SYSTEM_READ_RUNTIME_REGISTER, key)?;
            let succeeded = builder.ins().icmp_imm_s(IntCC::Equal, status, 0);
            let complete = builder.create_block();
            let failure = builder.create_block();
            builder.ins().brif(succeeded, complete, &[], failure, &[]);
            builder.switch_to_block(failure);
            emit_exit(builder, lowering, EXIT_INTERNAL, 0, source.pc.get(), 0, 0);
            builder.switch_to_block(complete);
            vec![load(
                builder,
                types::I64,
                lowering.frame,
                FRAME_OFFSETS.scratch_results,
            )?]
        }
        OperationKind::CacheMaintenance(operation) => {
            let address = operation
                .address
                .map(|address| operand(builder, values, address))
                .transpose()?
                .unwrap_or_else(|| builder.ins().iconst(types::I64, 0));
            let opcode = match operation.kind {
                CacheMaintenanceKind::InstructionInvalidate => SYSTEM_CACHE_INSTRUCTION_INVALIDATE,
                CacheMaintenanceKind::DataInvalidate => SYSTEM_CACHE_DATA_INVALIDATE,
                CacheMaintenanceKind::DataClean => SYSTEM_CACHE_DATA_CLEAN,
                CacheMaintenanceKind::DataCleanAndInvalidate => SYSTEM_CACHE_DATA_CLEAN_INVALIDATE,
                CacheMaintenanceKind::InstructionPrefetch => SYSTEM_CACHE_INSTRUCTION_PREFETCH,
            } | (u32::from(operation.address.is_some()) << 8);
            let status = call_system_helper(builder, lowering, opcode, address)?;
            let failed = builder.ins().icmp_imm_s(IntCC::NotEqual, status, 0);
            let failure = builder.create_block();
            let complete = builder.create_block();
            builder.ins().brif(failed, failure, &[], complete, &[]);
            builder.switch_to_block(failure);
            emit_exit(builder, lowering, EXIT_DATA_FAULT, 0, source.pc.get(), 0, 0);
            builder.switch_to_block(complete);
            if source.execution_state != ExecutionState::A64 {
                return Err(CompilerError::new(
                    "cache-maintenance continuation size is undefined outside A64",
                ));
            }
            let resumed = LocationDescriptor::new(
                source
                    .pc
                    .checked_add(4)
                    .ok_or_else(|| CompilerError::new("cache-maintenance PC overflow"))?,
                source.execution_state,
                source.profile_id,
            );
            set_current_location(builder, lowering, resumed)?;
            increment_retired(builder, lowering);
            emit_exit(builder, lowering, EXIT_DISPATCH, 0, resumed.pc.get(), 0, 0);
            let unreachable = builder.create_block();
            builder.switch_to_block(unreachable);
            Vec::new()
        }
        OperationKind::Exclusive(operation) => {
            lower_exclusive(builder, lowering, *operation, source, values)?
        }
        OperationKind::Atomic(operation) => {
            lower_atomic(builder, lowering, *operation, source, values)?
        }
        OperationKind::Helper(helper) => lower_named_helper(
            builder,
            lowering,
            helper,
            source,
            &operation
                .results
                .iter()
                .map(|value| value.ty)
                .collect::<Vec<_>>(),
            values,
        )?,
        OperationKind::FloatingPoint(_) => {
            return Err(CompilerError::new(format!(
                "exact helper lowering is not connected for {:?}",
                operation.kind
            )));
        }
    };
    let declared: Vec<_> = operation.results.iter().collect();
    if declared.len() != results.len() {
        return Err(CompilerError::new(
            "lowered result count differs from verified Nixe IR",
        ));
    }
    for (result, value) in declared.into_iter().zip(results) {
        values.insert(result.id, value);
    }
    Ok(())
}

fn lower_atomic(
    builder: &mut FunctionBuilder<'_>,
    lowering: &LoweringState,
    operation: AtomicOperation,
    source: LocationDescriptor,
    values: &BTreeMap<ValueId, ir::Value>,
) -> Result<Vec<ir::Value>, CompilerError> {
    if let AtomicOperation::CompareExchangePair {
        address,
        expected_low,
        expected_high,
        replacement_low,
        replacement_high,
        descriptor,
    } = operation
    {
        let address = operand(builder, values, address)?;
        let expected_low = operand(builder, values, expected_low)?;
        let expected_high = operand(builder, values, expected_high)?;
        let replacement_low = operand(builder, values, replacement_low)?;
        let replacement_high = operand(builder, values, replacement_high)?;
        let element_bytes = descriptor.access.size.bytes() / 2;
        let arguments = FRAME_OFFSETS.scratch_arguments;
        let zero = builder.ins().iconst(types::I64, 0);
        store_scratch(builder, lowering.frame, arguments, zero)?;
        store_scratch(builder, lowering.frame, arguments + 16, zero)?;
        store(builder, expected_low, lowering.frame, offset(arguments)?);
        store(
            builder,
            expected_high,
            lowering.frame,
            offset(arguments + element_bytes)?,
        );
        store(
            builder,
            replacement_low,
            lowering.frame,
            offset(arguments + 16)?,
        );
        store(
            builder,
            replacement_high,
            lowering.frame,
            offset(arguments + 16 + element_bytes)?,
        );
        let input = pointer_at(builder, lowering.frame, arguments)?;
        let output = pointer_at(builder, lowering.frame, FRAME_OFFSETS.scratch_results)?;
        let encoded = encode_access(descriptor.access) | (9_u64 << 32);
        let encoded = builder.ins().iconst(types::I64, encoded as i64);
        let status = call_helper(
            builder,
            lowering.frame,
            lowering.helper_call_conv,
            HELPER_OFFSETS.atomic,
            &[lowering.frame, address, encoded, input, output],
            5,
        )?;
        branch_on_helper_status(builder, lowering, source, status);
        let element_type = builder.func.dfg.value_type(expected_low);
        let low = load(
            builder,
            element_type,
            lowering.frame,
            FRAME_OFFSETS.scratch_results,
        )?;
        let high = load(
            builder,
            element_type,
            lowering.frame,
            FRAME_OFFSETS.scratch_results + element_bytes,
        )?;
        return Ok(vec![low, high]);
    }
    let (operation_code, address, first, second, descriptor) = match operation {
        AtomicOperation::ReadModifyWrite {
            kind,
            address,
            value,
            descriptor,
        } => {
            let code = match kind {
                nixe_cpu::memory::AtomicRmwKind::Add => 0,
                nixe_cpu::memory::AtomicRmwKind::Clear => 1,
                nixe_cpu::memory::AtomicRmwKind::Xor => 2,
                nixe_cpu::memory::AtomicRmwKind::Set => 3,
                nixe_cpu::memory::AtomicRmwKind::SignedMaximum => 4,
                nixe_cpu::memory::AtomicRmwKind::SignedMinimum => 5,
                nixe_cpu::memory::AtomicRmwKind::UnsignedMaximum => 6,
                nixe_cpu::memory::AtomicRmwKind::UnsignedMinimum => 7,
                nixe_cpu::memory::AtomicRmwKind::Swap => 8,
            };
            (code, address, value, None, descriptor)
        }
        AtomicOperation::CompareExchange {
            address,
            expected,
            replacement,
            descriptor,
        } => (9, address, expected, Some(replacement), descriptor),
        AtomicOperation::CompareExchangePair { .. } => unreachable!(),
    };
    let address = operand(builder, values, address)?;
    let first = operand(builder, values, first)?;
    let first = apply_byte_order(builder, first, descriptor.byte_order);
    store_scratch(
        builder,
        lowering.frame,
        FRAME_OFFSETS.scratch_arguments,
        first,
    )?;
    if let Some(second) = second {
        let second = operand(builder, values, second)?;
        let second = apply_byte_order(builder, second, descriptor.byte_order);
        store_scratch(
            builder,
            lowering.frame,
            FRAME_OFFSETS.scratch_arguments + 16,
            second,
        )?;
    }
    let input = pointer_at(builder, lowering.frame, FRAME_OFFSETS.scratch_arguments)?;
    let output = pointer_at(builder, lowering.frame, FRAME_OFFSETS.scratch_results)?;
    let encoded = encode_access(descriptor.access) | (operation_code << 32);
    let encoded = builder.ins().iconst(types::I64, encoded as i64);
    let status = call_helper(
        builder,
        lowering.frame,
        lowering.helper_call_conv,
        HELPER_OFFSETS.atomic,
        &[lowering.frame, address, encoded, input, output],
        5,
    )?;
    branch_on_helper_status(builder, lowering, source, status);
    let previous = load_scratch(
        builder,
        lowering.frame,
        FRAME_OFFSETS.scratch_results,
        descriptor.access.size,
    )?;
    Ok(vec![apply_byte_order(
        builder,
        previous,
        descriptor.byte_order,
    )])
}

fn lower_scalar(
    builder: &mut FunctionBuilder<'_>,
    operation: ScalarOperation,
    values: &BTreeMap<ValueId, ir::Value>,
) -> Result<Vec<ir::Value>, CompilerError> {
    Ok(match operation {
        ScalarOperation::Binary { kind, lhs, rhs } => {
            let lhs = operand(builder, values, lhs)?;
            let rhs = operand(builder, values, rhs)?;
            vec![lower_binary(builder, kind, lhs, rhs)]
        }
        ScalarOperation::AddWithCarry {
            lhs,
            rhs,
            carry_in,
            flags,
        } => {
            let lhs = operand(builder, values, lhs)?;
            let rhs = operand(builder, values, rhs)?;
            let carry = operand(builder, values, carry_in)?;
            let ty = builder.func.dfg.value_type(lhs);
            let carry = builder.ins().uextend(ty, carry);
            let (partial, carry0) = builder.ins().uadd_overflow(lhs, rhs);
            let (result, carry1) = builder.ins().uadd_overflow(partial, carry);
            let carry_out = builder.ins().bor(carry0, carry1);
            let (signed_partial, overflow0) = builder.ins().sadd_overflow(lhs, rhs);
            let (_, overflow1) = builder.ins().sadd_overflow(signed_partial, carry);
            let overflow = builder.ins().bor(overflow0, overflow1);
            let mut results = vec![result];
            if flags != ArithmeticFlagOutput::None {
                results.push(carry_out);
            }
            if flags == ArithmeticFlagOutput::CarryAndOverflow {
                results.push(overflow);
            }
            results
        }
        ScalarOperation::UnsignedOverflow {
            operation,
            lhs,
            rhs,
            result,
        } => {
            let lhs = operand(builder, values, lhs)?;
            let rhs = operand(builder, values, rhs)?;
            let result = operand(builder, values, result)?;
            let overflow = match operation {
                IntegerBinaryKind::Add => builder.ins().icmp(IntCC::UnsignedLessThan, result, lhs),
                IntegerBinaryKind::Subtract => {
                    builder
                        .ins()
                        .icmp(IntCC::UnsignedGreaterThanOrEqual, lhs, rhs)
                }
                _ => return Err(CompilerError::new("invalid unsigned-overflow operation")),
            };
            vec![overflow]
        }
        ScalarOperation::SignedOverflow {
            operation,
            lhs,
            rhs,
            result,
        } => {
            let lhs = operand(builder, values, lhs)?;
            let rhs = operand(builder, values, rhs)?;
            let result = operand(builder, values, result)?;
            let lhs_result = builder.ins().bxor(lhs, result);
            let other = match operation {
                IntegerBinaryKind::Add => builder.ins().bxor(rhs, result),
                IntegerBinaryKind::Subtract => builder.ins().bxor(lhs, rhs),
                _ => return Err(CompilerError::new("invalid signed-overflow operation")),
            };
            let combined = builder.ins().band(lhs_result, other);
            vec![builder.ins().icmp_imm_s(IntCC::SignedLessThan, combined, 0)]
        }
        ScalarOperation::Compare {
            predicate,
            lhs,
            rhs,
        } => {
            let lhs = operand(builder, values, lhs)?;
            let rhs = operand(builder, values, rhs)?;
            vec![builder.ins().icmp(int_condition(predicate), lhs, rhs)]
        }
        ScalarOperation::Select {
            condition,
            when_true,
            when_false,
        } => {
            let condition = operand(builder, values, condition)?;
            let when_true = operand(builder, values, when_true)?;
            let when_false = operand(builder, values, when_false)?;
            vec![builder.ins().select(condition, when_true, when_false)]
        }
        ScalarOperation::Shift {
            kind,
            value,
            amount,
        } => {
            let value = operand(builder, values, value)?;
            let amount = operand(builder, values, amount)?;
            vec![lower_shift(builder, kind, value, amount)]
        }
        ScalarOperation::CountLeadingZeros { value } => {
            let value = operand(builder, values, value)?;
            vec![builder.ins().clz(value)]
        }
        ScalarOperation::ReverseBits { value } => {
            let value = operand(builder, values, value)?;
            vec![builder.ins().bitrev(value)]
        }
        ScalarOperation::ZeroExtend { value, to } => {
            let value = operand(builder, values, value)?;
            vec![builder.ins().uextend(cranelift_type(to), value)]
        }
        ScalarOperation::SignExtend { value, to } => {
            let value = operand(builder, values, value)?;
            vec![builder.ins().sextend(cranelift_type(to), value)]
        }
        ScalarOperation::Truncate { value, to } => {
            let value = operand(builder, values, value)?;
            vec![builder.ins().ireduce(cranelift_type(to), value)]
        }
        ScalarOperation::Bitcast { value, to } => {
            let value = operand(builder, values, value)?;
            let flags = bitcast_flags(builder);
            vec![builder.ins().bitcast(cranelift_type(to), flags, value)]
        }
    })
}

fn lower_memory(
    builder: &mut FunctionBuilder<'_>,
    lowering: &LoweringState,
    operation: MemoryOperation,
    source: LocationDescriptor,
    values: &BTreeMap<ValueId, ir::Value>,
) -> Result<Vec<ir::Value>, CompilerError> {
    match operation {
        MemoryOperation::Load {
            address,
            descriptor,
        } => {
            let address = operand(builder, values, address)?;
            let value = memory_read(builder, lowering, source, address, descriptor.access)?;
            Ok(vec![apply_byte_order(
                builder,
                value,
                descriptor.byte_order,
            )])
        }
        MemoryOperation::Store {
            address,
            value,
            descriptor,
        } => {
            let address = operand(builder, values, address)?;
            let value = operand(builder, values, value)?;
            let value = apply_byte_order(builder, value, descriptor.byte_order);
            memory_write(builder, lowering, source, address, value, descriptor.access)?;
            Ok(Vec::new())
        }
        MemoryOperation::GuardedLoad {
            predicate,
            address,
            fallback,
            descriptor,
        } => {
            let predicate = operand(builder, values, predicate)?;
            let address = operand(builder, values, address)?;
            let fallback = operand(builder, values, fallback)?;
            let accessed = builder.create_block();
            let skipped = builder.create_block();
            let merged = builder.create_block();
            let ty = builder.func.dfg.value_type(fallback);
            builder.append_block_param(merged, ty);
            builder.ins().brif(predicate, accessed, &[], skipped, &[]);
            builder.switch_to_block(accessed);
            let value = memory_read(builder, lowering, source, address, descriptor.access)?;
            let value = apply_byte_order(builder, value, descriptor.byte_order);
            builder.ins().jump(merged, &[BlockArg::from(value)]);
            builder.switch_to_block(skipped);
            builder.ins().jump(merged, &[BlockArg::from(fallback)]);
            builder.switch_to_block(merged);
            Ok(vec![builder.block_params(merged)[0]])
        }
        MemoryOperation::GuardedStore {
            predicate,
            address,
            value,
            descriptor,
        } => {
            let predicate = operand(builder, values, predicate)?;
            let address = operand(builder, values, address)?;
            let value = operand(builder, values, value)?;
            let accessed = builder.create_block();
            let merged = builder.create_block();
            builder.ins().brif(predicate, accessed, &[], merged, &[]);
            builder.switch_to_block(accessed);
            let value = apply_byte_order(builder, value, descriptor.byte_order);
            memory_write(builder, lowering, source, address, value, descriptor.access)?;
            builder.ins().jump(merged, &[]);
            builder.switch_to_block(merged);
            Ok(Vec::new())
        }
    }
}

fn lower_processor_hint(
    builder: &mut FunctionBuilder<'_>,
    lowering: &mut LoweringState,
    operation: HintOperation,
    source: LocationDescriptor,
) -> Result<Vec<ir::Value>, CompilerError> {
    let resumed = LocationDescriptor::new(
        source
            .pc
            .checked_add(4)
            .ok_or_else(|| CompilerError::new("A64 processor-hint PC overflow"))?,
        source.execution_state,
        source.profile_id,
    );
    let exit = |builder: &mut FunctionBuilder<'_>,
                lowering: &mut LoweringState,
                detail: u32|
     -> Result<(), CompilerError> {
        set_current_location(builder, lowering, resumed)?;
        increment_retired(builder, lowering);
        emit_exit(
            builder,
            lowering,
            EXIT_SCHEDULED,
            detail,
            source.pc.get(),
            0,
            0,
        );
        Ok(())
    };
    match operation {
        HintOperation::NoOperation => {}
        HintOperation::Yield => {
            exit(builder, lowering, SCHEDULE_YIELD)?;
            let unreachable = builder.create_block();
            builder.switch_to_block(unreachable);
        }
        HintOperation::SendEvent => {
            exit(builder, lowering, SCHEDULE_SEND_EVENT)?;
            let unreachable = builder.create_block();
            builder.switch_to_block(unreachable);
        }
        HintOperation::SendEventLocal => {
            let argument = builder.ins().iconst(types::I64, 0);
            let status = call_system_helper(builder, lowering, SYSTEM_SEND_EVENT_LOCAL, argument)?;
            let succeeded = builder.ins().icmp_imm_s(IntCC::Equal, status, 0);
            let complete = builder.create_block();
            let failure = builder.create_block();
            builder.ins().brif(succeeded, complete, &[], failure, &[]);
            builder.switch_to_block(failure);
            emit_exit(builder, lowering, EXIT_INTERNAL, 0, source.pc.get(), 0, 0);
            builder.switch_to_block(complete);
        }
        HintOperation::WaitForEvent | HintOperation::WaitForInterrupt => {
            let (helper, detail) = match operation {
                HintOperation::WaitForEvent => (SYSTEM_WAIT_FOR_EVENT, SCHEDULE_WAIT_FOR_EVENT),
                HintOperation::WaitForInterrupt => {
                    (SYSTEM_WAIT_FOR_INTERRUPT, SCHEDULE_WAIT_FOR_INTERRUPT)
                }
                _ => unreachable!(),
            };
            let argument = builder.ins().iconst(types::I64, 0);
            let status = call_system_helper(builder, lowering, helper, argument)?;
            let succeeded = builder.ins().icmp_imm_s(IntCC::Equal, status, 0);
            let complete = builder.create_block();
            let classify_wait = builder.create_block();
            builder
                .ins()
                .brif(succeeded, complete, &[], classify_wait, &[]);
            builder.switch_to_block(classify_wait);
            let should_wait = builder.ins().icmp_imm_s(IntCC::Equal, status, 1);
            let wait = builder.create_block();
            let failure = builder.create_block();
            builder.ins().brif(should_wait, wait, &[], failure, &[]);
            builder.switch_to_block(wait);
            exit(builder, lowering, detail)?;
            builder.switch_to_block(failure);
            emit_exit(builder, lowering, EXIT_INTERNAL, 0, source.pc.get(), 0, 0);
            builder.switch_to_block(complete);
        }
    }
    Ok(Vec::new())
}

fn lower_named_helper(
    builder: &mut FunctionBuilder<'_>,
    lowering: &mut LoweringState,
    helper: &nixe_cpu::ir::op::HelperOperation,
    source: LocationDescriptor,
    result_types: &[IrType],
    values: &BTreeMap<ValueId, ir::Value>,
) -> Result<Vec<ir::Value>, CompilerError> {
    if let Some(results) = lower_native_named_helper(builder, helper, result_types, values)? {
        lowering.native_named_operations = lowering.native_named_operations.saturating_add(1);
        return Ok(results);
    }
    if !approved_semantic_slow_path(helper.helper.as_ref()) {
        return Err(CompilerError::new(format!(
            "named operation {} has no native lowering and is not an approved semantic slow path",
            helper.helper
        )));
    }
    let index = u32::try_from(lowering.semantic_calls.len())
        .map_err(|_| CompilerError::new("semantic helper metadata index overflow"))?;
    lowering.semantic_calls.push(SemanticCall {
        helper: helper.helper.clone(),
        result_types: result_types.to_vec().into_boxed_slice(),
    });
    if helper.arguments.len() > crate::abi::MAX_HELPER_ARGUMENTS
        || result_types.len() > crate::abi::MAX_HELPER_RESULTS
    {
        return Err(CompilerError::new(
            "semantic helper exceeds native scratch ABI",
        ));
    }
    for (argument_index, argument) in helper.arguments.iter().enumerate() {
        let value = operand(builder, values, *argument)?;
        store_scratch(
            builder,
            lowering.frame,
            FRAME_OFFSETS.scratch_arguments + argument_index * 16,
            value,
        )?;
    }
    let status = call_semantic_helper(builder, lowering.frame, lowering.helper_call_conv, index)?;
    let fp_trap = if semantic_helper_can_trap_fp(helper.helper.as_ref()) {
        Some(push_side_exit(
            lowering,
            SideExit::Architectural {
                source: compact_source_index(lowering, source)?,
                kind: ExceptionKind::FloatingPoint,
                syndrome: None,
            },
        )?)
    } else {
        None
    };
    branch_on_semantic_status(builder, lowering, source, status, fp_trap);
    result_types
        .iter()
        .enumerate()
        .map(|(result_index, ty)| {
            load_ir_scratch(
                builder,
                lowering.frame,
                FRAME_OFFSETS.scratch_results + result_index * 16,
                *ty,
            )
        })
        .collect()
}

fn approved_semantic_slow_path(name: &str) -> bool {
    matches!(
        name,
        "a64.simd.pair-memory"
            | "a64.simd.multiple-structure-memory"
            | "a64.simd.single-structure-memory"
            | "a64.fp-simd.semantic-vector"
            | "a64.fp.float-to-signed-int"
            | "a64.fp.float-to-unsigned-int"
            | "a64.fp.scalar-arithmetic"
            | "a64.fp.scalar-compare"
            | "a64.fp.semantic-conditional-compare"
            | "a64.fp.signed-int-to-float"
            | "a64.fp.unsigned-int-to-float"
            | "aarch32.vfp.binary32-vector"
    )
}

fn lower_native_named_helper(
    builder: &mut FunctionBuilder<'_>,
    helper: &nixe_cpu::ir::op::HelperOperation,
    result_types: &[IrType],
    values: &BTreeMap<ValueId, ir::Value>,
) -> Result<Option<Vec<ir::Value>>, CompilerError> {
    if helper.helper.as_ref() == "a64.fp-simd.semantic-vector"
        && let Some(results) = lower_native_a64_semantic_vector(builder, helper, values)?
    {
        return Ok(Some(results));
    }
    let native = matches!(
        helper.helper.as_ref(),
        "a64.extend-register"
            | "a64.load-store-register-offset"
            | "a64.sbfm"
            | "a64.bfm"
            | "a64.ubfm"
            | "a64.smaddl"
            | "a64.smsubl"
            | "a64.umaddl"
            | "a64.umsubl"
            | "a64.smulh"
            | "a64.umulh"
            | "a64.extr"
            | "a64.rev16"
            | "a64.rev32"
            | "a64.rev"
            | "a64.cls"
            | "a64.simd.zero-extend-load"
            | "a64.simd.low-bits"
            | "a64.simd.bitwise"
            | "a64.simd.integer-add-sub"
            | "a64.simd.unsigned-move-to-general"
            | "a64.fp.scalar-move"
            | "a64.fp.move-to-general"
            | "a64.fp.move-from-general"
            | "aarch32.vector.pack"
            | "aarch32.vector.unpack"
            | "aarch32.neon.bitwise"
            | "aarch32.shift"
            | "aarch32.data-processing"
            | "aarch32.multiply"
    );
    if !native {
        return Ok(None);
    }
    let arguments = helper
        .arguments
        .iter()
        .map(|argument| operand(builder, values, *argument))
        .collect::<Result<Vec<_>, _>>()?;
    let results = match helper.helper.as_ref() {
        "a64.extend-register" | "a64.load-store-register-offset" => {
            lower_native_extend(builder, helper, result_types, &arguments)?
        }
        "a64.sbfm" | "a64.bfm" | "a64.ubfm" => {
            lower_native_bitfield(builder, helper, result_types, &arguments)?
        }
        "a64.smaddl" | "a64.smsubl" | "a64.umaddl" | "a64.umsubl" => {
            lower_native_widening_multiply(builder, helper.helper.as_ref(), &arguments)?
        }
        "a64.smulh" => vec![builder.ins().smulhi(arguments[0], arguments[1])],
        "a64.umulh" => vec![builder.ins().umulhi(arguments[0], arguments[1])],
        "a64.extr" => lower_native_extract(builder, helper, result_types, &arguments)?,
        "a64.rev16" | "a64.rev32" | "a64.rev" | "a64.cls" => {
            lower_native_one_source(builder, helper.helper.as_ref(), &arguments)?
        }
        "a64.simd.zero-extend-load" => {
            vec![vector_from_low_integer(builder, arguments[0])]
        }
        "a64.simd.low-bits" => {
            vec![low_integer_from_vector(
                builder,
                arguments[0],
                cranelift_type(single_result_type(result_types)?),
            )]
        }
        "a64.simd.bitwise" => lower_native_a64_simd_bitwise(builder, helper, &arguments)?,
        "a64.simd.integer-add-sub" => lower_native_a64_simd_add_sub(builder, helper, &arguments)?,
        "a64.simd.unsigned-move-to-general" => {
            lower_native_a64_unsigned_move(builder, helper, result_types, &arguments)?
        }
        "a64.fp.scalar-move" => lower_native_a64_scalar_move(builder, helper, &arguments)?,
        "a64.fp.move-to-general" => {
            lower_native_a64_move_to_general(builder, helper, result_types, &arguments)?
        }
        "a64.fp.move-from-general" => {
            lower_native_a64_move_from_general(builder, helper, &arguments)?
        }
        "aarch32.vector.pack" => {
            vec![vector_from_halves(builder, arguments[0], arguments[1])]
        }
        "aarch32.vector.unpack" => {
            let (low, high) = vector_halves(builder, arguments[0]);
            vec![low, high]
        }
        "aarch32.neon.bitwise" => lower_native_aarch32_neon_bitwise(builder, helper, &arguments)?,
        "aarch32.shift" => lower_native_aarch32_shift(builder, helper, &arguments)?,
        "aarch32.data-processing" => {
            lower_native_aarch32_data_processing(builder, helper, &arguments)?
        }
        "aarch32.multiply" => lower_native_aarch32_multiply(builder, helper, &arguments)?,
        _ => unreachable!("native semantic helper classification is exhaustive"),
    };
    if results.len() != result_types.len() {
        return Err(CompilerError::new(format!(
            "native helper lowering for {} produced {} results, expected {}",
            helper.helper,
            results.len(),
            result_types.len()
        )));
    }
    Ok(Some(results))
}

fn single_result_type(result_types: &[IrType]) -> Result<IrType, CompilerError> {
    match result_types {
        [result] => Ok(*result),
        _ => Err(CompilerError::new("native helper requires one result")),
    }
}

fn helper_immediate(
    helper: &nixe_cpu::ir::op::HelperOperation,
    index: usize,
) -> Result<u64, CompilerError> {
    let value = helper.arguments.get(index).copied().ok_or_else(|| {
        CompilerError::new(format!("{} is missing argument {index}", helper.helper))
    })?;
    match value {
        Operand::Immediate(Immediate::I1(value)) => Ok(u64::from(value)),
        Operand::Immediate(Immediate::I8(value)) => Ok(u64::from(value)),
        Operand::Immediate(Immediate::I16(value)) => Ok(u64::from(value)),
        Operand::Immediate(Immediate::I32(value)) => Ok(u64::from(value)),
        Operand::Immediate(Immediate::I64(value)) => Ok(value),
        _ => Err(CompilerError::new(format!(
            "{} argument {index} must be an integer immediate",
            helper.helper
        ))),
    }
}

fn integer_constant(builder: &mut FunctionBuilder<'_>, ty: ir::Type, value: u64) -> ir::Value {
    builder.ins().iconst(ty, value as i64)
}

fn cast_integer(
    builder: &mut FunctionBuilder<'_>,
    value: ir::Value,
    to: ir::Type,
    signed: bool,
) -> ir::Value {
    let from = builder.func.dfg.value_type(value);
    match from.bits().cmp(&to.bits()) {
        std::cmp::Ordering::Less if signed => builder.ins().sextend(to, value),
        std::cmp::Ordering::Less => builder.ins().uextend(to, value),
        std::cmp::Ordering::Greater => builder.ins().ireduce(to, value),
        std::cmp::Ordering::Equal => value,
    }
}

fn lower_native_extend(
    builder: &mut FunctionBuilder<'_>,
    helper: &nixe_cpu::ir::op::HelperOperation,
    result_types: &[IrType],
    arguments: &[ir::Value],
) -> Result<Vec<ir::Value>, CompilerError> {
    let option = helper_immediate(helper, 1)? as u8;
    let shift = helper_immediate(helper, 2)? as u32;
    let source_bits = match option & 3 {
        0 => 8,
        1 => 16,
        2 => 32,
        3 => 64,
        _ => unreachable!(),
    };
    let source_ty = match source_bits {
        8 => types::I8,
        16 => types::I16,
        32 => types::I32,
        64 => types::I64,
        _ => unreachable!(),
    };
    let result_ty = cranelift_type(single_result_type(result_types)?);
    let narrowed = cast_integer(builder, arguments[0], source_ty, false);
    let extended = cast_integer(builder, narrowed, result_ty, option & 4 != 0);
    let result = if shift == 0 {
        extended
    } else {
        let amount = integer_constant(builder, result_ty, u64::from(shift));
        builder.ins().ishl(extended, amount)
    };
    Ok(vec![result])
}

fn lower_native_bitfield(
    builder: &mut FunctionBuilder<'_>,
    helper: &nixe_cpu::ir::op::HelperOperation,
    result_types: &[IrType],
    arguments: &[ir::Value],
) -> Result<Vec<ir::Value>, CompilerError> {
    let result_ty = cranelift_type(single_result_type(result_types)?);
    let width = result_ty.bits() as u8;
    let imm_r = helper_immediate(helper, 2)? as u8;
    let imm_s = helper_immediate(helper, 3)? as u8;
    let masks = decode_a64_bit_masks(width == 64, imm_r, imm_s, width, false)
        .map_err(|error| CompilerError::new(format!("invalid native bitfield masks: {error}")))?;
    let rotate = integer_constant(builder, result_ty, u64::from(imm_r));
    let rotated = builder.ins().rotr(arguments[1], rotate);
    let write = integer_constant(builder, result_ty, masks.write_mask);
    let rotated_write = builder.ins().band(rotated, write);
    let bottom = if helper.helper.as_ref() == "a64.bfm" {
        let inverse_write = integer_constant(builder, result_ty, !masks.write_mask);
        let preserved = builder.ins().band(arguments[0], inverse_write);
        builder.ins().bor(preserved, rotated_write)
    } else {
        rotated_write
    };
    let test = integer_constant(builder, result_ty, masks.test_mask);
    let bottom_test = builder.ins().band(bottom, test);
    let result = match helper.helper.as_ref() {
        "a64.ubfm" => bottom_test,
        "a64.bfm" => {
            let inverse_test = integer_constant(builder, result_ty, !masks.test_mask);
            let preserved = builder.ins().band(arguments[0], inverse_test);
            builder.ins().bor(preserved, bottom_test)
        }
        "a64.sbfm" => {
            let sign_shift = integer_constant(builder, result_ty, u64::from(imm_s));
            let sign = builder.ins().ushr(arguments[1], sign_shift);
            let sign = builder.ins().band_imm_u(sign, 1);
            let sign = builder.ins().ineg(sign);
            let inverse_test = integer_constant(builder, result_ty, !masks.test_mask);
            let top = builder.ins().band(sign, inverse_test);
            builder.ins().bor(top, bottom_test)
        }
        _ => unreachable!(),
    };
    Ok(vec![result])
}

fn lower_native_widening_multiply(
    builder: &mut FunctionBuilder<'_>,
    name: &str,
    arguments: &[ir::Value],
) -> Result<Vec<ir::Value>, CompilerError> {
    let signed = matches!(name, "a64.smaddl" | "a64.smsubl");
    let lhs = cast_integer(builder, arguments[0], types::I64, signed);
    let rhs = cast_integer(builder, arguments[1], types::I64, signed);
    let product = builder.ins().imul(lhs, rhs);
    let result = match name {
        "a64.smaddl" | "a64.umaddl" => builder.ins().iadd(product, arguments[2]),
        "a64.smsubl" | "a64.umsubl" => builder.ins().isub(arguments[2], product),
        _ => return Err(CompilerError::new("unknown widening multiply helper")),
    };
    Ok(vec![result])
}

fn lower_native_extract(
    builder: &mut FunctionBuilder<'_>,
    helper: &nixe_cpu::ir::op::HelperOperation,
    result_types: &[IrType],
    arguments: &[ir::Value],
) -> Result<Vec<ir::Value>, CompilerError> {
    let shift = helper_immediate(helper, 2)? as u32;
    if shift == 0 {
        return Ok(vec![arguments[1]]);
    }
    let ty = cranelift_type(single_result_type(result_types)?);
    let bits = ty.bits();
    let right_amount = integer_constant(builder, ty, u64::from(shift));
    let left_amount = integer_constant(builder, ty, u64::from(bits - shift));
    let low = builder.ins().ushr(arguments[1], right_amount);
    let high = builder.ins().ishl(arguments[0], left_amount);
    Ok(vec![builder.ins().bor(low, high)])
}

fn lower_native_one_source(
    builder: &mut FunctionBuilder<'_>,
    name: &str,
    arguments: &[ir::Value],
) -> Result<Vec<ir::Value>, CompilerError> {
    let value = arguments[0];
    let ty = builder.func.dfg.value_type(value);
    let result = match name {
        "a64.rev" => builder.ins().bswap(value),
        "a64.rev16" => {
            let low_mask = if ty == types::I64 {
                0x00ff_00ff_00ff_00ff
            } else {
                0x00ff_00ff
            };
            let mask = integer_constant(builder, ty, low_mask);
            let low = builder.ins().band(value, mask);
            let low = builder.ins().ishl_imm_u(low, 8);
            let high = builder.ins().ushr_imm_u(value, 8);
            let mask = integer_constant(builder, ty, low_mask);
            let high = builder.ins().band(high, mask);
            builder.ins().bor(low, high)
        }
        "a64.rev32" if ty == types::I32 => builder.ins().bswap(value),
        "a64.rev32" => {
            let swapped = builder.ins().bswap(value);
            builder.ins().rotl_imm_u(swapped, 32)
        }
        "a64.cls" => {
            let sign = builder.ins().sshr_imm_u(value, i64::from(ty.bits() - 1));
            let changed = builder.ins().bxor(value, sign);
            let leading = builder.ins().clz(changed);
            builder.ins().iadd_imm_s(leading, -1)
        }
        _ => return Err(CompilerError::new("unknown one-source native helper")),
    };
    Ok(vec![result])
}

fn vector_halves(builder: &mut FunctionBuilder<'_>, vector: ir::Value) -> (ir::Value, ir::Value) {
    let flags = bitcast_flags(builder);
    let integer = builder.ins().bitcast(types::I128, flags, vector);
    builder.ins().isplit(integer)
}

fn vector_from_halves(
    builder: &mut FunctionBuilder<'_>,
    low: ir::Value,
    high: ir::Value,
) -> ir::Value {
    let low = cast_integer(builder, low, types::I64, false);
    let high = cast_integer(builder, high, types::I64, false);
    let integer = builder.ins().iconcat(low, high);
    let flags = bitcast_flags(builder);
    builder.ins().bitcast(types::I8X16, flags, integer)
}

fn vector_from_low_integer(builder: &mut FunctionBuilder<'_>, value: ir::Value) -> ir::Value {
    let ty = builder.func.dfg.value_type(value);
    let integer = if ty == types::I128 {
        value
    } else {
        builder.ins().uextend(types::I128, value)
    };
    let flags = bitcast_flags(builder);
    builder.ins().bitcast(types::I8X16, flags, integer)
}

fn low_integer_from_vector(
    builder: &mut FunctionBuilder<'_>,
    vector: ir::Value,
    result_ty: ir::Type,
) -> ir::Value {
    let flags = bitcast_flags(builder);
    let integer = builder.ins().bitcast(types::I128, flags, vector);
    if result_ty == types::I128 {
        integer
    } else {
        builder.ins().ireduce(result_ty, integer)
    }
}

fn vector_mask(builder: &mut FunctionBuilder<'_>, bits: u32) -> ir::Value {
    let mask = match bits {
        0 => 0,
        1..=127 => (1_u128 << bits) - 1,
        128 => u128::MAX,
        _ => unreachable!("vector mask width is bounded"),
    };
    let integer = integer_128(builder, mask);
    let flags = bitcast_flags(builder);
    builder.ins().bitcast(types::I8X16, flags, integer)
}

fn semantic_fields(
    helper: &nixe_cpu::ir::op::HelperOperation,
    token_index: usize,
) -> Result<nixe_cpu::decode::a64::fp_simd::Operands, CompilerError> {
    Ok(semantic_instruction(helper_immediate(helper, token_index)?).operands())
}

fn lower_native_a64_semantic_vector(
    builder: &mut FunctionBuilder<'_>,
    helper: &nixe_cpu::ir::op::HelperOperation,
    values: &BTreeMap<ValueId, ir::Value>,
) -> Result<Option<Vec<ir::Value>>, CompilerError> {
    let token = helper_immediate(helper, 8)?;
    let instruction = semantic_instruction(token);
    let supported = matches!(
        instruction,
        A64FpSimdInstruction::DuplicateGeneral(_)
            | A64FpSimdInstruction::DuplicateElement(_)
            | A64FpSimdInstruction::ModifiedImmediate(_)
            | A64FpSimdInstruction::InsertElement(_)
            | A64FpSimdInstruction::InsertGeneral(_)
            | A64FpSimdInstruction::PermuteTwoSource(_)
            | A64FpSimdInstruction::Extract(_)
            | A64FpSimdInstruction::ExtractNarrow(_)
            | A64FpSimdInstruction::IntegerCompare(_)
            | A64FpSimdInstruction::IntegerPairwise(_)
            | A64FpSimdInstruction::IntegerMinMax(_)
            | A64FpSimdInstruction::VectorSignedShiftRegister(_)
            | A64FpSimdInstruction::VectorUnsignedShiftRegister(_)
            | A64FpSimdInstruction::ScalarAbsolute(_)
            | A64FpSimdInstruction::ScalarNegate(_)
            | A64FpSimdInstruction::ScalarFloatImmediate(_)
            | A64FpSimdInstruction::ScalarFloatConditionalSelect(_)
            | A64FpSimdInstruction::VectorFloatImmediate(_)
            | A64FpSimdInstruction::VectorFloatAbsolute(_)
            | A64FpSimdInstruction::VectorFloatNegate(_)
            | A64FpSimdInstruction::ShiftRightNarrow(_)
            | A64FpSimdInstruction::ScalarShiftRightImmediate(_)
            | A64FpSimdInstruction::VectorShiftRightImmediate(_)
            | A64FpSimdInstruction::ScalarShiftLeftImmediate(_)
            | A64FpSimdInstruction::VectorShiftLeftImmediate(_)
            | A64FpSimdInstruction::CountBits(_)
            | A64FpSimdInstruction::AddAcrossVector(_)
    );
    if !supported {
        return Ok(None);
    }
    let arguments = helper
        .arguments
        .iter()
        .map(|argument| operand(builder, values, *argument))
        .collect::<Result<Vec<_>, _>>()?;
    let fields = instruction.operands();
    let result = match instruction {
        A64FpSimdInstruction::DuplicateGeneral(_) => {
            let lane_bits = 8_u32 << fields.immediate_5.trailing_zeros();
            let lane = integer_lane_type(lane_bits)?;
            let source = cast_integer(builder, arguments[4], lane, false);
            let vector_ty = lane
                .by(128 / lane_bits)
                .ok_or_else(|| CompilerError::new("invalid DUP general vector shape"))?;
            let result = builder.ins().splat(vector_ty, source);
            opaque_vector_128(builder, result, fields.vector_128)?
        }
        A64FpSimdInstruction::DuplicateElement(_) => {
            let size_shift = fields.immediate_5.trailing_zeros();
            let lane_bits = 8_u32 << size_shift;
            let lane_index = fields.immediate_5 >> (size_shift + 1);
            let lane = integer_lane_type(lane_bits)?;
            let vector_ty = lane
                .by(128 / lane_bits)
                .ok_or_else(|| CompilerError::new("invalid DUP element vector shape"))?;
            let source = vector_bitcast(builder, arguments[0], vector_ty);
            let element = builder.ins().extractlane(source, lane_index);
            let result = builder.ins().splat(vector_ty, element);
            opaque_vector_128(builder, result, fields.vector_128)?
        }
        A64FpSimdInstruction::ModifiedImmediate(_) => {
            let immediate = expand_a64_modified_immediate(
                fields.cmode,
                fields.immediate_8,
                fields.operation_bit,
            )?;
            let replicated = u128::from(immediate) | (u128::from(immediate) << 64);
            let value = vector_constant(builder, replicated);
            let result = if fields.cmode <= 11 && fields.cmode & 1 != 0 {
                if fields.operation_bit {
                    builder.ins().band(arguments[3], value)
                } else {
                    builder.ins().bor(arguments[3], value)
                }
            } else {
                value
            };
            let active = vector_mask(builder, if fields.vector_128 { 128 } else { 64 });
            builder.ins().band(result, active)
        }
        A64FpSimdInstruction::InsertElement(_) => {
            let size_shift = fields.immediate_5.trailing_zeros();
            let lane_bits = 8_u32 << size_shift;
            let destination_lane = fields.immediate_5 >> (size_shift + 1);
            let source_lane = fields.immediate_4 >> size_shift;
            let lane = integer_lane_type(lane_bits)?;
            let vector_ty = lane
                .by(128 / lane_bits)
                .ok_or_else(|| CompilerError::new("invalid INS element vector shape"))?;
            let source = vector_bitcast(builder, arguments[0], vector_ty);
            let previous = vector_bitcast(builder, arguments[3], vector_ty);
            let element = builder.ins().extractlane(source, source_lane);
            let result = builder
                .ins()
                .insertlane(previous, element, destination_lane);
            vector_bitcast(builder, result, types::I8X16)
        }
        A64FpSimdInstruction::InsertGeneral(_) => {
            let size_shift = fields.immediate_5.trailing_zeros();
            let lane_bits = 8_u32 << size_shift;
            let destination_lane = fields.immediate_5 >> (size_shift + 1);
            let lane = integer_lane_type(lane_bits)?;
            let vector_ty = lane
                .by(128 / lane_bits)
                .ok_or_else(|| CompilerError::new("invalid INS general vector shape"))?;
            let previous = vector_bitcast(builder, arguments[3], vector_ty);
            let element = cast_integer(builder, arguments[4], lane, false);
            let result = builder
                .ins()
                .insertlane(previous, element, destination_lane);
            vector_bitcast(builder, result, types::I8X16)
        }
        A64FpSimdInstruction::Extract(_) => {
            lower_native_a64_vector_extract(builder, arguments[0], arguments[1], fields)?
        }
        A64FpSimdInstruction::PermuteTwoSource(_) => {
            lower_native_a64_permute(builder, arguments[0], arguments[1], fields)?
        }
        A64FpSimdInstruction::IntegerCompare(_) => {
            lower_native_a64_integer_compare(builder, arguments[0], arguments[1], fields)?
        }
        A64FpSimdInstruction::IntegerPairwise(_) => {
            lower_native_a64_integer_pairwise(builder, arguments[0], arguments[1], fields)?
        }
        A64FpSimdInstruction::IntegerMinMax(_) => {
            lower_native_a64_integer_min_max(builder, arguments[0], arguments[1], fields)?
        }
        A64FpSimdInstruction::ExtractNarrow(_) => {
            lower_native_a64_extract_narrow(builder, arguments[0], arguments[3], fields, false)?
        }
        A64FpSimdInstruction::ShiftRightNarrow(_) => {
            lower_native_a64_extract_narrow(builder, arguments[0], arguments[3], fields, true)?
        }
        A64FpSimdInstruction::ScalarShiftRightImmediate(_)
        | A64FpSimdInstruction::VectorShiftRightImmediate(_)
        | A64FpSimdInstruction::ScalarShiftLeftImmediate(_)
        | A64FpSimdInstruction::VectorShiftLeftImmediate(_) => {
            lower_native_a64_immediate_shift(builder, arguments[0], fields, instruction)?
        }
        A64FpSimdInstruction::VectorSignedShiftRegister(_)
        | A64FpSimdInstruction::VectorUnsignedShiftRegister(_) => lower_native_a64_register_shift(
            builder,
            arguments[0],
            arguments[1],
            fields,
            matches!(
                instruction,
                A64FpSimdInstruction::VectorSignedShiftRegister(_)
            ),
        )?,
        A64FpSimdInstruction::ScalarAbsolute(_) | A64FpSimdInstruction::ScalarNegate(_) => {
            let width = match fields.opc {
                0 => 32,
                1 => 64,
                3 => 16,
                _ => return Err(CompilerError::new("invalid scalar sign width")),
            };
            let active = vector_mask(builder, width);
            let source = builder.ins().band(arguments[0], active);
            let sign = vector_constant(builder, 1_u128 << (width - 1));
            if matches!(instruction, A64FpSimdInstruction::ScalarNegate(_)) {
                builder.ins().bxor(source, sign)
            } else {
                let ones = vector_mask(builder, 128);
                let not_sign = builder.ins().bxor(sign, ones);
                builder.ins().band(source, not_sign)
            }
        }
        A64FpSimdInstruction::VectorFloatAbsolute(_)
        | A64FpSimdInstruction::VectorFloatNegate(_) => {
            let lane_bits = if fields.opc & 1 == 0 { 32 } else { 64 };
            let vector_bits = if fields.vector_128 { 128 } else { 64 };
            let mut sign_mask = 0_u128;
            for offset in (0..vector_bits).step_by(lane_bits as usize) {
                sign_mask |= 1_u128 << (offset + lane_bits - 1);
            }
            let sign = vector_constant(builder, sign_mask);
            let active = vector_mask(builder, vector_bits);
            let source = builder.ins().band(arguments[0], active);
            if matches!(instruction, A64FpSimdInstruction::VectorFloatNegate(_)) {
                builder.ins().bxor(source, sign)
            } else {
                let ones = vector_mask(builder, 128);
                let not_sign = builder.ins().bxor(sign, ones);
                builder.ins().band(source, not_sign)
            }
        }
        A64FpSimdInstruction::ScalarFloatImmediate(_) => {
            let (exponent_bits, fraction_bits) = match fields.opc {
                0 => (8, 23),
                1 => (11, 52),
                3 => (5, 10),
                _ => return Err(CompilerError::new("invalid scalar FP immediate width")),
            };
            let immediate =
                expand_vfp_immediate(fields.fp_immediate_8, exponent_bits, fraction_bits);
            let immediate = integer_constant(builder, types::I64, immediate);
            vector_from_low_integer(builder, immediate)
        }
        A64FpSimdInstruction::VectorFloatImmediate(_) => {
            let (lane, lane_bits) = if fields.operation_bit {
                (expand_vfp_immediate(fields.immediate_8, 11, 52), 64)
            } else {
                (expand_vfp_immediate(fields.immediate_8, 8, 23), 32)
            };
            let value = if lane_bits == 64 {
                u128::from(lane) | (u128::from(lane) << 64)
            } else {
                let lane = u128::from(lane as u32);
                lane | (lane << 32) | (lane << 64) | (lane << 96)
            };
            let value = vector_constant(builder, value);
            let active = vector_mask(builder, if fields.vector_128 { 128 } else { 64 });
            builder.ins().band(value, active)
        }
        A64FpSimdInstruction::ScalarFloatConditionalSelect(_) => {
            let condition = evaluate_condition(
                builder,
                arguments[5],
                Condition::from_encoding(fields.condition),
                true,
            );
            let selected = builder.ins().select(condition, arguments[0], arguments[1]);
            let active = vector_mask(builder, if fields.opc == 0 { 32 } else { 64 });
            builder.ins().band(selected, active)
        }
        A64FpSimdInstruction::CountBits(_) => {
            let counted = builder.ins().popcnt(arguments[0]);
            let active = vector_mask(builder, if fields.vector_128 { 128 } else { 64 });
            builder.ins().band(counted, active)
        }
        A64FpSimdInstruction::AddAcrossVector(_) => {
            lower_native_a64_add_across(builder, arguments[0], fields)?
        }
        _ => unreachable!("native semantic-vector classification is exhaustive"),
    };
    Ok(Some(vec![result, arguments[7]]))
}

fn integer_lane_type(bits: u32) -> Result<ir::Type, CompilerError> {
    match bits {
        8 => Ok(types::I8),
        16 => Ok(types::I16),
        32 => Ok(types::I32),
        64 => Ok(types::I64),
        _ => Err(CompilerError::new("invalid integer vector lane width")),
    }
}

fn vector_bitcast(builder: &mut FunctionBuilder<'_>, value: ir::Value, to: ir::Type) -> ir::Value {
    if builder.func.dfg.value_type(value) == to {
        value
    } else {
        let flags = bitcast_flags(builder);
        builder.ins().bitcast(to, flags, value)
    }
}

fn opaque_vector_128(
    builder: &mut FunctionBuilder<'_>,
    value: ir::Value,
    full_width: bool,
) -> Result<ir::Value, CompilerError> {
    let result = vector_bitcast(builder, value, types::I8X16);
    if full_width {
        Ok(result)
    } else {
        let active = vector_mask(builder, 64);
        Ok(builder.ins().band(result, active))
    }
}

fn vector_constant(builder: &mut FunctionBuilder<'_>, value: u128) -> ir::Value {
    let integer = integer_128(builder, value);
    vector_bitcast(builder, integer, types::I8X16)
}

fn expand_a64_modified_immediate(
    cmode: u8,
    immediate: u8,
    operation_bit: bool,
) -> Result<u64, CompilerError> {
    let immediate = u64::from(immediate);
    let value = match cmode {
        0..=7 => {
            let lane = immediate << ((cmode >> 1) * 8);
            lane | (lane << 32)
        }
        8..=11 => {
            let lane = immediate << (((cmode >> 1) & 1) * 8);
            lane | (lane << 16) | (lane << 32) | (lane << 48)
        }
        12 => {
            let lane = (immediate << 8) | 0xff;
            lane | (lane << 32)
        }
        13 => {
            let lane = (immediate << 16) | 0xffff;
            lane | (lane << 32)
        }
        14 if !operation_bit => immediate * 0x0101_0101_0101_0101,
        14 => {
            let mut result = 0_u64;
            for bit in 0..8 {
                if immediate & (1 << bit) != 0 {
                    result |= 0xff << (bit * 8);
                }
            }
            result
        }
        _ => return Err(CompilerError::new("invalid SIMD modified immediate")),
    };
    Ok(if operation_bit && cmode != 14 {
        !value
    } else {
        value
    })
}

fn expand_vfp_immediate(immediate: u8, exponent_bits: u32, fraction_bits: u32) -> u64 {
    let sign = u64::from(immediate >> 7);
    let exponent_control = u64::from((immediate >> 6) & 1);
    let exponent_tail = u64::from((immediate >> 4) & 3);
    let fraction_head = u64::from(immediate & 0xf);
    let sign_shift = exponent_bits + fraction_bits;
    let repeated_count = exponent_bits - 3;
    let repeated = if exponent_control == 0 {
        0
    } else {
        (1_u64 << repeated_count) - 1
    };
    (sign << sign_shift)
        | ((exponent_control ^ 1) << (sign_shift - 1))
        | (repeated << (fraction_bits + 2))
        | (exponent_tail << fraction_bits)
        | (fraction_head << (fraction_bits - 4))
}

fn lower_native_a64_vector_extract(
    builder: &mut FunctionBuilder<'_>,
    first: ir::Value,
    second: ir::Value,
    fields: nixe_cpu::decode::a64::fp_simd::Operands,
) -> Result<ir::Value, CompilerError> {
    let offset_bits = u32::from(fields.immediate_4) * 8;
    let vector_bits = if fields.vector_128 { 128 } else { 64 };
    if offset_bits == 0 {
        let active = vector_mask(builder, vector_bits);
        return Ok(builder.ins().band(first, active));
    }
    if vector_bits == 64 {
        let (first, _) = vector_halves(builder, first);
        let (second, _) = vector_halves(builder, second);
        let low = builder.ins().ushr_imm_u(first, i64::from(offset_bits));
        let high = builder
            .ins()
            .ishl_imm_u(second, i64::from(64 - offset_bits));
        let result = builder.ins().bor(low, high);
        return Ok(vector_from_low_integer(builder, result));
    }
    let (first_low, first_high) = vector_halves(builder, first);
    let (second_low, second_high) = vector_halves(builder, second);
    let (result_low, result_high) = match offset_bits {
        1..=63 => (
            merge_shifted_halves(builder, first_low, first_high, offset_bits),
            merge_shifted_halves(builder, first_high, second_low, offset_bits),
        ),
        64 => (first_high, second_low),
        65..=127 => {
            let amount = offset_bits - 64;
            (
                merge_shifted_halves(builder, first_high, second_low, amount),
                merge_shifted_halves(builder, second_low, second_high, amount),
            )
        }
        _ => return Err(CompilerError::new("invalid 128-bit EXT offset")),
    };
    Ok(vector_from_halves(builder, result_low, result_high))
}

fn merge_shifted_halves(
    builder: &mut FunctionBuilder<'_>,
    low: ir::Value,
    high: ir::Value,
    amount: u32,
) -> ir::Value {
    debug_assert!((1..64).contains(&amount));
    let low = builder.ins().ushr_imm_u(low, i64::from(amount));
    let high = builder.ins().ishl_imm_u(high, i64::from(64 - amount));
    builder.ins().bor(low, high)
}

fn lower_native_a64_add_across(
    builder: &mut FunctionBuilder<'_>,
    source: ir::Value,
    fields: nixe_cpu::decode::a64::fp_simd::Operands,
) -> Result<ir::Value, CompilerError> {
    let bits = fields.helper_token.helper_abi_value();
    let lane_bits = 8_u32 << ((bits >> 22) & 3);
    let vector_bits = if fields.vector_128 { 128 } else { 64 };
    let lane_count = vector_bits / lane_bits;
    let lane = integer_lane_type(lane_bits)?;
    let vector_ty = lane
        .by(128 / lane_bits)
        .ok_or_else(|| CompilerError::new("invalid ADDV vector shape"))?;
    let source = vector_bitcast(builder, source, vector_ty);
    let mut result = builder.ins().iconst(lane, 0);
    for index in 0..lane_count {
        let value = builder.ins().extractlane(source, index as u8);
        result = builder.ins().iadd(result, value);
    }
    Ok(vector_from_low_integer(builder, result))
}

fn zero_lane_vector(
    builder: &mut FunctionBuilder<'_>,
    lane_bits: u32,
) -> Result<(ir::Type, ir::Value), CompilerError> {
    let lane = integer_lane_type(lane_bits)?;
    let vector_ty = lane
        .by(128 / lane_bits)
        .ok_or_else(|| CompilerError::new("invalid native vector shape"))?;
    let zero = builder.ins().iconst(lane, 0);
    Ok((vector_ty, builder.ins().splat(vector_ty, zero)))
}

fn finish_lane_vector(
    builder: &mut FunctionBuilder<'_>,
    value: ir::Value,
    full_width: bool,
) -> ir::Value {
    let value = vector_bitcast(builder, value, types::I8X16);
    if full_width {
        value
    } else {
        let active = vector_mask(builder, 64);
        builder.ins().band(value, active)
    }
}

fn lower_native_a64_permute(
    builder: &mut FunctionBuilder<'_>,
    first: ir::Value,
    second: ir::Value,
    fields: nixe_cpu::decode::a64::fp_simd::Operands,
) -> Result<ir::Value, CompilerError> {
    let lane_bits = 8_u32 << fields.opc;
    let vector_bits = if fields.vector_128 { 128 } else { 64 };
    let lane_count = vector_bits / lane_bits;
    let half = lane_count / 2;
    let (vector_ty, mut result) = zero_lane_vector(builder, lane_bits)?;
    let first = vector_bitcast(builder, first, vector_ty);
    let second = vector_bitcast(builder, second, vector_ty);
    let operation = fields
        .permute_operation
        .ok_or_else(|| CompilerError::new("SIMD permutation token has no operation"))?;
    for destination_lane in 0..lane_count {
        let (source, source_lane) = match operation {
            PermuteOperation::UnzipPrimary | PermuteOperation::UnzipSecondary => {
                let odd = u32::from(matches!(operation, PermuteOperation::UnzipSecondary));
                if destination_lane < half {
                    (first, destination_lane * 2 + odd)
                } else {
                    (second, (destination_lane - half) * 2 + odd)
                }
            }
            PermuteOperation::TransposePrimary | PermuteOperation::TransposeSecondary => {
                let odd = u32::from(matches!(operation, PermuteOperation::TransposeSecondary));
                let source = if destination_lane & 1 == 0 {
                    first
                } else {
                    second
                };
                (source, (destination_lane / 2) * 2 + odd)
            }
            PermuteOperation::ZipPrimary | PermuteOperation::ZipSecondary => {
                let upper = u32::from(matches!(operation, PermuteOperation::ZipSecondary));
                let source = if destination_lane & 1 == 0 {
                    first
                } else {
                    second
                };
                (source, destination_lane / 2 + upper * half)
            }
        };
        let lane = builder.ins().extractlane(source, source_lane as u8);
        result = builder
            .ins()
            .insertlane(result, lane, destination_lane as u8);
    }
    Ok(finish_lane_vector(builder, result, fields.vector_128))
}

fn lower_native_a64_integer_compare(
    builder: &mut FunctionBuilder<'_>,
    lhs: ir::Value,
    rhs: ir::Value,
    fields: nixe_cpu::decode::a64::fp_simd::Operands,
) -> Result<ir::Value, CompilerError> {
    let lane_bits = 8_u32 << fields.opc;
    let vector_bits = if fields.vector_128 { 128 } else { 64 };
    let lane_count = vector_bits / lane_bits;
    let lane = integer_lane_type(lane_bits)?;
    let (vector_ty, mut result) = zero_lane_vector(builder, lane_bits)?;
    let lhs = vector_bitcast(builder, lhs, vector_ty);
    let rhs = vector_bitcast(builder, rhs, vector_ty);
    let zero = builder.ins().iconst(lane, 0);
    let ones = builder.ins().iconst(lane, -1);
    let comparison = fields
        .integer_comparison
        .ok_or_else(|| CompilerError::new("SIMD comparison token has no predicate"))?;
    for index in 0..lane_count {
        let lhs_lane = builder.ins().extractlane(lhs, index as u8);
        let rhs_lane = if fields.compare_with_zero {
            zero
        } else {
            builder.ins().extractlane(rhs, index as u8)
        };
        let condition = match comparison {
            IntegerComparison::SignedGreaterThan => {
                builder
                    .ins()
                    .icmp(IntCC::SignedGreaterThan, lhs_lane, rhs_lane)
            }
            IntegerComparison::UnsignedGreaterThan => {
                builder
                    .ins()
                    .icmp(IntCC::UnsignedGreaterThan, lhs_lane, rhs_lane)
            }
            IntegerComparison::SignedGreaterThanOrEqual => {
                builder
                    .ins()
                    .icmp(IntCC::SignedGreaterThanOrEqual, lhs_lane, rhs_lane)
            }
            IntegerComparison::UnsignedGreaterThanOrEqual => {
                builder
                    .ins()
                    .icmp(IntCC::UnsignedGreaterThanOrEqual, lhs_lane, rhs_lane)
            }
            IntegerComparison::SignedLessThan => {
                builder
                    .ins()
                    .icmp(IntCC::SignedLessThan, lhs_lane, rhs_lane)
            }
            IntegerComparison::SignedLessThanOrEqual => {
                builder
                    .ins()
                    .icmp(IntCC::SignedLessThanOrEqual, lhs_lane, rhs_lane)
            }
            IntegerComparison::NonzeroBitTest => {
                let bits = builder.ins().band(lhs_lane, rhs_lane);
                builder.ins().icmp_imm_s(IntCC::NotEqual, bits, 0)
            }
            IntegerComparison::Equal => builder.ins().icmp(IntCC::Equal, lhs_lane, rhs_lane),
        };
        let lane_value = builder.ins().select(condition, ones, zero);
        result = builder.ins().insertlane(result, lane_value, index as u8);
    }
    Ok(finish_lane_vector(builder, result, fields.vector_128))
}

fn select_pairwise_lane(
    builder: &mut FunctionBuilder<'_>,
    lhs: ir::Value,
    rhs: ir::Value,
    operation: PairwiseOperation,
) -> Result<ir::Value, CompilerError> {
    Ok(match operation {
        PairwiseOperation::Add => builder.ins().iadd(lhs, rhs),
        PairwiseOperation::SignedMaximum => {
            let condition = builder
                .ins()
                .icmp(IntCC::SignedGreaterThanOrEqual, lhs, rhs);
            builder.ins().select(condition, lhs, rhs)
        }
        PairwiseOperation::SignedMinimum => {
            let condition = builder.ins().icmp(IntCC::SignedLessThanOrEqual, lhs, rhs);
            builder.ins().select(condition, lhs, rhs)
        }
        PairwiseOperation::UnsignedMaximum => {
            let condition = builder
                .ins()
                .icmp(IntCC::UnsignedGreaterThanOrEqual, lhs, rhs);
            builder.ins().select(condition, lhs, rhs)
        }
        PairwiseOperation::UnsignedMinimum => {
            let condition = builder.ins().icmp(IntCC::UnsignedLessThanOrEqual, lhs, rhs);
            builder.ins().select(condition, lhs, rhs)
        }
    })
}

fn lower_native_a64_integer_pairwise(
    builder: &mut FunctionBuilder<'_>,
    first: ir::Value,
    second: ir::Value,
    fields: nixe_cpu::decode::a64::fp_simd::Operands,
) -> Result<ir::Value, CompilerError> {
    let lane_bits = 8_u32 << fields.opc;
    let vector_bits = if fields.vector_128 { 128 } else { 64 };
    let lanes_per_source = vector_bits / lane_bits;
    let (vector_ty, mut result) = zero_lane_vector(builder, lane_bits)?;
    let first = vector_bitcast(builder, first, vector_ty);
    let second = vector_bitcast(builder, second, vector_ty);
    let operation = fields
        .pairwise_operation
        .ok_or_else(|| CompilerError::new("SIMD pairwise token has no operation"))?;
    for (source_index, source) in [first, second].into_iter().enumerate() {
        for pair in 0..(lanes_per_source / 2) {
            let lhs = builder.ins().extractlane(source, (pair * 2) as u8);
            let rhs = builder.ins().extractlane(source, (pair * 2 + 1) as u8);
            let reduced = select_pairwise_lane(builder, lhs, rhs, operation)?;
            let destination = source_index as u32 * (lanes_per_source / 2) + pair;
            result = builder.ins().insertlane(result, reduced, destination as u8);
        }
    }
    Ok(finish_lane_vector(builder, result, fields.vector_128))
}

fn lower_native_a64_integer_min_max(
    builder: &mut FunctionBuilder<'_>,
    lhs: ir::Value,
    rhs: ir::Value,
    fields: nixe_cpu::decode::a64::fp_simd::Operands,
) -> Result<ir::Value, CompilerError> {
    let lane_bits = 8_u32 << fields.opc;
    let vector_bits = if fields.vector_128 { 128 } else { 64 };
    let lane_count = vector_bits / lane_bits;
    let (vector_ty, mut result) = zero_lane_vector(builder, lane_bits)?;
    let lhs = vector_bitcast(builder, lhs, vector_ty);
    let rhs = vector_bitcast(builder, rhs, vector_ty);
    let operation = fields
        .pairwise_operation
        .ok_or_else(|| CompilerError::new("SIMD min/max token has no operation"))?;
    if operation == PairwiseOperation::Add {
        return Err(CompilerError::new(
            "SIMD min/max token contains pairwise add",
        ));
    }
    for index in 0..lane_count {
        let lhs_lane = builder.ins().extractlane(lhs, index as u8);
        let rhs_lane = builder.ins().extractlane(rhs, index as u8);
        let selected = select_pairwise_lane(builder, lhs_lane, rhs_lane, operation)?;
        result = builder.ins().insertlane(result, selected, index as u8);
    }
    Ok(finish_lane_vector(builder, result, fields.vector_128))
}

fn lower_native_a64_extract_narrow(
    builder: &mut FunctionBuilder<'_>,
    source: ir::Value,
    previous: ir::Value,
    fields: nixe_cpu::decode::a64::fp_simd::Operands,
    shift_right: bool,
) -> Result<ir::Value, CompilerError> {
    let (destination_lane_bits, shift) = if shift_right {
        let bits = fields.helper_token.helper_abi_value();
        let immediate_high = (bits >> 19) & 0xf;
        let immediate_low = (bits >> 16) & 7;
        let destination_lane_bits = 8_u32 << (31 - immediate_high.leading_zeros());
        let source_lane_bits = destination_lane_bits * 2;
        let immediate = (immediate_high << 3) | immediate_low;
        (destination_lane_bits, source_lane_bits - immediate)
    } else {
        (8_u32 << fields.opc, 0)
    };
    let source_lane_bits = destination_lane_bits * 2;
    let lane_count = 128 / source_lane_bits;
    let source_lane = integer_lane_type(source_lane_bits)?;
    let source_ty = source_lane
        .by(lane_count)
        .ok_or_else(|| CompilerError::new("invalid narrowing source shape"))?;
    let source = vector_bitcast(builder, source, source_ty);
    let destination_lane = integer_lane_type(destination_lane_bits)?;
    let destination_ty = destination_lane
        .by(128 / destination_lane_bits)
        .ok_or_else(|| CompilerError::new("invalid narrowing destination shape"))?;
    let mut result = if fields.vector_128 {
        vector_bitcast(builder, previous, destination_ty)
    } else {
        let zero = builder.ins().iconst(destination_lane, 0);
        builder.ins().splat(destination_ty, zero)
    };
    let first_destination = if fields.vector_128 { lane_count } else { 0 };
    for index in 0..lane_count {
        let value = builder.ins().extractlane(source, index as u8);
        let value = if shift == 0 {
            value
        } else {
            builder.ins().ushr_imm_u(value, i64::from(shift))
        };
        let value = builder.ins().ireduce(destination_lane, value);
        result = builder
            .ins()
            .insertlane(result, value, (first_destination + index) as u8);
    }
    Ok(vector_bitcast(builder, result, types::I8X16))
}

fn lower_native_a64_immediate_shift(
    builder: &mut FunctionBuilder<'_>,
    source: ir::Value,
    fields: nixe_cpu::decode::a64::fp_simd::Operands,
    instruction: A64FpSimdInstruction,
) -> Result<ir::Value, CompilerError> {
    let bits = fields.helper_token.helper_abi_value();
    let immediate = (bits >> 16) & 0x7f;
    let immediate_high = immediate >> 3;
    let lane_bits = 8_u32 << (31 - immediate_high.leading_zeros());
    let right = matches!(
        instruction,
        A64FpSimdInstruction::ScalarShiftRightImmediate(_)
            | A64FpSimdInstruction::VectorShiftRightImmediate(_)
    );
    let scalar = matches!(
        instruction,
        A64FpSimdInstruction::ScalarShiftRightImmediate(_)
            | A64FpSimdInstruction::ScalarShiftLeftImmediate(_)
    );
    let shift = if right {
        2 * lane_bits - immediate
    } else {
        immediate - lane_bits
    };
    let active_bits = if scalar || !fields.vector_128 {
        64
    } else {
        128
    };
    let lane_count = active_bits / lane_bits;
    let lane = integer_lane_type(lane_bits)?;
    let vector_ty = lane
        .by(128 / lane_bits)
        .ok_or_else(|| CompilerError::new("invalid immediate-shift vector shape"))?;
    let source = vector_bitcast(builder, source, vector_ty);
    let zero = builder.ins().iconst(lane, 0);
    let mut result = builder.ins().splat(vector_ty, zero);
    for index in 0..lane_count {
        let value = builder.ins().extractlane(source, index as u8);
        let shifted = if right && !fields.operation_bit {
            builder.ins().sshr_imm_u(value, i64::from(shift))
        } else if right {
            builder.ins().ushr_imm_u(value, i64::from(shift))
        } else {
            builder.ins().ishl_imm_u(value, i64::from(shift))
        };
        result = builder.ins().insertlane(result, shifted, index as u8);
    }
    Ok(vector_bitcast(builder, result, types::I8X16))
}

fn lower_native_a64_register_shift(
    builder: &mut FunctionBuilder<'_>,
    values: ir::Value,
    shifts: ir::Value,
    fields: nixe_cpu::decode::a64::fp_simd::Operands,
    signed: bool,
) -> Result<ir::Value, CompilerError> {
    let lane_bits = 8_u32 << fields.opc;
    let vector_bits = if fields.vector_128 { 128 } else { 64 };
    let lane_count = vector_bits / lane_bits;
    let lane = integer_lane_type(lane_bits)?;
    let vector_ty = lane
        .by(128 / lane_bits)
        .ok_or_else(|| CompilerError::new("invalid register-shift vector shape"))?;
    let values = vector_bitcast(builder, values, vector_ty);
    let shifts = vector_bitcast(builder, shifts, vector_ty);
    let zero = builder.ins().iconst(lane, 0);
    let mut result = builder.ins().splat(vector_ty, zero);
    for index in 0..lane_count {
        let value = builder.ins().extractlane(values, index as u8);
        let distance = builder.ins().extractlane(shifts, index as u8);
        let distance = cast_integer(builder, distance, types::I8, false);
        let distance = builder.ins().sextend(types::I32, distance);
        let nonnegative = builder
            .ins()
            .icmp_imm_s(IntCC::SignedGreaterThanOrEqual, distance, 0);
        let negative_distance = builder.ins().ineg(distance);
        let magnitude = builder
            .ins()
            .select(nonnegative, distance, negative_distance);
        let out_of_range = builder.ins().icmp_imm_u(
            IntCC::UnsignedGreaterThanOrEqual,
            magnitude,
            lane_bits as i64,
        );
        let lane_amount = cast_integer(builder, magnitude, lane, false);
        let left = builder.ins().ishl(value, lane_amount);
        let right = if signed {
            builder.ins().sshr(value, lane_amount)
        } else {
            builder.ins().ushr(value, lane_amount)
        };
        let right_fill = if signed {
            builder.ins().sshr_imm_u(value, i64::from(lane_bits - 1))
        } else {
            zero
        };
        let right = builder.ins().select(out_of_range, right_fill, right);
        let left = builder.ins().select(out_of_range, zero, left);
        let shifted = builder.ins().select(nonnegative, left, right);
        result = builder.ins().insertlane(result, shifted, index as u8);
    }
    Ok(finish_lane_vector(builder, result, fields.vector_128))
}

fn lower_native_a64_simd_bitwise(
    builder: &mut FunctionBuilder<'_>,
    helper: &nixe_cpu::ir::op::HelperOperation,
    arguments: &[ir::Value],
) -> Result<Vec<ir::Value>, CompilerError> {
    let fields = semantic_fields(helper, 3)?;
    let first = arguments[0];
    let second = arguments[1];
    let destination = arguments[2];
    let ones = vector_mask(builder, 128);
    let not_second = builder.ins().bxor(second, ones);
    let not_destination = builder.ins().bxor(destination, ones);
    let result = match fields
        .bitwise_operation
        .ok_or_else(|| CompilerError::new("SIMD bitwise token has no operation"))?
    {
        BitwiseOperation::And => builder.ins().band(first, second),
        BitwiseOperation::BitClear => builder.ins().band(first, not_second),
        BitwiseOperation::Or => builder.ins().bor(first, second),
        BitwiseOperation::OrNot => builder.ins().bor(first, not_second),
        BitwiseOperation::ExclusiveOr => builder.ins().bxor(first, second),
        BitwiseOperation::Select => {
            let selected_first = builder.ins().band(destination, first);
            let selected_second = builder.ins().band(not_destination, second);
            builder.ins().bor(selected_first, selected_second)
        }
        BitwiseOperation::InsertIfTrue => {
            let preserved = builder.ins().band(destination, not_second);
            let inserted = builder.ins().band(first, second);
            builder.ins().bor(preserved, inserted)
        }
        BitwiseOperation::InsertIfFalse => {
            let preserved = builder.ins().band(destination, second);
            let inserted = builder.ins().band(first, not_second);
            builder.ins().bor(preserved, inserted)
        }
    };
    let active = vector_mask(builder, if fields.vector_128 { 128 } else { 64 });
    Ok(vec![builder.ins().band(result, active)])
}

fn lower_native_a64_simd_add_sub(
    builder: &mut FunctionBuilder<'_>,
    helper: &nixe_cpu::ir::op::HelperOperation,
    arguments: &[ir::Value],
) -> Result<Vec<ir::Value>, CompilerError> {
    let fields = semantic_fields(helper, 3)?;
    let lane = match fields.opc {
        0 => types::I8,
        1 => types::I16,
        2 => types::I32,
        3 => types::I64,
        _ => return Err(CompilerError::new("invalid SIMD integer lane width")),
    };
    let vector_ty = lane
        .by(128 / lane.bits())
        .ok_or_else(|| CompilerError::new("unsupported SIMD integer arrangement"))?;
    let flags = bitcast_flags(builder);
    let lhs = builder.ins().bitcast(vector_ty, flags, arguments[0]);
    let flags = bitcast_flags(builder);
    let rhs = builder.ins().bitcast(vector_ty, flags, arguments[1]);
    let result = if fields.subtract {
        builder.ins().isub(lhs, rhs)
    } else {
        builder.ins().iadd(lhs, rhs)
    };
    let flags = bitcast_flags(builder);
    let result = builder.ins().bitcast(types::I8X16, flags, result);
    let active = vector_mask(builder, if fields.vector_128 { 128 } else { 64 });
    Ok(vec![builder.ins().band(result, active)])
}

fn lower_native_a64_unsigned_move(
    builder: &mut FunctionBuilder<'_>,
    helper: &nixe_cpu::ir::op::HelperOperation,
    result_types: &[IrType],
    arguments: &[ir::Value],
) -> Result<Vec<ir::Value>, CompilerError> {
    let fields = semantic_fields(helper, 1)?;
    let size_shift = fields.immediate_5.trailing_zeros();
    let lane_bits = 8_u32 << size_shift;
    let lane = u32::from(fields.immediate_5) >> (size_shift + 1);
    let shift = lane * lane_bits;
    let (low, high) = vector_halves(builder, arguments[0]);
    let half = if shift < 64 { low } else { high };
    let shifted = builder.ins().ushr_imm_u(half, i64::from(shift % 64));
    let result_ty = cranelift_type(single_result_type(result_types)?);
    let result = cast_integer(builder, shifted, result_ty, false);
    let mask = if lane_bits == 64 {
        u64::MAX
    } else {
        (1_u64 << lane_bits) - 1
    };
    let mask = integer_constant(builder, result_ty, mask);
    Ok(vec![builder.ins().band(result, mask)])
}

fn lower_native_a64_scalar_move(
    builder: &mut FunctionBuilder<'_>,
    helper: &nixe_cpu::ir::op::HelperOperation,
    arguments: &[ir::Value],
) -> Result<Vec<ir::Value>, CompilerError> {
    let fields = semantic_fields(helper, 1)?;
    let width = match fields.opc {
        0 => 32,
        1 => 64,
        3 => 16,
        _ => return Err(CompilerError::new("invalid scalar move width")),
    };
    let mask = vector_mask(builder, width);
    Ok(vec![builder.ins().band(arguments[0], mask)])
}

fn lower_native_a64_move_to_general(
    builder: &mut FunctionBuilder<'_>,
    helper: &nixe_cpu::ir::op::HelperOperation,
    result_types: &[IrType],
    arguments: &[ir::Value],
) -> Result<Vec<ir::Value>, CompilerError> {
    let fields = semantic_fields(helper, 1)?;
    let (low, high) = vector_halves(builder, arguments[0]);
    let (value, source_bits) = match (fields.size & 2 != 0, fields.opc) {
        (false, 0) => (low, 32),
        (false, 3) => (low, 16),
        (true, 1) => (low, 64),
        (true, 2) => (high, 64),
        _ => return Err(CompilerError::new("invalid move-to-general encoding")),
    };
    let result_ty = cranelift_type(single_result_type(result_types)?);
    let value = cast_integer(builder, value, result_ty, false);
    let mask = if source_bits == 64 {
        u64::MAX
    } else {
        (1_u64 << source_bits) - 1
    };
    let mask = integer_constant(builder, result_ty, mask);
    Ok(vec![builder.ins().band(value, mask)])
}

fn lower_native_a64_move_from_general(
    builder: &mut FunctionBuilder<'_>,
    helper: &nixe_cpu::ir::op::HelperOperation,
    arguments: &[ir::Value],
) -> Result<Vec<ir::Value>, CompilerError> {
    let fields = semantic_fields(helper, 2)?;
    let value = cast_integer(builder, arguments[0], types::I64, false);
    let zero = builder.ins().iconst(types::I64, 0);
    let result = match (fields.size & 2 != 0, fields.opc) {
        (false, 0) => {
            let masked = builder.ins().band_imm_u(value, i64::from(u32::MAX));
            vector_from_halves(builder, masked, zero)
        }
        (false, 3) => {
            let masked = builder.ins().band_imm_u(value, i64::from(u16::MAX));
            vector_from_halves(builder, masked, zero)
        }
        (true, 1) => vector_from_halves(builder, value, zero),
        (true, 2) => {
            let (previous_low, _) = vector_halves(builder, arguments[1]);
            vector_from_halves(builder, previous_low, value)
        }
        _ => return Err(CompilerError::new("invalid move-from-general encoding")),
    };
    Ok(vec![result])
}

fn lower_native_aarch32_neon_bitwise(
    builder: &mut FunctionBuilder<'_>,
    helper: &nixe_cpu::ir::op::HelperOperation,
    arguments: &[ir::Value],
) -> Result<Vec<ir::Value>, CompilerError> {
    let operation = helper_immediate(helper, 2)? as u32;
    let ones = match builder.func.dfg.value_type(arguments[0]).bits() {
        64 => {
            let bits = integer_constant(builder, types::I64, u64::MAX);
            let flags = bitcast_flags(builder);
            builder.ins().bitcast(types::I8X8, flags, bits)
        }
        128 => vector_mask(builder, 128),
        _ => return Err(CompilerError::new("invalid AArch32 NEON vector width")),
    };
    let not_rhs = builder.ins().bxor(arguments[1], ones);
    let result = match operation {
        0 => arguments[1],
        1 => builder.ins().band(arguments[0], arguments[1]),
        2 => builder.ins().band(arguments[0], not_rhs),
        3 => builder.ins().bor(arguments[0], arguments[1]),
        4 => builder.ins().bxor(arguments[0], arguments[1]),
        _ => return Err(CompilerError::new("invalid AArch32 NEON bitwise operation")),
    };
    Ok(vec![result])
}

fn lower_native_aarch32_shift(
    builder: &mut FunctionBuilder<'_>,
    helper: &nixe_cpu::ir::op::HelperOperation,
    arguments: &[ir::Value],
) -> Result<Vec<ir::Value>, CompilerError> {
    let value = arguments[0];
    let amount = arguments[1];
    let kind = helper_immediate(helper, 3)? as u8;
    let (shifted, _) = lower_aarch32_shift_with_carry(builder, value, amount, arguments[2], kind)?;
    Ok(vec![shifted])
}

fn lower_aarch32_shift_with_carry(
    builder: &mut FunctionBuilder<'_>,
    value: ir::Value,
    amount: ir::Value,
    cpsr: ir::Value,
    kind: u8,
) -> Result<(ir::Value, ir::Value), CompilerError> {
    let zero_value = builder.ins().iconst(types::I32, 0);
    let amount_is_zero = builder.ins().icmp_imm_s(IntCC::Equal, amount, 0);
    let amount_at_least_width =
        builder
            .ins()
            .icmp_imm_u(IntCC::UnsignedGreaterThanOrEqual, amount, 32);
    let amount_at_most_width = builder
        .ins()
        .icmp_imm_u(IntCC::UnsignedLessThanOrEqual, amount, 32);
    let carry_in_bits = builder.ins().ushr_imm_u(cpsr, 29);
    let carry_in_bits = builder.ins().band_imm_u(carry_in_bits, 1);
    let carry_in = builder.ins().icmp_imm_s(IntCC::NotEqual, carry_in_bits, 0);
    let (shifted, carry) = match kind {
        0 => {
            let candidate = builder.ins().ishl(value, amount);
            let bounded = builder
                .ins()
                .select(amount_at_least_width, zero_value, candidate);
            let result = builder.ins().select(amount_is_zero, value, bounded);
            let width = builder.ins().iconst(types::I32, 32);
            let carry_shift = builder.ins().isub(width, amount);
            let bit = builder.ins().ushr(value, carry_shift);
            let bit = builder.ins().band_imm_u(bit, 1);
            let bit = builder.ins().icmp_imm_s(IntCC::NotEqual, bit, 0);
            let within = builder.ins().band(amount_at_most_width, bit);
            let carry = builder.ins().select(amount_is_zero, carry_in, within);
            (result, carry)
        }
        1 => {
            let candidate = builder.ins().ushr(value, amount);
            let bounded = builder
                .ins()
                .select(amount_at_least_width, zero_value, candidate);
            let result = builder.ins().select(amount_is_zero, value, bounded);
            let carry_shift = builder.ins().iadd_imm_s(amount, -1);
            let bit = builder.ins().ushr(value, carry_shift);
            let bit = builder.ins().band_imm_u(bit, 1);
            let bit = builder.ins().icmp_imm_s(IntCC::NotEqual, bit, 0);
            let within = builder.ins().band(amount_at_most_width, bit);
            let carry = builder.ins().select(amount_is_zero, carry_in, within);
            (result, carry)
        }
        2 => {
            let candidate = builder.ins().sshr(value, amount);
            let sign = builder.ins().sshr_imm_u(value, 31);
            let bounded = builder.ins().select(amount_at_least_width, sign, candidate);
            let result = builder.ins().select(amount_is_zero, value, bounded);
            let carry_shift = builder.ins().iadd_imm_s(amount, -1);
            let bit = builder.ins().ushr(value, carry_shift);
            let bit = builder.ins().band_imm_u(bit, 1);
            let bit = builder.ins().icmp_imm_s(IntCC::NotEqual, bit, 0);
            let sign_bit = builder.ins().icmp_imm_s(IntCC::SignedLessThan, value, 0);
            let bounded_carry = builder.ins().select(amount_at_least_width, sign_bit, bit);
            let carry = builder
                .ins()
                .select(amount_is_zero, carry_in, bounded_carry);
            (result, carry)
        }
        3 => {
            let candidate = builder.ins().rotr(value, amount);
            let result = builder.ins().select(amount_is_zero, value, candidate);
            let bit = builder.ins().icmp_imm_s(IntCC::SignedLessThan, result, 0);
            let carry = builder.ins().select(amount_is_zero, carry_in, bit);
            (result, carry)
        }
        4 => {
            let carry = builder.ins().ishl_imm_u(carry_in_bits, 31);
            let shifted = builder.ins().ushr_imm_u(value, 1);
            let result = builder.ins().bor(carry, shifted);
            let carry = builder.ins().band_imm_u(value, 1);
            let carry = builder.ins().icmp_imm_s(IntCC::NotEqual, carry, 0);
            (result, carry)
        }
        _ => return Err(CompilerError::new("invalid AArch32 shift operation")),
    };
    Ok((shifted, carry))
}

fn native_add_with_carry(
    builder: &mut FunctionBuilder<'_>,
    lhs: ir::Value,
    rhs: ir::Value,
    carry: ir::Value,
) -> (ir::Value, ir::Value, ir::Value) {
    let carry_value = builder.ins().uextend(types::I32, carry);
    let (partial, carry0) = builder.ins().uadd_overflow(lhs, rhs);
    let (result, carry1) = builder.ins().uadd_overflow(partial, carry_value);
    let carry_out = builder.ins().bor(carry0, carry1);
    let (signed_partial, overflow0) = builder.ins().sadd_overflow(lhs, rhs);
    let (_, overflow1) = builder.ins().sadd_overflow(signed_partial, carry_value);
    let overflow = builder.ins().bor(overflow0, overflow1);
    (result, carry_out, overflow)
}

fn lower_native_aarch32_data_processing(
    builder: &mut FunctionBuilder<'_>,
    helper: &nixe_cpu::ir::op::HelperOperation,
    arguments: &[ir::Value],
) -> Result<Vec<ir::Value>, CompilerError> {
    let operation = helper_immediate(helper, 6)? as u8;
    let rotation = helper_immediate(helper, 11)? as u8;
    let carry_in_bits = builder.ins().ushr_imm_u(arguments[5], 29);
    let carry_in_bits = builder.ins().band_imm_u(carry_in_bits, 1);
    let carry_in = builder.ins().icmp_imm_s(IntCC::NotEqual, carry_in_bits, 0);
    let (operand, shifter_carry) = if rotation != 0 {
        let carry = builder
            .ins()
            .icmp_imm_s(IntCC::SignedLessThan, arguments[3], 0);
        (arguments[3], carry)
    } else {
        lower_aarch32_shift_with_carry(
            builder,
            arguments[3],
            arguments[4],
            arguments[5],
            helper_immediate(helper, 9)? as u8,
        )?
    };
    let all_ones = builder.ins().iconst(types::I32, -1);
    let not_operand = builder.ins().bxor(operand, all_ones);
    let carry_true = builder.ins().iconst(types::I8, 1);
    let carry_false = builder.ins().iconst(types::I8, 0);
    let mut arithmetic = None;
    let result = match operation {
        0 | 8 => builder.ins().band(arguments[2], operand),
        1 | 9 => builder.ins().bxor(arguments[2], operand),
        2 | 10 => {
            let value = native_add_with_carry(builder, arguments[2], not_operand, carry_true);
            arithmetic = Some((value.1, value.2));
            value.0
        }
        3 => {
            let not_lhs = builder.ins().bxor(arguments[2], all_ones);
            let value = native_add_with_carry(builder, operand, not_lhs, carry_true);
            arithmetic = Some((value.1, value.2));
            value.0
        }
        4 | 11 => {
            let value = native_add_with_carry(builder, arguments[2], operand, carry_false);
            arithmetic = Some((value.1, value.2));
            value.0
        }
        5 => {
            let value = native_add_with_carry(builder, arguments[2], operand, carry_in);
            arithmetic = Some((value.1, value.2));
            value.0
        }
        6 => {
            let value = native_add_with_carry(builder, arguments[2], not_operand, carry_in);
            arithmetic = Some((value.1, value.2));
            value.0
        }
        7 => {
            let not_lhs = builder.ins().bxor(arguments[2], all_ones);
            let value = native_add_with_carry(builder, operand, not_lhs, carry_in);
            arithmetic = Some((value.1, value.2));
            value.0
        }
        12 => builder.ins().bor(arguments[2], operand),
        13 => operand,
        14 => builder.ins().band(arguments[2], not_operand),
        15 => not_operand,
        _ => return Err(CompilerError::new("invalid AArch32 data operation")),
    };

    let set_flags = helper_immediate(helper, 7)? != 0;
    let updated_cpsr = if set_flags {
        let negative = builder.ins().icmp_imm_s(IntCC::SignedLessThan, result, 0);
        let zero = builder.ins().icmp_imm_s(IntCC::Equal, result, 0);
        let (carry, overflow) = arithmetic.unwrap_or_else(|| {
            let overflow_bits = builder.ins().ushr_imm_u(arguments[5], 28);
            let overflow_bits = builder.ins().band_imm_u(overflow_bits, 1);
            let overflow = builder.ins().icmp_imm_s(IntCC::NotEqual, overflow_bits, 0);
            (shifter_carry, overflow)
        });
        let negative = builder.ins().uextend(types::I32, negative);
        let zero = builder.ins().uextend(types::I32, zero);
        let carry = builder.ins().uextend(types::I32, carry);
        let overflow = builder.ins().uextend(types::I32, overflow);
        let negative = builder.ins().ishl_imm_u(negative, 31);
        let zero = builder.ins().ishl_imm_u(zero, 30);
        let carry = builder.ins().ishl_imm_u(carry, 29);
        let overflow = builder.ins().ishl_imm_u(overflow, 28);
        let nz = builder.ins().bor(negative, zero);
        let cv = builder.ins().bor(carry, overflow);
        let flags = builder.ins().bor(nz, cv);
        let preserved = builder
            .ins()
            .band_imm_u(arguments[5], i64::from(0x0fff_ffff_u32));
        let updated = builder.ins().bor(preserved, flags);
        let update = builder.ins().icmp_imm_s(IntCC::Equal, arguments[8], 0);
        builder.ins().select(update, updated, arguments[5])
    } else {
        arguments[5]
    };
    let selected_result = builder.ins().select(arguments[0], result, arguments[1]);
    let selected_cpsr = builder
        .ins()
        .select(arguments[0], updated_cpsr, arguments[5]);
    Ok(vec![selected_result, selected_cpsr])
}

fn lower_native_aarch32_multiply(
    builder: &mut FunctionBuilder<'_>,
    helper: &nixe_cpu::ir::op::HelperOperation,
    arguments: &[ir::Value],
) -> Result<Vec<ir::Value>, CompilerError> {
    let product = builder.ins().imul(arguments[2], arguments[3]);
    let computed = builder.ins().iadd(product, arguments[4]);
    let selected = builder.ins().select(arguments[0], computed, arguments[1]);
    let set_flags = helper_immediate(helper, 6)? != 0;
    let updated_cpsr = if set_flags {
        let negative = builder.ins().icmp_imm_s(IntCC::SignedLessThan, computed, 0);
        let zero = builder.ins().icmp_imm_s(IntCC::Equal, computed, 0);
        let negative = builder.ins().uextend(types::I32, negative);
        let zero = builder.ins().uextend(types::I32, zero);
        let negative = builder.ins().ishl_imm_u(negative, 31);
        let zero = builder.ins().ishl_imm_u(zero, 30);
        let flags = builder.ins().bor(negative, zero);
        let preserved = builder
            .ins()
            .band_imm_u(arguments[5], i64::from(!0xc000_0000_u32));
        let updated = builder.ins().bor(preserved, flags);
        let not_suppressed = builder.ins().icmp_imm_s(IntCC::Equal, arguments[7], 0);
        builder.ins().select(not_suppressed, updated, arguments[5])
    } else {
        arguments[5]
    };
    let cpsr = builder
        .ins()
        .select(arguments[0], updated_cpsr, arguments[5]);
    Ok(vec![selected, cpsr])
}

fn semantic_helper_can_trap_fp(name: &str) -> bool {
    matches!(
        name,
        "a64.fp-simd.semantic-vector"
            | "a64.fp.float-to-signed-int"
            | "a64.fp.float-to-unsigned-int"
            | "a64.fp.scalar-arithmetic"
            | "a64.fp.scalar-compare"
            | "a64.fp.semantic-conditional-compare"
            | "a64.fp.signed-int-to-float"
            | "a64.fp.unsigned-int-to-float"
            | "aarch32.vfp.binary32-vector"
    )
}

fn call_semantic_helper(
    builder: &mut FunctionBuilder<'_>,
    frame: ir::Value,
    call_conv: CallConv,
    operation: u32,
) -> Result<ir::Value, CompilerError> {
    let pointer_type = builder.func.dfg.value_type(frame);
    let table = load(builder, pointer_type, frame, FRAME_OFFSETS.helpers)?;
    let flags = trusted_mem_flags(builder);
    let callee = builder
        .ins()
        .load(pointer_type, flags, table, offset(HELPER_OFFSETS.semantic)?);
    let operation = builder.ins().iconst(types::I32, i64::from(operation));
    let mut signature = Signature::new(call_conv);
    signature.params.push(AbiParam::new(pointer_type));
    signature.params.push(AbiParam::new(types::I32));
    signature.returns.push(AbiParam::new(types::I32));
    let signature = builder.import_signature(signature);
    let call = builder
        .ins()
        .call_indirect(signature, callee, &[frame, operation]);
    Ok(builder.inst_results(call)[0])
}

fn branch_on_semantic_status(
    builder: &mut FunctionBuilder<'_>,
    lowering: &LoweringState,
    source: LocationDescriptor,
    status: ir::Value,
    fp_trap: Option<u32>,
) {
    let trapped = builder.ins().icmp_imm_s(IntCC::Equal, status, 4);
    let trap = builder.create_block();
    let classify_data_fault = builder.create_block();
    builder
        .ins()
        .brif(trapped, trap, &[], classify_data_fault, &[]);
    builder.switch_to_block(trap);
    increment_retired(builder, lowering);
    if let Some(side) = fp_trap {
        emit_exit(
            builder,
            lowering,
            EXIT_ARCHITECTURAL,
            side,
            source.pc.get(),
            0,
            0,
        );
    } else {
        emit_exit(builder, lowering, EXIT_INTERNAL, 0, source.pc.get(), 0, 0);
    }
    builder.switch_to_block(classify_data_fault);
    let data_fault = builder.ins().icmp_imm_s(IntCC::Equal, status, 3);
    let fault = builder.create_block();
    let classify_failure = builder.create_block();
    builder
        .ins()
        .brif(data_fault, fault, &[], classify_failure, &[]);
    builder.switch_to_block(fault);
    increment_retired(builder, lowering);
    emit_exit(builder, lowering, EXIT_DATA_FAULT, 0, source.pc.get(), 0, 0);

    builder.switch_to_block(classify_failure);
    let failed = builder.ins().icmp_imm_s(IntCC::NotEqual, status, 0);
    let failure = builder.create_block();
    let resume = builder.create_block();
    builder.ins().brif(failed, failure, &[], resume, &[]);
    builder.switch_to_block(failure);
    emit_exit(builder, lowering, EXIT_INTERNAL, 0, source.pc.get(), 0, 0);
    builder.switch_to_block(resume);
}

fn load_ir_scratch(
    builder: &mut FunctionBuilder<'_>,
    frame: ir::Value,
    frame_offset: usize,
    ty: IrType,
) -> Result<ir::Value, CompilerError> {
    let native = cranelift_type(ty);
    let low = load(builder, types::I64, frame, frame_offset)?;
    let integer_type = match native.bits() {
        8 => types::I8,
        16 => types::I16,
        32 => types::I32,
        64 => types::I64,
        128 => types::I128,
        bits => {
            return Err(CompilerError::new(format!(
                "unsupported helper result width {bits}"
            )));
        }
    };
    let integer = if integer_type.bits() > 64 {
        let high = load(builder, types::I64, frame, frame_offset + 8)?;
        builder.ins().iconcat(low, high)
    } else if integer_type.bits() < 64 {
        builder.ins().ireduce(integer_type, low)
    } else {
        low
    };
    if native.is_vector() || native.is_float() {
        let flags = bitcast_flags(builder);
        Ok(builder.ins().bitcast(native, flags, integer))
    } else {
        Ok(integer)
    }
}

fn memory_read(
    builder: &mut FunctionBuilder<'_>,
    lowering: &LoweringState,
    source: LocationDescriptor,
    address: ir::Value,
    access: nixe_cpu::memory::MemoryAccess,
) -> Result<ir::Value, CompilerError> {
    if access.class != nixe_cpu::memory::MemoryAccessClass::Normal
        || access.ordering != nixe_cpu::memory::MemoryOrdering::Relaxed
    {
        return memory_read_slow(builder, lowering, source, address, access);
    }
    let ty = memory_integer_type(access.size);
    let lookup = builder.create_block();
    let hit = builder.create_block();
    let visible_hit = builder.create_block();
    let hit_complete = builder.create_block();
    let slow = builder.create_block();
    let merged = builder.create_block();
    builder.append_block_param(merged, ty);

    let tlb_base = load(
        builder,
        types::I64,
        lowering.frame,
        FRAME_OFFSETS.memory_tlb_base,
    )?;
    let available = builder.ins().icmp_imm_s(IntCC::NotEqual, tlb_base, 0);
    builder.ins().brif(available, lookup, &[], slow, &[]);

    builder.switch_to_block(lookup);
    let entry = tlb_entry(builder, lowering, address, tlb_base)?;
    let valid = tlb_entry_matches(
        builder,
        lowering,
        address,
        entry,
        FLAG_READ | FLAG_ORDINARY | FLAG_CPU_VISIBLE,
        access.size,
    )?;
    builder.ins().brif(valid, hit, &[], slow, &[]);

    builder.switch_to_block(hit);
    let visible = direct_visibility_matches(builder, entry)?;
    builder.ins().brif(visible, visible_hit, &[], slow, &[]);
    builder.switch_to_block(visible_hit);
    let value = direct_load(builder, address, entry, access.size)?;
    let still_valid = direct_visibility_matches(builder, entry)?;
    builder
        .ins()
        .brif(still_valid, hit_complete, &[], slow, &[]);
    builder.switch_to_block(hit_complete);
    builder.ins().jump(merged, &[BlockArg::from(value)]);

    builder.switch_to_block(slow);
    let value = memory_read_slow(builder, lowering, source, address, access)?;
    builder.ins().jump(merged, &[BlockArg::from(value)]);
    builder.switch_to_block(merged);
    Ok(builder.block_params(merged)[0])
}

fn memory_read_slow(
    builder: &mut FunctionBuilder<'_>,
    lowering: &LoweringState,
    source: LocationDescriptor,
    address: ir::Value,
    access: nixe_cpu::memory::MemoryAccess,
) -> Result<ir::Value, CompilerError> {
    let output = pointer_at(builder, lowering.frame, FRAME_OFFSETS.scratch_results)?;
    let descriptor = builder
        .ins()
        .iconst(types::I64, encode_access(access) as i64);
    let status = call_helper(
        builder,
        lowering.frame,
        lowering.helper_call_conv,
        HELPER_OFFSETS.memory_read,
        &[lowering.frame, address, descriptor, output],
        4,
    )?;
    branch_on_helper_status(builder, lowering, source, status);
    load_scratch(
        builder,
        lowering.frame,
        FRAME_OFFSETS.scratch_results,
        access.size,
    )
}

fn lower_exclusive(
    builder: &mut FunctionBuilder<'_>,
    lowering: &LoweringState,
    operation: ExclusiveOperation,
    source: LocationDescriptor,
    values: &BTreeMap<ValueId, ir::Value>,
) -> Result<Vec<ir::Value>, CompilerError> {
    match operation {
        ExclusiveOperation::Load {
            address,
            descriptor,
        } => {
            let address = operand(builder, values, address)?;
            let value = exclusive_call(
                builder,
                lowering,
                source,
                0,
                address,
                None,
                descriptor.access,
            )?;
            Ok(vec![apply_byte_order(
                builder,
                value,
                descriptor.byte_order,
            )])
        }
        ExclusiveOperation::Store {
            address,
            value,
            descriptor,
        } => {
            let address = operand(builder, values, address)?;
            let value = operand(builder, values, value)?;
            let value = apply_byte_order(builder, value, descriptor.byte_order);
            let result = exclusive_call(
                builder,
                lowering,
                source,
                1,
                address,
                Some(value),
                descriptor.access,
            )?;
            Ok(vec![builder.ins().icmp_imm_s(IntCC::Equal, result, 0)])
        }
        ExclusiveOperation::GuardedLoad {
            predicate,
            address,
            fallback,
            descriptor,
        } => {
            let predicate = operand(builder, values, predicate)?;
            let address = operand(builder, values, address)?;
            let fallback = operand(builder, values, fallback)?;
            let accessed = builder.create_block();
            let skipped = builder.create_block();
            let merged = builder.create_block();
            builder.append_block_param(merged, builder.func.dfg.value_type(fallback));
            builder.ins().brif(predicate, accessed, &[], skipped, &[]);
            builder.switch_to_block(accessed);
            let value = exclusive_call(
                builder,
                lowering,
                source,
                0,
                address,
                None,
                descriptor.access,
            )?;
            let value = apply_byte_order(builder, value, descriptor.byte_order);
            builder.ins().jump(merged, &[BlockArg::from(value)]);
            builder.switch_to_block(skipped);
            builder.ins().jump(merged, &[BlockArg::from(fallback)]);
            builder.switch_to_block(merged);
            Ok(vec![builder.block_params(merged)[0]])
        }
        ExclusiveOperation::GuardedStore {
            predicate,
            address,
            value,
            fallback,
            descriptor,
        } => {
            let predicate = operand(builder, values, predicate)?;
            let address = operand(builder, values, address)?;
            let value = operand(builder, values, value)?;
            let fallback = operand(builder, values, fallback)?;
            let accessed = builder.create_block();
            let skipped = builder.create_block();
            let merged = builder.create_block();
            builder.append_block_param(merged, builder.func.dfg.value_type(fallback));
            builder.ins().brif(predicate, accessed, &[], skipped, &[]);
            builder.switch_to_block(accessed);
            let value = apply_byte_order(builder, value, descriptor.byte_order);
            let result = exclusive_call(
                builder,
                lowering,
                source,
                1,
                address,
                Some(value),
                descriptor.access,
            )?;
            let result = builder.ins().icmp_imm_s(IntCC::Equal, result, 0);
            builder.ins().jump(merged, &[BlockArg::from(result)]);
            builder.switch_to_block(skipped);
            builder.ins().jump(merged, &[BlockArg::from(fallback)]);
            builder.switch_to_block(merged);
            Ok(vec![builder.block_params(merged)[0]])
        }
        ExclusiveOperation::Clear => {
            let zero = builder.ins().iconst(types::I64, 0);
            let access = nixe_cpu::memory::MemoryAccess::new(
                nixe_cpu::memory::MemoryAccessSize::Byte,
                nixe_cpu::memory::MemoryAlignment::Unaligned,
                nixe_cpu::memory::MemoryOrdering::Relaxed,
                nixe_cpu::memory::MemoryAccessClass::Exclusive,
            );
            let _ = exclusive_call(builder, lowering, source, 2, zero, None, access)?;
            Ok(Vec::new())
        }
    }
}

fn exclusive_call(
    builder: &mut FunctionBuilder<'_>,
    lowering: &LoweringState,
    source: LocationDescriptor,
    operation: u64,
    address: ir::Value,
    value: Option<ir::Value>,
    access: nixe_cpu::memory::MemoryAccess,
) -> Result<ir::Value, CompilerError> {
    let input_offset = FRAME_OFFSETS.scratch_arguments;
    if let Some(value) = value {
        store_scratch(builder, lowering.frame, input_offset, value)?;
    } else {
        let zero = builder.ins().iconst(types::I64, 0);
        store_scratch(builder, lowering.frame, input_offset, zero)?;
    }
    let input = pointer_at(builder, lowering.frame, input_offset)?;
    let output = pointer_at(builder, lowering.frame, FRAME_OFFSETS.scratch_results)?;
    let descriptor = encode_access(access) | (operation << 32);
    let descriptor = builder.ins().iconst(types::I64, descriptor as i64);
    let status = call_helper(
        builder,
        lowering.frame,
        lowering.helper_call_conv,
        HELPER_OFFSETS.exclusive,
        &[lowering.frame, address, descriptor, input, output],
        5,
    )?;
    branch_on_helper_status(builder, lowering, source, status);
    load_scratch(
        builder,
        lowering.frame,
        FRAME_OFFSETS.scratch_results,
        access.size,
    )
}

fn memory_write(
    builder: &mut FunctionBuilder<'_>,
    lowering: &LoweringState,
    source: LocationDescriptor,
    address: ir::Value,
    value: ir::Value,
    access: nixe_cpu::memory::MemoryAccess,
) -> Result<(), CompilerError> {
    if access.class != nixe_cpu::memory::MemoryAccessClass::Normal
        || access.ordering != nixe_cpu::memory::MemoryOrdering::Relaxed
    {
        return memory_write_slow(builder, lowering, source, address, value, access);
    }
    let lookup = builder.create_block();
    let hit = builder.create_block();
    let slow = builder.create_block();
    let merged = builder.create_block();
    let tlb_base = load(
        builder,
        types::I64,
        lowering.frame,
        FRAME_OFFSETS.memory_tlb_base,
    )?;
    let available = builder.ins().icmp_imm_s(IntCC::NotEqual, tlb_base, 0);
    builder.ins().brif(available, lookup, &[], slow, &[]);

    builder.switch_to_block(lookup);
    let entry = tlb_entry(builder, lowering, address, tlb_base)?;
    let valid = tlb_entry_matches(
        builder,
        lowering,
        address,
        entry,
        FLAG_WRITE | FLAG_WRITE_ARMED | FLAG_ORDINARY | FLAG_CPU_VISIBLE,
        access.size,
    )?;
    builder.ins().brif(valid, hit, &[], slow, &[]);

    builder.switch_to_block(hit);
    direct_store(builder, address, entry, value, access.size, slow)?;
    builder.ins().jump(merged, &[]);

    builder.switch_to_block(slow);
    memory_write_slow(builder, lowering, source, address, value, access)?;
    builder.ins().jump(merged, &[]);
    builder.switch_to_block(merged);
    Ok(())
}

fn memory_write_slow(
    builder: &mut FunctionBuilder<'_>,
    lowering: &LoweringState,
    source: LocationDescriptor,
    address: ir::Value,
    value: ir::Value,
    access: nixe_cpu::memory::MemoryAccess,
) -> Result<(), CompilerError> {
    store_scratch(
        builder,
        lowering.frame,
        FRAME_OFFSETS.scratch_arguments,
        value,
    )?;
    let input = pointer_at(builder, lowering.frame, FRAME_OFFSETS.scratch_arguments)?;
    let descriptor = builder
        .ins()
        .iconst(types::I64, encode_access(access) as i64);
    let status = call_helper(
        builder,
        lowering.frame,
        lowering.helper_call_conv,
        HELPER_OFFSETS.memory_write,
        &[lowering.frame, address, descriptor, input],
        4,
    )?;
    branch_on_helper_status(builder, lowering, source, status);
    Ok(())
}

fn memory_integer_type(size: nixe_cpu::memory::MemoryAccessSize) -> ir::Type {
    match size {
        nixe_cpu::memory::MemoryAccessSize::Byte => types::I8,
        nixe_cpu::memory::MemoryAccessSize::Halfword => types::I16,
        nixe_cpu::memory::MemoryAccessSize::Word => types::I32,
        nixe_cpu::memory::MemoryAccessSize::Doubleword => types::I64,
        nixe_cpu::memory::MemoryAccessSize::Quadword => types::I128,
    }
}

fn tlb_entry(
    builder: &mut FunctionBuilder<'_>,
    lowering: &LoweringState,
    address: ir::Value,
    tlb_base: ir::Value,
) -> Result<ir::Value, CompilerError> {
    let guest_page = builder.ins().ushr_imm_u(address, i64::from(PAGE_BITS));
    let mask = load(
        builder,
        types::I32,
        lowering.frame,
        FRAME_OFFSETS.memory_tlb_index_mask,
    )?;
    let mask = builder.ins().uextend(types::I64, mask);
    let index = builder.ins().band(guest_page, mask);
    let byte_offset = builder.ins().imul_imm_u(index, TLB_ENTRY_SIZE as i64);
    Ok(builder.ins().iadd(tlb_base, byte_offset))
}

fn tlb_entry_matches(
    builder: &mut FunctionBuilder<'_>,
    lowering: &LoweringState,
    address: ir::Value,
    entry: ir::Value,
    required_flags: u32,
    size: nixe_cpu::memory::MemoryAccessSize,
) -> Result<ir::Value, CompilerError> {
    let flags = trusted_mem_flags(builder);
    let guest_page = builder.ins().ushr_imm_u(address, i64::from(PAGE_BITS));
    let cached_page = builder
        .ins()
        .load(types::I64, flags, entry, offset(TLB_OFFSETS.guest_page)?);
    let mut valid = builder.ins().icmp(IntCC::Equal, guest_page, cached_page);
    for (frame_offset, entry_offset) in [
        (
            FRAME_OFFSETS.memory_address_space,
            TLB_OFFSETS.address_space,
        ),
        (
            FRAME_OFFSETS.memory_mapping_epoch,
            TLB_OFFSETS.mapping_epoch,
        ),
    ] {
        let expected = load(builder, types::I64, lowering.frame, frame_offset)?;
        let observed = builder
            .ins()
            .load(types::I64, flags, entry, offset(entry_offset)?);
        let equal = builder.ins().icmp(IntCC::Equal, expected, observed);
        valid = builder.ins().band(valid, equal);
    }
    let observed_flags = builder
        .ins()
        .load(types::I32, flags, entry, offset(TLB_OFFSETS.flags)?);
    let masked = builder
        .ins()
        .band_imm_u(observed_flags, i64::from(required_flags));
    let allowed = builder
        .ins()
        .icmp_imm_s(IntCC::Equal, masked, i64::from(required_flags));
    valid = builder.ins().band(valid, allowed);

    let bytes = size.bytes() as u64;
    let page_offset = builder.ins().band_imm_u(address, (PAGE_SIZE - 1) as i64);
    let within_page = builder.ins().icmp_imm_u(
        IntCC::UnsignedLessThanOrEqual,
        page_offset,
        (PAGE_SIZE - bytes) as i64,
    );
    let alignment = builder.ins().band_imm_u(address, (bytes - 1) as i64);
    let aligned = builder.ins().icmp_imm_s(IntCC::Equal, alignment, 0);
    valid = builder.ins().band(valid, within_page);
    Ok(builder.ins().band(valid, aligned))
}

fn direct_visibility_matches(
    builder: &mut FunctionBuilder<'_>,
    entry: ir::Value,
) -> Result<ir::Value, CompilerError> {
    let flags = trusted_mem_flags(builder);
    let validity_address = builder.ins().load(
        types::I64,
        flags,
        entry,
        offset(TLB_OFFSETS.validity_address)?,
    );
    let current_visibility = builder
        .ins()
        .atomic_load(types::I64, flags, validity_address);
    let expected_visibility = builder.ins().load(
        types::I64,
        flags,
        entry,
        offset(TLB_OFFSETS.visibility_epoch)?,
    );
    let visible = builder
        .ins()
        .icmp(IntCC::Equal, current_visibility, expected_visibility);
    Ok(visible)
}

fn direct_word_pointer(
    builder: &mut FunctionBuilder<'_>,
    address: ir::Value,
    entry: ir::Value,
) -> Result<ir::Value, CompilerError> {
    let flags = trusted_mem_flags(builder);
    let base = builder.ins().load(
        types::I64,
        flags,
        entry,
        offset(TLB_OFFSETS.host_word_base)?,
    );
    let page_offset = builder.ins().band_imm_u(address, (PAGE_SIZE - 1) as i64);
    let word_offset = builder.ins().band_imm_s(page_offset, !7_i64);
    Ok(builder.ins().iadd(base, word_offset))
}

fn direct_load(
    builder: &mut FunctionBuilder<'_>,
    address: ir::Value,
    entry: ir::Value,
    size: nixe_cpu::memory::MemoryAccessSize,
) -> Result<ir::Value, CompilerError> {
    let pointer = direct_word_pointer(builder, address, entry)?;
    let flags = trusted_mem_flags(builder);
    let low = builder.ins().atomic_load(types::I64, flags, pointer);
    if size == nixe_cpu::memory::MemoryAccessSize::Quadword {
        let high_pointer = builder.ins().iadd_imm_s(pointer, 8);
        let high = builder.ins().atomic_load(types::I64, flags, high_pointer);
        return Ok(builder.ins().iconcat(low, high));
    }
    let ty = memory_integer_type(size);
    if ty == types::I64 {
        Ok(low)
    } else {
        let byte_offset = builder.ins().band_imm_u(address, 7);
        let shift = builder.ins().ishl_imm_u(byte_offset, 3);
        let shifted = builder.ins().ushr(low, shift);
        Ok(builder.ins().ireduce(ty, shifted))
    }
}

fn direct_store(
    builder: &mut FunctionBuilder<'_>,
    address: ir::Value,
    entry: ir::Value,
    value: ir::Value,
    size: nixe_cpu::memory::MemoryAccessSize,
    slow: ir::Block,
) -> Result<(), CompilerError> {
    let pointer = direct_word_pointer(builder, address, entry)?;
    let flags = trusted_mem_flags(builder);
    let sequence_address = builder.ins().load(
        types::I64,
        flags,
        entry,
        offset(TLB_OFFSETS.write_sequence_address)?,
    );
    let sequence = acquire_write_sequence(builder, sequence_address);
    let mut permitted = direct_visibility_matches(builder, entry)?;
    let generation_address = builder.ins().load(
        types::I64,
        flags,
        entry,
        offset(TLB_OFFSETS.generation_address)?,
    );
    let generation = builder
        .ins()
        .atomic_load(types::I64, flags, generation_address);
    let has_generation = builder.ins().icmp_imm_s(IntCC::NotEqual, generation, -1);
    permitted = builder.ins().band(permitted, has_generation);
    let content_epoch_address = builder.ins().load(
        types::I64,
        flags,
        entry,
        offset(TLB_OFFSETS.content_epoch_address)?,
    );
    let content_epoch = builder
        .ins()
        .atomic_load(types::I64, flags, content_epoch_address);
    let has_content_epoch = builder.ins().icmp_imm_s(IntCC::NotEqual, content_epoch, -1);
    permitted = builder.ins().band(permitted, has_content_epoch);
    let cpu_write_epoch_address = builder.ins().load(
        types::I64,
        flags,
        entry,
        offset(TLB_OFFSETS.cpu_write_epoch_address)?,
    );
    let cpu_write_epoch = builder
        .ins()
        .atomic_load(types::I64, flags, cpu_write_epoch_address);
    let has_cpu_write_epoch = builder
        .ins()
        .icmp_imm_s(IntCC::NotEqual, cpu_write_epoch, -1);
    permitted = builder.ins().band(permitted, has_cpu_write_epoch);
    let cpu_writes_active_address = builder.ins().load(
        types::I64,
        flags,
        entry,
        offset(TLB_OFFSETS.cpu_writes_active_address)?,
    );
    let store = builder.create_block();
    let revoked = builder.create_block();
    builder.ins().brif(permitted, store, &[], revoked, &[]);
    builder.switch_to_block(revoked);
    let completed = builder.ins().iadd_imm_s(sequence, 2);
    builder
        .ins()
        .atomic_store(flags, completed, sequence_address);
    builder.ins().jump(slow, &[]);
    builder.switch_to_block(store);
    let one = builder.ins().iconst(types::I64, 1);
    builder.ins().atomic_rmw(
        types::I64,
        flags,
        AtomicRmwOp::Add,
        cpu_writes_active_address,
        one,
    );
    if size == nixe_cpu::memory::MemoryAccessSize::Quadword {
        let (low, high) = builder.ins().isplit(value);
        builder.ins().atomic_store(flags, low, pointer);
        let high_pointer = builder.ins().iadd_imm_s(pointer, 8);
        builder.ins().atomic_store(flags, high, high_pointer);
    } else if size == nixe_cpu::memory::MemoryAccessSize::Doubleword {
        builder.ins().atomic_store(flags, value, pointer);
    } else {
        let bits = size.bytes() * 8;
        let extended = builder.ins().uextend(types::I64, value);
        let byte_offset = builder.ins().band_imm_u(address, 7);
        let shift = builder.ins().ishl_imm_u(byte_offset, 3);
        let shifted = builder.ins().ishl(extended, shift);
        let mask = builder
            .ins()
            .iconst(types::I64, ((1_u64 << bits) - 1) as i64);
        let mask = builder.ins().ishl(mask, shift);
        atomic_masked_store(builder, pointer, shifted, mask);
    }
    builder
        .ins()
        .atomic_rmw(types::I64, flags, AtomicRmwOp::Add, generation_address, one);
    builder.ins().atomic_rmw(
        types::I64,
        flags,
        AtomicRmwOp::Add,
        content_epoch_address,
        one,
    );
    builder.ins().atomic_rmw(
        types::I64,
        flags,
        AtomicRmwOp::Add,
        cpu_write_epoch_address,
        one,
    );
    let completed = builder.ins().iadd_imm_s(sequence, 2);
    builder
        .ins()
        .atomic_store(flags, completed, sequence_address);
    builder.ins().atomic_rmw(
        types::I64,
        flags,
        AtomicRmwOp::Sub,
        cpu_writes_active_address,
        one,
    );
    Ok(())
}

fn acquire_write_sequence(
    builder: &mut FunctionBuilder<'_>,
    sequence_address: ir::Value,
) -> ir::Value {
    let flags = trusted_mem_flags(builder);
    let retry = builder.create_block();
    let attempt = builder.create_block();
    let acquired = builder.create_block();
    builder.append_block_param(retry, types::I64);
    builder.append_block_param(attempt, types::I64);
    builder.append_block_param(acquired, types::I64);
    let observed = builder
        .ins()
        .atomic_load(types::I64, flags, sequence_address);
    builder.ins().jump(retry, &[BlockArg::from(observed)]);

    builder.switch_to_block(retry);
    let observed = builder.block_params(retry)[0];
    let busy = builder.ins().band_imm_u(observed, 1);
    let is_busy = builder.ins().icmp_imm_s(IntCC::NotEqual, busy, 0);
    let refreshed = builder
        .ins()
        .atomic_load(types::I64, flags, sequence_address);
    builder.ins().brif(
        is_busy,
        retry,
        &[BlockArg::from(refreshed)],
        attempt,
        &[BlockArg::from(observed)],
    );

    builder.switch_to_block(attempt);
    let observed = builder.block_params(attempt)[0];
    let writing = builder.ins().iadd_imm_s(observed, 1);
    let previous = builder
        .ins()
        .atomic_cas(flags, sequence_address, observed, writing);
    let success = builder.ins().icmp(IntCC::Equal, previous, observed);
    builder.ins().brif(
        success,
        acquired,
        &[BlockArg::from(observed)],
        retry,
        &[BlockArg::from(previous)],
    );
    builder.switch_to_block(acquired);
    builder.block_params(acquired)[0]
}

fn atomic_masked_store(
    builder: &mut FunctionBuilder<'_>,
    pointer: ir::Value,
    value: ir::Value,
    value_mask: ir::Value,
) {
    let flags = trusted_mem_flags(builder);
    let retry = builder.create_block();
    let done = builder.create_block();
    builder.append_block_param(retry, types::I64);
    let observed = builder.ins().atomic_load(types::I64, flags, pointer);
    builder.ins().jump(retry, &[BlockArg::from(observed)]);
    builder.switch_to_block(retry);
    let observed = builder.block_params(retry)[0];
    let inverted_mask = builder.ins().bnot(value_mask);
    let retained = builder.ins().band(observed, inverted_mask);
    let replacement = builder.ins().band(value, value_mask);
    let next = builder.ins().bor(retained, replacement);
    let previous = builder.ins().atomic_cas(flags, pointer, observed, next);
    let stored = builder.ins().icmp(IntCC::Equal, previous, observed);
    builder
        .ins()
        .brif(stored, done, &[], retry, &[BlockArg::from(previous)]);
    builder.switch_to_block(done);
}

fn call_helper(
    builder: &mut FunctionBuilder<'_>,
    frame: ir::Value,
    call_conv: CallConv,
    helper_offset: usize,
    arguments: &[ir::Value],
    pointer_arguments: usize,
) -> Result<ir::Value, CompilerError> {
    let pointer_type = builder.func.dfg.value_type(frame);
    let table = load(builder, pointer_type, frame, FRAME_OFFSETS.helpers)?;
    let flags = trusted_mem_flags(builder);
    let callee = builder
        .ins()
        .load(pointer_type, flags, table, offset(helper_offset)?);
    let mut signature = Signature::new(call_conv);
    for index in 0..pointer_arguments {
        let ty = if index == 1 || index == 2 {
            types::I64
        } else {
            pointer_type
        };
        signature.params.push(AbiParam::new(ty));
    }
    signature.returns.push(AbiParam::new(types::I32));
    let signature = builder.import_signature(signature);
    let call = builder.ins().call_indirect(signature, callee, arguments);
    Ok(builder.inst_results(call)[0])
}

fn branch_on_helper_status(
    builder: &mut FunctionBuilder<'_>,
    lowering: &LoweringState,
    source: LocationDescriptor,
    status: ir::Value,
) {
    let failed = builder.ins().icmp_imm_s(IntCC::NotEqual, status, 0);
    let fault = builder.create_block();
    let resume = builder.create_block();
    builder.ins().brif(failed, fault, &[], resume, &[]);
    builder.switch_to_block(fault);
    increment_retired(builder, lowering);
    emit_exit(builder, lowering, EXIT_DATA_FAULT, 0, source.pc.get(), 0, 0);
    builder.switch_to_block(resume);
}

fn pointer_at(
    builder: &mut FunctionBuilder<'_>,
    frame: ir::Value,
    frame_offset: usize,
) -> Result<ir::Value, CompilerError> {
    Ok(builder
        .ins()
        .iadd_imm_s(frame, i64::from(offset(frame_offset)?)))
}

fn store_scratch(
    builder: &mut FunctionBuilder<'_>,
    frame: ir::Value,
    frame_offset: usize,
    value: ir::Value,
) -> Result<(), CompilerError> {
    let ty = builder.func.dfg.value_type(value);
    let (low, high) = if ty.bits() > 64 {
        let integer = if ty.is_vector() {
            let flags = bitcast_flags(builder);
            builder.ins().bitcast(types::I128, flags, value)
        } else {
            value
        };
        builder.ins().isplit(integer)
    } else {
        let low = if ty.bits() < 64 {
            builder.ins().uextend(types::I64, value)
        } else {
            value
        };
        (low, builder.ins().iconst(types::I64, 0))
    };
    store(builder, low, frame, offset(frame_offset)?);
    store(builder, high, frame, offset(frame_offset + 8)?);
    Ok(())
}

fn load_scratch(
    builder: &mut FunctionBuilder<'_>,
    frame: ir::Value,
    frame_offset: usize,
    size: nixe_cpu::memory::MemoryAccessSize,
) -> Result<ir::Value, CompilerError> {
    let ty = match size {
        nixe_cpu::memory::MemoryAccessSize::Byte => types::I8,
        nixe_cpu::memory::MemoryAccessSize::Halfword => types::I16,
        nixe_cpu::memory::MemoryAccessSize::Word => types::I32,
        nixe_cpu::memory::MemoryAccessSize::Doubleword => types::I64,
        nixe_cpu::memory::MemoryAccessSize::Quadword => types::I128,
    };
    let low = load(builder, types::I64, frame, frame_offset)?;
    if ty == types::I128 {
        let high = load(builder, types::I64, frame, frame_offset + 8)?;
        Ok(builder.ins().iconcat(low, high))
    } else if ty == types::I64 {
        Ok(low)
    } else {
        Ok(builder.ins().ireduce(ty, low))
    }
}

fn apply_byte_order(
    builder: &mut FunctionBuilder<'_>,
    value: ir::Value,
    byte_order: nixe_cpu::ir::op::ByteOrder,
) -> ir::Value {
    if byte_order == nixe_cpu::ir::op::ByteOrder::Big
        && builder.func.dfg.value_type(value).bits() > 8
    {
        builder.ins().bswap(value)
    } else {
        value
    }
}

fn lower_binary(
    builder: &mut FunctionBuilder<'_>,
    kind: IntegerBinaryKind,
    lhs: ir::Value,
    rhs: ir::Value,
) -> ir::Value {
    match kind {
        IntegerBinaryKind::Add => builder.ins().iadd(lhs, rhs),
        IntegerBinaryKind::Subtract => builder.ins().isub(lhs, rhs),
        IntegerBinaryKind::Multiply => builder.ins().imul(lhs, rhs),
        IntegerBinaryKind::UnsignedDivide => safe_divide(builder, lhs, rhs, false),
        IntegerBinaryKind::SignedDivide => safe_divide(builder, lhs, rhs, true),
        IntegerBinaryKind::And => builder.ins().band(lhs, rhs),
        IntegerBinaryKind::Or => builder.ins().bor(lhs, rhs),
        IntegerBinaryKind::Xor => builder.ins().bxor(lhs, rhs),
    }
}

fn safe_divide(
    builder: &mut FunctionBuilder<'_>,
    lhs: ir::Value,
    rhs: ir::Value,
    signed: bool,
) -> ir::Value {
    let ty = builder.func.dfg.value_type(lhs);
    let zero = builder.ins().iconst(ty, 0);
    let one = builder.ins().iconst(ty, 1);
    let divisor_zero = builder.ins().icmp_imm_s(IntCC::Equal, rhs, 0);
    let overflow = if signed {
        let sign_shift = builder.ins().iconst(ty, i64::from(ty.bits() - 1));
        let minimum = builder.ins().ishl(one, sign_shift);
        let lhs_min = builder.ins().icmp(IntCC::Equal, lhs, minimum);
        let rhs_negative_one = builder.ins().icmp_imm_s(IntCC::Equal, rhs, -1);
        builder.ins().band(lhs_min, rhs_negative_one)
    } else {
        builder.ins().iconst(types::I8, 0)
    };
    let exceptional = builder.ins().bor(divisor_zero, overflow);
    let safe_rhs = builder.ins().select(exceptional, one, rhs);
    let quotient = if signed {
        builder.ins().sdiv(lhs, safe_rhs)
    } else {
        builder.ins().udiv(lhs, safe_rhs)
    };
    let exceptional_result = if signed {
        builder.ins().select(overflow, lhs, zero)
    } else {
        zero
    };
    builder
        .ins()
        .select(exceptional, exceptional_result, quotient)
}

fn lower_shift(
    builder: &mut FunctionBuilder<'_>,
    kind: ShiftKind,
    value: ir::Value,
    amount: ir::Value,
) -> ir::Value {
    let ty = builder.func.dfg.value_type(value);
    let amount_type = builder.func.dfg.value_type(amount);
    let amount = if amount_type == ty {
        amount
    } else if amount_type.bits() < ty.bits() {
        builder.ins().uextend(ty, amount)
    } else {
        builder.ins().ireduce(ty, amount)
    };
    let shifted = match kind {
        ShiftKind::LogicalLeft => builder.ins().ishl(value, amount),
        ShiftKind::LogicalRight => builder.ins().ushr(value, amount),
        ShiftKind::ArithmeticRight => builder.ins().sshr(value, amount),
        ShiftKind::RotateLeft => return builder.ins().rotl(value, amount),
        ShiftKind::RotateRight => return builder.ins().rotr(value, amount),
    };
    let outside =
        builder
            .ins()
            .icmp_imm_s(IntCC::UnsignedGreaterThanOrEqual, amount, ty.bits() as i64);
    let outside_value = if kind == ShiftKind::ArithmeticRight {
        builder.ins().sshr_imm_s(value, (ty.bits() - 1) as i64)
    } else {
        builder.ins().iconst(ty, 0)
    };
    builder.ins().select(outside, outside_value, shifted)
}

fn lower_address(
    builder: &mut FunctionBuilder<'_>,
    operation: AddressOperation,
    values: &BTreeMap<ValueId, ir::Value>,
) -> Result<ir::Value, CompilerError> {
    Ok(match operation {
        AddressOperation::FromInteger { value, width } => {
            let value = operand(builder, values, value)?;
            match width {
                GuestAddressWidth::Bits32 => {
                    let value = if builder.func.dfg.value_type(value) == types::I32 {
                        value
                    } else {
                        builder.ins().ireduce(types::I32, value)
                    };
                    builder.ins().uextend(types::I64, value)
                }
                GuestAddressWidth::Bits64 => value,
            }
        }
        AddressOperation::Offset {
            base,
            offset,
            width,
        } => {
            let base = operand(builder, values, base)?;
            let offset = operand(builder, values, offset)?;
            match width {
                GuestAddressWidth::Bits32 => {
                    let base = builder.ins().ireduce(types::I32, base);
                    let offset = if builder.func.dfg.value_type(offset) == types::I32 {
                        offset
                    } else {
                        builder.ins().ireduce(types::I32, offset)
                    };
                    let sum = builder.ins().iadd(base, offset);
                    builder.ins().uextend(types::I64, sum)
                }
                GuestAddressWidth::Bits64 => {
                    let offset = if builder.func.dfg.value_type(offset) == types::I64 {
                        offset
                    } else {
                        builder.ins().sextend(types::I64, offset)
                    };
                    builder.ins().iadd(base, offset)
                }
            }
        }
        AddressOperation::ToInteger { address, to } => {
            let address = operand(builder, values, address)?;
            if to == IrType::I32 {
                builder.ins().ireduce(types::I32, address)
            } else {
                address
            }
        }
    })
}

fn lower_flags(
    builder: &mut FunctionBuilder<'_>,
    operation: FlagOperation,
    values: &BTreeMap<ValueId, ir::Value>,
) -> Result<ir::Value, CompilerError> {
    Ok(match operation {
        FlagOperation::FromArithmetic {
            result,
            carry,
            overflow,
        } => {
            let result = operand(builder, values, result)?;
            let carry = operand(builder, values, carry)?;
            let overflow = operand(builder, values, overflow)?;
            flags_from_result(builder, result, carry, overflow)
        }
        FlagOperation::FromLogical { result, carry } => {
            let result = operand(builder, values, result)?;
            let carry = operand(builder, values, carry)?;
            let overflow = builder.ins().iconst(types::I8, 0);
            flags_from_result(builder, result, carry, overflow)
        }
        FlagOperation::FromPacked { value } => {
            let value = operand(builder, values, value)?;
            builder.ins().band_imm_s(value, 0xf000_0000_u32 as i64)
        }
        FlagOperation::Evaluate { flags, condition } => {
            let flags = operand(builder, values, flags)?;
            evaluate_condition(builder, flags, condition, true)
        }
        FlagOperation::EvaluateEncoded {
            flags,
            condition,
            nv_is_unconditional,
        } => {
            let flags = operand(builder, values, flags)?;
            let condition = operand(builder, values, condition)?;
            evaluate_encoded_condition(builder, flags, condition, nv_is_unconditional)
        }
        FlagOperation::Materialize { flags } => operand(builder, values, flags)?,
    })
}

fn flags_from_result(
    builder: &mut FunctionBuilder<'_>,
    result: ir::Value,
    carry: ir::Value,
    overflow: ir::Value,
) -> ir::Value {
    let negative = builder.ins().icmp_imm_s(IntCC::SignedLessThan, result, 0);
    let zero = builder.ins().icmp_imm_s(IntCC::Equal, result, 0);
    pack_flags(builder, negative, zero, carry, overflow)
}

fn pack_flags(
    builder: &mut FunctionBuilder<'_>,
    negative: ir::Value,
    zero: ir::Value,
    carry: ir::Value,
    overflow: ir::Value,
) -> ir::Value {
    let negative = builder.ins().uextend(types::I32, negative);
    let zero = builder.ins().uextend(types::I32, zero);
    let carry = builder.ins().uextend(types::I32, carry);
    let overflow = builder.ins().uextend(types::I32, overflow);
    let negative = builder.ins().ishl_imm_s(negative, 31);
    let zero = builder.ins().ishl_imm_s(zero, 30);
    let carry = builder.ins().ishl_imm_s(carry, 29);
    let overflow = builder.ins().ishl_imm_s(overflow, 28);
    let first = builder.ins().bor(negative, zero);
    let second = builder.ins().bor(carry, overflow);
    builder.ins().bor(first, second)
}

fn evaluate_condition(
    builder: &mut FunctionBuilder<'_>,
    flags: ir::Value,
    condition: Condition,
    nv_is_unconditional: bool,
) -> ir::Value {
    let n = flag_bit(builder, flags, 31);
    let z = flag_bit(builder, flags, 30);
    let c = flag_bit(builder, flags, 29);
    let v = flag_bit(builder, flags, 28);
    let one = builder.ins().iconst(types::I32, 1);
    let result = match condition {
        Condition::Eq => z,
        Condition::Ne => invert_bit(builder, z, one),
        Condition::Cs => c,
        Condition::Cc => invert_bit(builder, c, one),
        Condition::Mi => n,
        Condition::Pl => invert_bit(builder, n, one),
        Condition::Vs => v,
        Condition::Vc => invert_bit(builder, v, one),
        Condition::Hi => {
            let not_zero = invert_bit(builder, z, one);
            builder.ins().band(c, not_zero)
        }
        Condition::Ls => {
            let not_carry = invert_bit(builder, c, one);
            builder.ins().bor(not_carry, z)
        }
        Condition::Ge => {
            let condition = builder.ins().icmp(IntCC::Equal, n, v);
            bool_to_i32(builder, condition)
        }
        Condition::Lt => {
            let condition = builder.ins().icmp(IntCC::NotEqual, n, v);
            bool_to_i32(builder, condition)
        }
        Condition::Gt => {
            let not_zero = invert_bit(builder, z, one);
            let equal = builder.ins().icmp(IntCC::Equal, n, v);
            let equal = bool_to_i32(builder, equal);
            builder.ins().band(not_zero, equal)
        }
        Condition::Le => {
            let different = builder.ins().icmp(IntCC::NotEqual, n, v);
            let different = bool_to_i32(builder, different);
            builder.ins().bor(z, different)
        }
        Condition::Al => one,
        Condition::Nv if nv_is_unconditional => one,
        Condition::Nv => builder.ins().iconst(types::I32, 0),
    };
    builder.ins().icmp_imm_s(IntCC::NotEqual, result, 0)
}

fn flag_bit(builder: &mut FunctionBuilder<'_>, flags: ir::Value, shift: i64) -> ir::Value {
    let value = builder.ins().ushr_imm_s(flags, shift);
    builder.ins().band_imm_s(value, 1)
}

fn invert_bit(builder: &mut FunctionBuilder<'_>, value: ir::Value, one: ir::Value) -> ir::Value {
    builder.ins().bxor(value, one)
}

fn bool_to_i32(builder: &mut FunctionBuilder<'_>, value: ir::Value) -> ir::Value {
    builder.ins().uextend(types::I32, value)
}

fn evaluate_encoded_condition(
    builder: &mut FunctionBuilder<'_>,
    flags: ir::Value,
    condition: ir::Value,
    nv_is_unconditional: bool,
) -> ir::Value {
    let mut result = builder.ins().iconst(types::I8, 0);
    for encoding in 0..16_u8 {
        let matches = builder
            .ins()
            .icmp_imm_s(IntCC::Equal, condition, i64::from(encoding));
        let candidate = evaluate_condition(
            builder,
            flags,
            Condition::from_encoding(encoding),
            nv_is_unconditional,
        );
        result = builder.ins().select(matches, candidate, result);
    }
    result
}

fn lower_vector(
    builder: &mut FunctionBuilder<'_>,
    operation: VectorOperation,
    values: &BTreeMap<ValueId, ir::Value>,
) -> Result<Vec<ir::Value>, CompilerError> {
    match operation {
        VectorOperation::Arithmetic {
            kind,
            arrangement,
            lhs,
            rhs,
        } if matches!(kind, IntegerBinaryKind::Add | IntegerBinaryKind::Subtract) => {
            let lhs = operand(builder, values, lhs)?;
            let rhs = operand(builder, values, rhs)?;
            let lane = match arrangement.lane_type {
                nixe_cpu::ir::op::LaneType::I8 => types::I8,
                nixe_cpu::ir::op::LaneType::I16 => types::I16,
                nixe_cpu::ir::op::LaneType::I32 => types::I32,
                nixe_cpu::ir::op::LaneType::I64 => types::I64,
                _ => return Err(CompilerError::new("integer vector operation has FP lanes")),
            };
            let vector_type = lane
                .by(u32::from(arrangement.lane_count))
                .ok_or_else(|| CompilerError::new("unsupported vector arrangement"))?;
            let flags = bitcast_flags(builder);
            let lhs = builder.ins().bitcast(vector_type, flags, lhs);
            let rhs = builder.ins().bitcast(vector_type, flags, rhs);
            let result = if kind == IntegerBinaryKind::Add {
                builder.ins().iadd(lhs, rhs)
            } else {
                builder.ins().isub(lhs, rhs)
            };
            let opaque = if vector_type.bits() == 64 {
                types::I8X8
            } else {
                types::I8X16
            };
            let flags = bitcast_flags(builder);
            Ok(vec![builder.ins().bitcast(opaque, flags, result)])
        }
        _ => Err(CompilerError::new(format!(
            "vector operation requires helper lowering: {operation:?}"
        ))),
    }
}

fn lower_terminator(
    builder: &mut FunctionBuilder<'_>,
    region: &IrRegion,
    from: BlockId,
    block: &IrBlock,
    lowering: &mut LoweringState,
    values: &BTreeMap<ValueId, ir::Value>,
) -> Result<(), CompilerError> {
    match &block.terminator {
        Terminator::Direct { target }
        | Terminator::Indirect { target }
        | Terminator::Return { target } => {
            lower_target(builder, region, from, lowering, values, *target, None)
        }
        Terminator::Call {
            target,
            return_address,
        } => lower_target(
            builder,
            region,
            from,
            lowering,
            values,
            *target,
            Some(*return_address),
        ),
        Terminator::Conditional {
            condition,
            taken,
            fallthrough,
        } => {
            let condition = operand(builder, values, *condition)?;
            let taken_block = builder.create_block();
            let fallthrough_block = builder.create_block();
            builder
                .ins()
                .brif(condition, taken_block, &[], fallthrough_block, &[]);
            builder.switch_to_block(taken_block);
            lower_target(builder, region, from, lowering, values, *taken, None)?;
            builder.switch_to_block(fallthrough_block);
            lower_target(builder, region, from, lowering, values, *fallthrough, None)
        }
        Terminator::ConditionalCall {
            condition,
            target,
            fallthrough,
            return_address,
        } => {
            let condition = operand(builder, values, *condition)?;
            let taken_block = builder.create_block();
            let fallthrough_block = builder.create_block();
            builder
                .ins()
                .brif(condition, taken_block, &[], fallthrough_block, &[]);
            builder.switch_to_block(taken_block);
            lower_target(
                builder,
                region,
                from,
                lowering,
                values,
                *target,
                Some(*return_address),
            )?;
            builder.switch_to_block(fallthrough_block);
            lower_target(builder, region, from, lowering, values, *fallthrough, None)
        }
        Terminator::Exception {
            source,
            kind,
            syndrome,
        } => {
            let side = push_side_exit(
                lowering,
                SideExit::Architectural {
                    source: compact_source_index(lowering, *source)?,
                    kind: *kind,
                    syndrome: *syndrome,
                },
            )?;
            emit_exit(
                builder,
                lowering,
                EXIT_ARCHITECTURAL,
                side,
                source.pc.get(),
                syndrome.unwrap_or(0),
                0,
            );
            Ok(())
        }
        Terminator::ConditionalException {
            condition,
            source,
            kind,
            syndrome,
            fallthrough,
        } => {
            let condition = operand(builder, values, *condition)?;
            let exception = builder.create_block();
            let resume = builder.create_block();
            builder.ins().brif(condition, exception, &[], resume, &[]);
            builder.switch_to_block(exception);
            let side = push_side_exit(
                lowering,
                SideExit::Architectural {
                    source: compact_source_index(lowering, *source)?,
                    kind: *kind,
                    syndrome: *syndrome,
                },
            )?;
            emit_exit(
                builder,
                lowering,
                EXIT_ARCHITECTURAL,
                side,
                source.pc.get(),
                syndrome.unwrap_or(0),
                0,
            );
            builder.switch_to_block(resume);
            lower_target(builder, region, from, lowering, values, *fallthrough, None)
        }
        Terminator::InterpretOne {
            source,
            encoding: _,
            coverage_id: _,
        } => {
            decrement_retired(builder, lowering);
            emit_exit(
                builder,
                lowering,
                EXIT_INTERPRET_ONE,
                0,
                source.pc.get(),
                0,
                0,
            );
            Ok(())
        }
        Terminator::UnsupportedInstruction {
            source,
            encoding,
            coverage_id,
            disassembly,
            reason: _,
        } => {
            let side = push_side_exit(
                lowering,
                SideExit::Unsupported {
                    source: compact_source_index(lowering, *source)?,
                    encoding: *encoding,
                    coverage_id: *coverage_id,
                    disassembly: disassembly.clone(),
                },
            )?;
            decrement_retired(builder, lowering);
            emit_exit(
                builder,
                lowering,
                EXIT_UNSUPPORTED,
                side,
                source.pc.get(),
                0,
                0,
            );
            Ok(())
        }
        Terminator::Stop { source, reason } => {
            let kind = match reason {
                StopReason::DispatchBudgetExhausted | StopReason::TranslationLimit => {
                    EXIT_BUDGET_EXHAUSTED
                }
                StopReason::PendingEvent => EXIT_PENDING_EVENT,
                StopReason::DebugRequest | StopReason::ProcessExit => EXIT_SAFEPOINT,
            };
            emit_exit(builder, lowering, kind, 0, source.pc.get(), 0, 0);
            Ok(())
        }
    }
}

fn lower_target(
    builder: &mut FunctionBuilder<'_>,
    region: &IrRegion,
    from: BlockId,
    lowering: &mut LoweringState,
    values: &BTreeMap<ValueId, ir::Value>,
    target: ControlTarget,
    return_address: Option<GuestVirtualAddress>,
) -> Result<(), CompilerError> {
    if let ControlTarget::Internal { block } = target {
        if let Some(return_address) = return_address {
            install_call_link(builder, region, lowering, return_address)?;
        }
        if region.metadata.safepoints.iter().any(|safepoint| {
            safepoint.kind == RegionSafepointKind::BackwardEdge
                && safepoint.block == from
                && safepoint.target == Some(block)
        }) {
            let target_location = region.blocks[block.index() as usize].metadata.start;
            set_current_location(builder, lowering, target_location)?;
            emit_control_poll(builder, lowering, target_location)?;
        }
        builder
            .ins()
            .jump(lowering.blocks[block.index() as usize], &[]);
        return Ok(());
    }
    let source_state = region.metadata.start.execution_state;
    let (pc, state, metadata) = match target {
        ControlTarget::Direct {
            pc,
            execution_state,
        } => {
            let pc_value = builder.ins().iconst(types::I64, pc.get() as i64);
            if let Some(return_address) = return_address {
                install_call_link(builder, region, lowering, return_address)?;
            }
            let state = install_target(builder, lowering, pc_value, execution_state, source_state)?;
            (
                pc_value,
                state,
                LinkSiteMetadata {
                    kind: LinkKind::Direct,
                    direct_target: Some(LocationDescriptor::new(
                        pc,
                        execution_state,
                        region.metadata.start.profile_id,
                    )),
                },
            )
        }
        ControlTarget::Indirect {
            address,
            execution_state,
            source,
        } => {
            let pc = operand(builder, values, address)?;
            let alignment_mask = match execution_state {
                ExecutionState::A64 | ExecutionState::A32 => 3,
                ExecutionState::T32 => 1,
            };
            guard_computed_target_alignment(builder, lowering, pc, source, alignment_mask)?;
            if let Some(return_address) = return_address {
                install_call_link(builder, region, lowering, return_address)?;
            }
            let state = install_target(builder, lowering, pc, execution_state, source_state)?;
            (
                widen_pc(builder, pc),
                state,
                LinkSiteMetadata {
                    kind: LinkKind::Indirect,
                    direct_target: None,
                },
            )
        }
        ControlTarget::A32Interworking { address, source } => {
            let raw = operand(builder, values, address)?;
            let raw = if builder.func.dfg.value_type(raw) == types::I32 {
                raw
            } else {
                builder.ins().ireduce(types::I32, raw)
            };
            let thumb_bit = builder.ins().band_imm_s(raw, 1);
            let a32_target = builder.ins().icmp_imm_s(IntCC::Equal, thumb_bit, 0);
            let a32_misalignment = builder.ins().band_imm_s(raw, 2);
            let a32_misalignment = builder
                .ins()
                .icmp_imm_s(IntCC::NotEqual, a32_misalignment, 0);
            let misaligned = builder.ins().band(a32_target, a32_misalignment);
            guard_target_condition(builder, lowering, misaligned, source)?;
            if let Some(return_address) = return_address {
                install_call_link(builder, region, lowering, return_address)?;
            }
            let pc = builder.ins().band_imm_s(raw, -2);
            let pc = builder.ins().uextend(types::I64, pc);
            let a32 = builder.ins().iconst(types::I32, EXECUTION_STATE_A32 as i64);
            let t32 = builder.ins().iconst(types::I32, EXECUTION_STATE_T32 as i64);
            let thumb = builder.ins().icmp_imm_s(IntCC::NotEqual, thumb_bit, 0);
            let state = builder.ins().select(thumb, t32, a32);
            install_dynamic_target(builder, lowering, pc, state, source_state)?;
            (
                pc,
                state,
                LinkSiteMetadata {
                    kind: LinkKind::Indirect,
                    direct_target: None,
                },
            )
        }
        ControlTarget::Internal { .. } => unreachable!(),
    };
    emit_link(builder, lowering, metadata, pc, state)
}

fn install_call_link(
    builder: &mut FunctionBuilder<'_>,
    region: &IrRegion,
    lowering: &LoweringState,
    return_address: GuestVirtualAddress,
) -> Result<(), CompilerError> {
    let (register, value) = match region.metadata.start.execution_state {
        ExecutionState::A64 => (
            StateRegister::A64X(A64GeneralRegister::new(30).unwrap()),
            builder
                .ins()
                .iconst(types::I64, return_address.get() as i64),
        ),
        ExecutionState::A32 => (
            StateRegister::A32R(A32GeneralRegister::new(14).unwrap()),
            builder
                .ins()
                .iconst(types::I32, return_address.get() as i64),
        ),
        ExecutionState::T32 => (
            StateRegister::A32R(A32GeneralRegister::new(14).unwrap()),
            builder
                .ins()
                .iconst(types::I32, (return_address.get() | 1) as i64),
        ),
    };
    define_state(builder, lowering, register, value)
}

fn guard_computed_target_alignment(
    builder: &mut FunctionBuilder<'_>,
    lowering: &mut LoweringState,
    target: ir::Value,
    source: LocationDescriptor,
    alignment_mask: i64,
) -> Result<(), CompilerError> {
    let masked = builder.ins().band_imm_s(target, alignment_mask);
    let misaligned = builder.ins().icmp_imm_s(IntCC::NotEqual, masked, 0);
    guard_target_condition(builder, lowering, misaligned, source)
}

fn guard_target_condition(
    builder: &mut FunctionBuilder<'_>,
    lowering: &mut LoweringState,
    misaligned: ir::Value,
    source: LocationDescriptor,
) -> Result<(), CompilerError> {
    let fault = builder.create_block();
    let aligned = builder.create_block();
    builder.ins().brif(misaligned, fault, &[], aligned, &[]);
    builder.switch_to_block(fault);
    let side = push_side_exit(
        lowering,
        SideExit::Architectural {
            source: compact_source_index(lowering, source)?,
            kind: nixe_cpu::exception::ExceptionKind::AlignmentFault,
            syndrome: None,
        },
    )?;
    emit_exit(
        builder,
        lowering,
        EXIT_ARCHITECTURAL,
        side,
        source.pc.get(),
        0,
        0,
    );
    builder.switch_to_block(aligned);
    Ok(())
}

fn install_target(
    builder: &mut FunctionBuilder<'_>,
    lowering: &LoweringState,
    pc: ir::Value,
    target_state: ExecutionState,
    source_state: ExecutionState,
) -> Result<ir::Value, CompilerError> {
    let state = builder.ins().iconst(
        types::I32,
        match target_state {
            ExecutionState::A64 => EXECUTION_STATE_A64,
            ExecutionState::A32 => EXECUTION_STATE_A32,
            ExecutionState::T32 => EXECUTION_STATE_T32,
        } as i64,
    );
    install_dynamic_target(builder, lowering, pc, state, source_state)?;
    Ok(state)
}

fn widen_pc(builder: &mut FunctionBuilder<'_>, pc: ir::Value) -> ir::Value {
    if builder.func.dfg.value_type(pc) == types::I32 {
        builder.ins().uextend(types::I64, pc)
    } else {
        pc
    }
}

fn emit_link(
    builder: &mut FunctionBuilder<'_>,
    lowering: &mut LoweringState,
    metadata: LinkSiteMetadata,
    guest_pc: ir::Value,
    guest_state: ir::Value,
) -> Result<(), CompilerError> {
    let site = u32::try_from(lowering.link_sites.len())
        .map_err(|_| CompilerError::new("link-site metadata index overflow"))?;
    lowering.link_sites.push(metadata);
    store_architectural_state(builder, lowering)?;

    let pointer_type = builder.func.dfg.value_type(lowering.frame);
    let table = load(
        builder,
        pointer_type,
        lowering.frame,
        FRAME_OFFSETS.dispatch_link_table,
    )?;
    let site_offset = usize::try_from(site)
        .ok()
        .and_then(|site| site.checked_mul(LINK_OFFSETS.site_size))
        .ok_or_else(|| CompilerError::new("native link-table offset overflow"))?;
    let miss = builder.create_block();
    let ways = match metadata.kind {
        LinkKind::Direct => 1,
        LinkKind::Indirect => crate::links::INDIRECT_LINK_WAYS,
    };
    for way in 0..ways {
        let cell_offset = site_offset
            .checked_add(
                way.checked_mul(LINK_OFFSETS.cell_size)
                    .ok_or_else(|| CompilerError::new("native link-cell offset overflow"))?,
            )
            .and_then(|value| value.checked_add(LINK_OFFSETS.cell_target))
            .ok_or_else(|| CompilerError::new("native link-cell offset overflow"))?;
        let cell = builder.ins().iadd_imm_u(
            table,
            i64::try_from(cell_offset).map_err(|_| {
                CompilerError::new("native link-cell offset exceeds pointer arithmetic")
            })?,
        );
        let flags = plain_mem_flags(builder);
        let target = builder.ins().atomic_load(pointer_type, flags, cell);
        let populated = builder.ins().icmp_imm_s(IntCC::NotEqual, target, 0);
        let candidate = builder.create_block();
        let next = if way + 1 == ways {
            miss
        } else {
            builder.create_block()
        };
        builder.ins().brif(populated, candidate, &[], next, &[]);
        builder.switch_to_block(candidate);
        if metadata.kind == LinkKind::Indirect {
            let tag_pc = load(builder, types::I64, target, LINK_OFFSETS.target_guest_pc)?;
            let tag_state = load(builder, types::I32, target, LINK_OFFSETS.target_guest_state)?;
            let pc_matches = builder.ins().icmp(IntCC::Equal, tag_pc, guest_pc);
            let state_matches = builder.ins().icmp(IntCC::Equal, tag_state, guest_state);
            let matches = builder.ins().band(pc_matches, state_matches);
            let linked = builder.create_block();
            builder.ins().brif(matches, linked, &[], next, &[]);
            builder.switch_to_block(linked);
        }
        emit_link_tail_call(builder, lowering, target)?;
        if next != miss {
            builder.switch_to_block(next);
        }
    }
    builder.switch_to_block(miss);
    emit_exit(builder, lowering, EXIT_DISPATCH, site, 0, 0, 0);
    Ok(())
}

fn emit_link_tail_call(
    builder: &mut FunctionBuilder<'_>,
    lowering: &LoweringState,
    target: ir::Value,
) -> Result<(), CompilerError> {
    let pointer_type = builder.func.dfg.value_type(lowering.frame);
    let local = builder.use_var(lowering.retired);
    let carried = load(
        builder,
        types::I64,
        lowering.frame,
        FRAME_OFFSETS.dispatch_retired,
    )?;
    let retired = builder.ins().iadd(carried, local);
    store(
        builder,
        retired,
        lowering.frame,
        offset(FRAME_OFFSETS.dispatch_retired)?,
    );
    for (ty, source, destination) in [
        (
            pointer_type,
            LINK_OFFSETS.target_link_table,
            FRAME_OFFSETS.dispatch_link_table,
        ),
        (
            pointer_type,
            LINK_OFFSETS.target_metadata,
            FRAME_OFFSETS.dispatch_metadata,
        ),
        (
            types::I64,
            LINK_OFFSETS.target_region_id,
            FRAME_OFFSETS.dispatch_region_id,
        ),
    ] {
        let value = load(builder, ty, target, source)?;
        store(builder, value, lowering.frame, offset(destination)?);
    }
    let entry = load(builder, pointer_type, target, LINK_OFFSETS.target_entry)?;
    let mut signature = Signature::new(CallConv::Tail);
    signature.params.push(AbiParam::new(pointer_type));
    let signature = builder.import_signature(signature);
    builder
        .ins()
        .return_call_indirect(signature, entry, &[lowering.frame]);
    Ok(())
}

fn install_dynamic_target(
    builder: &mut FunctionBuilder<'_>,
    lowering: &LoweringState,
    pc: ir::Value,
    state: ir::Value,
    source_state: ExecutionState,
) -> Result<(), CompilerError> {
    builder.def_var(lowering.execution_state, state);
    match source_state {
        ExecutionState::A64 => define_state(builder, lowering, StateRegister::A64Pc, pc),
        ExecutionState::A32 | ExecutionState::T32 => {
            let pc = if builder.func.dfg.value_type(pc) == types::I32 {
                pc
            } else {
                builder.ins().ireduce(types::I32, pc)
            };
            define_state(builder, lowering, StateRegister::A32Pc, pc)?;
            let cpsr = state_value_for(builder, lowering, StateRegister::A32Cpsr)?;
            let cleared = builder.ins().band_imm_s(cpsr, !(1_i64 << 5));
            let is_thumb =
                builder
                    .ins()
                    .icmp_imm_s(IntCC::Equal, state, EXECUTION_STATE_T32 as i64);
            let thumb = builder.ins().iconst(types::I32, 1_i64 << 5);
            let zero = builder.ins().iconst(types::I32, 0);
            let thumb = builder.ins().select(is_thumb, thumb, zero);
            let cpsr = builder.ins().bor(cleared, thumb);
            define_state(builder, lowering, StateRegister::A32Cpsr, cpsr)
        }
    }
}

fn push_side_exit(lowering: &mut LoweringState, exit: SideExit) -> Result<u32, CompilerError> {
    let index = u32::try_from(lowering.side_exits.len())
        .map_err(|_| CompilerError::new("side-exit metadata index overflow"))?;
    lowering.side_exits.push(exit);
    Ok(index)
}

fn compact_source_index(
    lowering: &LoweringState,
    source: LocationDescriptor,
) -> Result<u32, CompilerError> {
    lowering
        .source_indices
        .get(&source)
        .copied()
        .ok_or_else(|| CompilerError::new("side exit source is absent from compact metadata"))
}

fn emit_exit(
    builder: &mut FunctionBuilder<'_>,
    lowering: &LoweringState,
    kind: u32,
    detail: u32,
    source_pc: u64,
    payload0: u64,
    payload1: u64,
) {
    let kind = builder.ins().iconst(types::I32, kind as i64);
    let detail = builder.ins().iconst(types::I32, detail as i64);
    let source_pc = builder.ins().iconst(types::I64, source_pc as i64);
    let payload0 = builder.ins().iconst(types::I64, payload0 as i64);
    let payload1 = builder.ins().iconst(types::I64, payload1 as i64);
    let retired = builder.use_var(lowering.retired);
    let arguments = [kind, detail, source_pc, payload0, payload1, retired].map(BlockArg::from);
    builder.ins().jump(lowering.exit, &arguments);
}

fn emit_exit_dynamic(
    builder: &mut FunctionBuilder<'_>,
    lowering: &LoweringState,
    kind: u32,
    detail: ir::Value,
    source_pc: u64,
) {
    let kind = builder.ins().iconst(types::I32, i64::from(kind));
    let source_pc = builder.ins().iconst(types::I64, source_pc as i64);
    let zero = builder.ins().iconst(types::I64, 0);
    let retired = builder.use_var(lowering.retired);
    let arguments = [kind, detail, source_pc, zero, zero, retired].map(BlockArg::from);
    builder.ins().jump(lowering.exit, &arguments);
}

fn emit_exit_with_payload(
    builder: &mut FunctionBuilder<'_>,
    lowering: &LoweringState,
    kind: u32,
    source_pc: u64,
    payload0: ir::Value,
) {
    let kind = builder.ins().iconst(types::I32, i64::from(kind));
    let detail = builder.ins().iconst(types::I32, 0);
    let source_pc = builder.ins().iconst(types::I64, source_pc as i64);
    let zero = builder.ins().iconst(types::I64, 0);
    let retired = builder.use_var(lowering.retired);
    let arguments = [kind, detail, source_pc, payload0, zero, retired].map(BlockArg::from);
    builder.ins().jump(lowering.exit, &arguments);
}

fn lower_exit_block(
    builder: &mut FunctionBuilder<'_>,
    lowering: &LoweringState,
) -> Result<(), CompilerError> {
    builder.switch_to_block(lowering.exit);
    store_architectural_state(builder, lowering)?;
    let params = builder.block_params(lowering.exit).to_vec();
    for (value, destination) in params[..5].iter().copied().zip([
        FRAME_OFFSETS.exit_kind,
        FRAME_OFFSETS.exit_detail,
        FRAME_OFFSETS.exit_source_pc,
        FRAME_OFFSETS.exit_payload0,
        FRAME_OFFSETS.exit_payload1,
    ]) {
        store(builder, value, lowering.frame, offset(destination)?);
    }
    let carried = load(
        builder,
        types::I64,
        lowering.frame,
        FRAME_OFFSETS.dispatch_retired,
    )?;
    let retired = builder.ins().iadd(carried, params[5]);
    store(
        builder,
        retired,
        lowering.frame,
        offset(FRAME_OFFSETS.exit_instructions_executed)?,
    );
    builder.ins().return_(&[]);
    Ok(())
}

fn store_architectural_state(
    builder: &mut FunctionBuilder<'_>,
    lowering: &LoweringState,
) -> Result<(), CompilerError> {
    for slot in &lowering.state {
        let value = builder.use_var(slot.variable);
        store(builder, value, lowering.frame, slot.offset);
    }
    let state = builder.use_var(lowering.execution_state);
    store(
        builder,
        state,
        lowering.frame,
        offset(FRAME_OFFSETS.execution_state)?,
    );
    Ok(())
}

fn declare_state(
    builder: &mut FunctionBuilder<'_>,
    execution_state: ExecutionState,
    frame: ir::Value,
) -> Result<Vec<StateSlot>, CompilerError> {
    let registers = state_registers(execution_state);
    let mut slots = Vec::with_capacity(registers.len());
    for register in registers {
        let ty = cranelift_type(register.ty());
        let variable = builder.declare_var(ty);
        let register_offset = state_offset(register)?;
        let flags = trusted_mem_flags(builder);
        let value = builder.ins().load(ty, flags, frame, register_offset);
        builder.def_var(variable, value);
        slots.push(StateSlot {
            register,
            variable,
            ty,
            offset: register_offset,
        });
    }
    Ok(slots)
}

fn state_registers(execution_state: ExecutionState) -> Vec<StateRegister> {
    match execution_state {
        ExecutionState::A64 => {
            let mut registers: Vec<_> = (0..31)
                .map(|index| StateRegister::A64X(A64GeneralRegister::new(index).unwrap()))
                .collect();
            registers.extend([
                StateRegister::A64Sp,
                StateRegister::A64Pc,
                StateRegister::A64Nzcv,
            ]);
            registers.extend((0..32).map(|index| {
                StateRegister::A64V(nixe_cpu::ir::op::RegisterIndex::new(index).unwrap())
            }));
            registers.extend([
                StateRegister::A64Fpcr,
                StateRegister::A64Fpsr,
                StateRegister::A64TpidrEl0,
                StateRegister::A64TpidrroEl0,
            ]);
            registers
        }
        ExecutionState::A32 | ExecutionState::T32 => {
            let mut registers: Vec<_> = (0..15)
                .map(|index| StateRegister::A32R(A32GeneralRegister::new(index).unwrap()))
                .collect();
            registers.extend([StateRegister::A32Pc, StateRegister::A32Cpsr]);
            registers.extend((0..32).map(|index| {
                StateRegister::A32D(nixe_cpu::ir::op::RegisterIndex::new(index).unwrap())
            }));
            registers.extend([
                StateRegister::A32Fpscr,
                StateRegister::A32Tpidrurw,
                StateRegister::A32Tpidruro,
            ]);
            registers
        }
    }
}

fn state_offset(register: StateRegister) -> Result<i32, CompilerError> {
    offset(match register {
        StateRegister::A64X(register) => FRAME_OFFSETS.a64_x + usize::from(register.index()) * 8,
        StateRegister::A64Sp => FRAME_OFFSETS.a64_sp,
        StateRegister::A64Pc => FRAME_OFFSETS.a64_pc,
        StateRegister::A64Nzcv => FRAME_OFFSETS.a64_nzcv,
        StateRegister::A64V(register) => {
            FRAME_OFFSETS.a64_vector + usize::from(register.get()) * 16
        }
        StateRegister::A64Fpcr => FRAME_OFFSETS.a64_fpcr,
        StateRegister::A64Fpsr => FRAME_OFFSETS.a64_fpsr,
        StateRegister::A64TpidrEl0 => FRAME_OFFSETS.a64_tpidr_el0,
        StateRegister::A64TpidrroEl0 => FRAME_OFFSETS.a64_tpidrro_el0,
        StateRegister::A32R(register) => FRAME_OFFSETS.a32_r + usize::from(register.index()) * 4,
        StateRegister::A32Pc => FRAME_OFFSETS.a32_pc,
        StateRegister::A32Cpsr => FRAME_OFFSETS.a32_cpsr,
        StateRegister::A32D(register) => FRAME_OFFSETS.a32_d + usize::from(register.get()) * 8,
        StateRegister::A32Fpscr => FRAME_OFFSETS.a32_fpscr,
        StateRegister::A32Tpidrurw => FRAME_OFFSETS.a32_tpidrurw,
        StateRegister::A32Tpidruro => FRAME_OFFSETS.a32_tpidruro,
    })
}

fn state_value_for(
    builder: &mut FunctionBuilder<'_>,
    lowering: &LoweringState,
    register: StateRegister,
) -> Result<ir::Value, CompilerError> {
    let slot = lowering
        .state
        .iter()
        .find(|slot| slot.register == register)
        .ok_or_else(|| CompilerError::new(format!("state register {register:?} unavailable")))?;
    Ok(builder.use_var(slot.variable))
}

fn define_state(
    builder: &mut FunctionBuilder<'_>,
    lowering: &LoweringState,
    register: StateRegister,
    value: ir::Value,
) -> Result<(), CompilerError> {
    let slot = lowering
        .state
        .iter()
        .find(|slot| slot.register == register)
        .ok_or_else(|| CompilerError::new(format!("state register {register:?} unavailable")))?;
    if builder.func.dfg.value_type(value) != slot.ty {
        return Err(CompilerError::new(format!(
            "state register {register:?} received the wrong Cranelift type"
        )));
    }
    builder.def_var(slot.variable, value);
    Ok(())
}

fn operand(
    builder: &mut FunctionBuilder<'_>,
    values: &BTreeMap<ValueId, ir::Value>,
    operand: Operand,
) -> Result<ir::Value, CompilerError> {
    match operand {
        Operand::Immediate(value) => Ok(immediate(builder, value)),
        Operand::Value(value) => values.get(&value.id).copied().ok_or_else(|| {
            CompilerError::new(format!("undefined lowered value %{}", value.id.index()))
        }),
    }
}

fn immediate(builder: &mut FunctionBuilder<'_>, value: Immediate) -> ir::Value {
    match value {
        Immediate::I1(value) => builder.ins().iconst(types::I8, i64::from(value)),
        Immediate::I8(value) => builder.ins().iconst(types::I8, i64::from(value)),
        Immediate::I16(value) => builder.ins().iconst(types::I16, i64::from(value)),
        Immediate::I32(value) => builder.ins().iconst(types::I32, i64::from(value)),
        Immediate::I64(value) => builder.ins().iconst(types::I64, value as i64),
        Immediate::I128(value) => integer_128(builder, value),
        Immediate::F16(value) => {
            let bits = builder.ins().iconst(types::I16, i64::from(value));
            let flags = bitcast_flags(builder);
            builder.ins().bitcast(types::F16, flags, bits)
        }
        Immediate::F32(value) => {
            let bits = builder.ins().iconst(types::I32, i64::from(value));
            let flags = bitcast_flags(builder);
            builder.ins().bitcast(types::F32, flags, bits)
        }
        Immediate::F64(value) => {
            let bits = builder.ins().iconst(types::I64, value as i64);
            let flags = bitcast_flags(builder);
            builder.ins().bitcast(types::F64, flags, bits)
        }
        Immediate::V64(value) => {
            let bits = builder.ins().iconst(types::I64, value as i64);
            let flags = bitcast_flags(builder);
            builder.ins().bitcast(types::I8X8, flags, bits)
        }
        Immediate::V128(value) => {
            let bits = integer_128(builder, value);
            let flags = bitcast_flags(builder);
            builder.ins().bitcast(types::I8X16, flags, bits)
        }
        Immediate::Address(value) => builder.ins().iconst(types::I64, value.get() as i64),
    }
}

fn integer_128(builder: &mut FunctionBuilder<'_>, value: u128) -> ir::Value {
    let low = builder.ins().iconst(types::I64, value as u64 as i64);
    let high = builder
        .ins()
        .iconst(types::I64, (value >> 64) as u64 as i64);
    builder.ins().iconcat(low, high)
}

fn int_condition(predicate: IntegerPredicate) -> IntCC {
    match predicate {
        IntegerPredicate::Equal => IntCC::Equal,
        IntegerPredicate::NotEqual => IntCC::NotEqual,
        IntegerPredicate::UnsignedLessThan => IntCC::UnsignedLessThan,
        IntegerPredicate::UnsignedLessThanOrEqual => IntCC::UnsignedLessThanOrEqual,
        IntegerPredicate::SignedLessThan => IntCC::SignedLessThan,
        IntegerPredicate::SignedLessThanOrEqual => IntCC::SignedLessThanOrEqual,
    }
}

fn cranelift_type(ty: IrType) -> ir::Type {
    match ty {
        IrType::I1 | IrType::I8 => types::I8,
        IrType::I16 => types::I16,
        IrType::I32 | IrType::Flags => types::I32,
        IrType::I64 | IrType::Address => types::I64,
        IrType::I128 => types::I128,
        IrType::F16 => types::F16,
        IrType::F32 => types::F32,
        IrType::F64 => types::F64,
        IrType::V64 => types::I8X8,
        IrType::V128 => types::I8X16,
    }
}

fn load(
    builder: &mut FunctionBuilder<'_>,
    ty: ir::Type,
    frame: ir::Value,
    frame_offset: usize,
) -> Result<ir::Value, CompilerError> {
    let flags = trusted_mem_flags(builder);
    Ok(builder.ins().load(ty, flags, frame, offset(frame_offset)?))
}

fn store(builder: &mut FunctionBuilder<'_>, value: ir::Value, frame: ir::Value, frame_offset: i32) {
    let flags = trusted_mem_flags(builder);
    builder.ins().store(flags, value, frame, frame_offset);
}

fn plain_mem_flags(_builder: &mut FunctionBuilder<'_>) -> MemFlagsData {
    MemFlagsData::new()
}

/// Cranelift requires an explicit byte order when a bitcast changes the lane
/// count. Nixe's opaque vector representation numbers byte zero from the low
/// end of the architectural register, matching the two-limb helper ABI.
fn bitcast_flags(_builder: &mut FunctionBuilder<'_>) -> MemFlagsData {
    MemFlagsData::new().with_endianness(ir::Endianness::Little)
}

fn trusted_mem_flags(_builder: &mut FunctionBuilder<'_>) -> MemFlagsData {
    MemFlagsData::trusted()
}

fn offset(value: usize) -> Result<i32, CompilerError> {
    i32::try_from(value).map_err(|_| CompilerError::new("native frame offset exceeds i32"))
}
