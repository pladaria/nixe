//! Executor-local reusable Cranelift state.

use cranelift_codegen::Context;
use cranelift_codegen::isa::OwnedTargetIsa;
use cranelift_frontend::FunctionBuilderContext;

use crate::abi::{FRAME_OFFSETS, HELPER_OFFSETS};

/// Scratch state retained by one executor across region compilation attempts.
pub(crate) struct CompilerContext {
    // JIT-006 consumes this retained state when real region compilation begins.
    // Until then it is deliberately inert: the fallback path must not fabricate
    // compilation attempts merely to exercise future storage.
    _isa: OwnedTargetIsa,
    _context: Context,
    _builder: FunctionBuilderContext,
}

impl CompilerContext {
    pub(crate) fn new(isa: OwnedTargetIsa) -> Self {
        debug_assert!(
            FRAME_OFFSETS
                .all()
                .into_iter()
                .chain(HELPER_OFFSETS.all())
                .all(|offset| i32::try_from(offset).is_ok()),
            "native frame offsets must fit Cranelift load/store immediates"
        );
        Self {
            _isa: isa,
            _context: Context::new(),
            _builder: FunctionBuilderContext::new(),
        }
    }
}
