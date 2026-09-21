//! HCQ target policy and backend completion using worker-owned scratch.

use super::*;
use crate::abi::CodeVersion;
use crate::frontend::target::{self, Policy};
use cranelift_codegen::{CodegenError, control::ControlPlane};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::hcq) enum Limit {
    Implementation,
    CodeSize,
}

#[derive(Debug)]
pub(in crate::hcq) enum Failure {
    Cancelled,
    Deferred,
    /// The consumer must reject only this captured version, releasing its
    /// reservation and retaining LCQ. This is not transient cache pressure.
    Rejected(Limit),
    Failed(Error),
}

impl From<crate::lifetime::background::workers::CompileError> for Failure {
    fn from(error: crate::lifetime::background::workers::CompileError) -> Self {
        use crate::lifetime::background::workers::CompileError;
        match error {
            CompileError::Cancelled => Self::Cancelled,
            CompileError::Deferred => Self::Deferred,
            CompileError::Failed(error) => Self::Failed(error),
        }
    }
}

impl From<crate::lifetime::Error> for Failure {
    fn from(error: crate::lifetime::Error) -> Self {
        crate::lifetime::background::workers::CompileError::from(error).into()
    }
}

impl Failure {
    fn backend(error: CodegenError) -> Self {
        match error {
            CodegenError::ImplLimitExceeded => Self::Rejected(Limit::Implementation),
            CodegenError::CodeTooLarge => Self::Rejected(Limit::CodeSize),
            // In particular, Unsupported is NOT an optimizer-shape rejection:
            // it may report a missing lowering or a forbidden backend call.
            error => Self::Failed(Error::internal(format!("HCQ Cranelift: {error:?}"))),
        }
    }
}

/// Immutable target policy can be shared; Context/FunctionBuilderContext stay
/// with each worker. No additional worker pool or compiler scratch is owned here.
pub(in crate::hcq) struct Compiler {
    abi: HostAbi,
    isa: Arc<dyn TargetIsa>,
    arena_size: usize,
}

impl Compiler {
    pub(in crate::hcq) fn new(abi: HostAbi, arena_size: usize) -> Result<Self, Error> {
        crate::frontend::arena_size(arena_size)?;
        Ok(Self {
            abi,
            isa: target::build(abi, Policy::Hcq)?,
            arena_size,
        })
    }

    pub(in crate::hcq) fn emit(
        &self,
        context: &mut Context,
        frontend: &mut FunctionBuilderContext,
        graph: &Graph,
        analysis: &Analysis,
        entries: &[usize],
    ) -> Result<Body, Error> {
        super::emit(
            self.abi,
            &*self.isa,
            Some(self.arena_size),
            context,
            frontend,
            graph,
            analysis,
            entries,
        )
    }

    pub(in crate::hcq) fn finish(
        &self,
        context: &mut Context,
        body: Body,
        graph: &Graph,
        version: CodeVersion,
    ) -> Result<stage::Staged, Failure> {
        let result = (|| {
            for (index, exit) in body.exits.iter().enumerate() {
                if exit.reason == NativeExitReason::Dispatch {
                    context
                        .func
                        .nixe_exit_costs
                        .insert(index as u64 + 1, exit.completed);
                }
            }
            context
                .compile(&*self.isa, &mut ControlPlane::default())
                .map_err(|error| Failure::backend(error.inner))?;
            let code = context.take_compiled_code().unwrap();
            body.stage(self.abi, code, &context.func, graph, version)
                .map_err(Failure::Failed)
        })();
        // Successful staging owns all output; failed/rejected compilation must
        // not leak partial code or exit-cost metadata into the next worker job.
        context.clear();
        result
    }
}

#[cfg(test)]
mod tests;
