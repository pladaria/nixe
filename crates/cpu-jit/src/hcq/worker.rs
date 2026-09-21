//! Real seed consumer for the existing fixed worker pool. The closure owns no
//! JitProcess/pool; each call borrows only that worker's private compiler scratch.

use super::{
    Graph,
    compiler::backend::{Compiler, Failure},
};
use crate::{
    abi::HostAbi,
    jit_error::Error,
    lifetime::{
        background::{
            Frozen, Observation, Work,
            workers::{CompileError, Resources},
        },
        unit::UnitHandle,
    },
};
use nixe_cpu::memory::ExecutionMemory;
use std::sync::Arc;

pub(crate) fn consumer(
    abi: HostAbi,
    arena_size: usize,
    memory: Arc<ExecutionMemory>,
) -> Result<impl Fn(&mut Resources, Work<'_>) -> Result<(), CompileError> + Send + Sync, Error> {
    let compiler = Compiler::new(abi, arena_size)?;
    Ok(move |resources: &mut Resources, work: Work<'_>| {
        if !matches!(work.observation(), Observation::Seed(_)) {
            return Err(Error::internal("HCQ reshape consumer is not implemented").into());
        }
        let graph = Graph::discover(&work)?;
        let frozen = work.reserve_candidate(graph)?.freeze()?;
        let result = compiler.publish(
            &mut resources.context,
            &mut resources.frontend,
            &frozen,
            &*memory,
        );
        complete(&frozen, result)
    })
}

fn complete(
    frozen: &Frozen<'_, '_>,
    result: Result<UnitHandle, Failure>,
) -> Result<(), CompileError> {
    match result {
        Ok(_) => Ok(()),
        Err(Failure::Cancelled) => Err(CompileError::Cancelled),
        Err(Failure::Deferred) => Err(CompileError::Deferred),
        Err(Failure::Rejected(_)) => frozen.reject().map_err(Into::into),
        Err(Failure::Failed(error)) => Err(CompileError::Failed(error)),
    }
}
