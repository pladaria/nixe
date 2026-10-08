//! Seed and reshape consumer for the existing fixed worker pool. The closure owns no
//! JitProcess/pool; each call borrows only that worker's private compiler scratch.

use super::{
    Graph,
    compiler::{
        backend::{Compiler, Failure},
        publication,
    },
};
use crate::{
    abi::HostAbi,
    jit_error::Error,
    lifetime::background::{
        Observation, Work,
        workers::{CompileError, Resources},
    },
};
use nixe_cpu::memory::ExecutionMemory;
use std::sync::Arc;

mod policy;

pub(crate) fn consumer(
    abi: HostAbi,
    arena_size: usize,
    memory: Arc<ExecutionMemory>,
) -> Result<impl Fn(&mut Resources, Work<'_>) -> Result<(), CompileError> + Send + Sync, Error> {
    let compiler = Compiler::new(abi, arena_size)?;
    Ok(move |resources: &mut Resources, work: Work<'_>| {
        let reshape = matches!(work.observation(), Observation::Reshape { .. });
        let graph = match Graph::discover(&work) {
            Ok(graph) => graph,
            Err(super::DiscoveryError::Interrupted(error)) => return Err(error),
            Err(super::DiscoveryError::Structural(result)) => {
                return publication::record_structural(&result, &*memory)
                    .map(|_| ())
                    .map_err(failure);
            }
        };
        let frozen = work.reserve_candidate(graph)?.freeze()?;
        if frozen.unchanged() {
            return publication::record_unchanged(&frozen, &*memory)
                .map(|_| ())
                .map_err(failure);
        }
        let (words, entries, count) = frozen.predecessor_size();
        let required = policy::required_samples(
            frozen.graph(),
            words,
            entries,
            count,
            frozen.entries().len(),
        );
        let observations = match work.observation() {
            Observation::Seed(seed) => seed.observations,
            Observation::Reshape { snapshot, .. } => snapshot.observations,
        };
        if observations < required {
            let _trace =
                nixe_trace::Span::new("cpu.hcq.defer_profit", observations as u64, required as u64);
            // This is not a structural negative: further executions can make
            // this exact candidate worthwhile without changing any code bytes.
            return Err(CompileError::Deferred);
        }
        let result = compiler.publish(
            &mut resources.context,
            &mut resources.frontend,
            &frozen,
            &*memory,
        );
        match result {
            Ok(_) => Ok(()),
            Err(Failure::Rejected(limit)) if reshape => {
                publication::record_backend_rejection(&frozen, limit, &*memory)
                    .map(|_| ())
                    .map_err(failure)
            }
            Err(Failure::Rejected(_)) => frozen.reject().map_err(Into::into),
            Err(error) => Err(failure(error)),
        }
    })
}

fn failure(error: Failure) -> CompileError {
    match error {
        Failure::Cancelled => CompileError::Cancelled,
        Failure::Deferred => CompileError::Deferred,
        Failure::Failed(error) => CompileError::Failed(error),
        Failure::Rejected(_) => Error::internal("unclassified HCQ backend rejection").into(),
    }
}
