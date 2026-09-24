use super::*;

mod installation;

fn branch(from: u64, to: u64) -> u32 {
    0x14000000 | (((to.wrapping_sub(from) / 4) as u32) & 0x03ff_ffff)
}

fn chain(process: &Lifetime, full: bool) -> Vec<UnitHandle> {
    let mut inputs = vec![
        publish_words(process, 0, &[branch(0, 0x1000)]),
        publish_words(process, 0x4000, &[branch(0x4000, 0x8000)]),
        publish_words(process, 0x8000, &[RET]),
    ];
    for (pc, next, count) in [
        (0x1000, 0x2000, 512),
        (0x2000, 0x3000, 512),
        (0x3000, 0x5000, 512),
        (
            0x5000,
            if full { 0x6000 } else { 0x4000 },
            if full { 509 } else { 512 },
        ),
    ] {
        let mut bits = vec![NOP; count];
        bits[count - 1] = branch(pc + (count as u64 - 1) * 4, next);
        inputs.push(publish_words(process, pc, &bits));
    }
    owned(
        process,
        &[(0, branch(0, 0x1000)), (0x4000, branch(0x4000, 0x8000))],
    );
    inputs
}

#[test]
fn disconnected_result_keeps_discarded_input_evidence_but_no_graph_or_code_pins() {
    let process = process();
    let inputs = [
        publish_words(&process, 0, &[RET]),
        publish_words(&process, 16, &[branch(16, 32)]),
        publish_words(&process, 32, &[RET]),
    ];
    owned(&process, &[(0, RET), (16, branch(16, 32))]);
    let work = reshape(&process, 0, 16, 32);
    let references = inputs
        .map(|input| Arc::strong_count(&process.lock().units.records.get(input.0).unwrap().code));
    let before = process.cache.usage().unwrap();
    let Err(DiscoveryError::Structural(result)) = Graph::discover(&work) else {
        panic!("expected disconnected discovery");
    };
    assert_eq!(result.reason(), StructuralReason::Disconnected);
    assert_eq!(result.inspected(), 3); // Connectivity trimming drops both 16 and 32.
    result.check().unwrap();
    for (input, references) in inputs.into_iter().zip(references) {
        assert_eq!(
            Arc::strong_count(&process.lock().units.records.get(input.0).unwrap().code),
            references
        );
    }
    let held = process.cache.usage().unwrap();
    assert!(held.metadata > before.metadata);
    assert_eq!(held.committed, before.committed);
    drop(result);
    assert_eq!(process.cache.usage().unwrap().metadata, before.metadata);
    // The result never consumed the Work's terminal-result slot/header.
    assert!(matches!(
        Graph::discover(&work),
        Err(DiscoveryError::Structural(_))
    ));
}

#[test]
fn cap_result_names_the_input_that_did_not_fit_and_revalidates_it() {
    let process = process();
    let inputs = chain(&process, false);
    let work = reshape(&process, 0, 0x4000, 0x8000);
    let Err(DiscoveryError::Structural(result)) = Graph::discover(&work) else {
        panic!("expected instruction ceiling outcome");
    };
    assert_eq!(result.reason(), StructuralReason::InstructionLimit);
    assert_eq!(result.inspected(), 7);
    result.check().unwrap();
    // The last 512-word input was acquired but contributed zero words, and
    // neither mandatory endpoint survived root-connectivity trimming.
    process.retire_unit(inputs[6]).unwrap();
    assert_eq!(result.check(), Err(Error::StalePublication));
}

#[test]
fn exactly_full_discovery_still_distinguishes_missing_frontier_from_cap_rejection() {
    let process = process();
    chain(&process, true);
    {
        let work = reshape(&process, 0, 0x4000, 0x8000);
        let before = process.cache.usage().unwrap().metadata;
        assert!(matches!(
            Graph::discover(&work),
            Err(DiscoveryError::Interrupted(CompileError::Deferred))
        ));
        assert_eq!(process.cache.usage().unwrap().metadata, before);
    }
    publish_words(&process, 0x6000, &[branch(0x6000, 0x4000)]);
    let work = reshape(&process, 0, 0x4000, 0x8000);
    let Err(DiscoveryError::Structural(result)) = Graph::discover(&work) else {
        panic!("demanded frontier should be a cap outcome, not missing input");
    };
    assert_eq!(result.reason(), StructuralReason::InstructionLimit);
    assert_eq!(result.inspected(), 8);
    result.check().unwrap();
}

#[test]
fn incomplete_disconnected_discovery_and_pressure_remain_retryable() {
    let process = process();
    publish_words(&process, 0, &[RET]);
    publish_words(&process, 16, &[branch(16, 32)]);
    publish_words(&process, 32, &[branch(32, 48)]);
    owned(&process, &[(0, RET), (16, branch(16, 32))]);
    {
        let work = reshape(&process, 0, 16, 32);
        assert!(matches!(
            Graph::discover(&work),
            Err(DiscoveryError::Interrupted(CompileError::Deferred))
        ));
    }
    publish_words(&process, 48, &[RET]);
    let work = reshape(&process, 0, 16, 32);
    let usage = process.cache.usage().unwrap();
    let pressure = process
        .cache
        .charge_metadata(crate::executable::SOFT_BYTES - usage.total() - 1, Tier::Lcq)
        .unwrap();
    assert!(matches!(
        Graph::discover(&work),
        Err(DiscoveryError::Interrupted(CompileError::Deferred))
    ));
    drop(pressure);
    assert_eq!(process.cache.usage().unwrap().metadata, usage.metadata);
    assert!(
        matches!(Graph::discover(&work), Err(DiscoveryError::Structural(result))
        if result.reason() == StructuralReason::Disconnected)
    );
    process.request_shutdown().unwrap();
    assert!(matches!(
        Graph::discover(&work),
        Err(DiscoveryError::Interrupted(CompileError::Cancelled))
    ));
}
