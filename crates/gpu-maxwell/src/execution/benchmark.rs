//! Isolate command-container retirement, excluding actual GPU work.
use super::*;
#[test]
fn partial_retirement_preserves_suffix_and_compilation_cursor_across_wraps() {
    let mut execution =
        MaxwellBackendExecution::<MaxwellLoweredWork>::begin(FrontendSubmissionId::new(1), None);
    for cycle in 0..64 {
        execution.append_steps(
            (0..32)
                .map(|_| MaxwellSubmissionExecutionStep::WaitForIdle)
                .collect(),
        );
        execution.compilation_cursor = execution.steps.len();
        let prior = execution.steps.len();
        execution.next_step = 24;
        assert_eq!(execution.retire_consumed_steps(), 24);
        assert_eq!(execution.steps.len(), prior - 24);
        assert_eq!(execution.compilation_cursor, prior - 24);
        assert_eq!(execution.next_step, 0);
        assert_eq!(execution.steps.len(), (cycle + 1) * 8);
        assert!(!execution.waiting_for_input());
    }
    execution.next_step = execution.steps.len();
    execution.retire_consumed_steps();
    assert!(execution.waiting_for_input());
}
#[test]
#[cfg(target_os = "linux")]
#[ignore = "opt-in prefix-retirement scaling, not complete GPU throughput"]
fn command_prefix_retirement_throughput() {
    let cpu_ns = || -> u64 {
        std::fs::read_to_string("/proc/thread-self/schedstat")
            .unwrap()
            .split_whitespace()
            .next()
            .unwrap()
            .parse()
            .unwrap()
    };
    for count in [1024usize, 8192] {
        let mut execution = MaxwellBackendExecution::<MaxwellLoweredWork>::begin(
            FrontendSubmissionId::new(1),
            None,
        );
        execution.append_steps(
            (0..count)
                .map(|_| MaxwellSubmissionExecutionStep::WaitForIdle)
                .collect(),
        );
        execution.compilation_cursor = count;
        let before = cpu_ns();
        let start = std::time::Instant::now();
        for _ in 0..count / 8 {
            execution.next_step = 8;
            std::hint::black_box(execution.retire_consumed_steps());
        }
        let seconds = start.elapsed().as_secs_f64();
        let cpu_nanoseconds = cpu_ns() - before;
        println!(
            "{{\"workload\":\"command_prefix_retirement\",\"steps\":{count},\"seconds\":{seconds},\"cpu_nanoseconds\":{cpu_nanoseconds}}}"
        );
    }
}
