use super::*;
use crate::executable::{HARD_BYTES, SOFT_BYTES, Tier};
use nixe_cpu::execution::{
    ArchitecturalTimer, CpuExit, CpuFaultKind, TimerSnapshot, VcpuEventState,
};

mod phases;

struct Timer;
impl ArchitecturalTimer for Timer {
    fn snapshot(&self) -> TimerSnapshot {
        TimerSnapshot {
            counter: 17,
            frequency: 19,
        }
    }
}

#[test]
fn cold_miss_reclaims_live_lcq_then_recaptures_without_charging_guest_work() {
    let mut worker = NativeWorker::default();
    let mut thread = budget::setup(&[0xd4200000, 0xd4200120], false); // BRK; BRK #9.
    let cache = thread.process.lifetime.executable_cache().clone();
    // Use the real accounting API, without allocating hundreds of MiB in a
    // test or changing production limits. One live code segment can cover this.
    let charge = cache
        .charge_metadata(
            SOFT_BYTES + 4096 - cache.usage().unwrap().total(),
            Tier::Lcq,
        )
        .unwrap();
    let mut state = A64State::default();
    state.set_pc(PC.get() + 4);
    let report = thread
        .run_slice(
            &mut crate::ReturnStack::default(),
            &mut worker,
            &mut state,
            10,
            &Timer,
            &VcpuEventState::default(),
        )
        .unwrap();
    assert!(matches!(
        report.stop,
        CpuExit::ArchitecturalException {
            syndrome: Some(9),
            ..
        }
    ));
    assert_eq!(report.progress, 1);
    assert_eq!(thread.sample_remaining, 4095);
    // The original demanded root was evicted, not just subtracted from usage.
    state.set_pc(PC.get());
    assert!(
        thread
            .invoke(
                &mut crate::ReturnStack::default(),
                &mut worker,
                &mut state,
                PollBudget::new(4096, 1).unwrap(),
                &VcpuEventState::default()
            )
            .unwrap()
            .0
            .is_none()
    );
    drop(charge);
    assert!(thread.process.try_shutdown().unwrap());
    assert_eq!(cache.usage().unwrap().committed, 0);
}

#[test]
fn unsatisfied_capacity_stops_once_with_exact_state_and_does_not_poison_the_cache() {
    let mut worker = NativeWorker::default();
    let process = Arc::new(JitProcess::new(cpu(), memory(DirectBackendPolicy::Required)).unwrap());
    let mut thread = JitThread::new(process.clone()).unwrap();
    let cache = process.lifetime.executable_cache();
    let charge = cache
        .charge_metadata(HARD_BYTES - cache.usage().unwrap().total(), Tier::Lcq)
        .unwrap();
    let mut state = A64State::default();
    state.set_pc(PC.get());
    let before = state.clone();
    let events = VcpuEventState::default();
    let error = thread
        .run_slice(
            &mut crate::ReturnStack::default(),
            &mut worker,
            &mut state,
            10,
            &Timer,
            &events,
        )
        .unwrap_err();
    assert_eq!(error.kind, CpuFaultKind::Unavailable);
    assert!(error.message.contains("LCQ capacity at"));
    assert!(error.message.contains("640 MiB code+metadata hard limit"));
    assert_eq!(error.progress, 0);
    assert_eq!(state, before);
    assert_eq!(*error.context, state.register_context());
    assert_eq!(cache.usage().unwrap().committed, 0);
    drop(charge);
    // Allocation authority remains usable after the external owner releases
    // its charge. No poisoned cache, stranded closure or legacy fallback.
    let report = thread
        .run_slice(
            &mut crate::ReturnStack::default(),
            &mut worker,
            &mut state,
            10,
            &Timer,
            &events,
        )
        .unwrap();
    assert!(matches!(
        report.stop,
        CpuExit::ArchitecturalException {
            syndrome: Some(1),
            ..
        }
    ));
    assert_eq!(report.progress, 2); // NOP; BRK #1.
}
