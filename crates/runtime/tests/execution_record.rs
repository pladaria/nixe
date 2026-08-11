use std::num::NonZeroUsize;

use nixe_runtime::{ExternalEvent, ReplayMismatchKind, RuntimeCoordinator};
use nixe_scheduler::{MachineSchedulerProfile, PriorityRange, VirtualCpuDescriptor, VirtualCpuId};

fn coordinator() -> RuntimeCoordinator {
    let profile = MachineSchedulerProfile::new(
        vec![VirtualCpuDescriptor::new(VirtualCpuId::new(0), 0)],
        PriorityRange::new(0, 63).unwrap(),
        100,
    )
    .unwrap();
    RuntimeCoordinator::new(profile)
}

#[test]
fn public_recording_api_preserves_external_order_and_reports_divergence() {
    let mut expected = coordinator();
    expected.enable_execution_recording(NonZeroUsize::new(8).unwrap());
    expected
        .event_sender()
        .submit(ExternalEvent::HostStop)
        .unwrap();
    expected.drain_external_events().unwrap();
    let expected = expected.take_execution_record().unwrap();

    let mut observed = coordinator();
    observed.enable_execution_recording(NonZeroUsize::new(8).unwrap());
    observed
        .event_sender()
        .submit(ExternalEvent::HostStop)
        .unwrap();
    observed
        .event_sender()
        .submit(ExternalEvent::HostStop)
        .unwrap();
    observed.drain_external_events().unwrap();
    let observed = observed.take_execution_record().unwrap();

    let mismatch = expected.compare(&observed).unwrap_err();
    assert_eq!(mismatch.index, 1);
    assert_eq!(mismatch.kind, ReplayMismatchKind::Length);
}
