use nixe_cpu::execution::CpuControl;
use nixe_cpu::memory::ExecutionMemory;
use nixe_runtime::{
    EventObject, ExternalEventSender, HandleObject, ReadableEventObject, RuntimeCoordinator,
    SharedMemoryObject, ThreadObject, WritableEventObject,
};

fn assert_send<T: Send>() {}
fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn future_worker_reachable_values_have_explicit_thread_traits() {
    assert_send_sync::<ExecutionMemory>();
    assert_send_sync::<CpuControl>();
    assert_send_sync::<EventObject>();
    assert_send_sync::<ReadableEventObject>();
    assert_send_sync::<WritableEventObject>();
    assert_send_sync::<ThreadObject>();
    assert_send_sync::<SharedMemoryObject>();
    assert_send_sync::<HandleObject>();
    assert_send_sync::<ExternalEventSender>();

    // The coordinator is transferred to its owner but deliberately not shared.
    assert_send::<RuntimeCoordinator>();
}
