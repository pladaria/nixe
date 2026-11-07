use super::*;
use nixe_memory::{AccessOperation, AccessResult, DeviceAccessEndpoint, DeviceAccessRequest};
use std::sync::{Arc, mpsc};

pub(super) struct Device(pub(super) mpsc::Sender<DeviceAccessRequest>);
impl DeviceAccessEndpoint for Device {
    fn submit(&self, request: DeviceAccessRequest) {
        self.0.send(request).unwrap();
    }
    fn cancel(&self, _: nixe_memory::AccessRequestId) {}
    fn allocation_released(&self, _: nixe_memory::MachineId, _: nixe_memory::AllocationId) {}
    fn physical_retired(
        &self,
        _: nixe_memory::MachineId,
        _: nixe_memory::AllocationId,
        _: Box<[nixe_memory::Extent]>,
    ) {
    }
    fn allocation_pressure(&self, _: nixe_memory::MachineId, _: u64) {}
}

#[test]
fn a_hle_descriptor_keeps_its_own_capture_when_it_aliases_a_remapped_command_buffer() {
    remapped_command_output(false);
}

#[test]
fn a_hle_map_alias_output_keeps_its_capture_separate_from_the_command() {
    remapped_command_output(true);
}

#[test]
fn kernel_transport_does_not_read_the_emitters_separate_receive_list() {
    capture_emitted_message(3, 128);
}

#[test]
fn kernel_transport_does_not_read_the_emitters_single_receive_descriptor() {
    capture_emitted_message(2, 0);
}

#[test]
fn a_pending_guest_receive_keeps_its_request_until_output_publication() {
    guest_receive(ReceiveEnd::Success);
}

#[test]
fn terminating_a_pending_guest_receive_completes_the_client_and_ignores_late_output() {
    guest_receive(ReceiveEnd::HostStop);
}

#[test]
fn failed_guest_receive_output_releases_its_request_and_preserves_the_host_fault() {
    guest_receive(ReceiveEnd::DeviceFailure);
}

enum ReceiveEnd {
    Success,
    HostStop,
    DeviceFailure,
}

#[test]
fn process_exit_cancels_a_pending_ipc_reply_without_publishing_its_handle() {
    pending_reply(Stop::Guest);
}

#[test]
fn a_pending_ipc_reply_publishes_its_handle_only_after_output_completes() {
    pending_reply(Stop::None);
}

#[test]
fn a_failed_server_reply_is_reported_even_when_the_client_process_exits() {
    pending_reply(Stop::DeviceFailure);
}

enum Stop {
    None,
    DeviceFailure,
    Guest,
    Host,
    CpuFault,
}

#[test]
fn host_stop_cancels_a_pending_ipc_reply_before_device_completion() {
    pending_reply(Stop::Host);
}

#[test]
fn terminal_cpu_failure_cancels_a_pending_reply_before_device_completion() {
    pending_reply(Stop::CpuFault);
}

#[test]
fn thread_creation_waits_for_device_owned_tls_initialization() {
    pending_thread_creation(false, false);
}

#[test]
fn process_exit_cancels_unpublished_thread_creation() {
    pending_thread_creation(true, false);
}

#[test]
fn failed_tls_initialization_reports_the_device_failure_without_publishing_a_thread() {
    pending_thread_creation(false, true);
}
