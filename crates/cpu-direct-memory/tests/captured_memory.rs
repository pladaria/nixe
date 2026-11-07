#![cfg(all(target_os = "linux", target_arch = "x86_64"))]

use nixe_cpu_direct_memory::{
    CapturedFault, FaultDisposition, InvocationOutcome, NativeInvocation, NativeWorker,
};
use nixe_memory::{
    Access, AccessOperation, AccessRequestId, AccessResult, AddressSpace, AddressWidth,
    AllocationId, AllocationRange, DeviceAccessEndpoint, DeviceAccessRequest, Extent,
    GuestVirtualAddress, Machine, MachineId, MemoryError, MemoryPermissions, NativeAdmission,
    PAGE_SIZE,
};
use std::sync::{Arc, mpsc};

// The only guest access is this attributed store. Escaping it crosses a gateway
// with no destructors; the exact operation commits later, after native return.
#[unsafe(naked)]
unsafe extern "C" fn store(_: *mut u8) {
    core::arch::naked_asm!("mov dword ptr [rdi], 0x44332211", "ret");
}
struct Call<'a> {
    space: &'a AddressSpace,
    admission: &'a NativeAdmission<'a>,
    address: GuestVirtualAddress,
    pointer: *mut u8,
    captured: Option<Vec<AllocationRange>>,
    error: Option<MemoryError>,
}
unsafe extern "C" fn gateway(opaque: *mut libc::c_void, _: usize) {
    let call = unsafe { &*opaque.cast::<Call<'_>>() };
    unsafe {
        store(call.pointer);
    }
}
unsafe extern "C" fn capture(
    opaque: *mut libc::c_void,
    fault: *mut CapturedFault<'_>,
) -> FaultDisposition {
    let call = unsafe { &mut *opaque.cast::<Call<'_>>() };
    let fault = unsafe { &*fault };
    if fault.native_pc() != store as *const () as usize
        || fault.fault_address() != call.pointer as usize
    {
        return FaultDisposition::FatalUnattributed;
    }
    match call
        .space
        .capture_native(call.admission, call.address, 4, MemoryPermissions::WRITE)
    {
        Ok(ranges) => call.captured = Some(ranges),
        Err(error) => call.error = Some(error),
    }
    FaultDisposition::Escape
}
struct Endpoint(mpsc::Sender<DeviceAccessRequest>);
impl DeviceAccessEndpoint for Endpoint {
    fn submit(&self, request: DeviceAccessRequest) {
        self.0.send(request).unwrap();
    }
    fn cancel(&self, _: AccessRequestId) {}
    fn allocation_pressure(&self, _: nixe_memory::MachineId, _: u64) {}
    fn allocation_released(&self, _: MachineId, _: AllocationId) {}
    fn physical_retired(&self, _: MachineId, _: AllocationId, _: Box<[Extent]>) {}
}

fn machine_write(space: &AddressSpace, ranges: &[AllocationRange]) {
    space
        .machine()
        .write_captured(
            &nixe_memory::CapturedMemory::new(ranges.iter().cloned()).unwrap(),
            0,
            &[0x11, 0x22, 0x33, 0x44],
        )
        .unwrap()
        .into_ready()
        .unwrap();
}
