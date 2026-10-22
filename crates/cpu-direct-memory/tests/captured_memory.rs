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

#[test]
fn native_store_capture_keeps_old_physical_input_after_return_and_remap() {
    let machine = Machine::new(2 * PAGE_SIZE).unwrap();
    let account = machine.account(2 * PAGE_SIZE).unwrap();
    let old = account.allocate(PAGE_SIZE).unwrap();
    let old_id = old.identity();
    let range = AllocationRange::new(old.pin(), 0, 4).unwrap();
    let space = AddressSpace::new(machine.clone(), AddressWidth::Bits32).unwrap();
    let address = GuestVirtualAddress::new(PAGE_SIZE);
    space
        .map(address, old, 0, PAGE_SIZE, MemoryPermissions::READ_WRITE)
        .unwrap();
    machine.publish_device_write(&[range], || Ok(())).unwrap();
    let (sender, receiver) = mpsc::channel();
    let endpoint: Arc<dyn DeviceAccessEndpoint> = Arc::new(Endpoint(sender));
    machine.set_device_endpoint(&endpoint);
    let core = space.register_core(Arc::new(|| {
        panic!("this test never mutates an active core")
    }));
    let admission = core.enter_native().unwrap();
    let arena = space.native_data_base().as_ptr() as usize;
    let mut call = Call {
        space: &space,
        admission: &admission,
        address,
        pointer: (arena + address.get() as usize) as *mut u8,
        captured: None,
        error: None,
    };
    let mut mxcsr = 0u32;
    unsafe {
        core::arch::asm!("stmxcsr [{}]", in(reg) &mut mxcsr, options(nostack, preserves_flags));
    }
    let mut worker = NativeWorker::default();
    let opaque = std::ptr::from_mut(&mut call).cast();
    let result = unsafe {
        worker.faults().unwrap().invoke_captured(
            arena..arena + space.width().bytes() as usize,
            [u64::from(mxcsr), 0],
            capture,
            opaque,
            NativeInvocation {
                gateway,
                context: opaque,
                entry: 0,
            },
        )
    }
    .unwrap();
    assert_eq!(result, InvocationOutcome::Escaped);
    assert!(call.error.is_none());
    let captured = call.captured.take().unwrap();
    assert_eq!(captured[0].allocation().identity(), old_id);
    admission.suspend();
    let replacement = account.allocate(PAGE_SIZE).unwrap();
    let replacement_id = replacement.identity();
    space
        .remap(
            address,
            replacement,
            0,
            PAGE_SIZE,
            MemoryPermissions::READ_WRITE,
        )
        .unwrap();
    assert_eq!(
        space
            .translate(address, MemoryPermissions::READ)
            .unwrap()
            .allocation,
        replacement_id
    );
    let Access::Pending(pending) = machine
        .access_captured(
            &captured,
            AccessOperation::Write(Box::new([0x11, 0x22, 0x33, 0x44])),
        )
        .unwrap()
    else {
        panic!("old device input must enter its owner");
    };
    let request = receiver.recv().unwrap();
    assert_eq!(request.ranges()[0].allocation().identity(), old_id);
    let result = machine
        .apply_resolved_access(request.ranges(), request.operation())
        .unwrap();
    assert_eq!(
        machine
            .apply_resolved_access(request.ranges(), &AccessOperation::Read)
            .unwrap(),
        AccessResult::Read(Box::new([0x11, 0x22, 0x33, 0x44]))
    );
    request.finish(Ok(result));
    assert_eq!(pending.wait().unwrap(), AccessResult::Written);
    assert!(matches!(
        space.read_value(address, 4).unwrap(),
        Access::Ready(0)
    ));
    assert_eq!(machine.statistics().allocated_bytes, 2 * PAGE_SIZE);
    drop(captured);
    assert_eq!(machine.statistics().allocated_bytes, PAGE_SIZE);
    worker.finish().unwrap();
}

#[test]
fn concurrent_first_native_writers_invalidate_one_shared_physical_observation() {
    let machine = Machine::new(PAGE_SIZE).unwrap();
    let allocation = machine
        .account(PAGE_SIZE)
        .unwrap()
        .allocate(PAGE_SIZE)
        .unwrap();
    let ranges = [AllocationRange::new(allocation.pin(), 0, 4).unwrap()];
    let observation = nixe_memory::Observation::default();
    machine
        .access_captured(
            &ranges,
            nixe_memory::AccessOperation::ReadObserved(observation.clone()),
        )
        .unwrap()
        .into_ready()
        .unwrap();
    let spaces: Vec<_> = [PAGE_SIZE, 2 * PAGE_SIZE]
        .into_iter()
        .map(|base| {
            let space = AddressSpace::new(machine.clone(), AddressWidth::Bits32).unwrap();
            space
                .map(
                    GuestVirtualAddress::new(base),
                    allocation.clone(),
                    0,
                    PAGE_SIZE,
                    MemoryPermissions::READ_WRITE,
                )
                .unwrap();
            (space, GuestVirtualAddress::new(base))
        })
        .collect();
    let start = Arc::new(std::sync::Barrier::new(2));
    let writers: Vec<_> = spaces.into_iter().map(|(space, address)| {
        let start = Arc::clone(&start);
        std::thread::spawn(move || {
            let core = space.register_core(Arc::new(|| panic!("data writes must not halt native cores")));
            let admission = core.enter_native().unwrap();
            let arena = space.native_data_base().as_ptr() as usize;
            let mut call = Call {
                space: &space, admission: &admission, address,
                pointer: (arena + address.get() as usize) as *mut u8,
                captured: None, error: None,
            };
            let mut mxcsr = 0u32;
            unsafe { core::arch::asm!("stmxcsr [{}]", in(reg) &mut mxcsr, options(nostack, preserves_flags)) };
            let mut worker = NativeWorker::default();
            let opaque = std::ptr::from_mut(&mut call).cast();
            start.wait();
            let outcome = unsafe {
                worker.faults().unwrap().invoke_captured(
                    arena..arena + space.width().bytes() as usize,
                    [u64::from(mxcsr), 0], capture, opaque,
                    NativeInvocation { gateway, context: opaque, entry: 0 },
                )
            }.unwrap();
            assert!(call.error.is_none());
            let captured = call.captured.take();
            drop(call);
            admission.suspend();
            match outcome {
                InvocationOutcome::Escaped => {
                    machine_write(&space, &captured.unwrap());
                }
                InvocationOutcome::Returned => assert!(captured.is_none()),
            }
            worker.finish().unwrap();
        })
    }).collect();
    for writer in writers {
        writer.join().unwrap();
    }
    assert!(observation.is_invalid());
    let mut bytes = [0; 4];
    machine
        .read_captured_into(
            &nixe_memory::CapturedMemory::new(ranges).unwrap(),
            0,
            &mut bytes,
        )
        .unwrap()
        .into_ready()
        .unwrap();
    assert_eq!(bytes, [0x11, 0x22, 0x33, 0x44]);
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
