use crate::abi::{ExecutionFrame, NativeEntryAddress, NativeGateway};

unsafe extern "C" {
    fn nixe_fastmem_execute(
        gateway: NativeGateway,
        frame: *mut ExecutionFrame,
        entry: NativeEntryAddress,
        base: usize,
        size: usize,
        reported_fault: *mut usize,
    ) -> std::ffi::c_int;
}

pub(crate) unsafe fn execute(
    gateway: NativeGateway,
    frame: *mut ExecutionFrame,
    entry: NativeEntryAddress,
    base: usize,
    size: usize,
) -> Option<usize> {
    let mut address = 0;
    let trapped =
        unsafe { nixe_fastmem_execute(gateway, frame, entry, base, size, &raw mut address) };
    (trapped != 0).then_some(address)
}

#[cfg(test)]
mod tests {
    use super::*;

    unsafe extern "C" {
        fn nixe_fastmem_test_gateway(frame: *mut ExecutionFrame, entry: NativeEntryAddress);
    }

    #[test]
    fn protected_arena_fault_returns_through_the_native_boundary() {
        let size = 4096;
        let mapping = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert_ne!(mapping, libc::MAP_FAILED);
        let address = mapping.addr();
        let fault = unsafe {
            execute(
                nixe_fastmem_test_gateway,
                std::ptr::null_mut(),
                address,
                address,
                size,
            )
        };
        assert_eq!(fault, Some(address));
        assert_eq!(unsafe { libc::munmap(mapping, size) }, 0);
    }
}
