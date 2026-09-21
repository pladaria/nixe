use super::*;
use crate::executable::tests::permissions;

#[test]
fn write_window_closes_rw_on_finish_and_discards_it_on_unwind() {
    for unwind in [false, true] {
        let cache = Cache::new().unwrap();
        let allocation = cache.allocate(16, 16, Tier::Lcq).unwrap();
        let state = cache.lock().unwrap();
        let rw = state.backing.as_ref().unwrap().rw.as_ref().unwrap();
        let rw_address = rw.base.as_ptr() as usize;
        // Exercise the private permission-window guard on unpublished test
        // storage. The live-code API additionally requires a ClosedCode permit.
        rw.protect(0, SEGMENT_BYTES, libc::PROT_READ | libc::PROT_WRITE)
            .unwrap();
        assert_eq!(permissions(allocation.address()), "r-xs");
        assert_eq!(permissions(rw_address), "rw-s");
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut window = Window {
                state,
                offset: 0,
                bytes: SEGMENT_BYTES,
                finished: false,
            };
            if unwind {
                panic!("abandon executable write window");
            }
            window.finish().unwrap();
        }));
        assert_eq!(result.is_err(), unwind);
        if unwind {
            let state = cache
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            assert!(state.failed);
            assert!(state.backing.as_ref().unwrap().rw.is_none());
        } else {
            assert_eq!(permissions(rw_address), "---s");
            assert!(cache.usage().is_ok());
        }
        assert_eq!(permissions(allocation.address()), "r-xs");
    }
}
