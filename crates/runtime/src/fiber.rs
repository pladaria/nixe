//! Guest host contexts switch through their physical-core scheduler context.
//! https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/common/fiber.cpp
use crate::{ExceptionDispatcher, GuestCpuState, ThreadObject};
use corosensei::{Coroutine, CoroutineResult, Yielder, stack::DefaultStack};
use nixe_scheduler::Completion;
use std::{cell::Cell, ptr, sync::Arc};

thread_local! {
    static CURRENT: Cell<*mut Control> = const { Cell::new(ptr::null_mut()) };
}
struct Control {
    yielder: *const Yielder<(), Completion>,
}
pub(crate) struct HostContext {
    coroutine: Coroutine<(), Completion, Completion>,
    control: Box<Control>,
}
// SAFETY: only the scheduler lease owner resumes a context. All suspended call
// stacks are emulator-owned Send data; physical-core/native borrows end before
// suspension. ExceptionDispatcher requires its implementations to preserve this
// stack contract, as required by corosensei::Coroutine.
// https://docs.rs/corosensei/0.3.4/corosensei/struct.Coroutine.html#send
unsafe impl Send for HostContext {}

impl HostContext {
    pub(crate) fn new<D: ExceptionDispatcher>(
        thread: Arc<ThreadObject>,
        dispatcher: Arc<D>,
    ) -> std::io::Result<Self> {
        let mut control = Box::new(Control {
            yielder: ptr::null(),
        });
        let pointer = ptr::from_mut(control.as_mut());
        let coroutine =
            Coroutine::with_stack(DefaultStack::new(512 * 4096)?, move |yielder, ()| {
                // The stable control box is owned beside this coroutine. The core
                // installs it in CURRENT before entry and on every later resume.
                unsafe { (*pointer).yielder = yielder };
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    crate::machine::core::run_guest(thread, dispatcher)
                }));
                match result {
                    Ok(result) => result,
                    Err(payload) if payload.is::<Cancelled>() => Completion::Ready,
                    Err(payload) => std::panic::resume_unwind(payload),
                }
            });
        Ok(Self { coroutine, control })
    }
    pub(crate) fn resume(&mut self) -> CoroutineResult<Completion, Completion> {
        struct Restore(*mut Control);
        impl Drop for Restore {
            fn drop(&mut self) {
                CURRENT.with(|current| current.set(self.0));
            }
        }
        let _restore = Restore(CURRENT.with(|current| current.replace(self.control.as_mut())));
        self.coroutine.resume(())
    }
}

struct Cancelled;
pub(crate) fn cancel_current() -> ! {
    std::panic::resume_unwind(Box::new(Cancelled))
}

/// Export the one register authority before yielding. The unavailable local
/// slot stays at its original stack address for any suspended SVC borrows.
pub(crate) fn suspend(
    thread: &Arc<ThreadObject>,
    state: &mut GuestCpuState,
    completion: Completion,
) {
    state.leave_core();
    thread.restore_state(state.take());
    let yielder = CURRENT.with(|current| {
        let control = current.get();
        assert!(
            !control.is_null(),
            "kernel suspension requires a guest fiber"
        );
        // SAFETY: resume owns the context and its control box until suspension.
        unsafe { (*control).yielder }
    });
    // SAFETY: the yielder belongs to the currently executing guest context. No
    // physical-core reference or scheduler lock spans this switch.
    unsafe { &*yielder }.suspend(completion);
    *state = thread
        .take_state()
        .expect("selected fiber owns saved registers");
}
