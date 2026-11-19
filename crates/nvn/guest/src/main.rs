//! Freestanding NVN procedures. The game calls these through its normal AAPCS64
//! function pointers; guest pointers and callbacks are used in its address space.
#![no_std]
#![no_main]

use arrayvec::ArrayVec;
use bytemuck::Pod;
use core::cell::Cell;
#[path = "../../src/abi.rs"]
pub mod abi;
use abi::*;
#[path = "../../src/layout.rs"]
pub mod layout;
use layout::Mip;
type MipList = ArrayVec<Mip, 32>;
mod buffer;
mod command;
mod device;
mod pool;
mod program;
mod queue;
mod sampler;
mod state;
mod texels;
mod texture;
mod vertex;
#[path = "../../src/view.rs"]
mod view;
core::arch::global_asm!(include_str!("memory.S"));

type Result<T> = core::result::Result<T, &'static str>;
struct Call {
    args: [u64; 8],
    float: [f32; 4],
    refill: Cell<Option<[u64; 4]>>,
}
impl Call {
    #[inline(always)]
    fn read<T: Pod>(&self, address: u64) -> Result<T> {
        Ok(unsafe { (address as *const T).read_unaligned() })
    }
    #[inline(always)]
    fn write<T: Pod>(&self, address: u64, value: &T) -> Result<()> {
        unsafe { (address as *mut T).write_unaligned(*value) };
        Ok(())
    }
    #[inline(always)]
    fn read_bytes(&self, address: u64, dst: &mut [u8]) -> Result<()> {
        if !dst.is_empty() {
            unsafe { core::ptr::copy(address as *const u8, dst.as_mut_ptr(), dst.len()) };
        }
        Ok(())
    }
    #[inline(always)]
    fn write_bytes(&self, address: u64, src: &[u8]) -> Result<()> {
        if !src.is_empty() {
            unsafe { core::ptr::copy(src.as_ptr(), address as *mut u8, src.len()) };
        }
        Ok(())
    }
}
mod debug {
    use super::*;
    // Debug labels are optional annotations on opaque objects.
    pub fn label(_: &Call) -> Result<u64> {
        Ok(0)
    }
}
core::arch::global_asm!(
    ".global nixe_nvn_error",
    ".type nixe_nvn_error,%function",
    "nixe_nvn_error:",
    "svc #0x4e56",
    "brk #0"
);
unsafe extern "C" {
    fn nixe_nvn_error(name: *const u8, n: usize, message: *const u8, m: usize) -> !;
}
#[cold]
fn fail(name: &str, message: &str) -> ! {
    unsafe { nixe_nvn_error(name.as_ptr(), name.len(), message.as_ptr(), message.len()) }
}
#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    fail("NVN", "guest procedure invariant failed")
}

macro_rules! exports {
    ($($family:ident { $($name:ident => $handler:ident),* $(,)? }),* $(,)?) => {
        $($(
            #[allow(non_snake_case, clippy::too_many_arguments)]
            #[unsafe(export_name = concat!("nvn", stringify!($name)))]
            pub extern "C" fn $name(
                x0: u64, x1: u64, x2: u64, x3: u64, x4: u64, x5: u64, x6: u64, x7: u64,
                s0: f32, s1: f32, s2: f32, s3: f32,
            ) -> u64 {
                let call = Call {
                    args: [x0, x1, x2, x3, x4, x5, x6, x7],
                    float: [s0, s1, s2, s3],
                    refill: Cell::new(None),
                };
                loop {
                    let result = $family::$handler(&call)
                        .unwrap_or_else(|reason| fail(concat!("nvn", stringify!($name)), reason));
                    let Some([target, event, bytes, data]) = call.refill.take() else {
                        return result;
                    };
                    let callback: unsafe extern "C" fn(u64, u64, u64, u64) =
                        unsafe { core::mem::transmute(target) };
                    unsafe { callback(x0, event, bytes, data) };
                }
            }
        )*)*
    }
}
include!("entries.rs");
