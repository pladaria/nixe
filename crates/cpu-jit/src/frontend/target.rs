//! Common native capabilities and ABI, with distinct compilation policies.

use super::fail;
use crate::abi::HostAbi;
use crate::jit_error::Error;
use cranelift_codegen::{
    isa::{self, TargetIsa},
    settings::{self, Configurable},
};
use std::sync::Arc;

#[derive(Clone, Copy)]
pub(crate) enum Policy {
    Lcq,
    Hcq,
}

pub(crate) fn build(abi: HostAbi, policy: Policy) -> Result<Arc<dyn TargetIsa>, Error> {
    let (optimization, allocator) = match policy {
        Policy::Lcq => ("none", "single_pass"),
        Policy::Hcq => ("speed", "backtracking"),
    };
    let mut flags = settings::builder();
    for (name, value) in [
        ("enable_pinned_reg", "true"),
        ("enable_nixe_abi", "true"),
        ("opt_level", optimization),
        ("regalloc_algorithm", allocator),
        ("machine_code_cfg_info", "true"),
        (
            "regalloc_checker",
            if cfg!(debug_assertions) {
                "true"
            } else {
                "false"
            },
        ),
    ] {
        flags.set(name, value).map_err(fail)?;
    }
    let triple = match abi {
        HostAbi::X86_64 => "x86_64-unknown-linux-gnu",
        HostAbi::Aarch64 => "aarch64-unknown-linux-gnu",
    };
    // Preserve actual host capabilities for both tiers: a baseline x86 target
    // can lower SIMD to libcalls, forbidden inside the frameless native ABI.
    let host = matches!(abi, HostAbi::X86_64) && cfg!(target_arch = "x86_64")
        || matches!(abi, HostAbi::Aarch64) && cfg!(target_arch = "aarch64");
    let mut target = if host {
        cranelift_native::builder().map_err(fail)?
    } else {
        isa::lookup(triple.parse().map_err(fail)?).map_err(fail)?
    };
    if abi == HostAbi::X86_64 {
        flags.set("enable_nixe_ibt", "true").map_err(fail)?;
    } else {
        target.set("use_bti", "true").map_err(fail)?;
    }
    target.finish(settings::Flags::new(flags)).map_err(fail)
}
