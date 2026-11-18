//! Resolve guest NVN entry points for HLE and optional call counting.
//! https://github.com/seedds/nxnative/blob/a366365/dm/gfx/nvn/hle_nvn.c
use super::{
    GuestCpuState, LaunchModuleImage, LaunchPlan, PreparedModule, ProcessMemory, RunnableProcess,
};
use crate::memory::{MappingError, MemoryKind};
use nixe_cpu::platform::TargetPlatform;
use nixe_cpu::state::a64::{A64GeneralRegister, A64Register};
use nixe_memory::{GuestVirtualAddress, MemoryPermissions};
use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};

pub(crate) const NVN_SVC: u32 = 0x4e56;
const BOOTSTRAP: &str = "nvnBootstrapLoader";
const DEVICE_RESOLVER: &str = "nvnDeviceGetProcAddress";

// Each unique (name, original address) gets an immutable two-instruction entry.
// Different targets for the same name share a counter. No code invalidation is needed.
const SLOTS: usize = 4096;
const ENTRIES: usize = 4096;
const HOST_ENTRIES: usize = ENTRIES + SLOTS * 8;
const DESCRIPTORS: usize = HOST_ENTRIES + SLOTS * 8;
const DESCRIPTOR_BYTES: usize = 3 * 8; // target, counter address, resolver kind
const COUNTERS: usize = DESCRIPTORS + SLOTS * DESCRIPTOR_BYTES;
const END: usize = COUNTERS + SLOTS * 8;
const DISPATCH_PC: usize = COUNT_CALL.len() * 4;
const RESOLVE_PC: u64 = (DISPATCH_PC + 8 * 4) as u64;

// AAPCS64 volatile temporaries only. Ordinary entries tail-call the original with
// the original SP, LR, flags, integer and vector arguments. Exclusive operations
// work on both consoles. Resolvers and HLE entries share a private SVC.
const COUNT_CALL: &[u32] = &[
    0xa9402a09, // ldp x9, x10, [x16]      target, per-name counter
    0xc85f7d4b, // retry: ldxr x11, [x10]
    0x9100056b, // add x11, x11, #1
    0xc80c7d4b, // stxr w12, x11, [x10]
    0x35ffffac, // cbnz w12, retry
];
const DISPATCH: &[u32] = &[
    0xb940120a, // ldr w10, [x16, #16]      resolver kind
    0x3400012a, // cbz w10, tail
    0xaa0003eb, // mov x11, x0              bootstrap takes name in x0
    0x3608004a, // tbz w10, #1, invoke
    0xaa0103eb, // mov x11, x1              device resolver takes name in x1
    0xa9bf7beb, // invoke: stp x11, x30, [sp, #-16]!
    0xd63f0120, // blr x9
    0xa8c17be1, // ldp x1, x30, [sp], #16
    0xd4000001 | NVN_SVC << 5,
    0xd65f03c0, // ret
    0xd61f0120, // tail: br x9
];
// The guest returns through LR after the private SVC writes x0.
const CALL_HOST: &[u32] = &[0xd4000001 | NVN_SVC << 5, 0xd65f03c0]; // svc; ret

// Query the linked SDK's two version constants once, during resolution.
// These queries accept a null device. Preserve the replacement and resolver LR.
const QUERY_VERSION: usize = 128;
const VERSION_WORDS: &[u32] = &[
    0xa9be7be0, // stp x0, x30, [sp, #-32]!
    0xf9000be1, // str x1, [sp, #16]   original SDK query
    0xaa0103e9, // mov x9, x1
    0xaa1f03e0, // mov x0, xzr
    0x2a1f03e1, // mov w1, wzr        major
    0x910063e2, // add x2, sp, #24
    0xd63f0120, // blr x9
    0xf9400be9, // ldr x9, [sp, #16]
    0xaa1f03e0, // mov x0, xzr
    0x52800021, // mov w1, #1         minor
    0x910073e2, // add x2, sp, #28
    0xd63f0120, // blr x9
    0x294307e0, // ldp w0, w1, [sp, #24]
    0xd4000001 | NVN_SVC << 5,
    0xa8c27be0, // ldp x0, x30, [sp], #32
    0xd65f03c0, // ret
];
type Procedure = (Box<str>, Option<nixe_nvn::Handler>);

struct Hle {
    session: nixe_nvn::Session,
    title: u64,
    platform: TargetPlatform,
    procedures: Box<[OnceLock<Procedure>]>,
}
impl Hle {
    fn new(platform: TargetPlatform, title: u64) -> Self {
        Self {
            session: nixe_nvn::Session::new(platform),
            platform,
            title,
            procedures: (0..SLOTS).map(|_| OnceLock::new()).collect(),
        }
    }
}
#[derive(Default)]
struct Registry {
    names: HashMap<Box<str>, usize>,
    slots: HashMap<(usize, u64), usize>,
    incomplete: bool,
}

pub(crate) struct NvnHooks {
    base: u64,
    bootstrap: Vec<u64>,
    registry: Mutex<Registry>,
    count_calls: bool,
    hle: Option<Hle>,
}

#[derive(Debug)]
pub(crate) enum Svc {
    Unknown,
    Returned,
    Wait(nixe_nvn::PendingWait),
}

pub(crate) fn wait_on_guest(
    context: &mut crate::ExceptionDispatchContext<'_>,
    pending: nixe_nvn::PendingWait,
) -> Result<(), Box<str>> {
    use nixe_nvn::PendingWait;
    use nixe_scheduler::{WaitOutcome, WaitStart};
    let scheduler = context.process().scheduler();
    let (event, timeout) = match &pending {
        PendingWait::Acquire(d) => (None, Some(d.as_nanos() as u64)),
        PendingWait::Fence(f, ns) => (f.event(scheduler), (*ns != u64::MAX).then_some(*ns)),
    };
    let expired = if event.is_none() && matches!(&pending, PendingWait::Fence(..)) {
        false // GPU completed between the poll and registration of its event.
    } else {
        let start = scheduler
            .wait_on_objects(context.thread().id(), event.as_slice(), timeout, false)
            .map_err(|e| e.to_string())?;
        let outcome = match start {
            WaitStart::Ready(outcome) => outcome,
            WaitStart::Pending(wait) => {
                let owned = wait.clone();
                context
                    .wait_future(async move { owned.wait().await }, Some(wait))
                    .map_err(|e| e.to_string())?
            }
        };
        matches!(outcome, WaitOutcome::TimedOut)
    };
    let value = pending.complete(expired)?;
    context.thread_mut().state_mut().write_x(
        A64Register::General(A64GeneralRegister::new(0).unwrap()),
        value,
    );
    Ok(())
}

impl NvnHooks {
    pub fn prepare(
        plan: &LaunchPlan,
        modules: &mut [PreparedModule],
        base: u64,
        count_calls: bool,
        hle: bool,
    ) -> Self {
        let mut bootstrap = Vec::new();
        for (module, prepared) in plan.modules().iter().zip(modules) {
            if let LaunchModuleImage::Nso(nso) = module.image() {
                prepared.redirect_nso_export(nso.metadata(), BOOTSTRAP.as_bytes(), |original| {
                    let slot = bootstrap
                        .iter()
                        .position(|v| *v == original)
                        .unwrap_or_else(|| {
                            bootstrap.push(original);
                            bootstrap.len() - 1
                        });
                    log::debug!("NVN bootstrap export in module {}", module.name());
                    Some(base + (ENTRIES + slot * 8) as u64)
                });
            }
        }
        Self {
            base,
            bootstrap,
            registry: Mutex::default(),
            count_calls,
            hle: hle.then(|| {
                Hle::new(
                    plan.platform(),
                    plan.packaged_identity()
                        .map_or(0, |i| i.application_id().get()),
                )
            }),
        }
    }

    pub fn end(&self) -> u64 {
        self.base
            + if self.bootstrap.is_empty() {
                0
            } else if self.count_calls {
                END as u64
            } else {
                COUNTERS as u64
            }
    }

    pub fn install(&self, memory: &ProcessMemory) -> Result<(), MappingError> {
        if self.bootstrap.is_empty() {
            log::info!("NVN bootstrap export not found in launch modules");
            return Ok(());
        }
        let mut code = vec![0u8; DESCRIPTORS];
        let mut emit = |offset: usize, words: &[u32]| {
            for (word, dst) in words.iter().zip(code[offset..].chunks_exact_mut(4)) {
                dst.copy_from_slice(&word.to_le_bytes());
            }
        };
        let entry_pc = if self.count_calls {
            emit(0, COUNT_CALL);
            0
        } else {
            // Only the two resolvers need wrappers when call counting is off.
            emit(DISPATCH_PC - 4, &[0xf9400209]); // ldr x9, [x16]
            DISPATCH_PC - 4
        };
        emit(DISPATCH_PC, DISPATCH);
        if self.hle.is_some() {
            emit(QUERY_VERSION, VERSION_WORDS);
            for slot in 0..SLOTS {
                emit(HOST_ENTRIES + slot * 8, CALL_HOST);
            }
        }
        for slot in 0..SLOTS {
            let pc = ENTRIES + slot * 8;
            let diff = (DESCRIPTORS + slot * DESCRIPTOR_BYTES - pc) as u32;
            let adr_x16 = 0x10000010 | ((diff & 3) << 29) | ((diff >> 2) << 5);
            let branch_common =
                0x14000000 | (((entry_pc as i32 - pc as i32 - 4) / 4) as u32 & 0x03ff_ffff);
            code[pc..pc + 4].copy_from_slice(&adr_x16.to_le_bytes());
            code[pc + 4..pc + 8].copy_from_slice(&branch_common.to_le_bytes());
        }
        memory.map_initialized(
            GuestVirtualAddress::new(self.base),
            &code,
            // The guest rtld enumerates static code mappings as NSO modules. This
            // process-private trampoline has no module header and must not be enumerated.
            super::builder::kernel_mapping(MemoryKind::Data, MemoryPermissions::READ_EXECUTE),
        )?;
        memory.map_zeroed(
            GuestVirtualAddress::new(self.base + DESCRIPTORS as u64),
            self.end() - self.base - DESCRIPTORS as u64,
            super::builder::kernel_mapping(MemoryKind::Data, MemoryPermissions::READ_WRITE),
        )?;
        let mut registry = self.registry.lock().unwrap();
        for &original in &self.bootstrap {
            self.bind(memory, &mut registry, BOOTSTRAP, original)?;
        }
        log::info!(
            "NVN bootstrap exports={}, call counting={}, HLE={}",
            self.bootstrap.len(),
            self.count_calls,
            self.hle.is_some()
        );
        Ok(())
    }

    fn original_target(&self, registry: &Registry, address: u64) -> u64 {
        let first = self.base + ENTRIES as u64;
        let Some(offset) = address.checked_sub(first) else {
            return address;
        };
        if !offset.is_multiple_of(8) || offset / 8 >= registry.slots.len() as u64 {
            return address;
        }
        registry
            .slots
            .iter()
            .find_map(|(&(_, target), &slot)| (slot as u64 == offset / 8).then_some(target))
            .unwrap_or(address)
    }

    fn bind(
        &self,
        memory: &ProcessMemory,
        reg: &mut Registry,
        name: &str,
        target: u64,
    ) -> Result<u64, MappingError> {
        let kind: u64 = match name {
            BOOTSTRAP => 1,
            DEVICE_RESOLVER => 2,
            _ => 0,
        };
        let replace = self.hle.is_some() && kind == 0;
        if !self.count_calls && !replace && kind == 0 {
            return Ok(target);
        }
        // Some SDKs may return a pointer that already went through the dynamic
        // linker. Forward to the underlying guest function exactly once, including
        // when two API names are aliases for the same implementation.
        let target = self.original_target(reg, target);
        let name_id = reg.names.get(name).copied().unwrap_or(reg.names.len());
        let effective = if replace {
            let hle = self.hle.as_ref().unwrap();
            let entry = hle
                .procedures
                .get(name_id)
                .ok_or(MappingError::InvalidRange)?;
            entry.get_or_init(|| (name.into(), nixe_nvn::resolve(hle.platform, name)));
            self.base + (HOST_ENTRIES + name_id * 8) as u64
        } else {
            target
        };
        if replace && !self.count_calls {
            if name_id == reg.names.len() {
                reg.names.insert(name.into(), name_id);
            }
            return Ok(effective);
        }
        let slot = if let Some(slot) = reg.slots.get(&(name_id, target)) {
            *slot
        } else {
            let slot = reg.slots.len();
            if slot == SLOTS {
                return Err(MappingError::InvalidRange);
            }
            let words = [
                effective,
                if self.count_calls {
                    self.base + (COUNTERS + name_id * 8) as u64
                } else {
                    0
                },
                kind,
            ];
            let mut descriptor = [0u8; DESCRIPTOR_BYTES];
            for (dst, word) in descriptor.chunks_exact_mut(8).zip(words) {
                dst.copy_from_slice(&word.to_le_bytes());
            }
            memory
                .space()
                .write(
                    GuestVirtualAddress::new(
                        self.base + (DESCRIPTORS + slot * DESCRIPTOR_BYTES) as u64,
                    ),
                    &descriptor,
                )?
                .into_ready()
                .map_err(|_| MappingError::InvalidRange)?;
            if name_id == reg.names.len() {
                reg.names.insert(name.into(), name_id);
            }
            reg.slots.insert((name_id, target), slot);
            slot
        };
        Ok(self.base + (ENTRIES + slot * 8) as u64)
    }

    pub fn handle_svc(
        &self,
        pc: u64,
        state: &mut GuestCpuState,
        memory: &ProcessMemory,
    ) -> Result<Svc, Box<str>> {
        if self.bootstrap.is_empty() {
            return Ok(Svc::Unknown);
        }
        let x = |i| A64Register::General(A64GeneralRegister::new(i).unwrap());
        if pc == self.base + RESOLVE_PC {
            if self.resolve(state, memory)? {
                state.set_pc(self.base + QUERY_VERSION as u64);
                return Ok(Svc::Returned);
            }
        } else if let Some(hle) = &self.hle
            && pc == self.base + (QUERY_VERSION + 13 * 4) as u64
        {
            let version = [state.read_x(x(0)) as u32, state.read_x(x(1)) as u32];
            hle.session.api_version.get_or_init(|| version);
        } else if let Some(hle) = &self.hle
            && let Some(offset) = pc.checked_sub(self.base + HOST_ENTRIES as u64)
            && offset.is_multiple_of(8)
            && let Some((name, handler)) = hle
                .procedures
                .get((offset / 8) as usize)
                .and_then(OnceLock::get)
        {
            let caller = state.read_x(x(30));
            let invoke = || {
                let handler =
                    handler.ok_or_else(|| Box::<str>::from("function is not implemented"))?;
                let call = nixe_nvn::Call {
                    session: &hle.session,
                    memory: memory.space(),
                    args: std::array::from_fn(|i| state.read_x(x(i as u8))),
                    pending: Default::default(),
                };
                let value = handler(&call)?;
                Ok::<_, Box<str>>((value, call.pending.take()))
            };
            let (result, pending) = invoke().map_err(|e| {
                format!(
                    "NVN HLE {name} (title={:016X}, caller={caller:#x}): {e}",
                    hle.title
                )
            })?;
            state.write_x(x(0), result);
            if let Some(wait) = pending {
                state.set_pc(pc + 4);
                return Ok(Svc::Wait(wait));
            }
        } else {
            return Ok(Svc::Unknown);
        }
        state.set_pc(pc + 4);
        Ok(Svc::Returned)
    }

    fn resolve(&self, state: &mut GuestCpuState, memory: &ProcessMemory) -> Result<bool, Box<str>> {
        let x = |i| A64Register::General(A64GeneralRegister::new(i).unwrap());
        let target = state.read_x(x(0));
        if target != 0 {
            let name = read_name(memory, state.read_x(x(1)));
            let mut reg = self.registry.lock().unwrap();
            let wrapped = name
                .as_deref()
                .and_then(|name| self.bind(memory, &mut reg, name, target).ok());
            if let Some(wrapped) = wrapped {
                state.write_x(x(0), wrapped);
                if name.as_deref() == Some("nvnDeviceGetInteger")
                    && self.hle.as_ref().is_some_and(|h| {
                        h.platform == TargetPlatform::Switch1
                            && h.session.api_version.get().is_none()
                    })
                {
                    state.write_x(x(1), self.original_target(&reg, target));
                    return Ok(true);
                }
            } else if self.hle.is_some() {
                return Err(format!(
                    "NVN HLE could not intercept resolved procedure at {target:#x}, caller={:#x}",
                    state.read_x(x(30))
                )
                .into());
            } else if !reg.incomplete {
                log::warn!("Could not intercept a resolved NVN function; using the SDK entry");
                reg.incomplete = true;
            }
        }
        Ok(false)
    }
}

fn read_name(memory: &ProcessMemory, mut address: u64) -> Option<String> {
    let mut bytes = Vec::new();
    let mut buf = [0u8; 64];
    while bytes.len() < 1024 {
        let n = buf.len().min(4096 - (address % 4096) as usize);
        memory
            .space()
            .read_into(GuestVirtualAddress::new(address), &mut buf[..n])
            .ok()?
            .into_ready()
            .ok()?;
        if let Some(end) = buf[..n].iter().position(|&b| b == 0) {
            bytes.extend_from_slice(&buf[..end]);
            return String::from_utf8(bytes).ok();
        }
        bytes.extend_from_slice(&buf[..n]);
        address = address.checked_add(n as u64)?;
    }
    None
}

#[derive(Debug, Default)]
pub struct NvnCallReport {
    pub bootstrap_found: bool,
    pub complete: bool,
    pub resolved_entries: usize,
    pub calls: Vec<(String, u64)>,
}

impl RunnableProcess {
    pub fn nvn_session(&self) -> Option<&nixe_nvn::Session> {
        self.nvn.as_ref()?.hle.as_ref().map(|h| &h.session)
    }
    /// Snapshot after the vCPUs quiesce, while the diagnostic pages are still mapped.
    pub fn nvn_call_report(&self) -> Option<Result<NvnCallReport, MappingError>> {
        let trace = self.nvn.as_ref().filter(|hooks| hooks.count_calls)?;
        Some(trace.report(&self.memory))
    }
}

impl NvnHooks {
    fn report(&self, memory: &ProcessMemory) -> Result<NvnCallReport, MappingError> {
        let registry = self.registry.lock().unwrap();
        let mut data = vec![0; registry.names.len() * 8];
        if !data.is_empty() {
            memory
                .space()
                .read_into(
                    GuestVirtualAddress::new(self.base + COUNTERS as u64),
                    &mut data,
                )?
                .into_ready()
                .map_err(|_| MappingError::InvalidRange)?;
        }
        let mut calls = Vec::new();
        for (name, &index) in &registry.names {
            let offset = index * 8;
            let count = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
            if count != 0 {
                calls.push((name.to_string(), count));
            }
        }
        calls.sort_unstable_by(|(a_name, a_count), (b_name, b_count)| {
            b_count.cmp(a_count).then_with(|| a_name.cmp(b_name))
        });
        Ok(NvnCallReport {
            bootstrap_found: !self.bootstrap.is_empty(),
            complete: !registry.incomplete,
            resolved_entries: registry.slots.len(),
            calls,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn process_memory() -> ProcessMemory {
        let machine = nixe_memory::Machine::new(0x10_0000).unwrap();
        let account = machine.account(machine.arena().capacity()).unwrap();
        ProcessMemory::new(
            account.clone(),
            account,
            nixe_memory::AddressWidth::Bits32,
            crate::ProcessVirtualRegion::new(GuestVirtualAddress::new(0x100_0000), 0x40_0000),
            0,
        )
        .unwrap()
    }

    #[test]
    fn linker_sees_private_memory_and_rebinding_a_wrapper_forwards_only_once() {
        let memory = process_memory();
        let trace = NvnHooks {
            base: 0x40_0000,
            bootstrap: vec![0x20_0000],
            registry: Mutex::default(),
            count_calls: true,
            hle: None,
        };
        trace.install(&memory).unwrap();
        let code = memory
            .query(GuestVirtualAddress::new(trace.base))
            .unwrap()
            .mapping
            .unwrap();
        assert_eq!(code.kind, MemoryKind::Data);
        assert_eq!(code.permissions, MemoryPermissions::READ_EXECUTE);
        let mut registry = trace.registry.lock().unwrap();
        let first = trace
            .bind(&memory, &mut registry, "nvnQueueSubmitCommands", 0x21_0000)
            .unwrap();
        assert_eq!(
            trace
                .bind(&memory, &mut registry, "nvnQueueSubmitCommands", first)
                .unwrap(),
            first
        );
        let other_device = trace
            .bind(&memory, &mut registry, "nvnQueueSubmitCommands", 0x22_0000)
            .unwrap();
        assert_ne!(first, other_device);
        assert_eq!(registry.slots.len(), 3);
        assert_eq!(registry.names.len(), 2);
    }

    #[test]
    fn report_lists_only_invoked_functions_by_descending_frequency() {
        let memory = process_memory();
        let hooks = NvnHooks {
            base: 0x40_0000,
            bootstrap: vec![0x20_0000],
            registry: Mutex::default(),
            count_calls: true,
            hle: None,
        };
        hooks.install(&memory).unwrap();
        let mut reg = hooks.registry.lock().unwrap();
        for (name, count) in [("nvnZ", 1_u64), ("nvnA", 1), ("nvnB", u64::MAX)] {
            hooks.bind(&memory, &mut reg, name, 0x21_0000).unwrap();
            let addr = hooks.base + (COUNTERS + reg.names[name] * 8) as u64;
            memory
                .space()
                .write(GuestVirtualAddress::new(addr), &count.to_le_bytes())
                .unwrap()
                .into_ready()
                .unwrap();
        }
        drop(reg);
        assert_eq!(
            hooks.report(&memory).unwrap().calls,
            [
                ("nvnB".into(), u64::MAX),
                ("nvnA".into(), 1),
                ("nvnZ".into(), 1)
            ]
        );
    }

    #[test]
    fn unsupported_procedures_resolve_and_fail_only_when_invoked() {
        for count_calls in [false, true] {
            let mem = process_memory();
            let hooks = NvnHooks {
                base: 0x400000,
                bootstrap: vec![0x200000],
                registry: Mutex::default(),
                count_calls,
                hle: Some(Hle::new(TargetPlatform::Switch1, 42)),
            };
            hooks.install(&mem).unwrap();
            let mut reg = hooks.registry.lock().unwrap();
            hooks
                .bind(&mem, &mut reg, "nvnUnimplemented", 0x210000)
                .unwrap();
            let index = reg.names["nvnUnimplemented"];
            drop(reg);
            let mut cpu = GuestCpuState::default();
            let error = hooks
                .handle_svc(
                    hooks.base + (HOST_ENTRIES + index * 8) as u64,
                    &mut cpu,
                    &mem,
                )
                .unwrap_err();
            assert!(error.contains("nvnUnimplemented") && error.contains("000000000000002A"));
        }
    }
}
