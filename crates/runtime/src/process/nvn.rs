//! Count calls made through the guest NVN bootstrap and device resolvers.
//! https://github.com/seedds/nxnative/blob/a366365/dm/gfx/nvn/hle_nvn.c
use super::{
    GuestCpuState, LaunchModuleImage, LaunchPlan, PreparedModule, ProcessMemory, RunnableProcess,
};
use crate::memory::{MappingError, MemoryKind};
use nixe_cpu::state::a64::{A64GeneralRegister, A64Register};
use nixe_memory::{Access, GuestVirtualAddress, MemoryPermissions};
use std::{collections::HashMap, sync::Mutex};

pub(crate) const RESOLVE_SVC: u32 = 0x4e56;
const BOOTSTRAP: &str = "nvnBootstrapLoader";
const DEVICE_RESOLVER: &str = "nvnDeviceGetProcAddress";
// Each unique (name, original address) gets an immutable two-instruction entry.
// Different targets for the same name share a counter. No code invalidation is needed.
const SLOTS: usize = 4096;
const ENTRIES: usize = 4096;
const DATA: usize = ENTRIES + SLOTS * 8;
const DESCRIPTORS: usize = DATA + 32;
const COUNTERS: usize = DESCRIPTORS + SLOTS * 32;
const END: usize = (COUNTERS + SLOTS * 16).next_multiple_of(4096);
const RESOLVE_PC: u64 = 0x64;

// AAPCS64 volatile temporaries only. Ordinary entries tail-call the original with
// the original SP, LR, flags, integer and vector arguments. Exclusive operations
// work on both consoles. Only a resolver return crosses into the host.
const COMMON: &[u32] = &[
    0xa9402a09, // ldp x9, x10, [x16]      target, per-name counters
    0xc8dffd4b, // ldar x11, [x10]         first invocation ordinal
    0xb500014b, // cbnz x11, increment
    0xf9400e0d, // ldr x13, [x16, #24]     sequence shared by all cores
    0xc85f7dab, // ticket: ldxr x11, [x13]
    0x9100056b, // add x11, x11, #1
    0xc80c7dab, // stxr w12, x11, [x13]
    0x35ffffac, // cbnz w12, ticket
    0xc85f7d4c, // publish: ldxr x12, [x10]
    0xb500006c, // cbnz x12, increment      another caller published first
    0xc80cfd4b, // stlxr w12, x11, [x10]
    0x35ffffac, // cbnz w12, publish
    0x9100214a, // increment: add x10, x10, #8
    0xc85f7d4b, // retry: ldxr x11, [x10]
    0x9100056b, // add x11, x11, #1
    0xc80c7d4b, // stxr w12, x11, [x10]
    0x35ffffac, // cbnz w12, retry
    0xb940120a, // ldr w10, [x16, #16]      resolver kind
    0x3400012a, // cbz w10, tail
    0xaa0003eb, // mov x11, x0              bootstrap takes name in x0
    0x3608004a, // tbz w10, #1, invoke
    0xaa0103eb, // mov x11, x1              device resolver takes name in x1
    0xa9bf7beb, // invoke: stp x11, x30, [sp, #-16]!
    0xd63f0120, // blr x9
    0xa8c17be1, // ldp x1, x30, [sp], #16
    0xd4000001 | RESOLVE_SVC << 5,
    0xd65f03c0, // ret
    0xd61f0120, // tail: br x9
];

#[derive(Default)]
struct Registry {
    names: HashMap<Box<str>, usize>,
    slots: HashMap<(usize, u64), usize>,
    incomplete: bool,
}

pub(crate) struct CallTrace {
    base: u64,
    bootstrap: Vec<u64>,
    registry: Mutex<Registry>,
}

impl CallTrace {
    pub fn prepare(plan: &LaunchPlan, modules: &mut [PreparedModule], base: u64) -> Self {
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
        }
    }

    pub fn end(&self) -> u64 {
        self.base
            + if self.bootstrap.is_empty() {
                0
            } else {
                END as u64
            }
    }

    pub fn install(&self, memory: &ProcessMemory) -> Result<(), MappingError> {
        if self.bootstrap.is_empty() {
            log::info!("NVN call counting: bootstrap export not found in launch modules");
            return Ok(());
        }
        let mut code = vec![0u8; DATA];
        for (word, dst) in COMMON.iter().zip(code.chunks_exact_mut(4)) {
            dst.copy_from_slice(&word.to_le_bytes());
        }
        for slot in 0..SLOTS {
            let pc = ENTRIES + slot * 8;
            let diff = (DESCRIPTORS + slot * 32 - pc) as u32;
            let adr_x16 = 0x10000010 | ((diff & 3) << 29) | ((diff >> 2) << 5);
            let branch_common =
                0x14000000 | ((0u32.wrapping_sub((pc as u32 + 4) / 4)) & 0x03ff_ffff);
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
            GuestVirtualAddress::new(self.base + DATA as u64),
            (END - DATA) as u64,
            super::builder::kernel_mapping(MemoryKind::Data, MemoryPermissions::READ_WRITE),
        )?;
        let mut registry = self.registry.lock().unwrap();
        for &original in &self.bootstrap {
            self.bind(memory, &mut registry, BOOTSTRAP, original)?;
        }
        log::info!(
            "NVN call counting active for {} bootstrap export(s)",
            self.bootstrap.len()
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
        // Some SDKs may return a pointer that already went through the dynamic
        // linker. Forward to the underlying guest function exactly once, including
        // when two API names are aliases for the same implementation.
        let target = self.original_target(reg, target);
        let name_id = reg.names.get(name).copied().unwrap_or(reg.names.len());
        let slot = if let Some(slot) = reg.slots.get(&(name_id, target)) {
            *slot
        } else {
            let slot = reg.slots.len();
            if slot == SLOTS {
                return Err(MappingError::InvalidRange);
            }
            let kind: u64 = match name {
                BOOTSTRAP => 1,
                DEVICE_RESOLVER => 2,
                _ => 0,
            };
            let words = [
                target,
                self.base + (COUNTERS + name_id * 16) as u64,
                kind,
                self.base + DATA as u64,
            ];
            let mut descriptor = [0u8; 32];
            for (dst, word) in descriptor.chunks_exact_mut(8).zip(words) {
                dst.copy_from_slice(&word.to_le_bytes());
            }
            memory
                .space()
                .write(
                    GuestVirtualAddress::new(self.base + (DESCRIPTORS + slot * 32) as u64),
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

    pub fn resolve(&self, pc: u64, state: &mut GuestCpuState, memory: &ProcessMemory) -> bool {
        if self.bootstrap.is_empty() || pc != self.base + RESOLVE_PC {
            return false;
        }
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
            } else if !reg.incomplete {
                log::warn!("NVN call trace is partial: could not register a resolved function");
                reg.incomplete = true;
            }
        }
        state.set_pc(pc + 4);
        true
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
    /// Snapshot after the vCPUs quiesce, while the diagnostic pages are still mapped.
    pub fn nvn_call_report(&self) -> Option<Result<NvnCallReport, MappingError>> {
        let trace = self.nvn.as_ref()?;
        let registry = trace.registry.lock().unwrap();
        let mut data = vec![0; registry.names.len() * 16];
        let read = self.memory.space().read_into(
            GuestVirtualAddress::new(trace.base + COUNTERS as u64),
            &mut data,
        );
        if !matches!(read, Ok(Access::Ready(()))) {
            return Some(Err(match read {
                Err(error) => error.into(),
                _ => MappingError::InvalidRange,
            }));
        }
        let mut rows = Vec::new();
        for (name, &index) in &registry.names {
            let word = |k| u64::from_le_bytes(data[k..k + 8].try_into().unwrap());
            let count = word(index * 16 + 8);
            if count != 0 {
                rows.push((word(index * 16), name.to_string(), count));
            }
        }
        rows.sort_unstable_by_key(|row| row.0);
        Some(Ok(NvnCallReport {
            bootstrap_found: !trace.bootstrap.is_empty(),
            complete: !registry.incomplete,
            resolved_entries: registry.slots.len(),
            calls: rows.into_iter().map(|(_, n, c)| (n, c)).collect(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linker_sees_private_memory_and_rebinding_a_wrapper_forwards_only_once() {
        let machine = nixe_memory::Machine::new(0x10_0000).unwrap();
        let account = machine.account(machine.arena().capacity()).unwrap();
        let memory = ProcessMemory::new(
            account.clone(),
            account,
            nixe_memory::AddressWidth::Bits32,
            crate::ProcessVirtualRegion::new(GuestVirtualAddress::new(0x100_0000), 0x40_0000),
            0,
        )
        .unwrap();
        let trace = CallTrace {
            base: 0x40_0000,
            bootstrap: vec![0x20_0000],
            registry: Mutex::default(),
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
}
