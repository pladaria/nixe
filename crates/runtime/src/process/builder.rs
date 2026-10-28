use super::*;

/// Builds an emulated process from a prepared launch plan.
pub struct ProcessBuilder {
    config: ProcessBuildConfig,
    virtual_clock: crate::VirtualClock,
    sd_card_root: Option<PathBuf>,
    machine: nixe_memory::Machine,
    system: nixe_memory::MemoryAccount,
}

impl std::fmt::Debug for ProcessBuilder {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ProcessBuilder")
            .field("config", &self.config)
            .field("virtual_clock", &self.virtual_clock)
            .field("sd_card_root", &self.sd_card_root)
            .field("machine", &self.machine.identity())
            .finish()
    }
}

impl ProcessBuilder {
    /// Creates a process builder using Switch 1 defaults.
    ///
    #[must_use]
    pub fn new(machine: nixe_memory::Machine, system: nixe_memory::MemoryAccount) -> Self {
        Self {
            config: ProcessBuildConfig::default(),
            virtual_clock: crate::VirtualClock::default(),
            sd_card_root: None,
            machine,
            system,
        }
    }

    #[must_use]
    pub const fn with_config(mut self, config: ProcessBuildConfig) -> Self {
        self.config = config;
        self
    }

    /// Selects the clock source shared by CPU architectural timers and services.
    #[must_use]
    pub fn with_virtual_clock(mut self, virtual_clock: crate::VirtualClock) -> Self {
        self.virtual_clock = virtual_clock;
        self
    }

    /// Selects the canonical host directory exposed to the guest as `sdmc:`.
    #[must_use]
    pub fn with_sd_card_root(mut self, root: PathBuf) -> Self {
        self.sd_card_root = Some(root);
        self
    }

    /// Prepares, maps, and initializes one runnable process.
    ///
    /// Packaged NSOs retain their dynamic relocations for the guest `rtld`.
    /// Standalone NROs likewise enter through their guest startup ABI.
    pub fn build(&self, plan: &LaunchPlan) -> Result<RunnableProcess, ProcessBuildError> {
        if self.config.architectural_timer_frequency == 0 {
            return Err(ProcessBuildError::new(
                ProcessBuildStage::Metadata,
                "architectural timer frequency must be nonzero",
            ));
        }
        let metadata = process_metadata(plan);
        let address_space = metadata.address_space;
        let stack_size = metadata.stack_size;
        let abi = metadata.abi;
        let random_entropy = generate_process_entropy()?;
        let placements = module_placements(plan, self.config.image_base, address_space)?;
        let modules = prepare_modules(plan, &placements, address_space)?;
        let process_code_start = modules
            .iter()
            .map(PreparedModule::image_base)
            .min()
            .ok_or_else(|| error(ProcessBuildStage::Placement, "launch plan has no modules"))?;
        let process_code_end = modules
            .iter()
            .map(|module| module.image_base().saturating_add(module.image_extent()))
            .max()
            .ok_or_else(|| error(ProcessBuildStage::Placement, "launch plan has no modules"))?;
        let memory_layout = ProcessMemoryLayout::for_address_space(
            self.config.memory_layout_profile,
            address_space,
            process_code_start,
            process_code_end,
            self.config.physical_memory_limit,
        )?;
        let entry_module = plan.entry_module_index();
        // ACI0 is the process-requested policy after ACID authorization has
        // already been checked by the NPDM loader. Horizon uses its
        // HandleTableSize descriptor as the live per-process handle limit:
        // https://switchbrew.org/w/index.php?title=NPDM&oldid=14486#HandleTableSize
        let handles = plan
            .effective_policy()
            .and_then(|policy| policy.handle_table_size())
            .map_or_else(crate::HandleTable::new, |size| {
                crate::HandleTable::with_capacity_limit(usize::from(size))
            });
        let main_thread_object = Arc::new(crate::ThreadObject::new(1));
        let main_thread_handle = handles
            .insert_object(crate::HandleObject::from_arc(main_thread_object.clone()))
            .map_err(|error| {
                ProcessBuildError::new(ProcessBuildStage::ThreadInitialization, error)
            })?;

        let user_limit = self
            .config
            .physical_memory_limit
            .checked_sub(metadata.system_resource_size)
            .ok_or_else(|| {
                error(
                    ProcessBuildStage::Mapping,
                    "system resource reservation exceeds process RAM budget",
                )
            })?;
        let account = self
            .machine
            .account(user_limit)
            .map_err(|error| ProcessBuildError::new(ProcessBuildStage::Mapping, error))?;
        let memory = crate::memory::ProcessMemory::new(
            account,
            self.system.clone(),
            address_space.width(),
            memory_layout.heap(),
            metadata.system_resource_size,
        )
        .map_err(|error| ProcessBuildError::new(ProcessBuildStage::Mapping, error))?;
        let cpu =
            ProcessCpuContext::for_platform(self.config.target_platform, memory.space().identity());
        for module in &modules {
            crate::install_prepared_module(
                &memory,
                module,
                matches!(abi, InitialProcessAbi::Homebrew),
            )
            .map_err(|error| ProcessBuildError::new(ProcessBuildStage::Mapping, error))?;
        }

        let stack_size = align_up(stack_size.max(PAGE_SIZE), PAGE_SIZE)?;
        if stack_size + (RESOURCE_GUARD_SIZE * 3) + (PAGE_SIZE * 2) > memory_layout.stack().size() {
            return Err(error(
                ProcessBuildStage::Placement,
                "main-thread resources exceed the reserved stack region",
            ));
        }
        let stack_bottom = memory_layout.stack().base();
        let stack_top = stack_bottom
            .checked_add(stack_size)
            .ok_or_else(|| error(ProcessBuildStage::Placement, "main stack overflows"))?;
        let local_page = stack_top
            .checked_add(RESOURCE_GUARD_SIZE)
            .ok_or_else(|| error(ProcessBuildStage::Placement, "TLS base overflows"))?;
        validate_range(address_space, local_page.get(), PAGE_SIZE)?;
        memory
            .map_zeroed(
                stack_bottom,
                stack_size,
                kernel_mapping(
                    crate::memory::MemoryKind::Stack,
                    MemoryPermissions::READ_WRITE,
                ),
            )
            .map_err(|error| ProcessBuildError::new(ProcessBuildStage::Mapping, error))?;
        memory
            .map_zeroed(
                local_page,
                PAGE_SIZE,
                kernel_mapping(
                    crate::memory::MemoryKind::ThreadLocal,
                    MemoryPermissions::READ_WRITE,
                ),
            )
            .map_err(|error| ProcessBuildError::new(ProcessBuildStage::Mapping, error))?;
        // Reserve the first 0x200-byte region for the process before allocating
        // the main thread's TLS, as KProcess::Initialize/CreateThreadLocalRegion.
        // Both regions share the same kernel-owned page and allocation index.
        // https://github.com/eden-emulator/mirror/blob/master/src/core/hle/kernel/k_process.cpp
        let process_local_region = local_page;
        let mut thread_local = thread_local::ThreadLocalPages::new(
            local_page,
            memory_layout.stack().base(),
            memory_layout
                .stack()
                .base()
                .checked_add(memory_layout.stack().size())
                .expect("validated stack region has a finite end"),
        );
        let tls_base = thread_local
            .allocate(&memory)
            .map_err(|error| ProcessBuildError::new(ProcessBuildStage::Mapping, error))?;
        let (abi_context, loader_return) = if matches!(abi, InitialProcessAbi::Homebrew) {
            let address = local_page
                .checked_add(PAGE_SIZE + RESOURCE_GUARD_SIZE)
                .ok_or_else(|| error(ProcessBuildStage::Placement, "ABI context overflows"))?;
            validate_range(address_space, address.get(), nixe_memory::PAGE_SIZE)?;
            install_homebrew_context(
                &memory,
                address,
                main_thread_handle,
                plan.homebrew_identity().ok_or_else(|| {
                    error(
                        ProcessBuildStage::Metadata,
                        "homebrew launch has no retained executable identity",
                    )
                })?,
            )?;
            let loader_return = address
                .checked_add(nixe_memory::PAGE_SIZE + RESOURCE_GUARD_SIZE)
                .ok_or_else(|| error(ProcessBuildStage::Placement, "loader return overflows"))?;
            validate_range(address_space, loader_return.get(), nixe_memory::PAGE_SIZE)?;
            install_homebrew_loader_return(&memory, loader_return)?;
            (Some(address), Some(loader_return))
        } else {
            (None, None)
        };

        let entry = GuestVirtualAddress::new(modules[entry_module].entry_address());
        let mut state = GuestCpuState::default();
        initialize_thread(
            &mut state,
            entry,
            stack_top,
            tls_base,
            main_thread_handle,
            abi_context,
            loader_return,
        )?;
        let main_thread = crate::GuestThread {
            // The coordinator replaces this process-local provisional identity
            // with a runtime-global identity before scheduler publication.
            id: nixe_scheduler::GuestThreadId::new(1),
            object: Arc::downgrade(&main_thread_object),
            running_reference: Some(main_thread_object),
            exit: None,
            state: Some(state),
            handle: main_thread_handle,
            stack_bottom,
            stack_top,
            tls_base,
            abi_context,
            loader_return,
        };
        let main_thread_id = main_thread.id();
        let mut threads = crate::ThreadTable::new();
        threads.insert(main_thread).map_err(|error| {
            ProcessBuildError::new(ProcessBuildStage::ThreadInitialization, error)
        })?;
        let cpu_process_id = execution::allocate_cpu_process_id().ok_or_else(|| {
            ProcessBuildError::new(
                ProcessBuildStage::CpuInitialization,
                "CPU process identity exhausted",
            )
        })?;
        let memory = std::sync::Arc::new(memory);
        let execution = execution::ProcessExecutionControl::new(
            execution::ProcessExecutionConfiguration {
                virtual_clock: self.virtual_clock.clone(),
                timer_frequency: self.config.architectural_timer_frequency,
                cpu,
            },
            std::sync::Arc::clone(memory.space()),
            cpu_process_id,
        )
        .map_err(|error| ProcessBuildError::new(ProcessBuildStage::CpuInitialization, error))?;
        let process = RunnableProcess {
            scheduler: std::sync::OnceLock::new(),
            process_id: self.config.process_id,
            lifecycle: nixe_scheduler::ProcessLifecycle::Running,
            process_exit: None,
            num_running_threads: 1,
            cpu,
            address_space,
            memory_layout,
            random_entropy,
            memory,
            modules: modules.into_boxed_slice(),
            entry_module,
            main_thread_id,
            initial_thread_priority: metadata.priority,
            initial_ideal_vcpu: metadata.ideal_vcpu,
            thread_policy: metadata.thread_policy,
            thread_local,
            process_local_region,
            threads,
            mounts: std::sync::Arc::new(crate::ProcessMountNamespace::from_launch_plan(
                plan,
                self.sd_card_root.clone(),
            )),
            handles: std::sync::Arc::new(handles),
            address_waits: std::sync::Arc::default(),
            execution,
        };
        Ok(process)
    }
}

fn generate_process_entropy() -> Result<[u64; 4], ProcessBuildError> {
    let mut bytes = [0_u8; size_of::<[u64; 4]>()];
    getrandom::fill(&mut bytes).map_err(|error| {
        ProcessBuildError::new(
            ProcessBuildStage::Metadata,
            format_args!("cannot obtain host entropy for the guest process: {error}"),
        )
    })?;
    Ok(std::array::from_fn(|index| {
        let offset = index * size_of::<u64>();
        u64::from_le_bytes(bytes[offset..offset + size_of::<u64>()].try_into().unwrap())
    }))
}

#[derive(Clone, Copy)]
enum InitialProcessAbi {
    Packaged,
    Homebrew,
}

#[derive(Clone, Copy)]
struct ProcessMetadata {
    address_space: ProcessAddressSpace,
    stack_size: u64,
    abi: InitialProcessAbi,
    priority: i32,
    ideal_vcpu: nixe_scheduler::VirtualCpuId,
    thread_policy: Option<ThreadPolicy>,
    system_resource_size: u64,
}

#[derive(Clone, Copy)]
pub(crate) struct ThreadPolicy {
    pub(super) highest_priority: i32,
    pub(super) lowest_priority: i32,
    pub(super) min_core: u32,
    pub(super) max_core: u32,
}

impl ThreadPolicy {
    /// Core permissions loaded from the executable's effective NPDM policy.
    #[must_use]
    pub fn allowed_core_mask(&self) -> Option<u64> {
        let policy = self;
        let low = 1_u64.checked_shl(policy.min_core)?.wrapping_sub(1);
        let high = if policy.max_core == 63 {
            u64::MAX
        } else {
            1_u64.checked_shl(policy.max_core + 1)?.wrapping_sub(1)
        };
        Some(high & !low)
    }

    /// Priority permissions shared with thread creation and policy validation.
    #[must_use]
    pub fn allowed_priority_mask(&self) -> Option<u64> {
        let policy = self;
        let low = 1_u64
            .checked_shl(policy.highest_priority as u32)?
            .wrapping_sub(1);
        let high = if policy.lowest_priority == 63 {
            u64::MAX
        } else {
            1_u64
                .checked_shl(policy.lowest_priority as u32 + 1)?
                .wrapping_sub(1)
        };
        Some(high & !low)
    }
}

fn process_metadata(plan: &LaunchPlan) -> ProcessMetadata {
    match plan.kind() {
        LaunchKind::Packaged(identity) => {
            let npdm = identity.npdm();
            ProcessMetadata {
                address_space: ProcessAddressSpace::from_npdm(npdm.flags().address_space()),
                stack_size: u64::from(npdm.main_thread_stack_size()),
                abi: InitialProcessAbi::Packaged,
                priority: i32::from(npdm.main_thread_priority()),
                ideal_vcpu: nixe_scheduler::VirtualCpuId::new(u32::from(npdm.default_cpu_core())),
                thread_policy: npdm
                    .effective_policy()
                    .kernel_capabilities()
                    .thread_info()
                    .map(|(lowest, highest, min_core, max_core)| ThreadPolicy {
                        highest_priority: i32::from(highest),
                        lowest_priority: i32::from(lowest),
                        min_core: u32::from(min_core),
                        max_core: u32::from(max_core),
                    }),
                system_resource_size: u64::from(npdm.system_resource_size()),
            }
        }
        LaunchKind::Homebrew(_) => ProcessMetadata {
            address_space: ProcessAddressSpace::Bit64,
            stack_size: DEFAULT_HOME_BREW_STACK_SIZE,
            abi: InitialProcessAbi::Homebrew,
            priority: 44,
            ideal_vcpu: nixe_scheduler::VirtualCpuId::new(0),
            thread_policy: None,
            system_resource_size: 0,
        },
    }
}

fn module_placements(
    plan: &LaunchPlan,
    first_base: GuestVirtualAddress,
    address_space: ProcessAddressSpace,
) -> Result<Vec<PreparationConfig>, ProcessBuildError> {
    let limit = address_space.exclusive_limit();
    let mut next = align_up(first_base.get(), nixe_memory::PAGE_SIZE)?;
    let mut placements = Vec::with_capacity(plan.modules().len());
    for module in plan.modules() {
        let extent = image_extent(module.image())?;
        validate_range(address_space, next, extent)?;
        placements.push(PreparationConfig {
            image_base: next,
            address_limit: limit,
        });
        next = align_up(
            next.checked_add(extent)
                .and_then(|end| end.checked_add(MODULE_GUARD_SIZE))
                .ok_or_else(|| error(ProcessBuildStage::Placement, "module placement overflows"))?,
            nixe_memory::PAGE_SIZE,
        )?;
    }
    Ok(placements)
}

fn image_extent(image: &LaunchModuleImage) -> Result<u64, ProcessBuildError> {
    let executable = match image {
        LaunchModuleImage::Nso(image) => image.executable(),
        LaunchModuleImage::Nro(image) => image.executable(),
    };
    executable
        .segments()
        .iter()
        .map(|segment| segment.memory_offset().checked_add(segment.mapping_size()))
        .try_fold(0_u64, |extent, end| {
            Ok(extent.max(
                end.ok_or_else(|| error(ProcessBuildStage::Placement, "module extent overflows"))?,
            ))
        })
}

fn prepare_modules(
    plan: &LaunchPlan,
    placements: &[PreparationConfig],
    address_space: ProcessAddressSpace,
) -> Result<Vec<PreparedModule>, ProcessBuildError> {
    let unresolved = |_: ExternalSymbol<'_>| SymbolResolution::Unresolved;
    plan.modules()
        .iter()
        .zip(placements)
        .map(|(module, config)| {
            let prepared = match module.image() {
                LaunchModuleImage::Nso(image) => {
                    image.prepare_for_guest_relocation(*config, &unresolved)
                }
                LaunchModuleImage::Nro(image) => {
                    image.prepare_for_guest_relocation(*config, &unresolved)
                }
            }
            .map_err(|error| ProcessBuildError::new(ProcessBuildStage::Preparation, error))?;
            validate_range(
                address_space,
                prepared.image_base(),
                prepared.image_extent(),
            )?;
            Ok(prepared)
        })
        .collect()
}

fn kernel_mapping(
    kind: crate::memory::MemoryKind,
    permissions: MemoryPermissions,
) -> crate::memory::Mapping {
    crate::memory::Mapping {
        kind,
        permissions,
        uncached: false,
        locked: false,
        permission_locked: false,
    }
}

fn install_homebrew_context(
    memory: &crate::memory::ProcessMemory,
    address: GuestVirtualAddress,
    main_thread_handle: u32,
    identity: &crate::HomebrewIdentity,
) -> Result<(), ProcessBuildError> {
    let mut page = [0_u8; nixe_memory::PAGE_SIZE as usize];
    page[..4].copy_from_slice(&HOME_BREW_MAIN_THREAD_HANDLE_KEY.to_le_bytes());
    page[8..16].copy_from_slice(&u64::from(main_thread_handle).to_le_bytes());
    let argv_entry = HOME_BREW_CONFIG_ENTRY_SIZE;
    page[argv_entry..argv_entry + 4].copy_from_slice(&HOME_BREW_ARGV_KEY.to_le_bytes());
    let argv_address = address
        .checked_add(HOME_BREW_ARGV_OFFSET as u64)
        .ok_or_else(|| {
            error(
                ProcessBuildStage::Mapping,
                "homebrew argv address overflows",
            )
        })?;
    page[argv_entry + 16..argv_entry + 24].copy_from_slice(&argv_address.get().to_le_bytes());
    let argv = identity.argv0().as_bytes();
    let argv_end = HOME_BREW_ARGV_OFFSET
        .checked_add(argv.len())
        .and_then(|end| end.checked_add(1))
        .filter(|end| *end <= page.len())
        .ok_or_else(|| {
            error(
                ProcessBuildStage::Metadata,
                "homebrew argv exceeds ABI page",
            )
        })?;
    page[HOME_BREW_ARGV_OFFSET..argv_end - 1].copy_from_slice(argv);
    // The zeroed third 24-byte entry is EntryType_EndOfList and the argv
    // string is NUL-terminated by the zero-initialized ABI page.
    memory
        .map_initialized(
            address,
            &page,
            kernel_mapping(crate::memory::MemoryKind::Data, MemoryPermissions::READ),
        )
        .map_err(|error| ProcessBuildError::new(ProcessBuildStage::Mapping, error))
}

fn install_homebrew_loader_return(
    memory: &crate::memory::ProcessMemory,
    address: GuestVirtualAddress,
) -> Result<(), ProcessBuildError> {
    let mut page = [0_u8; nixe_memory::PAGE_SIZE as usize];
    // Execute a normal ExitProcess SVC on return. Runtime completion recognizes
    // this thread's stub source and retains X0 as the loader result; CPU backends
    // and native links need no special return-address interception.
    page[..4].copy_from_slice(&HOME_BREW_EXIT_PROCESS_INSTRUCTION.to_le_bytes());
    memory
        .map_initialized(
            address,
            &page,
            kernel_mapping(
                crate::memory::MemoryKind::Code,
                MemoryPermissions::READ_EXECUTE,
            ),
        )
        .map_err(|error| ProcessBuildError::new(ProcessBuildStage::Mapping, error))
}

pub(super) fn initialize_thread(
    state: &mut GuestCpuState,
    entry: GuestVirtualAddress,
    stack_top: GuestVirtualAddress,
    tls_base: GuestVirtualAddress,
    main_thread_handle: u32,
    abi_context: Option<GuestVirtualAddress>,
    loader_return: Option<GuestVirtualAddress>,
) -> Result<(), ProcessBuildError> {
    state.set_pc(entry.get());
    state.write_x(A64Register::StackPointer, stack_top.get());
    state.set_tpidr_el0(tls_base.get());
    state.set_tpidrro_el0_from_runtime(tls_base.get());
    state.write_x(
        A64Register::General(a64_register(0)),
        abi_context.map_or(0, GuestVirtualAddress::get),
    );
    state.write_x(
        A64Register::General(a64_register(1)),
        if abi_context.is_some() {
            u64::MAX
        } else {
            u64::from(main_thread_handle)
        },
    );
    state.write_x(
        A64Register::General(a64_register(30)),
        loader_return.map_or(0, GuestVirtualAddress::get),
    );
    Ok(())
}

pub(super) fn initialize_created_thread(
    state: &mut GuestCpuState,
    request: &ThreadCreateRequest,
    tls_base: GuestVirtualAddress,
) -> Result<(), ProcessBuildError> {
    state.set_pc(request.entry.get());
    state.write_x(A64Register::StackPointer, request.stack_top.get());
    state.set_tpidr_el0(tls_base.get());
    state.set_tpidrro_el0_from_runtime(tls_base.get());
    state.write_x(A64Register::General(a64_register(0)), request.argument);
    Ok(())
}

pub(super) fn validate_range(
    address_space: ProcessAddressSpace,
    start: u64,
    size: u64,
) -> Result<(), ProcessBuildError> {
    let end = start
        .checked_add(size)
        .ok_or_else(|| error(ProcessBuildStage::Placement, "guest range overflows"))?;
    if end > address_space.exclusive_limit() {
        return Err(error(
            ProcessBuildStage::Placement,
            "guest range exceeds the NPDM-selected address width",
        ));
    }
    Ok(())
}

pub(super) fn align_up(value: u64, alignment: u64) -> Result<u64, ProcessBuildError> {
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or_else(|| error(ProcessBuildStage::Placement, "alignment overflows"))
}

pub(super) fn a64_register(index: u8) -> nixe_cpu::state::a64::A64GeneralRegister {
    nixe_cpu::state::a64::A64GeneralRegister::new(index).expect("valid ABI register")
}

pub(super) fn error(stage: ProcessBuildStage, cause: impl Display) -> ProcessBuildError {
    ProcessBuildError::new(stage, cause)
}

#[cfg(test)]
mod thread_policy_tests {
    use super::ThreadPolicy;

    #[test]
    fn permission_masks_cover_inclusive_ranges_and_bit_63() {
        for (min, max, expected) in [(0, 2, 7), (1, 1, 2), (63, 63, 1 << 63), (0, 63, u64::MAX)] {
            let policy = ThreadPolicy {
                highest_priority: min as i32,
                lowest_priority: max as i32,
                min_core: min,
                max_core: max,
            };
            assert_eq!(policy.allowed_core_mask(), Some(expected));
            assert_eq!(policy.allowed_priority_mask(), Some(expected));
        }
        let unrepresentable = ThreadPolicy {
            highest_priority: 0,
            lowest_priority: 64,
            min_core: 0,
            max_core: 64,
        };
        assert_eq!(unrepresentable.allowed_core_mask(), None);
        assert_eq!(unrepresentable.allowed_priority_mask(), None);
    }
}
