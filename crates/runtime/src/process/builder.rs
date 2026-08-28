use super::*;

/// Builds an emulated process from a prepared launch plan.
#[derive(Default)]
pub struct ProcessBuilder {
    config: ProcessBuildConfig,
    virtual_clock: crate::VirtualClock,
    sd_card_root: Option<PathBuf>,
    cpu_backend: execution::CpuBackendConfig,
}

impl std::fmt::Debug for ProcessBuilder {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ProcessBuilder")
            .field("config", &self.config)
            .field("virtual_clock", &self.virtual_clock)
            .field("sd_card_root", &self.sd_card_root)
            .field("cpu_backend", &self.cpu_backend)
            .finish()
    }
}

impl ProcessBuilder {
    /// Creates a process builder using Switch 1 defaults.
    ///
    #[must_use]
    pub fn new() -> Self {
        Self::default()
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

    /// Selects the concrete CPU backend used by this process.
    #[must_use]
    pub fn with_cpu_backend(mut self, backend: execution::CpuBackendConfig) -> Self {
        self.cpu_backend = backend;
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
        let cpu = ProcessCpuContext::for_platform(
            self.config.target_platform,
            self.config.address_space_id,
        );
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
        let mut handles = plan
            .effective_policy()
            .and_then(|policy| policy.handle_table_size())
            .map_or_else(crate::HandleTable::new, |size| {
                crate::HandleTable::with_capacity_limit(usize::from(size))
            });
        let main_thread_object = crate::ThreadObject::new(1);
        let main_thread_handle = handles
            .insert(main_thread_object.clone())
            .map_err(|error| {
                ProcessBuildError::new(ProcessBuildStage::ThreadInitialization, error)
            })?;

        let mut memory = ExecutionMemory::new();
        for module in &modules {
            install_prepared_module(&mut memory, self.config.address_space_id, module)
                .map_err(|error| ProcessBuildError::new(ProcessBuildStage::Mapping, error))?;
            for mapping in module.mappings() {
                let mutable = mapping.permissions().is_writable();
                let purpose = match (matches!(abi, InitialProcessAbi::Homebrew), mutable) {
                    (true, true) => MemoryMappingPurpose::ModuleCodeMutable,
                    (true, false) => MemoryMappingPurpose::ModuleCodeStatic,
                    (false, true) => MemoryMappingPurpose::CodeMutable,
                    (false, false) => MemoryMappingPurpose::CodeStatic,
                };
                if !memory.set_mapping_purpose(
                    self.config.address_space_id,
                    GuestVirtualAddress::new(mapping.guest_address()),
                    mapping.bytes().len() as u64,
                    purpose,
                ) {
                    return Err(error(
                        ProcessBuildStage::Mapping,
                        "installed module mapping could not retain its purpose",
                    ));
                }
            }
        }
        let code_size = u64::try_from(memory.physical_page_count())
            .ok()
            .and_then(|pages| pages.checked_mul(SYNTHETIC_PAGE_SIZE as u64))
            .ok_or_else(|| error(ProcessBuildStage::Mapping, "process code size overflows"))?;

        let stack_size = align_up(stack_size.max(SYNTHETIC_PAGE_SIZE as u64), TLS_SIZE)?;
        if stack_size + (RESOURCE_GUARD_SIZE * 3) + (TLS_SIZE * 2) > memory_layout.stack().size() {
            return Err(error(
                ProcessBuildStage::Placement,
                "main-thread resources exceed the reserved stack region",
            ));
        }
        let stack_bottom = memory_layout.stack().base();
        let stack_top = stack_bottom
            .checked_add(stack_size)
            .ok_or_else(|| error(ProcessBuildStage::Placement, "main stack overflows"))?;
        let tls_base = stack_top
            .checked_add(RESOURCE_GUARD_SIZE)
            .ok_or_else(|| error(ProcessBuildStage::Placement, "TLS base overflows"))?;
        validate_range(address_space, tls_base.get(), TLS_SIZE)?;
        install_zero_pages(
            &mut memory,
            self.config.address_space_id,
            stack_bottom,
            stack_size,
        )?;
        install_zero_pages(
            &mut memory,
            self.config.address_space_id,
            tls_base,
            TLS_SIZE,
        )?;
        if !memory.set_mapping_purpose(
            self.config.address_space_id,
            tls_base,
            TLS_SIZE,
            MemoryMappingPurpose::ThreadLocal,
        ) {
            return Err(error(
                ProcessBuildStage::Mapping,
                "installed TLS mapping could not retain its purpose",
            ));
        }
        let (abi_context, loader_return) = if matches!(abi, InitialProcessAbi::Homebrew) {
            let address = tls_base
                .checked_add(TLS_SIZE + RESOURCE_GUARD_SIZE)
                .ok_or_else(|| error(ProcessBuildStage::Placement, "ABI context overflows"))?;
            validate_range(address_space, address.get(), SYNTHETIC_PAGE_SIZE as u64)?;
            install_homebrew_context(
                &mut memory,
                self.config.address_space_id,
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
                .checked_add(SYNTHETIC_PAGE_SIZE as u64 + RESOURCE_GUARD_SIZE)
                .ok_or_else(|| error(ProcessBuildStage::Placement, "loader return overflows"))?;
            validate_range(
                address_space,
                loader_return.get(),
                SYNTHETIC_PAGE_SIZE as u64,
            )?;
            install_homebrew_loader_return(
                &mut memory,
                self.config.address_space_id,
                loader_return,
            )?;
            (Some(address), Some(loader_return))
        } else {
            (None, None)
        };

        let entry = GuestVirtualAddress::new(modules[entry_module].entry_address());
        let mut state = ThreadCpuState::default();
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
            object: main_thread_object,
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
        let initial_mapped_size = u64::try_from(memory.physical_page_count())
            .ok()
            .and_then(|pages| pages.checked_mul(SYNTHETIC_PAGE_SIZE as u64))
            .ok_or_else(|| {
                error(
                    ProcessBuildStage::Mapping,
                    "process memory accounting overflows",
                )
            })?;
        let memory_accounting = ProcessMemoryAccounting::new(
            memory_layout.memory_capacity(),
            initial_mapped_size,
            code_size,
            stack_size,
            metadata.system_resource_size,
        )
        .ok_or_else(|| {
            error(
                ProcessBuildStage::Mapping,
                "initial process mappings and system resource exceed the configured physical-memory limit",
            )
        })?;
        let cpu_process_id = execution::allocate_cpu_process_id().ok_or_else(|| {
            ProcessBuildError::new(
                ProcessBuildStage::CpuInitialization,
                "CPU process identity exhausted",
            )
        })?;
        let execution = execution::ProcessExecutionControl::new(
            execution::ProcessExecutionConfiguration {
                virtual_clock: self.virtual_clock.clone(),
                timer_frequency: self.config.architectural_timer_frequency,
                cpu,
                address_space_end: nixe_memory::GuestVirtualAddress::new(
                    address_space.exclusive_limit(),
                ),
            },
            &memory,
            cpu_process_id,
            &self.cpu_backend,
        )
        .map_err(|error| ProcessBuildError::new(ProcessBuildStage::CpuInitialization, error))?;
        let process = RunnableProcess {
            process_id: self.config.process_id,
            lifecycle: nixe_scheduler::ProcessLifecycle::Running,
            process_exit: None,
            cpu,
            address_space,
            memory_layout,
            random_entropy,
            memory_accounting,
            memory: std::sync::Arc::new(memory),
            modules: modules.into_boxed_slice(),
            entry_module,
            main_thread_id,
            initial_thread_priority: metadata.priority,
            initial_ideal_vcpu: metadata.ideal_vcpu,
            thread_policy: metadata.thread_policy,
            next_thread_tls: memory_layout
                .stack()
                .base()
                .checked_add(memory_layout.stack().size())
                .expect("validated stack region has a finite end"),
            free_thread_tls: std::collections::BTreeSet::new(),
            threads,
            mounts: crate::ProcessMountNamespace::from_launch_plan(plan, self.sd_card_root.clone()),
            handles,
            address_waits: crate::AddressWaitRegistry::default(),
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
pub(super) struct ThreadPolicy {
    pub(super) highest_priority: i32,
    pub(super) lowest_priority: i32,
    pub(super) min_core: u32,
    pub(super) max_core: u32,
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
    let mut next = align_up(first_base.get(), SYNTHETIC_PAGE_SIZE as u64)?;
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
            SYNTHETIC_PAGE_SIZE as u64,
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

fn install_zero_pages(
    memory: &mut ExecutionMemory,
    address_space: AddressSpaceId,
    start: GuestVirtualAddress,
    size: u64,
) -> Result<(), ProcessBuildError> {
    let zero = [0_u8; SYNTHETIC_PAGE_SIZE];
    let page_count = usize::try_from(size / SYNTHETIC_PAGE_SIZE as u64).map_err(|_| {
        error(
            ProcessBuildStage::Mapping,
            "resource page count is too large",
        )
    })?;
    let requests = (0..page_count)
        .map(|index| SyntheticRamPage {
            virtual_address: start
                .checked_add((index * SYNTHETIC_PAGE_SIZE) as u64)
                .expect("validated resource range"),
            bytes: &zero,
            permissions: MemoryPermissions::READ_WRITE,
        })
        .collect::<Vec<_>>();
    memory
        .install_ram_pages_atomic(address_space, &requests)
        .map_err(|failure| ProcessBuildError::new(ProcessBuildStage::Mapping, failure.reason))
}

fn install_homebrew_context(
    memory: &mut ExecutionMemory,
    address_space: AddressSpaceId,
    address: GuestVirtualAddress,
    main_thread_handle: u32,
    identity: &crate::HomebrewIdentity,
) -> Result<(), ProcessBuildError> {
    let mut page = [0_u8; SYNTHETIC_PAGE_SIZE];
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
        .install_ram_pages_atomic(
            address_space,
            &[SyntheticRamPage {
                virtual_address: address,
                bytes: &page,
                permissions: MemoryPermissions::READ,
            }],
        )
        .map_err(|failure| ProcessBuildError::new(ProcessBuildStage::Mapping, failure.reason))
}

fn install_homebrew_loader_return(
    memory: &mut ExecutionMemory,
    address_space: AddressSpaceId,
    address: GuestVirtualAddress,
) -> Result<(), ProcessBuildError> {
    let mut page = [0_u8; SYNTHETIC_PAGE_SIZE];
    // If a CPU backend misses the runtime return-address boundary, the
    // mapped fallback still performs the ABI-prescribed process exit.
    page[..4].copy_from_slice(&HOME_BREW_EXIT_PROCESS_INSTRUCTION.to_le_bytes());
    memory
        .install_ram_pages_atomic(
            address_space,
            &[SyntheticRamPage {
                virtual_address: address,
                bytes: &page,
                permissions: MemoryPermissions::READ_EXECUTE,
            }],
        )
        .map_err(|failure| ProcessBuildError::new(ProcessBuildStage::Mapping, failure.reason))?;
    if !memory.set_mapping_purpose(
        address_space,
        address,
        SYNTHETIC_PAGE_SIZE as u64,
        MemoryMappingPurpose::CodeStatic,
    ) {
        return Err(error(
            ProcessBuildStage::Mapping,
            "loader return mapping could not retain its purpose",
        ));
    }
    Ok(())
}

pub(super) fn initialize_thread(
    state: &mut ThreadCpuState,
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
    state: &mut ThreadCpuState,
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
