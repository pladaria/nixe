use super::*;

/// Builds an emulated process from a prepared launch plan.
#[derive(Default)]
pub struct ProcessBuilder {
    diagnostics: crate::DiagnosticsPolicy,
    config: ProcessBuildConfig,
    virtual_clock: crate::VirtualClock,
    sd_card_root: Option<PathBuf>,
    engine_provider: Option<Arc<dyn nixe_cpu_engine::EngineProvider>>,
    fallback_engine_provider: Option<Arc<dyn nixe_cpu_engine::EngineProvider>>,
}

impl std::fmt::Debug for ProcessBuilder {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ProcessBuilder")
            .field("diagnostics", &self.diagnostics)
            .field("config", &self.config)
            .field("virtual_clock", &self.virtual_clock)
            .field("sd_card_root", &self.sd_card_root)
            .field(
                "engine_provider",
                &self
                    .engine_provider
                    .as_ref()
                    .map(|provider| provider.descriptor()),
            )
            .field(
                "fallback_engine_provider",
                &self
                    .fallback_engine_provider
                    .as_ref()
                    .map(|provider| provider.descriptor()),
            )
            .finish()
    }
}

impl ProcessBuilder {
    /// Creates a process builder using detailed diagnostics and Switch 1 defaults.
    ///
    /// The application must inject a selected CPU engine provider before
    /// [`Self::build`]; runtime deliberately has no concrete-engine default.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub const fn with_diagnostics(mut self, diagnostics: crate::DiagnosticsPolicy) -> Self {
        self.diagnostics = diagnostics;
        self
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

    /// Injects the provider used to create the process-local execution domain.
    #[must_use]
    pub fn with_engine_provider(
        mut self,
        provider: Arc<dyn nixe_cpu_engine::EngineProvider>,
    ) -> Self {
        self.engine_provider = Some(provider);
        self
    }

    /// Injects the semantic engine used for exact `InterpretOne` exits.
    #[must_use]
    pub fn with_fallback_engine_provider(
        mut self,
        provider: Arc<dyn nixe_cpu_engine::EngineProvider>,
    ) -> Self {
        self.fallback_engine_provider = Some(provider);
        self
    }

    /// Prepares, maps, and initializes one runnable process.
    ///
    /// Packaged NSOs retain their dynamic relocations for the guest `rtld`.
    /// Standalone NROs likewise enter through their guest startup ABI.
    pub fn build(&self, plan: &LaunchPlan) -> Result<RunnableProcess, ProcessBuildError> {
        let engine_provider = self.engine_provider.as_deref().ok_or_else(|| {
            ProcessBuildError::new(
                ProcessBuildStage::EngineInitialization,
                "a CPU engine provider must be selected by the application",
            )
        })?;
        if self.config.architectural_timer_frequency == 0 {
            return Err(ProcessBuildError::new(
                ProcessBuildStage::Metadata,
                "architectural timer frequency must be nonzero",
            ));
        }
        let metadata = process_metadata(plan);
        let execution_state = metadata.execution_state;
        let address_space = metadata.address_space;
        let stack_size = metadata.stack_size;
        let abi = metadata.abi;
        let random_entropy = generate_process_entropy()?;
        let cpu = ProcessCpuContext::new(self.config.cpu_profile, self.config.address_space_id);
        let required = engine_requirements(execution_state, self.diagnostics.instruction_trace);
        let report = engine_provider.probe(self.config.cpu_profile, required);
        if !report.available
            || report.descriptor != engine_provider.descriptor()
            || !report.descriptor.capabilities.is_coherent()
            || !report.descriptor.capabilities.contains(required)
            || !report
                .descriptor
                .capabilities
                .supports_profile(self.config.cpu_profile, required)
        {
            return Err(ProcessBuildError::new(
                ProcessBuildStage::EngineInitialization,
                format!(
                    "CPU engine {} rejected guest profile {}: {:?}",
                    report.descriptor.name, self.config.cpu_profile, report.rejections
                ),
            ));
        }
        if let Some(fallback) = self.fallback_engine_provider.as_deref() {
            let report = fallback.probe(
                self.config.cpu_profile,
                engine_requirements(execution_state, false),
            );
            let fallback_required = engine_requirements(execution_state, false);
            if !report.available
                || report.descriptor != fallback.descriptor()
                || !report.descriptor.capabilities.is_coherent()
                || !report.descriptor.capabilities.contains(fallback_required)
                || !report
                    .descriptor
                    .capabilities
                    .supports_profile(self.config.cpu_profile, fallback_required)
                || report.descriptor.capabilities.interpret_one_fallback
            {
                return Err(ProcessBuildError::new(
                    ProcessBuildStage::EngineInitialization,
                    format!(
                        "fallback CPU engine {} rejected guest profile {}",
                        report.descriptor.name, self.config.cpu_profile
                    ),
                ));
            }
        }
        let thread_configuration = cpu
            .thread_configuration(execution_state)
            .map_err(|error| ProcessBuildError::new(ProcessBuildStage::Metadata, error))?;
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
        let mut state = ThreadCpuState::new(thread_configuration);
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
        let initial_memory_size = u64::try_from(memory.physical_page_count())
            .ok()
            .and_then(|pages| pages.checked_mul(SYNTHETIC_PAGE_SIZE as u64))
            .ok_or_else(|| {
                error(
                    ProcessBuildStage::Mapping,
                    "process memory accounting overflows",
                )
            })?;
        if initial_memory_size > memory_layout.memory_capacity() {
            return Err(error(
                ProcessBuildStage::Mapping,
                "initial process mappings exceed the configured physical-memory limit",
            ));
        }
        let domain_id = execution::allocate_engine_domain_id().ok_or_else(|| {
            ProcessBuildError::new(
                ProcessBuildStage::EngineInitialization,
                "engine domain identity exhausted",
            )
        })?;
        let fallback_domain = self
            .fallback_engine_provider
            .as_deref()
            .map(|provider| {
                execution::allocate_engine_domain_id()
                    .map(|id| (id, provider))
                    .ok_or_else(|| {
                        ProcessBuildError::new(
                            ProcessBuildStage::EngineInitialization,
                            "fallback engine domain identity exhausted",
                        )
                    })
            })
            .transpose()?;
        let execution = execution::ProcessExecutionControl::with_provider(
            execution::ProcessExecutionConfiguration {
                diagnostics: self.diagnostics,
                virtual_clock: self.virtual_clock.clone(),
                timer_frequency: self.config.architectural_timer_frequency,
                cpu,
                address_space_end: nixe_memory::GuestVirtualAddress::new(
                    address_space.exclusive_limit(),
                ),
            },
            &memory,
            domain_id,
            engine_provider,
            fallback_domain,
        )
        .map_err(|error| ProcessBuildError::new(ProcessBuildStage::EngineInitialization, error))?;
        let process = RunnableProcess {
            process_id: self.config.process_id,
            lifecycle: nixe_scheduler::ProcessLifecycle::Running,
            process_exit: None,
            cpu,
            address_space,
            memory_layout,
            random_entropy,
            heap_size: 0,
            initial_memory_size,
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
        process.translate_entry()?;
        Ok(process)
    }
}

fn engine_requirements(
    state: ExecutionState,
    instruction_trace: bool,
) -> nixe_cpu_engine::EngineCapabilities {
    nixe_cpu_engine::EngineCapabilities {
        a64: state == ExecutionState::A64,
        a32: state == ExecutionState::A32,
        t32: state == ExecutionState::T32,
        instruction_trace,
        deterministic_execution: true,
        ..Default::default()
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
    execution_state: ExecutionState,
    address_space: ProcessAddressSpace,
    stack_size: u64,
    abi: InitialProcessAbi,
    priority: i32,
    ideal_vcpu: nixe_scheduler::VirtualCpuId,
    thread_policy: Option<ThreadPolicy>,
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
            let state = if npdm.flags().is_64_bit_instruction() {
                ExecutionState::A64
            } else {
                ExecutionState::A32
            };
            ProcessMetadata {
                execution_state: state,
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
            }
        }
        LaunchKind::Homebrew(_) => ProcessMetadata {
            execution_state: ExecutionState::A64,
            address_space: ProcessAddressSpace::Bit64,
            stack_size: DEFAULT_HOME_BREW_STACK_SIZE,
            abi: InitialProcessAbi::Homebrew,
            priority: 44,
            ideal_vcpu: nixe_scheduler::VirtualCpuId::new(0),
            thread_policy: None,
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
    // If an execution engine misses the runtime return-address boundary, the
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
    match state {
        ThreadCpuState::A64(state) => {
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
        }
        ThreadCpuState::A32(state) => {
            let entry = u32::try_from(entry.get()).map_err(|_| {
                error(
                    ProcessBuildStage::ThreadInitialization,
                    "A32 PC exceeds 32 bits",
                )
            })?;
            let stack_top = u32::try_from(stack_top.get()).map_err(|_| {
                error(
                    ProcessBuildStage::ThreadInitialization,
                    "A32 SP exceeds 32 bits",
                )
            })?;
            let tls_base = u32::try_from(tls_base.get()).map_err(|_| {
                error(
                    ProcessBuildStage::ThreadInitialization,
                    "A32 TLS exceeds 32 bits",
                )
            })?;
            state.set_instruction_address(entry).map_err(|error| {
                ProcessBuildError::new(ProcessBuildStage::ThreadInitialization, error)
            })?;
            state.write_r(a32_register(13), stack_top);
            state.set_tpidrurw(tls_base);
            state.set_tpidruro_from_runtime(tls_base);
            state.write_r(a32_register(0), 0);
            state.write_r(a32_register(1), main_thread_handle);
        }
    }
    Ok(())
}

pub(super) fn initialize_created_thread(
    state: &mut ThreadCpuState,
    request: &ThreadCreateRequest,
    tls_base: GuestVirtualAddress,
) -> Result<(), ProcessBuildError> {
    match state {
        ThreadCpuState::A64(state) => {
            state.set_pc(request.entry.get());
            state.write_x(A64Register::StackPointer, request.stack_top.get());
            state.set_tpidr_el0(tls_base.get());
            state.set_tpidrro_el0_from_runtime(tls_base.get());
            state.write_x(A64Register::General(a64_register(0)), request.argument);
        }
        ThreadCpuState::A32(state) => {
            let entry = u32::try_from(request.entry.get()).map_err(|_| {
                error(
                    ProcessBuildStage::ThreadInitialization,
                    "A32 PC exceeds 32 bits",
                )
            })?;
            let stack = u32::try_from(request.stack_top.get()).map_err(|_| {
                error(
                    ProcessBuildStage::ThreadInitialization,
                    "A32 SP exceeds 32 bits",
                )
            })?;
            let tls = u32::try_from(tls_base.get()).map_err(|_| {
                error(
                    ProcessBuildStage::ThreadInitialization,
                    "A32 TLS exceeds 32 bits",
                )
            })?;
            state.set_instruction_address(entry).map_err(|error| {
                ProcessBuildError::new(ProcessBuildStage::ThreadInitialization, error)
            })?;
            state.write_r(a32_register(0), request.argument as u32);
            state.write_r(a32_register(13), stack);
            state.set_tpidrurw(tls);
            state.set_tpidruro_from_runtime(tls);
        }
    }
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

pub(super) fn a32_register(index: u8) -> A32GeneralRegister {
    A32GeneralRegister::new(index).expect("valid ABI register")
}

pub(super) fn error(stage: ProcessBuildStage, cause: impl Display) -> ProcessBuildError {
    ProcessBuildError::new(stage, cause)
}
