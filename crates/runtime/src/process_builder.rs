//! Construction of a runnable CPU process from an immutable launch plan.

use std::error::Error;
use std::fmt::{Display, Formatter};

use swiitx_cpu::address::{AddressSpaceId, GuestVirtualAddress};
use swiitx_cpu::ir::block::IrBlock;
use swiitx_cpu::ir::print::{IrPrintOptions, print_block};
use swiitx_cpu::location::{ExecutionState, LocationDescriptor};
use swiitx_cpu::memory::{
    MemoryPermissions, SYNTHETIC_PAGE_SIZE, SyntheticMemory, SyntheticRamPage,
};
use swiitx_cpu::profile::{GuestCpuProfile, ProcessCpuContext};
use swiitx_cpu::state::{ThreadCpuState, a32::A32GeneralRegister, a64::A64Register};
use swiitx_cpu::translate::{BlockTranslationConfig, translate_block};
use swiitx_loader_executable::{
    AddressSpaceType, ExternalSymbol, NsoBatchModule, PreparationConfig, PreparedModule,
    SymbolResolution, prepare_nso_batch,
};

use crate::{LaunchKind, LaunchModuleImage, LaunchPlan, install_prepared_module};

const DEFAULT_IMAGE_BASE: u64 = 0x7100_0000;
const DEFAULT_HOME_BREW_STACK_SIZE: u64 = 0x10_0000;
const MODULE_GUARD_SIZE: u64 = 0x1_0000;
const RESOURCE_GUARD_SIZE: u64 = 0x1_0000;
const TLS_SIZE: u64 = SYNTHETIC_PAGE_SIZE as u64;
const MAIN_THREAD_HANDLE: u32 = 1;
const HOME_BREW_CONFIG_ENTRY_SIZE: usize = 24;
const HOME_BREW_MAIN_THREAD_HANDLE_KEY: u32 = 1;

/// Runtime interpretation of the address-space selector validated by NPDM.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProcessAddressSpace {
    Bit32,
    Bit32NoReserved,
    Bit64Old,
    Bit64,
}

impl ProcessAddressSpace {
    const fn from_npdm(value: AddressSpaceType) -> Self {
        match value {
            AddressSpaceType::AddressSpace32Bit => Self::Bit32,
            AddressSpaceType::AddressSpace32BitNoReserved => Self::Bit32NoReserved,
            AddressSpaceType::AddressSpace64BitOld => Self::Bit64Old,
            AddressSpaceType::AddressSpace64Bit => Self::Bit64,
        }
    }

    const fn exclusive_limit(self) -> u64 {
        match self {
            Self::Bit32 | Self::Bit32NoReserved => 1_u64 << 32,
            Self::Bit64Old | Self::Bit64 => u64::MAX,
        }
    }
}

/// Caller-controlled process identities and relocatable image placement.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProcessBuildConfig {
    pub address_space_id: AddressSpaceId,
    pub cpu_profile: GuestCpuProfile,
    pub image_base: GuestVirtualAddress,
}

impl Default for ProcessBuildConfig {
    fn default() -> Self {
        Self {
            address_space_id: AddressSpaceId::new(1),
            cpu_profile: GuestCpuProfile::switch_1(),
            image_base: GuestVirtualAddress::new(DEFAULT_IMAGE_BASE),
        }
    }
}

/// Fully initialized main thread returned by [`ProcessBuilder`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MainThread {
    pub state: ThreadCpuState,
    pub handle: u32,
    pub stack_bottom: GuestVirtualAddress,
    pub stack_top: GuestVirtualAddress,
    pub tls_base: GuestVirtualAddress,
    pub abi_context: Option<GuestVirtualAddress>,
}

/// A process whose executable bytes are visible only through process memory.
pub struct RunnableProcess {
    cpu: ProcessCpuContext,
    address_space: ProcessAddressSpace,
    memory: SyntheticMemory,
    modules: Box<[PreparedModule]>,
    entry_module: usize,
    main_thread: MainThread,
}

impl RunnableProcess {
    #[must_use]
    pub const fn cpu_context(&self) -> ProcessCpuContext {
        self.cpu
    }

    #[must_use]
    pub const fn address_space(&self) -> ProcessAddressSpace {
        self.address_space
    }

    #[must_use]
    pub const fn memory(&self) -> &SyntheticMemory {
        &self.memory
    }

    #[must_use]
    pub fn modules(&self) -> &[PreparedModule] {
        &self.modules
    }

    #[must_use]
    pub fn entry_module(&self) -> &PreparedModule {
        &self.modules[self.entry_module]
    }

    #[must_use]
    pub const fn main_thread(&self) -> &MainThread {
        &self.main_thread
    }

    /// Translates and verifies the initialized entry block through process memory.
    pub fn translate_entry(&self) -> Result<IrBlock, ProcessBuildError> {
        let start = LocationDescriptor::new(
            GuestVirtualAddress::new(self.entry_module().entry_address()),
            self.main_thread.state.execution_state(),
            self.cpu.profile().id(),
        );
        translate_block(
            BlockTranslationConfig::default(),
            &self.cpu.profile(),
            self.cpu.address_space_id(),
            start,
            &self.memory,
        )
        .map_err(|error| ProcessBuildError::new(ProcessBuildStage::EntryTranslation, error))
    }

    /// Produces the deterministic verified-IR dump used by the first integration milestone.
    pub fn print_entry_ir(&self) -> Result<String, ProcessBuildError> {
        Ok(print_block(
            &self.translate_entry()?,
            IrPrintOptions::default(),
        ))
    }
}

/// Stage at which process construction failed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProcessBuildStage {
    Metadata,
    Placement,
    Preparation,
    Mapping,
    ThreadInitialization,
    EntryTranslation,
}

/// Fail-closed process construction error.
#[derive(Debug)]
pub struct ProcessBuildError {
    stage: ProcessBuildStage,
    cause: Box<str>,
}

impl ProcessBuildError {
    fn new(stage: ProcessBuildStage, cause: impl Display) -> Self {
        Self {
            stage,
            cause: cause.to_string().into_boxed_str(),
        }
    }

    #[must_use]
    pub const fn stage(&self) -> ProcessBuildStage {
        self.stage
    }

    #[must_use]
    pub const fn cause(&self) -> &str {
        &self.cause
    }
}

impl Display for ProcessBuildError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "cannot build process during {:?}: {}",
            self.stage, self.cause
        )
    }
}

impl Error for ProcessBuildError {}

/// Builds an emulated process from a prepared launch plan.
#[derive(Debug, Default)]
pub struct ProcessBuilder {
    diagnostics: crate::DiagnosticsPolicy,
    config: ProcessBuildConfig,
}

impl ProcessBuilder {
    /// Creates a process builder using detailed diagnostics and Switch 1 defaults.
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

    #[must_use]
    pub const fn diagnostics(&self) -> crate::DiagnosticsPolicy {
        self.diagnostics
    }

    #[must_use]
    pub const fn cpu_diagnostics(&self) -> swiitx_cpu::coverage::CpuDiagnosticsConfig {
        self.diagnostics.cpu()
    }

    /// Prepares, relocates, maps, and initializes one runnable process.
    pub fn build(&self, plan: &LaunchPlan) -> Result<RunnableProcess, ProcessBuildError> {
        let (execution_state, address_space, stack_size, abi) = process_metadata(plan);
        let cpu = ProcessCpuContext::new(self.config.cpu_profile, self.config.address_space_id);
        let thread_configuration = cpu
            .thread_configuration(execution_state)
            .map_err(|error| ProcessBuildError::new(ProcessBuildStage::Metadata, error))?;
        let placements = module_placements(plan, self.config.image_base, address_space)?;
        let modules = prepare_modules(plan, &placements, address_space)?;
        let entry_module = plan.entry_module_index();

        let mut memory = SyntheticMemory::new();
        for module in &modules {
            install_prepared_module(&mut memory, self.config.address_space_id, module)
                .map_err(|error| ProcessBuildError::new(ProcessBuildStage::Mapping, error))?;
        }

        let resource_start = align_up(
            modules
                .iter()
                .map(|module| module.image_base() + module.image_extent())
                .max()
                .ok_or_else(|| error(ProcessBuildStage::Placement, "launch plan has no modules"))?
                .checked_add(RESOURCE_GUARD_SIZE)
                .ok_or_else(|| error(ProcessBuildStage::Placement, "resource base overflows"))?,
            SYNTHETIC_PAGE_SIZE as u64,
        )?;
        let stack_size = align_up(stack_size.max(SYNTHETIC_PAGE_SIZE as u64), TLS_SIZE)?;
        let stack_bottom = GuestVirtualAddress::new(resource_start);
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
        let abi_context = if matches!(abi, InitialProcessAbi::Homebrew) {
            let address = tls_base
                .checked_add(TLS_SIZE + RESOURCE_GUARD_SIZE)
                .ok_or_else(|| error(ProcessBuildStage::Placement, "ABI context overflows"))?;
            validate_range(address_space, address.get(), SYNTHETIC_PAGE_SIZE as u64)?;
            install_homebrew_context(
                &mut memory,
                self.config.address_space_id,
                address,
                MAIN_THREAD_HANDLE,
            )?;
            Some(address)
        } else {
            None
        };

        let entry = GuestVirtualAddress::new(modules[entry_module].entry_address());
        let mut state = ThreadCpuState::new(thread_configuration);
        initialize_thread(
            &mut state,
            entry,
            stack_top,
            tls_base,
            MAIN_THREAD_HANDLE,
            abi_context,
        )?;
        let main_thread = MainThread {
            state,
            handle: MAIN_THREAD_HANDLE,
            stack_bottom,
            stack_top,
            tls_base,
            abi_context,
        };
        let process = RunnableProcess {
            cpu,
            address_space,
            memory,
            modules: modules.into_boxed_slice(),
            entry_module,
            main_thread,
        };
        process.translate_entry()?;
        Ok(process)
    }
}

#[derive(Clone, Copy)]
enum InitialProcessAbi {
    Packaged,
    Homebrew,
}

fn process_metadata(
    plan: &LaunchPlan,
) -> (ExecutionState, ProcessAddressSpace, u64, InitialProcessAbi) {
    match plan.kind() {
        LaunchKind::Packaged(identity) => {
            let npdm = identity.npdm();
            let state = if npdm.flags().is_64_bit_instruction() {
                ExecutionState::A64
            } else {
                ExecutionState::A32
            };
            (
                state,
                ProcessAddressSpace::from_npdm(npdm.flags().address_space()),
                u64::from(npdm.main_thread_stack_size()),
                InitialProcessAbi::Packaged,
            )
        }
        LaunchKind::Homebrew => (
            ExecutionState::A64,
            ProcessAddressSpace::Bit64,
            DEFAULT_HOME_BREW_STACK_SIZE,
            InitialProcessAbi::Homebrew,
        ),
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
    let all_nso = plan
        .modules()
        .iter()
        .all(|module| matches!(module.image(), LaunchModuleImage::Nso(_)));
    if all_nso {
        let modules = plan
            .modules()
            .iter()
            .zip(placements)
            .map(|(module, config)| match module.image() {
                LaunchModuleImage::Nso(image) => Ok(NsoBatchModule {
                    image,
                    config: *config,
                }),
                LaunchModuleImage::Nro(_) => unreachable!(),
            })
            .collect::<Result<Vec<_>, ProcessBuildError>>()?;
        return prepare_nso_batch(&modules, plan.symbol_scope(), &[])
            .map_err(|error| ProcessBuildError::new(ProcessBuildStage::Preparation, error));
    }
    if plan.modules().len() != 1 {
        return Err(error(
            ProcessBuildStage::Preparation,
            "mixed NRO/NSO launch plans are unsupported",
        ));
    }
    let LaunchModuleImage::Nro(image) = plan.modules()[0].image() else {
        unreachable!();
    };
    let unresolved = |_: ExternalSymbol<'_>| SymbolResolution::Unresolved;
    let module = image
        .prepare(placements[0], &unresolved)
        .map_err(|error| ProcessBuildError::new(ProcessBuildStage::Preparation, error))?;
    validate_range(address_space, module.image_base(), module.image_extent())?;
    Ok(vec![module])
}

fn install_zero_pages(
    memory: &mut SyntheticMemory,
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
    memory: &mut SyntheticMemory,
    address_space: AddressSpaceId,
    address: GuestVirtualAddress,
    main_thread_handle: u32,
) -> Result<(), ProcessBuildError> {
    let mut page = [0_u8; SYNTHETIC_PAGE_SIZE];
    page[..4].copy_from_slice(&HOME_BREW_MAIN_THREAD_HANDLE_KEY.to_le_bytes());
    page[8..16].copy_from_slice(&u64::from(main_thread_handle).to_le_bytes());
    // The following zeroed 24-byte entry is EntryType_EndOfList.
    debug_assert!(HOME_BREW_CONFIG_ENTRY_SIZE * 2 <= page.len());
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

fn initialize_thread(
    state: &mut ThreadCpuState,
    entry: GuestVirtualAddress,
    stack_top: GuestVirtualAddress,
    tls_base: GuestVirtualAddress,
    main_thread_handle: u32,
    abi_context: Option<GuestVirtualAddress>,
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

fn validate_range(
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

fn align_up(value: u64, alignment: u64) -> Result<u64, ProcessBuildError> {
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or_else(|| error(ProcessBuildStage::Placement, "alignment overflows"))
}

fn a64_register(index: u8) -> swiitx_cpu::state::a64::A64GeneralRegister {
    swiitx_cpu::state::a64::A64GeneralRegister::new(index).expect("valid ABI register")
}

fn a32_register(index: u8) -> A32GeneralRegister {
    A32GeneralRegister::new(index).expect("valid ABI register")
}

fn error(stage: ProcessBuildStage, cause: impl Display) -> ProcessBuildError {
    ProcessBuildError::new(stage, cause)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use swiitx_cpu::ir::terminator::{ControlTarget, Terminator};
    use swiitx_cpu::memory::{
        CpuMemory, InstructionMemory, MemoryAccess, MemoryAccessSize, MemoryValue,
    };

    use super::*;
    use crate::{Launcher, LauncherInput};

    fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn synthetic_nro() -> Vec<u8> {
        let mut bytes = vec![0; 0x2800];
        bytes[..4].copy_from_slice(&0x1400_0020_u32.to_le_bytes()); // B entry + 0x80
        bytes[0x10..0x14].copy_from_slice(b"NRO0");
        put_u32(&mut bytes, 0x18, 0x2800);
        put_u32(&mut bytes, 0x20, 0);
        put_u32(&mut bytes, 0x24, 0x1000);
        put_u32(&mut bytes, 0x28, 0x1000);
        put_u32(&mut bytes, 0x2c, 0x1000);
        put_u32(&mut bytes, 0x30, 0x2000);
        put_u32(&mut bytes, 0x34, 0x800);
        put_u32(&mut bytes, 0x38, 0x800);
        bytes[0x40..0x60].fill(0x5a);
        bytes
    }

    fn plan() -> (tempfile::TempDir, LaunchPlan) {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("synthetic.nro");
        fs::write(&path, synthetic_nro()).unwrap();
        let plan = Launcher::build(LauncherInput::new(&path)).unwrap();
        (directory, plan)
    }

    #[test]
    fn builder_propagates_runtime_diagnostics_to_cpu_resources() {
        let builder = ProcessBuilder::new();
        assert_eq!(
            builder.cpu_diagnostics().report_detail,
            swiitx_cpu::coverage::MissingInstructionReportDetail::Detailed
        );
    }

    #[test]
    fn npdm_address_space_values_keep_distinct_runtime_meanings() {
        assert_eq!(
            ProcessAddressSpace::from_npdm(AddressSpaceType::AddressSpace32Bit),
            ProcessAddressSpace::Bit32
        );
        assert_eq!(
            ProcessAddressSpace::from_npdm(AddressSpaceType::AddressSpace32BitNoReserved),
            ProcessAddressSpace::Bit32NoReserved
        );
        assert_eq!(
            ProcessAddressSpace::from_npdm(AddressSpaceType::AddressSpace64BitOld),
            ProcessAddressSpace::Bit64Old
        );
        assert_eq!(
            ProcessAddressSpace::from_npdm(AddressSpaceType::AddressSpace64Bit),
            ProcessAddressSpace::Bit64
        );
        assert!(validate_range(ProcessAddressSpace::Bit32, u64::from(u32::MAX), 2).is_err());
    }

    #[test]
    fn a32_thread_initialization_uses_32_bit_pc_stack_and_tls() {
        let cpu = ProcessCpuContext::new(GuestCpuProfile::switch_1(), AddressSpaceId::new(7));
        let configuration = cpu.thread_configuration(ExecutionState::A32).unwrap();
        let mut state = ThreadCpuState::new(configuration);
        initialize_thread(
            &mut state,
            GuestVirtualAddress::new(0x0020_0000),
            GuestVirtualAddress::new(0x0080_0000),
            GuestVirtualAddress::new(0x0090_0000),
            MAIN_THREAD_HANDLE,
            None,
        )
        .unwrap();
        let ThreadCpuState::A32(state) = state else {
            panic!("A32 metadata must create AArch32 state");
        };
        assert_eq!(state.instruction_address(), 0x0020_0000);
        assert_eq!(state.read_r(a32_register(13)), 0x0080_0000);
        assert_eq!(state.tpidrurw(), 0x0090_0000);
        assert_eq!(state.tpidruro(), 0x0090_0000);
        assert_eq!(state.read_r(a32_register(1)), MAIN_THREAD_HANDLE);
    }

    #[test]
    fn synthetic_launch_translates_entry_only_through_process_memory() {
        let (_directory, plan) = plan();
        let process = ProcessBuilder::new().build(&plan).unwrap();
        let entry = GuestVirtualAddress::new(process.entry_module().entry_address());
        assert_eq!(
            process
                .memory()
                .fetch32(process.cpu_context().address_space_id(), entry)
                .unwrap()
                .bits,
            0x1400_0020
        );
        let dump = process.print_entry_ir().unwrap();
        assert!(dump.contains(" A64 "));
        assert!(dump.contains("raw=0x14000020"));
        assert_eq!(
            process.main_thread().state.execution_state(),
            ExecutionState::A64
        );
        let ThreadCpuState::A64(state) = &process.main_thread().state else {
            panic!("homebrew fixture must initialize A64");
        };
        assert_eq!(state.pc(), entry.get());
        assert_eq!(
            state.read_x(A64Register::StackPointer),
            process.main_thread().stack_top.get()
        );
        assert_eq!(state.tpidr_el0(), process.main_thread().tls_base.get());
        let context = process.main_thread().abi_context.unwrap();
        assert_eq!(
            state.read_x(A64Register::General(a64_register(0))),
            context.get()
        );
        assert_eq!(
            state.read_x(A64Register::General(a64_register(1))),
            u64::MAX
        );
        assert_eq!(
            process
                .memory()
                .read(
                    process.cpu_context().address_space_id(),
                    context,
                    MemoryAccess::normal(MemoryAccessSize::Word),
                )
                .unwrap()
                .value,
            MemoryValue::U32(HOME_BREW_MAIN_THREAD_HANDLE_KEY)
        );
        assert_eq!(
            process
                .memory()
                .read(
                    process.cpu_context().address_space_id(),
                    context.checked_add(8).unwrap(),
                    MemoryAccess::normal(MemoryAccessSize::Doubleword),
                )
                .unwrap()
                .value,
            MemoryValue::U64(u64::from(MAIN_THREAD_HANDLE))
        );
        assert_eq!(
            process
                .memory()
                .read(
                    process.cpu_context().address_space_id(),
                    context
                        .checked_add(HOME_BREW_CONFIG_ENTRY_SIZE as u64)
                        .unwrap(),
                    MemoryAccess::normal(MemoryAccessSize::Word),
                )
                .unwrap()
                .value,
            MemoryValue::U32(0)
        );
    }

    #[test]
    fn image_base_is_relocatable_without_changing_pc_relative_translation() {
        let (_directory, plan) = plan();
        let first = ProcessBuilder::new()
            .with_config(ProcessBuildConfig {
                image_base: GuestVirtualAddress::new(0x7100_0000),
                ..ProcessBuildConfig::default()
            })
            .build(&plan)
            .unwrap();
        let second = ProcessBuilder::new()
            .with_config(ProcessBuildConfig {
                image_base: GuestVirtualAddress::new(0x7200_0000),
                ..ProcessBuildConfig::default()
            })
            .build(&plan)
            .unwrap();
        assert_eq!(
            second.entry_module().entry_address() - first.entry_module().entry_address(),
            0x0100_0000
        );
        let first_block = first.translate_entry().unwrap();
        let second_block = second.translate_entry().unwrap();
        let direct_target = |block: &IrBlock| match block.terminator {
            Terminator::Direct {
                target: ControlTarget::Direct { pc, .. },
            } => pc.get(),
            ref terminator => panic!("unexpected terminator {terminator:?}"),
        };
        assert_eq!(
            direct_target(&second_block) - direct_target(&first_block),
            0x0100_0000
        );
        assert_eq!(
            second.modules()[0].mappings()[0].guest_address()
                - first.modules()[0].mappings()[0].guest_address(),
            0x0100_0000
        );
    }

    #[test]
    fn writable_code_alias_updates_the_fetched_generation() {
        let (_directory, plan) = plan();
        let mut process = ProcessBuilder::new().build(&plan).unwrap();
        let space = process.cpu.address_space_id();
        let entry = GuestVirtualAddress::new(process.entry_module().entry_address());
        let before = process.memory.fetch32(space, entry).unwrap().dependencies;
        let mapping = process.memory.mapping_info(space, entry).unwrap();
        let alias = GuestVirtualAddress::new(0x7000_0000);
        assert!(process.memory.map_page(
            space,
            alias,
            mapping.physical_page,
            MemoryPermissions::READ_WRITE
        ));
        process
            .memory
            .write(
                space,
                alias,
                MemoryAccess::normal(MemoryAccessSize::Word),
                MemoryValue::U32(0xd503_201f),
            )
            .unwrap();
        let after = process.memory.fetch32(space, entry).unwrap().dependencies;
        assert_ne!(before, after);
    }
}
