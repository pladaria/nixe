use nixe_scheduler::{MachineSchedulerProfile, PriorityRange, VirtualCpuDescriptor, VirtualCpuId};

// A retail 4 GiB Switch assigns 3285 MiB to the application memory pool:
// https://github.com/Atmosphere-NX/Atmosphere/blob/e468f59c9d369b8ebbffa040f4c9fc201b9f75a8/libraries/libmesosphere/source/board/nintendo/nx/kern_k_system_control.cpp#L319-L340
const SWITCH_1_APPLICATION_MEMORY_CAPACITY: u64 = 3285 * 1024 * 1024;

/// Atomic Horizon machine composition. Keeping CPU, scheduler, memory, and
/// timer policy together prevents mixed-generation configurations while
/// leaving their neutral component types in their owning crates.
#[derive(Clone, Debug)]
pub struct HorizonMachineProfile {
    scheduler: MachineSchedulerProfile,
    platform: nixe_cpu::platform::TargetPlatform,
    memory_layout: nixe_runtime::ProcessMemoryLayoutProfile,
    application_memory_capacity: u64,
    physical_memory_capacity: u64,
    graphics: GraphicsGeneration,
}

/// Guest command-engine generation, independent of the host graphics API.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphicsGeneration {
    Maxwell,
    Ampere,
}

impl HorizonMachineProfile {
    pub fn for_platform(platform: nixe_cpu::platform::TargetPlatform) -> Self {
        match platform {
            nixe_cpu::platform::TargetPlatform::Switch1 => switch_1_machine_profile(),
            nixe_cpu::platform::TargetPlatform::Switch2 => switch_2_machine_profile(),
        }
    }

    pub const fn physical_memory_capacity(&self) -> u64 {
        self.physical_memory_capacity
    }

    /// Kernel/HLE shared objects are charged separately from the application pool.
    pub const fn system_memory_capacity(&self) -> u64 {
        self.physical_memory_capacity - self.application_memory_capacity
    }

    pub const fn graphics(&self) -> GraphicsGeneration {
        self.graphics
    }

    #[must_use]
    pub const fn scheduler(&self) -> &MachineSchedulerProfile {
        &self.scheduler
    }

    #[must_use]
    pub const fn platform(&self) -> nixe_cpu::platform::TargetPlatform {
        self.platform
    }

    #[must_use]
    pub fn process_build_config(&self) -> nixe_runtime::ProcessBuildConfig {
        nixe_runtime::ProcessBuildConfig {
            target_platform: self.platform,
            memory_layout_profile: self.memory_layout,
            // The current frontend launches applications and homebrew in the
            // application slot. Select the distinct applet and system pools
            // here once those process classes can be launched independently.
            physical_memory_limit: self.application_memory_capacity,
            ..nixe_runtime::ProcessBuildConfig::default()
        }
    }
}

/// Returns the verified Switch 1 CPU/kernel/runtime composition.
#[must_use]
pub fn switch_1_machine_profile() -> HorizonMachineProfile {
    HorizonMachineProfile {
        scheduler: switch_1_scheduler_profile(),
        platform: nixe_cpu::platform::TargetPlatform::Switch1,
        memory_layout: nixe_runtime::ProcessMemoryLayoutProfile::Horizon2Plus,
        application_memory_capacity: SWITCH_1_APPLICATION_MEMORY_CAPACITY,
        physical_memory_capacity: 4 * 1024 * 1024 * 1024,
        graphics: GraphicsGeneration::Maxwell,
    }
}

/// Returns the verified Horizon scheduler topology for Nintendo Switch 1.
///
/// The retail system exposes four Cortex-A57 application cores and Horizon
/// thread priorities 0 through 63.
///
/// Sources:
/// - <https://developer.nvidia.com/embedded/jetson-tx1>
/// - <https://switchbrew.org/wiki/SVC#svcSetThreadPriority>
#[must_use]
pub fn switch_1_scheduler_profile() -> MachineSchedulerProfile {
    MachineSchedulerProfile::new(
        (0..4)
            .map(|id| {
                VirtualCpuDescriptor::new(VirtualCpuId::new(id), 0)
                    .with_rotation_priority(if id == 3 { 63 } else { 59 })
            })
            .collect(),
        PriorityRange::new(0, 63).expect("the verified Switch 1 priority range is valid"),
    )
    .expect("the verified Switch 1 scheduler topology is valid")
    // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/global_scheduler_context.cpp#L41-L55
    .with_preemption(10_000_000, 2)
    .expect("the verified Switch 1 preemption policy is valid")
}

/// Native application composition: six schedulable cores and a 12 GiB RAM arena.
/// These application-pool and scheduling policies follow the reference kernel's
/// 12 GiB configuration, without its title-specific pool growth or extra GPU RAM.
/// https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/core/hle/kernel/board/nintendo/nx/k_system_control.cpp
/// https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/core/hle/kernel/global_scheduler_context.cpp
fn switch_2_machine_profile() -> HorizonMachineProfile {
    let scheduler = MachineSchedulerProfile::new(
        [59, 59, 59, 63, 59, 59]
            .into_iter()
            .enumerate()
            .map(|(id, priority)| {
                VirtualCpuDescriptor::new(VirtualCpuId::new(id as u32), 0)
                    .with_rotation_priority(priority)
            })
            .collect(),
        PriorityRange::new(0, 63).expect("valid Horizon priorities"),
    )
    .expect("valid native application topology")
    .with_preemption(10_000_000, 2)
    .expect("valid Horizon preemption policy");
    HorizonMachineProfile {
        scheduler,
        platform: nixe_cpu::platform::TargetPlatform::Switch2,
        memory_layout: nixe_runtime::ProcessMemoryLayoutProfile::Switch2,
        application_memory_capacity: 9809 * 1024 * 1024,
        physical_memory_capacity: 12 * 1024 * 1024 * 1024,
        graphics: GraphicsGeneration::Ampere,
    }
}
