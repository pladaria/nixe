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
    architectural_timer_frequency: u64,
}

impl HorizonMachineProfile {
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
            architectural_timer_frequency: self.architectural_timer_frequency,
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
        architectural_timer_frequency: 19_200_000,
    }
}

/// Returns the verified Horizon scheduler topology for Nintendo Switch 1.
///
/// The retail system exposes four Cortex-A57 application cores and Horizon
/// thread priorities 0 through 63. Switch 2 deliberately has no corresponding
/// built-in profile until its topology and kernel policy are publicly verified.
///
/// Sources:
/// - <https://developer.nvidia.com/embedded/jetson-tx1>
/// - <https://switchbrew.org/wiki/SVC#svcSetThreadPriority>
#[must_use]
pub fn switch_1_scheduler_profile() -> MachineSchedulerProfile {
    MachineSchedulerProfile::new(
        (0..4)
            .map(|id| VirtualCpuDescriptor::new(VirtualCpuId::new(id), 0))
            .collect(),
        PriorityRange::new(0, 63).expect("the verified Switch 1 priority range is valid"),
        10_000,
    )
    .expect("the verified Switch 1 scheduler topology is valid")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn switch_1_profile_records_verified_machine_properties() {
        let machine = switch_1_machine_profile();
        let profile = machine.scheduler();
        assert_eq!(profile.vcpus().len(), 4);
        assert_eq!(profile.priorities().highest(), 0);
        assert_eq!(profile.priorities().lowest(), 63);
        assert_eq!(profile.all_cores().len(), 4);
        assert_eq!(
            machine.platform(),
            nixe_cpu::platform::TargetPlatform::Switch1
        );
        assert_eq!(
            machine.process_build_config().architectural_timer_frequency,
            19_200_000
        );
        assert_eq!(
            machine.process_build_config().physical_memory_limit,
            SWITCH_1_APPLICATION_MEMORY_CAPACITY
        );
    }
}
