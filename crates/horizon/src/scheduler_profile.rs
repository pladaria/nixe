use nixe_scheduler::{MachineSchedulerProfile, PriorityRange, VirtualCpuDescriptor, VirtualCpuId};

/// Atomic Horizon machine composition. Keeping CPU, scheduler, memory-layout,
/// and timer policy together prevents mixed-generation configurations while
/// leaving their neutral component types in their owning crates.
#[derive(Clone, Debug)]
pub struct HorizonMachineProfile {
    scheduler: MachineSchedulerProfile,
    cpu: nixe_cpu::profile::GuestCpuProfile,
    memory_layout: nixe_runtime::ProcessMemoryLayoutProfile,
    architectural_timer_frequency: u64,
}

impl HorizonMachineProfile {
    #[must_use]
    pub const fn scheduler(&self) -> &MachineSchedulerProfile {
        &self.scheduler
    }

    #[must_use]
    pub const fn cpu(&self) -> nixe_cpu::profile::GuestCpuProfile {
        self.cpu
    }

    #[must_use]
    pub fn process_build_config(&self) -> nixe_runtime::ProcessBuildConfig {
        nixe_runtime::ProcessBuildConfig {
            cpu_profile: self.cpu,
            memory_layout_profile: self.memory_layout,
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
        cpu: nixe_cpu::profile::GuestCpuProfile::switch_1(),
        memory_layout: nixe_runtime::ProcessMemoryLayoutProfile::Horizon2Plus,
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
    fn switch_1_profile_records_verified_topology() {
        let machine = switch_1_machine_profile();
        let profile = machine.scheduler();
        assert_eq!(profile.vcpus().len(), 4);
        assert_eq!(profile.priorities().highest(), 0);
        assert_eq!(profile.priorities().lowest(), 63);
        assert_eq!(profile.all_cores().len(), 4);
        assert_eq!(
            machine.cpu(),
            nixe_cpu::profile::GuestCpuProfile::switch_1()
        );
        assert_eq!(
            machine.process_build_config().architectural_timer_frequency,
            19_200_000
        );
    }
}
