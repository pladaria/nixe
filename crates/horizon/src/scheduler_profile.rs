use nixe_scheduler::{MachineSchedulerProfile, PriorityRange, VirtualCpuDescriptor, VirtualCpuId};

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
        let profile = switch_1_scheduler_profile();
        assert_eq!(profile.vcpus().len(), 4);
        assert_eq!(profile.priorities().highest(), 0);
        assert_eq!(profile.priorities().lowest(), 63);
        assert_eq!(profile.all_cores().len(), 4);
    }
}
