use std::error::Error;
use std::fmt::{Display, Formatter};
use std::sync::Arc;

use crate::VirtualCpuId;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct PriorityRange {
    highest: i32,
    lowest: i32,
}

impl PriorityRange {
    pub fn new(highest: i32, lowest: i32) -> Result<Self, MachineSchedulerProfileError> {
        if highest > lowest {
            return Err(MachineSchedulerProfileError::InvalidPriorityRange { highest, lowest });
        }
        Ok(Self { highest, lowest })
    }

    #[must_use]
    pub const fn highest(self) -> i32 {
        self.highest
    }

    #[must_use]
    pub const fn lowest(self) -> i32 {
        self.lowest
    }

    #[must_use]
    pub const fn contains(self, priority: i32) -> bool {
        priority >= self.highest && priority <= self.lowest
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct VirtualCpuDescriptor {
    id: VirtualCpuId,
    cluster: u32,
}

impl VirtualCpuDescriptor {
    #[must_use]
    pub const fn new(id: VirtualCpuId, cluster: u32) -> Self {
        Self { id, cluster }
    }

    #[must_use]
    pub const fn id(&self) -> VirtualCpuId {
        self.id
    }

    #[must_use]
    pub const fn cluster(&self) -> u32 {
        self.cluster
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MachineSchedulerProfile {
    vcpus: Arc<[VirtualCpuDescriptor]>,
    priorities: PriorityRange,
    default_timeslice_instructions: u64,
}

impl MachineSchedulerProfile {
    pub fn new(
        mut vcpus: Vec<VirtualCpuDescriptor>,
        priorities: PriorityRange,
        default_timeslice_instructions: u64,
    ) -> Result<Self, MachineSchedulerProfileError> {
        if vcpus.is_empty() {
            return Err(MachineSchedulerProfileError::EmptyTopology);
        }
        if default_timeslice_instructions == 0 {
            return Err(MachineSchedulerProfileError::ZeroTimeslice);
        }
        vcpus.sort_by_key(VirtualCpuDescriptor::id);
        if let Some(duplicate) = vcpus
            .windows(2)
            .find(|pair| pair[0].id() == pair[1].id())
            .map(|pair| pair[0].id())
        {
            return Err(MachineSchedulerProfileError::DuplicateVirtualCpu(duplicate));
        }
        Ok(Self {
            vcpus: vcpus.into(),
            priorities,
            default_timeslice_instructions,
        })
    }

    #[must_use]
    pub fn vcpus(&self) -> &[VirtualCpuDescriptor] {
        &self.vcpus
    }

    #[must_use]
    pub const fn priorities(&self) -> PriorityRange {
        self.priorities
    }

    #[must_use]
    pub const fn default_timeslice_instructions(&self) -> u64 {
        self.default_timeslice_instructions
    }

    #[must_use]
    pub fn contains(&self, id: VirtualCpuId) -> bool {
        self.vcpus
            .binary_search_by_key(&id, VirtualCpuDescriptor::id)
            .is_ok()
    }

    pub fn all_cores(&self) -> CoreSet {
        CoreSet {
            cores: self.vcpus.iter().map(VirtualCpuDescriptor::id).collect(),
        }
    }

    pub fn core_set<I>(&self, cores: I) -> Result<CoreSet, CoreSetError>
    where
        I: IntoIterator<Item = VirtualCpuId>,
    {
        CoreSet::new(self, cores)
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct CoreSet {
    cores: Arc<[VirtualCpuId]>,
}

impl CoreSet {
    pub fn new<I>(profile: &MachineSchedulerProfile, cores: I) -> Result<Self, CoreSetError>
    where
        I: IntoIterator<Item = VirtualCpuId>,
    {
        let mut cores: Vec<_> = cores.into_iter().collect();
        cores.sort_unstable();
        cores.dedup();
        if cores.is_empty() {
            return Err(CoreSetError::Empty);
        }
        if let Some(core) = cores.iter().find(|core| !profile.contains(**core)) {
            return Err(CoreSetError::UnknownVirtualCpu(*core));
        }
        Ok(Self {
            cores: cores.into(),
        })
    }

    #[must_use]
    pub fn contains(&self, id: VirtualCpuId) -> bool {
        self.cores.binary_search(&id).is_ok()
    }

    #[must_use]
    pub fn iter(&self) -> impl ExactSizeIterator<Item = VirtualCpuId> + '_ {
        self.cores.iter().copied()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.cores.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cores.is_empty()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MachineSchedulerProfileError {
    EmptyTopology,
    DuplicateVirtualCpu(VirtualCpuId),
    InvalidPriorityRange { highest: i32, lowest: i32 },
    ZeroTimeslice,
}

impl Display for MachineSchedulerProfileError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyTopology => formatter.write_str("machine topology has no virtual CPUs"),
            Self::DuplicateVirtualCpu(id) => {
                write!(formatter, "duplicate virtual CPU identity {id}")
            }
            Self::InvalidPriorityRange { highest, lowest } => write!(
                formatter,
                "invalid priority range: highest {highest} is numerically greater than lowest {lowest}"
            ),
            Self::ZeroTimeslice => formatter.write_str("scheduler timeslice must be non-zero"),
        }
    }
}

impl Error for MachineSchedulerProfileError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CoreSetError {
    Empty,
    UnknownVirtualCpu(VirtualCpuId),
}

impl Display for CoreSetError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => formatter.write_str("core set cannot be empty"),
            Self::UnknownVirtualCpu(id) => {
                write!(formatter, "virtual CPU {id} is not in the machine topology")
            }
        }
    }
}

impl Error for CoreSetError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile(count: u32) -> MachineSchedulerProfile {
        MachineSchedulerProfile::new(
            (0..count)
                .map(|id| VirtualCpuDescriptor::new(VirtualCpuId::new(id), 0))
                .collect(),
            PriorityRange::new(0, 63).unwrap(),
            10_000,
        )
        .unwrap()
    }

    #[test]
    fn topology_and_core_sets_are_checked_by_identity() {
        let profile = profile(6);
        let set = profile
            .core_set([
                VirtualCpuId::new(5),
                VirtualCpuId::new(1),
                VirtualCpuId::new(5),
            ])
            .unwrap();
        assert_eq!(
            set.iter().collect::<Vec<_>>(),
            [VirtualCpuId::new(1), VirtualCpuId::new(5)]
        );
        assert_eq!(
            profile.core_set([VirtualCpuId::new(6)]),
            Err(CoreSetError::UnknownVirtualCpu(VirtualCpuId::new(6)))
        );
    }

    #[test]
    fn arbitrary_non_contiguous_identities_do_not_become_indices() {
        let profile = MachineSchedulerProfile::new(
            vec![
                VirtualCpuDescriptor::new(VirtualCpuId::new(9), 1),
                VirtualCpuDescriptor::new(VirtualCpuId::new(2), 0),
            ],
            PriorityRange::new(-4, 12).unwrap(),
            1,
        )
        .unwrap();
        assert!(profile.contains(VirtualCpuId::new(9)));
        assert!(!profile.contains(VirtualCpuId::new(1)));
    }
}
