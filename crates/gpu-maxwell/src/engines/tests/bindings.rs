use super::*;
use crate::engines::threed::MaxwellThreeDResourceSelection;

pub(super) fn fixture_roles(
    state: &MaxwellThreeDState,
    space: &crate::MaxwellGpuMemory,
    roles: &[MaxwellThreeDResourceRole],
    access: &mut nixe_gpu::MemoryAccess<'_>,
) -> Result<MaxwellThreeDResolvedResources, MaxwellThreeDResourceError> {
    let mut roles = roles.to_vec();
    roles.sort_unstable();
    roles.dedup();
    super::super::threed::MaxwellThreeDResolvedResourceCache::new(1)
        .select(
            &MaxwellThreeDResourceSelection::new(&roles, &[]),
            Default::default(),
        )
        .resolve(state, space, 4096, access)
        .cloned()
}
