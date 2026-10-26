use super::*;

// Fixtures explicitly consume the resources they program through the same
// retained binding cache used by draw preparation.
pub(super) fn fixture_bindings(
    state: &MaxwellThreeDState,
    space: &crate::MaxwellGpuMemory,
    access: &mut nixe_gpu::MemoryAccess<'_>,
) -> Result<MaxwellThreeDResolvedResources, MaxwellThreeDResourceError> {
    use MaxwellThreeDResourceRole as R;
    let mut roles = Vec::new();
    roles.extend((0..state.vertex_input().streams().len()).map(|i| R::VertexStream(i as u8)));
    let index = state.vertex_input().index();
    if [
        index.address_upper().raw(),
        index.address_lower().raw(),
        index.limit_upper().raw(),
        index.limit_lower().raw(),
    ]
    .iter()
    .any(Option::is_some)
    {
        roles.push(R::IndexBuffer);
    }
    for (group, bindings) in state.shader_bindings().groups().iter().enumerate() {
        roles.extend(
            (0..bindings.constant_buffers().len()).map(|slot| R::ConstantBuffer {
                group: group as u8,
                slot: slot as u8,
            }),
        );
    }
    roles.extend((0..state.render_targets().color().len()).map(|i| R::ColorTarget(i as u8)));
    let depth = state.render_targets().depth_stencil();
    if state.render_targets().depth_target_count().value()
        != Some(&MaxwellThreeDDepthTargetCount::None)
        && [
            depth.address_upper().raw(),
            depth.address_lower().raw(),
            depth.format().raw(),
            depth.layout().raw(),
            depth.width().raw(),
            depth.height().raw(),
            depth.third_dimension().raw(),
        ]
        .iter()
        .any(Option::is_some)
    {
        roles.push(R::DepthStencilTarget);
    }
    fixture_roles(state, space, &roles, access)
}

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
        .resolve(state, space, &roles, 4096, access)
        .cloned()
}
