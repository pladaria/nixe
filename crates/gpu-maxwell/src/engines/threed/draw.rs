//! Primitive execution follows Maxwell DrawManager: counters program state,
//! END executes general/inline draws, and instance sequences finish at a new
//! first BEGIN or the end of their macro.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/engines/draw_manager.cpp
use super::*;
use crate::MaxwellMethodSource;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum Mode {
    #[default]
    General,
    Instance,
    Inline,
}
/// Current hardware draw state, owned by the channel's DrawManager. Native
/// consumers borrow this state; compilation keys never stand in for it.
/// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/engines/maxwell_3d.h
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellThreeDDrawState {
    pub topology: u32,
    pub indexed: bool,
    pub vertex_first: u32,
    pub vertex_count: u32,
    pub index: MaxwellThreeDDrawIndexBuffer,
    pub base_vertex: u32,
    pub base_instance: u32,
    pub instance_count: u32,
    mode: Mode,
    instances: u32,
    source: Option<MaxwellMethodSource>,
    preparing: Option<MaxwellMethodSource>,
    pub(super) inline_indices: Vec<u8>,
    pub(super) inline_draw_pending: bool,
}
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MaxwellThreeDDrawIndexBuffer {
    pub address: u64,
    pub limit: u64,
    pub format: u32,
    pub first: u32,
    pub count: u32,
}
impl MaxwellThreeDDrawState {
    pub fn inline_indices(&self) -> Option<&[u8]> {
        self.inline_draw_pending
            .then_some(self.inline_indices.as_slice())
    }
}

pub(in crate::engines) fn is_pending(state: &MaxwellThreeDFrontendState) -> bool {
    state.draw.mode == Mode::Instance && state.draw.instances != 0
}

/// Only a new first BEGIN consumes a deferred instance sequence. Ordinary
/// register writes do not probe/compare the previous draw's state.
#[inline]
pub(in crate::engines) fn admit_method(
    state: &mut MaxwellThreeDFrontendState,
    method: GpuMethodId,
    argument: u32,
) -> bool {
    if method.0 != 0x1618 || !is_pending(state) {
        return true;
    }
    let argument = state.mme().resolve_shadow_argument(method, argument);
    MaxwellThreeDBegin::parse(argument)
        .is_none_or(|begin| begin.instance() != MaxwellThreeDBeginInstance::First)
}

pub(super) fn begin(
    state: &mut MaxwellThreeDFrontendState,
    argument: u32,
    value: MaxwellThreeDBegin,
    origin: impl FnOnce() -> MaxwellMethodSource,
) {
    match value.instance() {
        MaxwellThreeDBeginInstance::First => {
            debug_assert!(!is_pending(state));
            state.draw.instances = 0;
            state.draw.mode = Mode::General;
        }
        MaxwellThreeDBeginInstance::Subsequent => {
            state.draw.instances = state.draw.instances.wrapping_add(1);
            state.draw.mode = Mode::Instance;
            state.draw.source = Some(origin());
        }
        MaxwellThreeDBeginInstance::Unchanged => {}
    }
    state.draw.topology = u32::from(value.topology());
    state.begin_primitive(argument, value);
}
pub(super) fn index_count(state: &mut MaxwellThreeDFrontendState) {
    state.draw.indexed = true;
}
pub(super) fn inline(state: &mut MaxwellThreeDFrontendState) {
    state.draw.mode = Mode::Inline;
}
fn capture(state: &mut MaxwellThreeDFrontendState, instances: Option<u32>) {
    let r = state.operation_state().register_words();
    let indexed = state.draw.indexed || state.draw.mode == Mode::Inline;
    let vertices = (!indexed).then(|| (r[0x0d74 / 4], r[0x0d78 / 4]));
    let index = indexed.then(|| MaxwellThreeDDrawIndexBuffer {
        address: (u64::from(r[0x17c8 / 4]) << 32) | u64::from(r[0x17cc / 4]),
        limit: (u64::from(r[0x17d0 / 4]) << 32) | u64::from(r[0x17d4 / 4]),
        format: r[0x17d8 / 4],
        first: r[0x17dc / 4],
        count: r[0x17e0 / 4],
    });
    let base_vertex = r[0x1434 / 4];
    let base_instance = r[0x1438 / 4];
    let topology = if r[0x1948 / 4] == 1 {
        match r[0x1970 / 4] {
            0 => state.draw.topology,
            1 => 0,
            2 => 1,
            value => value,
        }
    } else {
        state.draw.topology
    };
    state.draw.topology = topology;
    if let Some((first, count)) = vertices {
        state.draw.vertex_first = first;
        state.draw.vertex_count = count;
    }
    if let Some(index) = index {
        state.draw.index = index;
    }
    state.draw.base_vertex = base_vertex;
    state.draw.base_instance = base_instance;
    state.draw.instance_count = instances.unwrap_or(1);
    if state.draw.mode == Mode::Inline {
        state.draw.indexed = true;
        state.draw.index.format = 2;
        state.draw.index.first = 0;
        state.draw.index.count = state.publish_inline_indices();
    }
}
fn complete(state: &mut MaxwellThreeDFrontendState) {
    state.complete_inline_draw();
    if state.draw.mode == Mode::General {
        state.draw.indexed = false;
    }
}
pub(super) fn end<L: crate::MaxwellFrontend, H: crate::MaxwellCommandHost>(
    state: &mut MaxwellThreeDFrontendState,
    argument: u32,
    execution: &mut crate::execution::ChannelExecution<'_, L, H>,
    origin: impl Fn() -> MaxwellMethodSource,
) -> Result<bool, MaxwellEngineStreamError> {
    if argument > 1 {
        return Err(invalid_encoding(origin(), "END", "expected boolean 0 or 1").into());
    }
    state.end_primitive(argument);
    if state.draw.mode == Mode::Instance {
        state.draw.source = Some(origin());
        return Ok(true);
    }
    Ok(submit(state, None, execution, origin)?)
}
pub(super) fn clear<L: crate::MaxwellFrontend, H: crate::MaxwellCommandHost>(
    state: &mut MaxwellThreeDFrontendState,
    argument: u32,
    execution: &mut crate::execution::ChannelExecution<'_, L, H>,
    origin: impl Fn() -> MaxwellMethodSource,
) -> Result<bool, MaxwellEngineStreamError> {
    let value = MaxwellThreeDClearSurface::parse(argument).ok_or_else(|| {
        invalid_encoding(
            origin(),
            "CLEAR_SURFACE",
            "invalid target, layer, or reserved bits",
        )
    })?;
    if value.depth() || value.stencil() || value.color_mask() != 0 {
        execution.clear(state.operation_state(), origin)?;
    }
    Ok(true)
}
fn submit<L: crate::MaxwellFrontend, H: crate::MaxwellCommandHost>(
    state: &mut MaxwellThreeDFrontendState,
    instances: Option<u32>,
    execution: &mut crate::execution::ChannelExecution<'_, L, H>,
    origin: impl Fn() -> MaxwellMethodSource,
) -> Result<bool, crate::MaxwellSubmissionExecutionError> {
    debug_assert!(state.draw.preparing.is_none());
    capture(state, instances);
    if execution.draw(state.operation_state(), &state.draw, &origin)? {
        complete(state);
        Ok(true)
    } else {
        state.draw.preparing = Some(origin());
        Ok(false)
    }
}
/// Only native preparation retries. Registers, uploads and signals are never replayed.
pub(in crate::engines) async fn wait_preparation<
    L: crate::MaxwellFrontend,
    H: crate::MaxwellCommandHost,
>(
    state: &mut MaxwellThreeDFrontendState,
    execution: &mut crate::execution::ChannelExecution<'_, L, H>,
) -> Result<(), crate::MaxwellSubmissionExecutionError> {
    let source = state.draw.preparing.expect("unavailable native draw");
    execution
        .wait_draw(state.operation_state(), &state.draw, source)
        .await?;
    state.draw.preparing = None;
    complete(state);
    Ok(())
}
/// A new first BEGIN or macro completion consumes the live deferred instance draw.
pub(in crate::engines) fn flush<L: crate::MaxwellFrontend, H: crate::MaxwellCommandHost>(
    state: &mut MaxwellThreeDFrontendState,
    execution: &mut crate::execution::ChannelExecution<'_, L, H>,
) -> Result<bool, crate::MaxwellSubmissionExecutionError> {
    if !is_pending(state) {
        return Ok(true);
    }
    let source = state
        .draw
        .source
        .expect("deferred instance BEGIN/END source");
    if !submit(
        state,
        Some(state.draw.instances.wrapping_add(1)),
        execution,
        || source,
    )? {
        return Ok(false);
    }
    state.draw.instances = 0;
    state.draw.indexed = false;
    Ok(true)
}
pub(in crate::engines) async fn wait_flush<
    L: crate::MaxwellFrontend,
    H: crate::MaxwellCommandHost,
>(
    state: &mut MaxwellThreeDFrontendState,
    execution: &mut crate::execution::ChannelExecution<'_, L, H>,
) -> Result<(), crate::MaxwellSubmissionExecutionError> {
    wait_preparation(state, execution).await?;
    state.draw.instances = 0;
    state.draw.indexed = false;
    Ok(())
}
