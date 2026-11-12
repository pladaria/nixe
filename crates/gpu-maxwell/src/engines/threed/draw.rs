//! Primitive execution follows Maxwell DrawManager: counters program state,
//! END executes general/inline draws, and instance sequences finish at a new
//! first BEGIN, state changes, or the end of their macro.
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
    drawn_instances: u32,
    source: Option<MaxwellMethodSource>,
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

pub(in crate::engines) fn is_pending(state: &MaxwellThreeDEngineState) -> bool {
    state.draw.mode == Mode::Instance && state.draw.instances != 0
}

/// Only these methods belong to the deferred instance loop itself.
/// https://github.com/zortos293/Lichii/blob/b4b830c8faf1fb5d7b2ad1b892117f4cfd708546/src/video_core/engines/draw_manager.cpp#L116-L140
#[inline]
pub(super) fn instance_loop_method(method: GpuMethodId) -> bool {
    matches!(
        method.0,
        0x1618
            | 0x1614
            | 0x0d74
            | 0x0d78
            | 0x17dc
            | 0x17e0
            | 0x17e4
            | 0x17e8
            | 0x17ec
            | 0x17f0
            | 0x17f4
            | 0x17f8
            | 0x15e8
            | 0x15ec
            | 0x15f0
            | 0x1300
            | 0x1304
            | 0x1214
            | 0x1218
    )
}

#[inline]
pub(crate) fn before_write<L: crate::MaxwellFrontend, H: crate::MaxwellCommandHost>(
    state: &mut MaxwellThreeDEngineState,
    method: GpuMethodId,
    execution: &mut crate::execution::ChannelExecution<'_, L, H>,
) -> Result<(), crate::MaxwellSubmissionExecutionError> {
    if is_pending(state) && !instance_loop_method(method) {
        flush(state, execution, true)?;
    }
    Ok(())
}

pub(super) fn begin(
    state: &mut MaxwellThreeDEngineState,
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
pub(super) fn index_count(state: &mut MaxwellThreeDEngineState) {
    state.draw.indexed = true;
}
pub(super) fn inline(state: &mut MaxwellThreeDEngineState) {
    state.draw.mode = Mode::Inline;
}
fn capture(state: &mut MaxwellThreeDEngineState, instances: Option<u32>) {
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
    let mut base_instance = r[0x1438 / 4];
    if state.draw.mode == Mode::Instance && state.draw.drawn_instances != 0 {
        base_instance = base_instance
            .wrapping_add(state.draw.drawn_instances)
            .wrapping_add(1);
    }
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
fn complete(state: &mut MaxwellThreeDEngineState) {
    state.complete_inline_draw();
    if state.draw.mode == Mode::General {
        state.draw.indexed = false;
    }
}
pub(super) fn end<L: crate::MaxwellFrontend, H: crate::MaxwellCommandHost>(
    state: &mut MaxwellThreeDEngineState,
    argument: u32,
    execution: &mut crate::execution::ChannelExecution<'_, L, H>,
    origin: impl Fn() -> MaxwellMethodSource,
) -> Result<(), MaxwellEngineStreamError> {
    if argument > 1 {
        return Err(invalid_encoding(origin(), "END", "expected boolean 0 or 1").into());
    }
    state.end_primitive(argument);
    if state.draw.mode == Mode::Instance {
        state.draw.source = Some(origin());
        return Ok(());
    }
    Ok(submit(state, None, execution, origin)?)
}
pub(super) fn clear<L: crate::MaxwellFrontend, H: crate::MaxwellCommandHost>(
    state: &mut MaxwellThreeDEngineState,
    argument: u32,
    execution: &mut crate::execution::ChannelExecution<'_, L, H>,
    origin: impl Fn() -> MaxwellMethodSource,
) -> Result<(), MaxwellEngineStreamError> {
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
    Ok(())
}
fn submit<L: crate::MaxwellFrontend, H: crate::MaxwellCommandHost>(
    state: &mut MaxwellThreeDEngineState,
    instances: Option<u32>,
    execution: &mut crate::execution::ChannelExecution<'_, L, H>,
    origin: impl Fn() -> MaxwellMethodSource,
) -> Result<(), crate::MaxwellSubmissionExecutionError> {
    capture(state, instances);
    execution.draw(state.operation_state(), &state.draw, &origin)?;
    complete(state);
    Ok(())
}
/// Split a loop before a state write, or finish it at a new BEGIN/macro boundary.
/// Continuations draw only the remaining instances and advance their IDs.
pub(in crate::engines) fn flush<L: crate::MaxwellFrontend, H: crate::MaxwellCommandHost>(
    state: &mut MaxwellThreeDEngineState,
    execution: &mut crate::execution::ChannelExecution<'_, L, H>,
    loop_continues: bool,
) -> Result<(), crate::MaxwellSubmissionExecutionError> {
    if !is_pending(state) {
        if !loop_continues {
            state.draw.drawn_instances = 0;
        }
        return Ok(());
    }
    let source = state
        .draw
        .source
        .expect("deferred instance BEGIN/END source");
    submit(
        state,
        Some(
            state
                .draw
                .instances
                .wrapping_add(u32::from(state.draw.drawn_instances == 0)),
        ),
        execution,
        || source,
    )?;
    state.draw.drawn_instances = if loop_continues {
        state
            .draw
            .drawn_instances
            .wrapping_add(state.draw.instances)
    } else {
        0
    };
    state.draw.instances = 0;
    state.draw.indexed = false;
    Ok(())
}
