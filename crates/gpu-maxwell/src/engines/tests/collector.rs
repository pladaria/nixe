use std::sync::Arc;

use super::*;

/// Test observation of validated methods and owned operation snapshots.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaxwellEnginePacketDispatch {
    pub(super) methods: Box<[MaxwellEngineMethodDispatch]>,
    ordered_operations: Box<[MaxwellEngineOperation]>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MaxwellEngineOperation {
    Notification {
        address: u64,
        source: MaxwellMethodSource,
    },
    TwoDBlit(twod::MaxwellTwoDBlitOperation),
    TwoDSolid(twod::MaxwellTwoDSolidOperation),
    HostSynchronization(MaxwellHostSynchronizationOperation),
    ComputeInlineToMemory(MaxwellComputeInlineToMemoryUpload),
    InlineToMemory(MaxwellInlineToMemoryUpload),
    MemoryCopy(MaxwellMemoryCopyOperation),
    ComputeSynchronization(MaxwellComputeOperationTrigger),
    ComputeLaunch {
        launch: MaxwellComputeLaunch,
        state: Box<MaxwellComputeState>,
    },
    ThreeDInlineConstantBuffer(MaxwellThreeDInlineConstantBufferUpload),
    ThreeD(Box<MaxwellThreeDTriggeredOperation>),
    ThreeDRenderCondition(MaxwellMethodSource),
    ThreeDSynchronization(Box<MaxwellThreeDSynchronizationOperation>),
}

/// Test snapshot of one 3D trigger.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaxwellThreeDTriggeredOperation {
    trigger: MaxwellThreeDOperationTrigger,
    state: Arc<MaxwellThreeDState>,
}

impl MaxwellThreeDTriggeredOperation {
    #[must_use]
    pub const fn trigger(&self) -> MaxwellThreeDOperationTrigger {
        self.trigger
    }

    #[must_use]
    pub fn state(&self) -> &MaxwellThreeDState {
        &self.state
    }
}

impl MaxwellEnginePacketDispatch {
    #[must_use]
    pub fn methods(&self) -> &[MaxwellEngineMethodDispatch] {
        &self.methods
    }

    #[must_use]
    pub fn ordered_operations(&self) -> &[MaxwellEngineOperation] {
        &self.ordered_operations
    }
}

/// Test collector for method semantics and owned trigger observations.
pub fn dispatch_maxwell_engine_packet(
    channel: &mut MaxwellGpuChannel,
    submission: FrontendSubmissionId,
    packet: &crate::MaxwellDecodedPacket,
) -> Result<MaxwellEnginePacketDispatch, MaxwellEngineDispatchError> {
    let mut methods = Vec::new();
    let mut ordered_operations = Vec::new();
    let mut mme_methods = None;
    let mut mme_parameters = Vec::new();
    let mut observe = async |event: MaxwellEngineEvent<'_>| {
        let operation = match event.operation {
            PendingEngineOperation::Notification { address, source } => {
                MaxwellEngineOperation::Notification { address, source }
            }
            PendingEngineOperation::TwoDSolid(rect) => MaxwellEngineOperation::TwoDSolid(rect),
            PendingEngineOperation::TwoDBlit(resolve) => MaxwellEngineOperation::TwoDBlit(resolve),
            PendingEngineOperation::HostSynchronization(operation) => {
                MaxwellEngineOperation::HostSynchronization(operation)
            }
            PendingEngineOperation::ComputeInlineToMemory(upload) => {
                MaxwellEngineOperation::ComputeInlineToMemory(upload)
            }
            PendingEngineOperation::InlineToMemory(upload) => {
                MaxwellEngineOperation::InlineToMemory(upload)
            }
            PendingEngineOperation::MemoryCopy(operation) => {
                MaxwellEngineOperation::MemoryCopy(operation)
            }
            PendingEngineOperation::ComputeSynchronization(operation) => {
                MaxwellEngineOperation::ComputeSynchronization(operation)
            }
            PendingEngineOperation::ComputeLaunch(launch) => {
                MaxwellEngineOperation::ComputeLaunch {
                    launch,
                    state: Box::new(
                        event
                            .compute
                            .expect("compute launch has live state")
                            .clone(),
                    ),
                }
            }
            PendingEngineOperation::ThreeDInlineConstantBuffer(upload) => {
                MaxwellEngineOperation::ThreeDInlineConstantBuffer(upload)
            }
            PendingEngineOperation::ThreeD(trigger) => {
                MaxwellEngineOperation::ThreeD(Box::new(MaxwellThreeDTriggeredOperation {
                    trigger,
                    state: Arc::new(event.three_d.clone()),
                }))
            }
            PendingEngineOperation::ThreeDRenderCondition(source) => {
                MaxwellEngineOperation::ThreeDRenderCondition(source)
            }
            PendingEngineOperation::ThreeDSynchronization(trigger) => {
                MaxwellEngineOperation::ThreeDSynchronization(Box::new(
                    MaxwellThreeDSynchronizationOperation {
                        trigger,
                        state: Arc::new(event.three_d.clone()),
                    },
                ))
            }
        };
        ordered_operations.push(operation);
        Ok::<(), std::convert::Infallible>(())
    };
    let streamed = crate::test_memory::ready(stream_maxwell_engine_packet(
        channel,
        submission,
        packet.view(),
        Some(&mut methods),
        &mut mme_methods,
        &mut mme_parameters,
        &mut observe,
    ))
    .map_err(|error| match error {
        MaxwellEngineStreamError::Dispatch(error) => *error,
        MaxwellEngineStreamError::Consumer(never) => match never {},
    });
    if streamed.is_ok() {
        match crate::test_memory::ready(finish_maxwell_engine_stream(channel, &mut observe)) {
            Ok(()) => {}
            Err(never) => match never {},
        }
    }
    streamed?;
    Ok(MaxwellEnginePacketDispatch {
        methods: methods.into_boxed_slice(),
        ordered_operations: ordered_operations.into_boxed_slice(),
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaxwellThreeDSynchronizationOperation {
    trigger: MaxwellThreeDSynchronizationTrigger,
    state: Arc<MaxwellThreeDState>,
}

impl MaxwellThreeDSynchronizationOperation {
    pub const fn trigger(&self) -> MaxwellThreeDSynchronizationTrigger {
        self.trigger
    }

    pub fn state(&self) -> &MaxwellThreeDState {
        &self.state
    }
}
