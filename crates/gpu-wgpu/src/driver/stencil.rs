//! WebGPU shares stencil masks/reference across faces. Preserve independent
//! face operations, and merge only parameters which are actually consumed.
//! https://www.w3.org/TR/webgpu/#stencil-state
use super::*;
use nixe_gpu::{
    DepthCompareOperation as Compare, StencilFaceState as Face, StencilOperation as Op,
};

pub(super) fn parameters(
    state: nixe_gpu::StencilTestState,
) -> Result<(StencilState, u32), BackendDriverError> {
    let mut read = None;
    let mut write = None;
    let mut reference = None;
    let mut merge = |face: Face| {
        if !matches!(face.compare, Compare::Always | Compare::Never) {
            merge_parameter(
                &mut read,
                face.read_mask,
                "per-face stencil comparison masks",
            )?;
        }
        if face.writes() {
            merge_parameter(&mut write, face.write_mask, "per-face stencil write masks")?;
        }
        if face.uses_reference() {
            merge_parameter(
                &mut reference,
                face.reference,
                "per-face stencil references",
            )?;
        }
        Ok::<_, BackendDriverError>(())
    };
    merge(state.front)?;
    merge(state.back)?;
    Ok((
        StencilState {
            front: face(state.front),
            back: face(state.back),
            read_mask: u32::from(read.unwrap_or(255)),
            write_mask: u32::from(write.unwrap_or(0)),
        },
        u32::from(reference.unwrap_or(0)),
    ))
}

fn merge_parameter(
    target: &mut Option<u8>,
    value: u8,
    detail: &'static str,
) -> Result<(), BackendDriverError> {
    if target.is_some_and(|current| current != value) {
        return Err(unsupported(detail));
    }
    *target = Some(value);
    Ok(())
}

fn face(state: Face) -> StencilFaceState {
    // A zero per-face write mask must remain read-only even when the other
    // face supplies a nonzero shared mask. KEEP preserves every stencil bit.
    let op = |value| {
        if state.write_mask == 0 {
            StencilOperation::Keep
        } else {
            operation(value)
        }
    };
    StencilFaceState {
        compare: compare_function(state.compare),
        fail_op: op(state.fail),
        depth_fail_op: op(state.depth_fail),
        pass_op: op(state.pass),
    }
}

pub(super) const fn operation(value: Op) -> StencilOperation {
    match value {
        Op::Keep => StencilOperation::Keep,
        Op::Zero => StencilOperation::Zero,
        Op::Replace => StencilOperation::Replace,
        Op::IncrementClamp => StencilOperation::IncrementClamp,
        Op::DecrementClamp => StencilOperation::DecrementClamp,
        Op::Invert => StencilOperation::Invert,
        Op::IncrementWrap => StencilOperation::IncrementWrap,
        Op::DecrementWrap => StencilOperation::DecrementWrap,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shared_parameters_preserve_consumed_values_and_disable_masked_face_writes() {
        let front = Face {
            pass: Op::Replace,
            write_mask: 15,
            ..Face::read_only(Compare::Always, 3, 17)
        };
        let back = Face {
            pass: Op::Invert,
            ..Face::read_only(Compare::NotEqual, 7, 17)
        };
        let (state, reference) = parameters(nixe_gpu::StencilTestState { front, back }).unwrap();
        assert_eq!((state.read_mask, state.write_mask, reference), (7, 15, 17));
        assert_eq!(state.front.pass_op, StencilOperation::Replace);
        assert_eq!(state.back.pass_op, StencilOperation::Keep);
        let back = Face {
            write_mask: 15,
            ..back
        };
        assert!(parameters(nixe_gpu::StencilTestState { front, back }).is_ok());
        assert!(
            parameters(nixe_gpu::StencilTestState {
                front,
                back: Face {
                    write_mask: 240,
                    ..back
                }
            })
            .is_err()
        );
        assert!(
            parameters(nixe_gpu::StencilTestState {
                front,
                back: Face {
                    reference: 18,
                    ..back
                }
            })
            .is_err()
        );
    }
}
