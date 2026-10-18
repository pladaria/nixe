use super::*;

#[test]
fn separate_alpha_initial_state_is_consistent_across_targets_and_shadow_replay() {
    for _ in 0..2 {
        let mut channel = three_d_channel();
        for target in 0..=8 {
            let (register, method) = if target == 8 {
                (
                    channel
                        .three_d()
                        .fixed_function()
                        .register(MaxwellThreeDFixedFunctionRegister::BlendSeparateAlpha),
                    0x133c,
                )
            } else {
                (
                    &channel.three_d().fixed_function().per_target_blend()[target][0],
                    0x1e00 + target as u32 * 0x20,
                )
            };
            assert_eq!(
                register.origin(),
                MaxwellThreeDRegisterOrigin::VerifiedReset
            );
            assert_eq!(register.raw(), Some(1));
            assert_eq!(
                register.value(),
                Some(&MaxwellThreeDFixedFunctionValue::Boolean(true))
            );
            assert_eq!(register.source(), None);
            let raw = channel
                .three_d_mut()
                .raw_register(GpuMethodId(method))
                .unwrap();
            assert_eq!(raw.origin(), MaxwellThreeDRegisterOrigin::VerifiedReset);
            assert_eq!(raw.value(), Some(&1));
            assert_eq!(raw.source(), None);
        }
        // Replay an unprogrammed selector: the supplied invalid argument must
        // be replaced by the initial shadow value for common and per-RT state.
        program_three_d(&mut channel, 0x124, 3);
        program_three_d(&mut channel, 0x133c, u32::MAX);
        assert_eq!(
            channel
                .three_d()
                .fixed_function()
                .register(MaxwellThreeDFixedFunctionRegister::BlendSeparateAlpha)
                .raw(),
            Some(1)
        );
        for target in 0..8 {
            program_three_d(&mut channel, 0x1e00 + target * 0x20, u32::MAX);
            assert_eq!(
                channel.three_d().fixed_function().per_target_blend()[target as usize][0].raw(),
                Some(1)
            );
        }
        // Passthrough changes live state, not the initial shadow value.
        program_three_d(&mut channel, 0x124, 2);
        program_three_d(&mut channel, 0x1e00, 0);
        program_three_d(&mut channel, 0x124, 3);
        program_three_d(&mut channel, 0x1e00, 0);
        assert_eq!(
            channel.three_d().fixed_function().per_target_blend()[0][0].raw(),
            Some(1)
        );
        // A tracked write must subsequently override that initial value.
        program_three_d(&mut channel, 0x124, 0);
        program_three_d(&mut channel, 0x1e00, 0);
        program_three_d(&mut channel, 0x124, 3);
        program_three_d(&mut channel, 0x1e00, 1);
        assert_eq!(
            channel.three_d().fixed_function().per_target_blend()[0][0].raw(),
            Some(0)
        );
    }
}
