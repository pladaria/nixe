use super::*;

#[test]
fn inline_tails_use_direct_near_branches_and_embed_full_far_targets() {
    let source = 0x2_0000_0000;
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let width = if abi == HostAbi::X86_64 { 8 } else { 4 };
        for target in [source - 128, source, source + 128] {
            let bytes = inline_tail(abi, source, target).unwrap();
            assert_eq!(destination(abi, source, &bytes[..width]), target);
        }
        for target in [0, u64::MAX - 15] {
            let bytes = inline_tail(abi, source, target).unwrap();
            match abi {
                HostAbi::X86_64 => {
                    assert_eq!(&bytes[..2], &[0x49, 0xbb]);
                    assert_eq!(&bytes[2..10], &target.to_le_bytes());
                    assert_eq!(&bytes[10..13], &[0x41, 0xff, 0xe3]);
                }
                HostAbi::Aarch64 => {
                    assert_eq!(&bytes[..4], &0x58000050_u32.to_le_bytes());
                    assert_eq!(&bytes[4..8], &0xd61f0200_u32.to_le_bytes());
                    assert_eq!(&bytes[8..], &target.to_le_bytes());
                }
            }
        }
        assert!(inline_tail(abi, source + 1, 0).is_err());
        assert!(inline_tail(abi, u64::MAX - 7, 0).is_err());
    }
    assert!(inline_tail(HostAbi::Aarch64, source, 3).is_err());
}

fn destination(abi: HostAbi, source: u64, patch: &[u8]) -> u64 {
    match abi {
        HostAbi::X86_64 => {
            assert_eq!(patch.len(), 8);
            assert_eq!(patch[0], 0xe9);
            assert_eq!(&patch[5..], &[0x90; 3]);
            let delta = i32::from_le_bytes(patch[1..5].try_into().unwrap());
            (i128::from(source) + 5 + i128::from(delta)) as u64
        }
        HostAbi::Aarch64 => {
            assert_eq!(patch.len(), 4);
            let word = u32::from_le_bytes(patch.try_into().unwrap());
            assert_eq!(word & 0xfc000000, 0x14000000);
            let delta = ((word << 6) as i32 >> 6) * 4;
            (i128::from(source) + i128::from(delta)) as u64
        }
    }
}

#[test]
fn direct_branches_include_exact_positive_and_negative_limits() {
    let source = 0x2_0000_0000;
    for (abi, deltas) in [
        (
            HostAbi::X86_64,
            [5 + i64::from(i32::MIN), 5 + i64::from(i32::MAX), 0],
        ),
        (HostAbi::Aarch64, [-(1 << 27), (1 << 27) - 4, 0]),
    ] {
        for delta in deltas {
            let target = (source as i64 + delta) as u64;
            // An unused reservation does not affect the direct edge.
            let branch = emit(abi, source, target, u64::MAX).unwrap();
            assert!(branch.island.is_none());
            assert_eq!(destination(abi, source, branch.patch()), target);
        }
    }
}

#[test]
fn far_branches_use_only_the_local_island_and_preserve_the_full_target() {
    let source = 0x2_0000_0000;
    let island = source + 0xff_0000;
    for (abi, deltas) in [
        (
            HostAbi::X86_64,
            [4 + i64::from(i32::MIN), 6 + i64::from(i32::MAX)],
        ),
        (HostAbi::Aarch64, [-(1 << 27) - 4, 1 << 27]),
    ] {
        for target in deltas
            .map(|delta| (source as i64 + delta) as u64)
            .into_iter()
            .chain([0, u64::MAX - 15])
        {
            let branch = emit(abi, source, target, island).unwrap();
            assert_eq!(destination(abi, source, branch.patch()), island);
            let bytes = branch.island.unwrap();
            match abi {
                HostAbi::X86_64 => {
                    assert_eq!(&bytes[..2], &[0x49, 0xbb]);
                    assert_eq!(&bytes[2..10], &target.to_le_bytes());
                    assert_eq!(&bytes[10..], &[0x41, 0xff, 0xe3, 0x90, 0x90, 0x90]);
                }
                HostAbi::Aarch64 => {
                    assert_eq!(&bytes[..4], &0x58000050_u32.to_le_bytes());
                    assert_eq!(&bytes[4..8], &0xd61f0200_u32.to_le_bytes());
                    assert_eq!(&bytes[8..], &target.to_le_bytes());
                }
            }
            assert!(abi.reserved().link_scratch.contains(&match abi {
                HostAbi::X86_64 => 11,
                HostAbi::Aarch64 => 16,
            }));
        }
    }
}

#[test]
fn invalid_addresses_do_not_wrap_or_substitute_an_unreachable_island() {
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        assert!(emit(abi, 1, 16, 32).is_err());
        assert!(emit(abi, u64::MAX - 7, 0, 16).is_err());
        assert!(emit(abi, 0, u64::MAX - 15, 17).is_err());
        assert!(emit(abi, 0, u64::MAX - 15, u64::MAX - 15).is_err());
        assert!(emit(abi, 0, u64::MAX - 15, 0x4_0000_0000).is_err());
    }
    assert!(emit(HostAbi::Aarch64, 0, 2, 16).is_err());
    assert!(emit(HostAbi::Aarch64, 0, (1 << 27) + 2, 16).is_err());
    // AArch64's last aligned patch cannot overflow its full four-byte extent.
    assert!(emit(HostAbi::Aarch64, u64::MAX - 3, 0, 16).is_err());
    assert!(emit(HostAbi::X86_64, 0, 1, 16).is_ok());
}
