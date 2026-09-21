use super::*;
use crate::abi::HostAbi;
use cranelift_codegen::{
    Context,
    control::ControlPlane,
    cursor::{Cursor, FuncCursor},
    ir::{self, InstBuilder, MemFlagsData, types},
    isa::{self, CallConv},
    settings::{self, Configurable},
};
use output::{Metadata, Relocation};
use std::os::fd::AsRawFd;

#[test]
fn background_usage_snapshot_defers_instead_of_waiting_for_allocator() {
    let cache = Cache::new().unwrap();
    let state = cache.state.lock().unwrap();
    assert_eq!(cache.try_usage().unwrap(), None);
    let expected = state.usage;
    drop(state);
    assert_eq!(cache.try_usage().unwrap(), Some(expected));
}

#[test]
fn inline_bridge_tail_uses_no_island_and_returns_failed_storage() {
    let cache = Cache::new().unwrap();
    let target = cache
        .install(return_value(91), Tier::Lcq, |_| None)
        .unwrap();
    let (bytes, tail) = crate::native::link::bridge(host(), &[]);
    let mut bytes = bytes.into_vec();
    bytes.resize(tail + 16, 0);
    let installed = cache
        .install_with_inline_branch(
            output(host(), bytes),
            Tier::Lcq,
            tail,
            target.allocation.address(),
        )
        .unwrap();
    assert_eq!(installed.allocation.island_count(), 0);
    assert_eq!(permissions(installed.allocation.address()), "r-xs");
    assert_eq!(unsafe { execute(&installed) }, 91);
    let address = installed.allocation.address();
    drop(installed);
    let before = cache.usage().unwrap();
    for offset in [1, 8, usize::MAX] {
        assert!(matches!(
            cache.install_with_inline_branch(
                output(host(), vec![0; 16]),
                Tier::Lcq,
                offset,
                target.allocation.address()
            ),
            Err(Error::Output(_))
        ));
        assert_eq!(cache.usage().unwrap(), before);
        let reused = cache.allocate_with_islands(32, 16, Tier::Lcq, 0).unwrap();
        assert_eq!(reused.address(), address);
        assert_eq!(reused.island_count(), 0);
    }
}

#[test]
fn final_bridge_tail_executes_with_closed_write_view_and_rejects_invalid_extents() {
    let cache = Cache::new().unwrap();
    let target = cache
        .install(return_value(73), Tier::Lcq, |_| None)
        .unwrap();
    let transfer = match host() {
        HostAbi::X86_64 => vec![0x90],
        HostAbi::Aarch64 => 0xd503201fu32.to_le_bytes().to_vec(),
    };
    let (bytes, tail) = crate::native::link::bridge(host(), &transfer);
    let bridge = cache
        .install_with_branch(
            output(host(), bytes.into_vec()),
            Tier::Lcq,
            tail,
            target.allocation.address(),
        )
        .unwrap();
    let address = bridge.allocation.address();
    let island = bridge.allocation.island_address(0);
    let rw = cache
        .lock()
        .unwrap()
        .backing
        .as_ref()
        .unwrap()
        .rw
        .as_ref()
        .unwrap()
        .base
        .as_ptr() as usize
        + address
        - cache.base;
    assert_eq!(permissions(address), "r-xs");
    assert_eq!(permissions(rw), "---s");
    assert_eq!(unsafe { execute(&bridge) }, 73);
    drop(bridge);
    let before = cache.usage().unwrap();
    for invalid in [1, 8, usize::MAX] {
        assert!(matches!(
            cache.install_with_branch(
                output(host(), vec![0; 8]),
                Tier::Lcq,
                invalid,
                target.allocation.address()
            ),
            Err(Error::Output(_))
        ));
        assert_eq!(cache.usage().unwrap(), before);
        let reuse = cache.allocate_with_islands(8, 16, Tier::Lcq, 1).unwrap();
        assert_eq!(reuse.address(), address);
        assert_eq!(reuse.island_address(0), island);
    }
}

#[test]
fn static_branch_bytes_execute_through_owned_segment_islands() {
    use crate::native::link;

    let cache = Cache::new().unwrap();
    let mut previous_island = None;
    for value in [17, 93] {
        let target = cache
            .install(return_value(value), Tier::Lcq, |_| None)
            .unwrap();
        let source = cache.allocate_with_islands(16, 16, Tier::Lcq, 1).unwrap();
        let source_pc = source.address();
        let island_pc = source.island_address(0).unwrap();
        if let Some(previous) = previous_island {
            assert_eq!(island_pc, previous);
        }
        previous_island = Some(island_pc);

        // Obtain the actual far-target payload using an artificial source PC:
        // x86's single 2047 MiB cache normally needs no far static branches.
        // The executable source below deliberately routes to its reserved slot
        // to exercise those bytes on both hosts, without fixed-address mmap.
        let far_pc = target.allocation.address() as u64 + (1 << 32);
        let far = link::emit(
            host(),
            far_pc,
            target.allocation.address() as u64,
            far_pc + 16,
        )
        .unwrap();
        let island_bytes = far.island.unwrap();
        let patch = link::emit(
            host(),
            source_pc as u64 + 8,
            island_pc as u64,
            island_pc as u64,
        )
        .unwrap();
        assert!(patch.island.is_none());
        let mut bytes = return_value(0).bytes.into_vec();
        bytes.resize(16, 0);
        // Host-call landing, padding, then aligned static source patch.
        match host() {
            HostAbi::X86_64 => bytes[4..8].fill(0x90),
            HostAbi::Aarch64 => bytes[4..8].copy_from_slice(&0xd503201f_u32.to_le_bytes()),
        }
        bytes[8..8 + patch.patch().len()].copy_from_slice(patch.patch());

        let rw_address;
        {
            // Both spans are unpublished and test-owned. This is NOT a live
            // patch API; no dispatch root or other reader can reach the source.
            let state = cache.lock().unwrap();
            let rw = state.backing.as_ref().unwrap().rw.as_ref().unwrap();
            rw_address = rw.base.as_ptr() as usize;
            rw.protect(0, SEGMENT_BYTES, libc::PROT_READ | libc::PROT_WRITE)
                .unwrap();
            assert_eq!(permissions(source_pc), "r-xs");
            assert_eq!(permissions(rw_address), "rw-s");
            unsafe {
                linux::copy(
                    rw.base.as_ptr().add(island_pc - cache.base),
                    island_pc as *const u8,
                    &island_bytes,
                );
                linux::copy(
                    rw.base.as_ptr().add(source_pc - cache.base),
                    source_pc as *const u8,
                    &bytes,
                );
            }
            rw.protect(0, SEGMENT_BYTES, libc::PROT_NONE).unwrap();
        }
        wasmtime_internal_jit_icache_coherence::pipeline_flush_mt().unwrap();
        assert_eq!(permissions(rw_address), "---s");
        assert_eq!(permissions(island_pc), "r-xs");
        // Source and target allocations stay alive through the complete call.
        // JMP/BR must preserve the host return continuation (no extra call).
        let function: unsafe extern "C" fn() -> u32 = unsafe { std::mem::transmute(source_pc) };
        assert_eq!(unsafe { function() }, u32::from(value));
        drop(source);
        drop(target);
    }
}

#[test]
fn code_and_islands_share_one_reusable_span_without_a_slot_ceiling() {
    let cache = Cache::new().unwrap();
    let source = Arc::new(
        cache
            .install_with_islands(return_value(17), Tier::Lcq, 4096, |_| None)
            .unwrap(),
    );
    let address = source.allocation.address();
    let island = source.allocation.island_address(0).unwrap();
    assert_eq!(source.allocation.segment, 0);
    assert_eq!(
        island,
        address + source.allocation.len().next_multiple_of(16)
    );
    assert_eq!(source.allocation.island_count(), 4096);
    assert_eq!(source.allocation.island_address(4096), None);
    assert_eq!(source.allocation.island_address(usize::MAX), None);
    assert_eq!(unsafe { execute(&source) }, 17);
    // Previously this forced another 16 MiB segment despite ample code space.
    let next = cache
        .install_with_islands(return_value(23), Tier::Lcq, 4096, |_| None)
        .unwrap();
    assert_eq!(next.allocation.segment, 0);
    assert!(next.allocation.address() >= island + 4096 * 16);
    assert_eq!(cache.usage().unwrap().committed, SEGMENT_BYTES);
    assert_eq!(unsafe { execute(&next) }, 23);
    let retained = source.clone();
    drop(source);
    assert!(!unsafe { cache.decommit_empty(0) }.unwrap());
    drop(retained);
    let reused = cache
        .install_with_islands(return_value(41), Tier::Lcq, 4096, |_| None)
        .unwrap();
    assert_eq!(reused.allocation.address(), address);
    assert_eq!(reused.allocation.island_address(0), Some(island));
    assert_eq!(unsafe { execute(&reused) }, 41);
    drop(reused);
    drop(next);
    assert!(unsafe { cache.decommit_empty(0) }.unwrap());
    assert_eq!(cache.usage().unwrap().committed, 0);
}

#[test]
fn code_island_padding_and_extent_are_checked_before_allocation() {
    let cache = Cache::new().unwrap();
    let before = cache.usage().unwrap();
    for count in [usize::MAX, SEGMENT_BYTES / 16] {
        assert!(cache.allocate_with_islands(1, 1, Tier::Lcq, count).is_err());
        assert_eq!(cache.usage().unwrap(), before);
    }
    let allocation = cache.allocate_with_islands(17, 1, Tier::Lcq, 2).unwrap();
    assert_eq!(allocation.len(), 17);
    assert_eq!(
        allocation.island_address(0),
        Some(allocation.address() + 32)
    );
    assert_eq!(
        allocation.island_address(1),
        Some(allocation.address() + 48)
    );
    let next = cache.allocate(1, 1, Tier::Lcq).unwrap();
    assert_eq!(next.address(), allocation.address() + 64);
    drop(allocation);
    let reused = cache.allocate_with_islands(17, 1, Tier::Lcq, 2).unwrap();
    assert_eq!(reused.address() + 64, next.address());
}

#[test]
fn write_permissions_cover_only_the_owned_spans_host_pages() {
    let cache = Cache::new().unwrap();
    let pages = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
    let allocation = cache
        .allocate_with_islands(pages - 8, 16, Tier::Lcq, 2)
        .unwrap();
    let state = cache.lock().unwrap();
    let backing = state.backing.as_ref().unwrap();
    let offset = allocation.segment * SEGMENT_BYTES + allocation.span.start;
    let (start, len) = backing.write_window(offset, allocation.span.len);
    assert_eq!((start, len), (0, 2 * pages)); // The trailing islands cross a page.
    for (offset, len, expected) in [
        (pages + 16, 16, (pages, pages)),
        (2 * pages - 1, 2, (pages, 2 * pages)),
        (WINDOW_BYTES - 16, 16, (WINDOW_BYTES - pages, pages)),
    ] {
        assert_eq!(backing.write_window(offset, len), expected);
    }
    let rw = backing.rw.as_ref().unwrap();
    rw.protect(start, len, libc::PROT_READ | libc::PROT_WRITE)
        .unwrap();
    assert_eq!(permissions(rw.base.as_ptr() as usize), "rw-s");
    assert_eq!(permissions(rw.base.as_ptr() as usize + len), "---s");
    assert_eq!(permissions(allocation.address()), "r-xs");
    rw.protect(start, len, libc::PROT_NONE).unwrap();
    drop(state);
}

#[test]
fn island_capacity_failure_and_tier_separation_do_not_lose_reservations() {
    let cache = Cache::new().unwrap();
    let before = cache.usage().unwrap();
    assert!(matches!(
        cache.install_with_islands(return_value(1), Tier::Lcq, SEGMENT_BYTES / 16, |_| None),
        Err(Error::Capacity(_))
    ));
    assert_eq!(cache.usage().unwrap(), before);
    let lcq = cache
        .install_with_islands(return_value(2), Tier::Lcq, 65, |_| None)
        .unwrap();
    let hcq = cache
        .install_with_islands(return_value(3), Tier::Hcq, 65, |_| None)
        .unwrap();
    assert_ne!(lcq.allocation.segment, hcq.allocation.segment);
    let address = lcq.allocation.island_address(0);
    drop(lcq);
    let borrowed = cache
        .install_with_islands(return_value(4), Tier::Hcq, 65, |_| None)
        .unwrap();
    assert_eq!(borrowed.allocation.island_address(0), address);
    assert_eq!(unsafe { execute(&borrowed) }, 4);
}

#[test]
fn final_short_segment_islands_remain_inside_the_reserved_window() {
    let cache = Cache::new().unwrap();
    // Restrict the synthetic allocator state without committing 127 segments
    // (which would exceed the real budget). No code or mapping uses those bumps.
    {
        let mut state = cache.lock().unwrap();
        for segment in &mut state.segments[..SEGMENTS - 1] {
            segment.bump = SEGMENT_BYTES;
        }
        state.segments[SEGMENTS - 1].bump = segment_size(SEGMENTS - 1) - (16 + 4096 * 16);
    }
    let code = cache
        .install_with_islands(return_value(42), Tier::Lcq, 4096, |_| None)
        .unwrap();
    assert_eq!(code.allocation.segment, SEGMENTS - 1);
    assert_eq!(
        code.allocation.island_address(0),
        Some(code.allocation.address() + code.allocation.len().next_multiple_of(16))
    );
    assert_eq!(
        code.allocation.island_address(4096 - 1).unwrap() + 16,
        cache.base + WINDOW_BYTES
    );
    assert_eq!(unsafe { execute(&code) }, 42);
    let generation = code.allocation.generation;
    drop(code);
    assert!(unsafe { cache.decommit_empty(SEGMENTS - 1) }.unwrap());
    let replacement = cache
        .install_with_islands(return_value(43), Tier::Lcq, 4096, |_| None)
        .unwrap();
    assert_ne!(replacement.allocation.generation, generation);
    assert_eq!(replacement.allocation.island_count(), 4096);
    assert_eq!(unsafe { execute(&replacement) }, 43);
}

fn host() -> HostAbi {
    if cfg!(target_arch = "x86_64") {
        HostAbi::X86_64
    } else {
        HostAbi::Aarch64
    }
}

fn output(abi: HostAbi, bytes: Vec<u8>) -> Output {
    Output {
        bytes: bytes.into_boxed_slice(),
        alignment: 16,
        metadata: Metadata {
            abi,
            frame_extent: crate::abi::TRANSFER_BYTES,
            entries: Box::new([]),
            states: Box::new([]),
            faults: Box::new([]),
            traps: Box::new([]),
            relocations: Box::new([]),
        },
    }
}

fn return_value(value: u8) -> Output {
    let bytes = match host() {
        HostAbi::X86_64 => vec![0xf3, 0x0f, 0x1e, 0xfa, 0xb8, value, 0, 0, 0, 0xc3],
        HostAbi::Aarch64 => [
            0xd503245f_u32,
            0x52800000 | (u32::from(value) << 5),
            0xd65f03c0,
        ]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect(),
    };
    output(host(), bytes)
}

pub(super) fn permissions(address: usize) -> String {
    let maps = std::fs::read_to_string("/proc/self/maps").unwrap();
    for line in maps.lines() {
        let mut fields = line.split_whitespace();
        let range = fields.next().unwrap();
        let (start, end) = range.split_once('-').unwrap();
        if (usize::from_str_radix(start, 16).unwrap()..usize::from_str_radix(end, 16).unwrap())
            .contains(&address)
        {
            return fields.next().unwrap().to_owned();
        }
    }
    panic!("address {address:#x} is not mapped");
}

unsafe fn execute(code: &Installed) -> u32 {
    // This fixture is a host System-ABI leaf, not a native guest unit. Its
    // immutable lease stays alive through the call; no JITModule is involved.
    let function: unsafe extern "C" fn() -> u32 =
        unsafe { std::mem::transmute(code.allocation.address()) };
    unsafe { function() }
}

#[test]
fn synthetic_code_executes_from_rx_with_the_write_view_closed() {
    let cache = Cache::new().unwrap();
    let initial = cache.usage().unwrap();
    let code = cache
        .install(return_value(42), Tier::Lcq, |_| None)
        .unwrap();
    assert_eq!(unsafe { execute(&code) }, 42);
    assert!(permissions(code.allocation.address()).starts_with("r-x"));
    let rw = cache
        .lock()
        .unwrap()
        .backing
        .as_ref()
        .unwrap()
        .rw
        .as_ref()
        .unwrap()
        .base
        .as_ptr() as usize;
    assert!(permissions(rw).starts_with("---"));
    assert_eq!(cache.usage().unwrap().committed, SEGMENT_BYTES);
    assert!(cache.usage().unwrap().metadata > initial.metadata);
    for line in std::fs::read_to_string("/proc/self/maps")
        .unwrap()
        .lines()
        .filter(|line| line.contains("memfd:nixe-jit"))
    {
        let mode = line.split_whitespace().nth(1).unwrap();
        assert!(!(mode.contains('w') && mode.contains('x')), "{line}");
    }
    drop(code);
    // Free spans do not pretend already-committed backing has been released.
    assert_eq!(cache.usage().unwrap().committed, SEGMENT_BYTES);
    assert!(unsafe { cache.decommit_empty(0) }.unwrap());
    assert_eq!(cache.usage().unwrap(), initial);
}

#[test]
fn individual_spans_reuse_and_empty_segments_release_real_backing() {
    let cache = Cache::new().unwrap();
    let first = cache.install(return_value(1), Tier::Lcq, |_| None).unwrap();
    let address = first.allocation.address();
    let generation = first.allocation.generation;
    let neighbor = cache.install(return_value(2), Tier::Lcq, |_| None).unwrap();
    assert!(!unsafe { cache.decommit_empty(0) }.unwrap());
    drop(first);
    let replacement = cache.install(return_value(3), Tier::Lcq, |_| None).unwrap();
    assert_eq!(replacement.allocation.address(), address);
    assert_eq!(unsafe { execute(&replacement) }, 3);
    assert_eq!(unsafe { execute(&neighbor) }, 2);
    drop((replacement, neighbor));
    let blocks = || {
        let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
        assert_eq!(
            unsafe {
                libc::fstat(
                    cache
                        .lock()
                        .unwrap()
                        .backing
                        .as_ref()
                        .unwrap()
                        .fd
                        .as_raw_fd(),
                    stat.as_mut_ptr(),
                )
            },
            0
        );
        unsafe { stat.assume_init().st_blocks }
    };
    assert!(blocks() > 0);
    assert!(unsafe { cache.decommit_empty(0) }.unwrap());
    assert_eq!(blocks(), 0);
    assert!(permissions(address).starts_with("---"));
    let reused = cache.install(return_value(4), Tier::Lcq, |_| None).unwrap();
    assert_eq!(reused.allocation.address(), address);
    assert_ne!(reused.allocation.generation, generation);
    let expected = return_value(4);
    assert_eq!(
        unsafe { std::slice::from_raw_parts(address as *const u8, expected.bytes.len()) },
        &*expected.bytes
    );
    assert_eq!(unsafe { execute(&reused) }, 4);
}

#[test]
fn best_fit_then_lowest_address_and_coalescing_match_policy() {
    let cache = Cache::new().unwrap();
    let a = cache.allocate(80, 16, Tier::Lcq).unwrap();
    let b = cache.allocate(16, 16, Tier::Lcq).unwrap();
    let c = cache.allocate(48, 16, Tier::Lcq).unwrap();
    let d = cache.allocate(16, 16, Tier::Lcq).unwrap();
    let e = cache.allocate(80, 16, Tier::Lcq).unwrap();
    let f = cache.allocate(16, 16, Tier::Lcq).unwrap();
    let (low, small) = (a.address(), c.address());
    drop((a, c, e));
    let fit = cache.allocate(40, 16, Tier::Lcq).unwrap();
    assert_eq!(fit.address(), small);
    let tie = cache.allocate(60, 16, Tier::Lcq).unwrap();
    assert_eq!(tie.address(), low);
    drop((b, d, f, fit, tie));
    let state = cache.lock().unwrap();
    assert_eq!(
        (
            state.segments[0].bump,
            state.segments[0].free_len,
            state.segments[0].live
        ),
        (0, 0, 0)
    );
}

#[test]
fn alignment_padding_and_fragmentation_are_reusable() {
    let cache = Cache::new().unwrap();
    let a = cache.allocate(3, 1, Tier::Lcq).unwrap();
    let b = cache.allocate(19, 4096, Tier::Lcq).unwrap();
    assert_eq!(b.address() % 4096, 0);
    let gap = cache.allocate(16, 16, Tier::Lcq).unwrap();
    assert!(a.address() < gap.address() && gap.address() < b.address());
    drop((a, b, gap));
    assert_eq!(cache.lock().unwrap().segments[0].bump, 0);
    assert!(cache.allocate(1, 3, Tier::Lcq).is_err());
    let mut malformed = return_value(1);
    malformed.alignment = 3;
    assert!(cache.install(malformed, Tier::Lcq, |_| None).is_err());
}

#[test]
fn address_alignment_preserves_exact_overflow_boundary() {
    for alignment in [1, 2, 16, 4096, SEGMENT_BYTES] {
        assert_eq!(align(0, alignment).unwrap(), 0);
        assert_eq!(align(alignment, alignment).unwrap(), alignment);
        let last = usize::MAX & !(alignment - 1);
        assert_eq!(align(last, alignment).unwrap(), last);
        if alignment > 1 {
            assert_eq!(align(1, alignment).unwrap(), alignment);
            assert!(matches!(
                align(last + 1, alignment),
                Err(Error::Capacity("aligned address overflow"))
            ));
        }
    }
}

#[test]
fn tiers_do_not_share_live_segments_but_borrow_empty_ones() {
    let cache = Cache::new().unwrap();
    let lcq = cache.allocate(16, 16, Tier::Lcq).unwrap();
    let hcq = cache.allocate(16, 16, Tier::Hcq).unwrap();
    assert_ne!(lcq.segment, hcq.segment);
    let first_segment = lcq.segment;
    drop((lcq, hcq));
    let borrowed = cache.allocate(16, 16, Tier::Hcq).unwrap();
    assert_eq!(borrowed.segment, first_segment);
}

#[test]
fn reservation_bounds_and_islands_exclude_unallocatable_bytes() {
    let cache = Cache::new().unwrap();
    assert_eq!(
        (SEGMENTS - 1) * SEGMENT_BYTES + segment_size(127),
        WINDOW_BYTES
    );
    assert_eq!(segment_size(127), 15 * MIB);
    let base = cache.executable_base();
    assert_eq!(cache.segment_for_pc(base - 1), None);
    assert_eq!(cache.segment_for_pc(base), Some(0));
    assert_eq!(cache.segment_for_pc(base + WINDOW_BYTES - 1), Some(127));
    assert_eq!(cache.segment_for_pc(base + WINDOW_BYTES), None);
    assert_eq!(cache.segment_for_pc(usize::MAX), None);
    let whole = cache.allocate(SEGMENT_BYTES, 16, Tier::Lcq).unwrap();
    let next = cache.allocate(16, 16, Tier::Lcq).unwrap();
    assert_eq!((whole.segment, next.segment), (0, 1));
    assert!(whole.address() + whole.len() <= base + SEGMENT_BYTES);
    assert!(cache.allocate(SEGMENT_BYTES + 1, 16, Tier::Lcq).is_err());
}

#[test]
fn budget_charges_overlap_and_preserves_lcq_capacity() {
    let usage = Usage {
        committed: 600 * MIB,
        metadata: 8 * MIB,
    };
    assert!(usage.check(1, Tier::Hcq).is_err());
    assert!(usage.check(LCQ_RESERVE, Tier::Lcq).is_ok());
    assert!(usage.check(LCQ_RESERVE + 1, Tier::Lcq).is_err());
    assert!(usage.needs_reclamation());
    assert!(
        Usage {
            committed: SOFT_BYTES,
            metadata: 0
        }
        .check(1, Tier::Hcq)
        .is_err()
    );
    let cache = Cache::new().unwrap();
    let initial = cache.usage().unwrap();
    let old = cache.charge_metadata(1024, Tier::Lcq).unwrap();
    let new = cache.charge_metadata(2048, Tier::Lcq).unwrap();
    assert_eq!(cache.usage().unwrap().metadata, initial.metadata + 3072);
    drop(old);
    assert_eq!(cache.usage().unwrap().metadata, initial.metadata + 2048);
    drop(new);
    assert_eq!(cache.usage().unwrap(), initial);
}

#[test]
fn relocation_failure_returns_the_unpublished_span_and_metadata() {
    let cache = Cache::new().unwrap();
    let warmup = cache
        .allocate_with_islands(16, 16, Tier::Lcq, 4096)
        .unwrap();
    let address = warmup.address();
    let island = warmup.island_address(0);
    drop(warmup);
    let initial = cache.usage().unwrap();
    let mut bad = output(host(), vec![0; 16]);
    bad.metadata.relocations = vec![Relocation {
        offset: 0,
        kind: Reloc::Abs8,
        target: Target::User {
            namespace: 1,
            index: 2,
        },
        addend: 0,
    }]
    .into_boxed_slice();
    assert!(matches!(
        cache.install_with_islands(bad, Tier::Lcq, 4096, |_| None),
        Err(Error::Relocation { .. })
    ));
    assert_eq!(cache.usage().unwrap(), initial);
    let code = cache
        .install_with_islands(return_value(5), Tier::Lcq, 4096, |_| None)
        .unwrap();
    assert_eq!(code.allocation.address(), address);
    assert_eq!(code.allocation.island_address(0), island);
    assert_eq!(unsafe { execute(&code) }, 5);
}

#[test]
fn validation_releases_backend_storage_and_only_its_exact_charge() {
    use cranelift_codegen::nixe::{LocatedValue, Location, StateMap};
    let cache = Cache::new().unwrap();
    let mut output = return_value(29);
    output.metadata.states = Box::new([StateMap {
        id: 1,
        offset: 0,
        entry: true,
        patch_bytes: 0,
        fault_bytes: 0,
        poll: None,
        values: vec![
            LocatedValue {
                ty: cranelift_codegen::ir::types::I64,
                location: Location::Spill {
                    offset: crate::abi::TRANSFER_BYTES
                },
            };
            32
        ],
    }]);
    let heap = output.metadata.bytes();
    let mut installed = cache.install(output, Tier::Lcq, |_| None).unwrap();
    let before = cache.usage().unwrap();
    let address = installed.allocation.address();
    installed.finish_validation();
    let after = cache.usage().unwrap();
    assert_eq!(after.committed, before.committed);
    assert_eq!(before.metadata - after.metadata, heap);
    assert!(installed.proofs.is_none());
    assert_eq!(installed.allocation.address(), address);
    assert_eq!(unsafe { execute(&installed) }, 29);
    installed.finish_validation();
    assert_eq!(cache.usage().unwrap(), after);
}

#[test]
fn local_absolute_relocations_use_rx_not_rw_addresses() {
    let cache = Cache::new().unwrap();
    let mut data = output(host(), vec![0; 16]);
    data.metadata.relocations = vec![Relocation {
        offset: 0,
        kind: Reloc::Abs8,
        target: Target::Local(8),
        addend: 3,
    }]
    .into_boxed_slice();
    let code = cache
        .install(data, Tier::Lcq, |_| {
            panic!("local target must not call resolver")
        })
        .unwrap();
    let stored = unsafe { (code.allocation.address() as *const u64).read_unaligned() };
    assert_eq!(stored as usize, code.allocation.address() + 11);
}

#[test]
fn relocation_ranges_alignment_and_instruction_fields_are_checked() {
    let mut x86 = output(HostAbi::X86_64, vec![0; 8]);
    x86.metadata.relocations = vec![Relocation {
        offset: 0,
        kind: Reloc::X86CallPCRel4,
        target: Target::User {
            namespace: 1,
            index: 2,
        },
        addend: -4,
    }]
    .into_boxed_slice();
    x86.relocate(0x1000, |_| Some(0x2000)).unwrap();
    assert_eq!(
        i32::from_le_bytes(x86.bytes[..4].try_into().unwrap()),
        0xffc
    );
    assert!(x86.relocate(0x1000, |_| Some(usize::MAX)).is_err());
    x86.metadata.relocations[0].kind = Reloc::X86GOTPCRel4;
    assert!(x86.relocate(0x1000, |_| Some(0x2000)).is_err());
    let mut arm = output(HostAbi::Aarch64, 0x97ff_ffffu32.to_le_bytes().to_vec());
    arm.metadata.relocations = vec![Relocation {
        offset: 0,
        kind: Reloc::Arm64Call,
        target: Target::User {
            namespace: 0,
            index: 0,
        },
        addend: 0,
    }]
    .into_boxed_slice();
    arm.relocate(0x1000, |_| Some(0x1004)).unwrap();
    assert_eq!(
        u32::from_le_bytes(arm.bytes[..4].try_into().unwrap()),
        0x94000001
    );
    assert!(arm.relocate(0x1000, |_| Some(0x1002)).is_err());
    assert!(arm.relocate(0x1000, |_| Some(0x1000 + (1 << 27))).is_err());
    arm.metadata.relocations[0].offset = 2;
    assert!(arm.relocate(0x1000, |_| Some(0x1004)).is_err());
}

#[test]
fn adrp_handles_the_full_signed_four_gibibyte_range_and_replaces_immediates() {
    let mut arm = output(
        HostAbi::Aarch64,
        [0x90000000u32, 0x91000000]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect(),
    );
    arm.metadata.relocations = vec![
        Relocation {
            offset: 0,
            kind: Reloc::Aarch64AdrPrelPgHi21,
            target: Target::User {
                namespace: 0,
                index: 0,
            },
            addend: 0,
        },
        Relocation {
            offset: 4,
            kind: Reloc::Aarch64AddAbsLo12Nc,
            target: Target::User {
                namespace: 0,
                index: 0,
            },
            addend: 0,
        },
    ]
    .into_boxed_slice();
    let base = 1usize << 33;
    for target in [base + (1 << 32) - 1, base - (1 << 32), base + 0xabc] {
        arm.relocate(base, |_| Some(target)).unwrap();
        let page = u32::from_le_bytes(arm.bytes[..4].try_into().unwrap());
        let immediate = ((page >> 29) & 3) | (((page >> 5) & 0x7ffff) << 2);
        let signed = ((immediate << 11) as i32) >> 11;
        assert_eq!(
            (base as i128) + ((signed as i128) << 12),
            (target & !4095) as i128
        );
        let low = u32::from_le_bytes(arm.bytes[4..].try_into().unwrap());
        assert_eq!((low >> 10) & 4095, target as u32 & 4095);
    }
    assert!(arm.relocate(base, |_| Some(base + (1 << 32))).is_err());
}

#[test]
fn backend_output_survives_context_reset_with_exact_labels_and_fault_maps() {
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        for allocator in ["single_pass", "backtracking"] {
            let mut function = ir::Function::new();
            let block = function.dfg.make_block();
            function.layout.append_block(block);
            let signature = function.import_signature(ir::Signature::new(CallConv::SystemV));
            let mut cursor = FuncCursor::new(&mut function).at_bottom(block);
            cursor.ins().nixe_entry(signature, 10);
            let address = cursor.ins().get_pinned_reg(types::I64);
            let name = cursor
                .func
                .declare_imported_user_function(ir::UserExternalName::new(42, 7));
            let external = cursor.func.import_function(ir::ExtFuncData {
                name: ir::ExternalName::user(name),
                signature,
                colocated: false,
                patchable: false,
            });
            let external_address = cursor.ins().func_addr(types::I64, external);
            cursor
                .ins()
                .store(MemFlagsData::new(), external_address, address, 16);
            cursor.ins().nixe_fault_start(30, &[]);
            let value = cursor
                .ins()
                .load(types::I64, MemFlagsData::new(), address, 0);
            cursor.ins().store(MemFlagsData::new(), value, address, 8);
            cursor.ins().nixe_fault_end(30, &[]);
            cursor.ins().nixe_exit(20, &[]);
            cranelift_codegen::nixe::set_entries(&mut function, &[block]).unwrap();
            let mut flags = settings::builder();
            for (name, value) in [
                ("enable_nixe_abi", "true"),
                ("enable_pinned_reg", "true"),
                ("regalloc_algorithm", allocator),
                ("opt_level", "none"),
            ] {
                flags.set(name, value).unwrap();
            }
            let triple = if abi == HostAbi::X86_64 {
                "x86_64-unknown-linux-gnu"
            } else {
                "aarch64-unknown-linux-gnu"
            };
            let isa = isa::lookup(triple.parse().unwrap())
                .unwrap()
                .finish(settings::Flags::new(flags))
                .unwrap();
            let mut context = Context::for_function(function);
            context
                .compile(&*isa, &mut ControlPlane::default())
                .unwrap();
            let code = context.take_compiled_code().unwrap();
            let expected = code.code_buffer().to_vec();
            let offsets = code.buffer.nixe_entries.clone();
            let faults = code.buffer.nixe_faults.clone();
            let owned = Output::from_backend(abi, code, &context.func).unwrap();
            context.clear();
            assert_eq!(&*owned.bytes, &expected);
            assert_eq!(&*owned.metadata.entries, &offsets);
            assert_eq!(owned.metadata.faults.len(), 2);
            assert_eq!(&*owned.metadata.faults, &faults);
            assert!(!owned.metadata.states.is_empty());
            assert!(!owned.metadata.relocations.is_empty());
            assert!(
                owned
                    .metadata
                    .relocations
                    .iter()
                    .any(|relocation| relocation.target
                        == Target::User {
                            namespace: 42,
                            index: 7
                        })
            );
            assert!(owned.metadata.frame_extent <= crate::abi::SPILL_BYTES);
            let cache = Cache::new().unwrap();
            let symbol = cache.executable_base() + 0x12340;
            let installed = cache
                .install(owned, Tier::Lcq, |target| {
                    (*target
                        == Target::User {
                            namespace: 42,
                            index: 7,
                        })
                    .then_some(symbol)
                })
                .unwrap();
            assert_eq!(&*installed.proofs.as_ref().unwrap().entries, &offsets);
            assert_eq!(&*installed.proofs.as_ref().unwrap().faults, &faults);
            assert!(permissions(installed.allocation.address()).starts_with("r-x"));
        }
    }
}

#[test]
fn span_churn_never_overlaps_live_allocations_and_returns_to_empty() {
    let cache = Cache::new().unwrap();
    let mut slots: [Option<Allocation>; 32] = std::array::from_fn(|_| None);
    let mut random = 1u64;
    for _ in 0..2000 {
        random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
        let index = (random >> 32) as usize % slots.len();
        if slots[index].take().is_some() {
            continue;
        }
        let length = (random as usize % 1024) + 1;
        let alignment = 1 << (4 + (random >> 48) as usize % 9);
        let allocation = cache.allocate(length, alignment, Tier::Lcq).unwrap();
        assert_eq!(allocation.address() % alignment, 0);
        for other in slots.iter().flatten() {
            assert!(
                allocation.address() + allocation.len() <= other.address()
                    || other.address() + other.len() <= allocation.address()
            );
        }
        slots[index] = Some(allocation);
    }
    drop(slots);
    let state = cache.lock().unwrap();
    assert!(
        state
            .segments
            .iter()
            .all(|segment| segment.live == 0 && segment.bump == 0 && segment.free_len == 0)
    );
}

#[test]
fn hard_limit_rejection_does_not_commit_a_segment_or_consume_reserve() {
    let cache = Cache::new().unwrap();
    let initial = cache.usage().unwrap();
    // Exercise the accounting authority without physically allocating 640 MiB.
    let charge = cache
        .charge_metadata(HARD_BYTES - initial.total(), Tier::Lcq)
        .unwrap();
    assert!(cache.install(return_value(1), Tier::Lcq, |_| None).is_err());
    assert_eq!(cache.usage().unwrap().committed, 0);
    assert_eq!(cache.usage().unwrap().total(), HARD_BYTES);
    drop(charge);
    assert_eq!(cache.usage().unwrap(), initial);
}

#[test]
fn code_is_coherent_when_published_and_reused_on_another_thread() {
    let cache = Cache::new().unwrap();
    let (send, receive) = std::sync::mpsc::sync_channel::<Installed>(0);
    let (done, acknowledged) = std::sync::mpsc::sync_channel(0);
    std::thread::scope(|scope| {
        scope.spawn(move || {
            for expected in 0..32 {
                let code = receive.recv().unwrap();
                assert_eq!(unsafe { execute(&code) }, expected);
                drop(code);
                done.send(()).unwrap();
            }
        });
        for value in 0..32 {
            let code = cache
                .install(return_value(value), Tier::Lcq, |_| None)
                .unwrap();
            send.send(code).unwrap();
            acknowledged.recv().unwrap();
        }
    });
}
