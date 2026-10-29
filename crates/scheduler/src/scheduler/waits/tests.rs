use super::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Wait {
    Mutex(GuestThreadId, u64),
    Ordered(OrderedWaitKind, u64),
}
struct Expected {
    id: GuestThreadId,
    process: ProcessId,
    priority: i32,
    order: u64,
    wait: Option<Wait>,
}
fn thread(s: &SchedulerState, process: ProcessId, priority: i32) -> GuestThreadId {
    let id = s
        .create_thread(ScheduledThreadConfig {
            process,
            base_priority: priority,
            ideal_vcpu: Some(VirtualCpuId::new(0)),
            affinity: s.profile().all_cores(),
        })
        .unwrap();
    s.make_ready(id).unwrap();
    id
}
fn scheduler() -> std::sync::Arc<SchedulerState> {
    SchedulerState::new(
        MachineSchedulerProfile::new(
            vec![crate::VirtualCpuDescriptor::new(VirtualCpuId::new(0), 0)],
            crate::PriorityRange::new(0, 63).unwrap(),
        )
        .unwrap(),
        crate::VirtualClock::new(crate::VirtualClockMode::Fixed { unix_seconds: 0 }),
    )
}
fn check_tree(state: &State, tree: &WaitTree, expected: &[GuestThreadId], seen: &mut [bool]) {
    fn visit(
        state: &State,
        index: Option<Index>,
        parent: Option<Index>,
        seen: &mut [bool],
        order: &mut Vec<GuestThreadId>,
    ) -> usize {
        let Some(index) = index else {
            return 1;
        };
        assert!(
            !std::mem::replace(&mut seen[slot_index(index)], true),
            "duplicate or cyclic wait node"
        );
        let node = state.slots[index].wait_node;
        assert_eq!(node.parent, parent);
        if node.red {
            for child in [node.left, node.right].into_iter().flatten() {
                assert!(!state.slots[child].wait_node.red, "consecutive red nodes");
            }
        }
        let left = visit(state, node.left, Some(index), seen, order);
        order.push(state.slots[index].id);
        let right = visit(state, node.right, Some(index), seen, order);
        assert_eq!(left, right, "unequal black heights");
        left + usize::from(!node.red)
    }
    if let Some(root) = tree.root {
        assert!(!state.slots[root].wait_node.red);
    }
    let mut actual = Vec::new();
    visit(state, tree.root, None, seen, &mut actual);
    assert_eq!(actual, expected);
    assert_eq!(tree.len, expected.len());
}
fn check(s: &SchedulerState, model: &[Expected]) {
    let state = s.state.lock().unwrap();
    let mut seen = vec![false; state.slots.entries.len()];
    for slot in &state.locks {
        let Some(lock) = &slot.value else {
            continue;
        };
        let mut expected: Vec<_> = model
            .iter()
            .filter(|m| {
                m.process == lock.process
                    && m.wait
                        == Some(Wait::Mutex(
                            state.slots[lock.owner].id,
                            match lock.key {
                                LockKey::User(key) => key,
                                _ => panic!("user lock fixture"),
                            },
                        ))
            })
            .collect();
        expected.sort_by_key(|m| (m.priority, m.order));
        check_tree(
            &state,
            &lock.waiters,
            &expected.into_iter().map(|m| m.id).collect::<Vec<_>>(),
            &mut seen,
        );
    }
    for (&process, record) in &state.processes {
        for (kind, tree) in [
            OrderedWaitKind::ConditionVariable,
            OrderedWaitKind::AddressArbiter,
        ]
        .into_iter()
        .zip(&record.ordered)
        {
            let mut expected: Vec<_> = model
                .iter()
                .filter_map(|m| match m.wait {
                    Some(Wait::Ordered(k, address)) if m.process == process && k == kind => {
                        Some((address, m.priority, m.order, m.id))
                    }
                    _ => None,
                })
                .collect();
            expected.sort_by_key(|&(address, priority, order, _)| (address, priority, order));
            check_tree(
                &state,
                tree,
                &expected
                    .into_iter()
                    .map(|(_, _, _, id)| id)
                    .collect::<Vec<_>>(),
                &mut seen,
            );
        }
    }
    for slot in &state.slots.entries {
        let Some(t) = &slot.thread else {
            continue;
        };
        let ordered = matches!(
            t.waiting_on,
            Some(Membership::Mutex(_) | Membership::Ordered(..))
        );
        assert_eq!(seen[slot_index(state.slots.index(t.id).unwrap())], ordered);
        if !ordered {
            assert_eq!(t.wait_node, WaitNode::default());
        }
    }
}

#[test]
fn mixed_waits_preserve_fifo_balance_and_identity_through_growth_and_reuse() {
    for seed in [1, 0x12345678, 0xbad5eed, 0xdeadbeef] {
        let s = scheduler();
        let owners = [
            thread(&s, ProcessId::new(1), 63),
            thread(&s, ProcessId::new(2), 63),
        ];
        let mut random = seed;
        let mut next = || {
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            random
        };
        let mut model = Vec::new();
        for step in 1..=2400u64 {
            // Growth while nodes are linked must not invalidate their indices.
            if step == 1 || step % 400 == 0 && model.len() < 96 {
                for i in 0..16 {
                    let process = ProcessId::new(1 + i % 2);
                    let priority = (next() % 64) as i32;
                    model.push(Expected {
                        id: thread(&s, process, priority),
                        process,
                        priority,
                        order: 0,
                        wait: None,
                    });
                }
            }
            let index = next() as usize % model.len();
            let m = &mut model[index];
            match next() % 6 {
                0..=2 => {
                    s.remove_waiter(m.id);
                    let address = (next() % 8) * 8;
                    let wait = match next() % 3 {
                        0 => Wait::Mutex(owners[next() as usize % 2], address),
                        1 => Wait::Ordered(OrderedWaitKind::ConditionVariable, address),
                        _ => Wait::Ordered(OrderedWaitKind::AddressArbiter, address),
                    };
                    match wait {
                        Wait::Mutex(owner, address) => s
                            .add_waiter(m.process, LockKey::User(address), owner, m.id, Some(m.id))
                            .unwrap(),
                        Wait::Ordered(kind, address) => s
                            .edit(|state| {
                                let index = state.slots.index(m.id)?;
                                state.insert_ordered_wait(index, kind, address)
                            })
                            .unwrap(),
                    }
                    m.wait = Some(wait);
                    m.order = step;
                }
                3 => {
                    s.remove_waiter(m.id);
                    m.wait = None;
                }
                4 => {
                    let priority = (next() % 64) as i32;
                    s.set_priority(m.id, priority, None).unwrap();
                    if priority != m.priority {
                        m.order = step;
                    }
                    m.priority = priority;
                }
                _ => {
                    let old = m.id;
                    s.terminate(old, false).unwrap();
                    s.unregister(old).unwrap();
                    m.id = thread(&s, m.process, m.priority);
                    assert_ne!(old, m.id);
                    m.wait = None;
                }
            }
            check(&s, &model);
        }
        // Actual ownership handoff must retain FIFO through the whole queue.
        while let Some(m) = model
            .iter()
            .find(|m| matches!(m.wait, Some(Wait::Mutex(..))))
        {
            let process = m.process;
            let Some(Wait::Mutex(mut owner, address)) = m.wait else {
                unreachable!()
            };
            loop {
                let expected = model
                    .iter()
                    .filter(|m| m.process == process && m.wait == Some(Wait::Mutex(owner, address)))
                    .min_by_key(|m| (m.priority, m.order))
                    .map(|m| m.id)
                    .unwrap();
                let handoff = s
                    .release_lock(process, LockKey::User(address), owner)
                    .unwrap();
                assert_eq!(handoff.0.next, Some(expected));
                for m in &mut model {
                    if m.process == process && m.wait == Some(Wait::Mutex(owner, address)) {
                        m.wait = (m.id != expected).then_some(Wait::Mutex(expected, address));
                    }
                }
                owner = expected;
                check(&s, &model);
                if !handoff.0.more_waiters {
                    break;
                }
            }
        }
        for m in &mut model {
            s.remove_waiter(m.id);
            m.wait = None;
        }
        check(&s, &model);
        assert_eq!(s.priority_donation_count(), 0);
    }
}

#[test]
fn ordered_bounds_and_reprioritization_preserve_process_key_and_fifo_order() {
    let s = scheduler();
    let mut rows = Vec::new();
    for process in [ProcessId::new(1), ProcessId::new(2)] {
        for kind in [
            OrderedWaitKind::ConditionVariable,
            OrderedWaitKind::AddressArbiter,
        ] {
            for address in [0, 8, u64::MAX] {
                let mut threads = Vec::new();
                for _ in 0..24 {
                    let id = thread(&s, process, 20);
                    s.edit(|state| {
                        let index = state.slots.index(id)?;
                        state.insert_ordered_wait(index, kind, address)
                    })
                    .unwrap();
                    threads.push(id);
                }
                rows.push((process, kind, address, threads));
            }
        }
    }
    for (process, kind, address, mut threads) in rows {
        let changed = threads.remove(7);
        s.set_priority(changed, 5, None).unwrap();
        assert_eq!(
            s.first_ordered_waiter(process, kind, address),
            Some(changed)
        );
        s.set_priority(changed, 20, None).unwrap();
        threads.push(changed);
        assert_eq!(
            s.ordered_waiters(process, kind, address, usize::MAX),
            threads
        );
        assert_eq!(s.ordered_waiters(process, kind, address, 1), threads[..1]);
        assert!(s.ordered_waiters(process, kind, address, 0).is_empty());
        assert!(s.first_ordered_waiter(process, kind, 9).is_none());
        for id in threads {
            s.remove_waiter(id);
        }
        assert!(s.first_ordered_waiter(process, kind, address).is_none());
    }
}
