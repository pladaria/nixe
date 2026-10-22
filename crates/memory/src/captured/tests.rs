use super::*;
use crate::{
    AddressSpace, AddressWidth, ArenaOffset, GuestVirtualAddress, Machine, MemoryPermissions,
    PAGE_SIZE,
};

#[test]
fn fragmented_allocation_subranges_match_physical_bytes_at_every_boundary() {
    let machine = Machine::new(64 * PAGE_SIZE).unwrap();
    let account = machine.account(64 * PAGE_SIZE).unwrap();
    let mut occupied: Vec<_> = (0..64)
        .map(|_| Some(account.allocate(PAGE_SIZE).unwrap()))
        .collect();
    for index in (0..64).step_by(2) {
        occupied[index] = None;
    }
    let allocation = account.allocate(32 * PAGE_SIZE).unwrap();
    let full = AllocationRange::new(allocation.pin(), 0, allocation.bytes()).unwrap();
    assert_eq!(full.physical_extents().count(), 32);
    for page in 0..32 {
        for within in [0, 17, PAGE_SIZE - 1] {
            let offset = page * PAGE_SIZE + within;
            let bytes = (PAGE_SIZE + 7).min(full.bytes() - offset);
            let selected = full.slice(offset, bytes).unwrap();
            let actual: Vec<_> = selected.physical_extents().collect();
            let mut expected = Vec::new();
            let mut cursor = offset;
            while cursor < offset + bytes {
                let count = (PAGE_SIZE - cursor % PAGE_SIZE).min(offset + bytes - cursor);
                expected.push(Extent {
                    start: ArenaOffset::new(
                        2 * (cursor / PAGE_SIZE) * PAGE_SIZE + cursor % PAGE_SIZE,
                    ),
                    bytes: count,
                });
                cursor += count;
            }
            assert_eq!(actual, expected, "offset={offset} bytes={bytes}");
        }
    }
}

#[test]
fn accepted_spans_seek_in_logical_order_with_unequal_lengths_and_offsets() {
    let machine = Machine::new(32 * PAGE_SIZE).unwrap();
    let account = machine.account(32 * PAGE_SIZE).unwrap();
    let mut ranges = Vec::new();
    let mut content = Vec::new();
    for i in 0..32 {
        let allocation = account.allocate(PAGE_SIZE).unwrap();
        let bytes = vec![i as u8 + 1; 17 + i % 5];
        let range =
            AllocationRange::new(allocation.pin(), i as u64 + 3, bytes.len() as u64).unwrap();
        machine
            .write_resolved(std::slice::from_ref(&range), &bytes)
            .unwrap();
        ranges.push(range);
        content.push(bytes);
    }
    // Physical addresses descend; only logical end positions may be searched.
    ranges.reverse();
    content.reverse();
    let mut expected: Vec<_> = content.into_iter().flatten().collect();
    let captured = CapturedMemory::new(ranges).unwrap();
    assert_eq!(captured.len(), 32);
    for offset in 0..=expected.len() {
        for bytes in [0, 1, 7, 33] {
            let end = (offset + bytes).min(expected.len());
            let mut output = vec![0; end - offset];
            machine
                .read_captured_into(&captured, offset as u64, &mut output)
                .unwrap()
                .into_ready()
                .unwrap();
            assert_eq!(output, expected[offset..end]);
        }
    }
    let offset = expected.len() - 19;
    machine
        .write_captured(&captured, offset as u64, &[0xfe; 19])
        .unwrap()
        .into_ready()
        .unwrap();
    expected[offset..].fill(0xfe);
    let mut actual = vec![0; expected.len()];
    machine
        .read_captured_into(&captured, 0, &mut actual)
        .unwrap()
        .into_ready()
        .unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn exact_tail_capture_keeps_only_selected_pages_and_original_mapping() {
    let machine = Machine::new(32 * PAGE_SIZE).unwrap();
    let allocation = machine
        .account(32 * PAGE_SIZE)
        .unwrap()
        .allocate(32 * PAGE_SIZE)
        .unwrap();
    let space = AddressSpace::new(machine.clone(), AddressWidth::Bits32).unwrap();
    let base = GuestVirtualAddress::new(PAGE_SIZE);
    space
        .map(
            base,
            allocation,
            0,
            32 * PAGE_SIZE,
            MemoryPermissions::READ_WRITE,
        )
        .unwrap();
    let original = space
        .capture(base, 32 * PAGE_SIZE, MemoryPermissions::READ_WRITE)
        .unwrap();
    let mapping = original[0].mapping().unwrap();
    let captured =
        CapturedMemory::new((0..32).map(|i| original[0].slice(i * PAGE_SIZE, PAGE_SIZE).unwrap()))
            .unwrap();
    drop(original);
    let offset = 31 * PAGE_SIZE + 9;
    let selected = captured.capture(offset, 7).unwrap();
    assert_eq!(selected.len(), 1);
    let version = selected[0].mapping().unwrap();
    assert_eq!(version.start, base.checked_add(offset).unwrap());
    assert_eq!(version.generation, mapping.generation);
    assert_eq!(selected[0].offset(), offset);
    space.unmap(base, 32 * PAGE_SIZE).unwrap();
    drop(captured);
    assert_eq!(machine.statistics().allocated_bytes, PAGE_SIZE);
    let selected = CapturedMemory::new(selected).unwrap();
    machine
        .write_captured(&selected, 0, &[7; 7])
        .unwrap()
        .into_ready()
        .unwrap();
    let mut output = [0; 7];
    machine
        .read_captured_into(&selected, 0, &mut output)
        .unwrap()
        .into_ready()
        .unwrap();
    assert_eq!(output, [7; 7]);
    drop(selected);
    assert_eq!(machine.statistics().allocated_bytes, 0);
}

#[test]
fn captured_bounds_and_machine_identity_are_checked_before_access() {
    let machine = Machine::new(PAGE_SIZE).unwrap();
    let other = Machine::new(PAGE_SIZE).unwrap();
    let make = |machine: &Machine| {
        let allocation = machine
            .account(PAGE_SIZE)
            .unwrap()
            .allocate(PAGE_SIZE)
            .unwrap();
        AllocationRange::new(allocation.pin(), 1, 7).unwrap()
    };
    let range = make(&machine);
    assert!(CapturedMemory::new([range.clone(), make(&other)]).is_err());
    let captured = CapturedMemory::from(range);
    let mut output = [0xaa; 2];
    for offset in [6, 8, u64::MAX] {
        assert!(
            machine
                .read_captured_into(&captured, offset, &mut output)
                .is_err()
        );
        assert_eq!(output, [0xaa; 2]);
        assert!(machine.write_captured(&captured, offset, &[9; 2]).is_err());
        assert!(captured.capture(offset, 2).is_err());
    }
    assert!(other.read_captured_into(&captured, 0, &mut output).is_err());
    assert!(other.write_captured(&captured, 0, &[9; 2]).is_err());
    assert!(captured.capture(7, 0).unwrap().is_empty());
    machine
        .read_captured_into(&captured, 7, &mut [])
        .unwrap()
        .into_ready()
        .unwrap();
    machine
        .write_captured(&captured, 7, &[])
        .unwrap()
        .into_ready()
        .unwrap();
    let empty = CapturedMemory::new([]).unwrap();
    assert!(empty.is_empty());
    assert_eq!(empty.bytes(), 0);
    machine
        .read_captured_into(&empty, 0, &mut [])
        .unwrap()
        .into_ready()
        .unwrap();
    machine
        .write_captured(&empty, 0, &[])
        .unwrap()
        .into_ready()
        .unwrap();
    assert!(empty.capture(1, 0).is_err());
}
