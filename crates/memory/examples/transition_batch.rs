//! Diagnostic singleton/batch comparison; no production mode switch.
use nixe_memory::*;
use std::{collections::BTreeMap, sync::Arc};

struct Cache;
impl VisibilityCoordinator for Cache {
    fn cache_cpu_page(
        &self,
        _: DeviceVisibilityRequest,
        _: &[u8],
    ) -> Result<(), VisibilityCoordinatorError> {
        Ok(())
    }
    fn make_cpu_visible(
        &self,
        _: CpuVisibilityRequest,
    ) -> Result<Box<[u8]>, VisibilityCoordinatorError> {
        panic!("this workload must not download GPU data")
    }
}

fn counters() -> BTreeMap<&'static str, u64> {
    metrics::snapshot().into_iter().collect()
}
fn report(phase: &str, before: BTreeMap<&'static str, u64>) {
    let after = counters();
    println!(
        "{{\"phase\":\"{phase}\",\"gates\":{},\"protection_calls\":{},\"protection_bytes\":{},\"snapshot_bytes\":{},\"page_ownership_updates\":{}}}",
        after["GateExclusiveAcquisitions"] - before["GateExclusiveAcquisitions"],
        after["DirectProtectionCalls"] - before["DirectProtectionCalls"],
        after["DirectProtectionBytes"] - before["DirectProtectionBytes"],
        after["SnapshotCopiedBytes"] - before["SnapshotCopiedBytes"],
        after["PageOwnershipUpdates"] - before["PageOwnershipUpdates"],
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let serial = match std::env::args().nth(1).as_deref() {
        Some("serial") => true,
        Some("batch") | None => false,
        _ => return Err("expected serial or batch".into()),
    };
    let store = CanonicalBackingStore::allocate()?;
    let pages = (0..33)
        .map(|index| {
            CanonicalBackingPage::zeroed(
                &store,
                GuestPhysicalPageId::new(index),
                4096,
                ContentGeneration::INITIAL,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let arena = DirectArena::new(34 * 4096)?;
    for (index, page) in pages.iter().enumerate() {
        let host = page.direct_backing()?;
        let address = (index as u64 + 1) * 4096;
        arena.map_pages(&[DirectMapRequest {
            guest_address: address,
            backing: &host,
            protection: DirectProtection::Read,
        }])?;
        page.register_direct_alias(&arena, address, DirectProtection::ReadWrite)?;
    }
    let ranges = (0..32)
        .map(|index| {
            CanonicalBackingRange::new(
                pages[index..index + 2]
                    .iter()
                    .map(|page| {
                        CanonicalBackingSegment::new(
                            page.clone(),
                            0,
                            4096,
                            MemoryPermissions::READ_WRITE,
                            MappingGeneration::INITIAL,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let dependencies = ranges
        .iter()
        .map(CanonicalCpuWriteDependency::capture)
        .collect::<Result<Vec<_>, _>>()?;
    {
        let mut transition = store.execution_gate().acquire_exclusive();
        transition.commit();
        for (index, page) in pages.iter().enumerate() {
            let generation = page.content_generation();
            page.write_preflighted(0, &[index as u8 + 1], generation, generation.next()?)?;
        }
    }
    let coordinator: Arc<dyn VisibilityCoordinator> = Arc::new(Cache);
    let read = DeviceAccessDeclaration::read(NonCpuDeviceId::new(1), DeviceVisibilityPoint::new(1));
    let prepare = || -> Result<(), VisibilityError> {
        if serial {
            for range in &ranges {
                CanonicalBackingRange::prepare_resident_device_accesses(
                    [(range, read)],
                    coordinator.clone(),
                )?;
            }
        } else {
            CanonicalBackingRange::prepare_resident_device_accesses(
                ranges.iter().map(|range| (range, read)),
                coordinator.clone(),
            )?;
        }
        Ok(())
    };
    let requests = ranges
        .iter()
        .zip(&dependencies)
        .map(|(range, dependency)| CpuWriteSnapshotRequest {
            range,
            dependency,
            selection: CpuWriteSnapshotSelection::DirtyPages,
            alignment: 4,
        })
        .collect::<Vec<_>>();
    let snapshot = || -> Result<(), CanonicalRangeAccessError> {
        let mut resolve = |_: &dyn VisibilityCoordinator,
                           _: CpuVisibilityRequest|
         -> Result<Box<[u8]>, VisibilityCoordinatorError> {
            panic!("no GPU producer in this workload")
        };
        if serial {
            for request in &requests {
                CanonicalCpuWriteDependency::snapshot_batch_with_resolver(
                    std::slice::from_ref(request),
                    &mut resolve,
                )?;
            }
        } else {
            CanonicalCpuWriteDependency::snapshot_batch_with_resolver(&requests, &mut resolve)?;
        }
        Ok(())
    };
    let before = counters();
    prepare()?;
    report("dirty_prepare", before);
    let before = counters();
    snapshot()?;
    report("dirty_snapshot", before);
    let before = counters();
    prepare()?;
    snapshot()?;
    report("clean_resident", before);
    let write = |point| {
        DeviceAccessDeclaration::write(
            NonCpuDeviceId::new(1),
            DeviceVisibilityPoint::new(point),
            DeviceVisibilityPoint::new(point),
        )
        .unwrap()
    };
    let publish = |point| -> Result<(), VisibilityError> {
        if serial {
            for range in &ranges {
                CanonicalBackingRange::publish_device_writes(
                    [(range, write(point))],
                    coordinator.clone(),
                )?;
            }
        } else {
            CanonicalBackingRange::publish_device_writes(
                ranges.iter().map(|range| (range, write(point))),
                coordinator.clone(),
            )?;
        }
        Ok(())
    };
    // Preparation/snapshots left every page clean and its CPU bytes unchanged.
    // Publication itself neither transfers bytes nor waits for device work.
    let before = counters();
    publish(2)?;
    report("accepted_write_publish", before);
    let before = counters();
    publish(3)?;
    report("resident_alias_publish", before);
    let whole = CanonicalBackingRange::new(
        ranges
            .iter()
            .flat_map(|range| range.segments().iter().cloned())
            .collect(),
    )?;
    CanonicalBackingRange::publish_device_writes([(&whole, write(4))], coordinator.clone())?;
    let before = counters();
    CanonicalBackingRange::publish_device_writes([(&whole, write(5))], coordinator.clone())?;
    report("whole_range_resident_publish", before);
    let before = counters();
    if serial {
        for range in &ranges {
            CanonicalBackingRange::invalidate_visibility_ranges([range])?;
        }
    } else {
        CanonicalBackingRange::invalidate_visibility_ranges(ranges.iter())?;
    }
    report("terminal_invalidation", before);
    Ok(())
}
