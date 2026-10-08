use super::*;

fn region(start: u32, end: u32, pc: Option<u64>, category: &'static str) -> Region {
    Region {
        start,
        end,
        pc,
        category,
    }
}

#[test]
fn source_intervals_do_not_assign_gaps_or_terminal_patches_to_guest_execution() {
    let actual = partition_regions(
        48,
        32,
        vec![
            region(8, 20, Some(0x1000), "guest_lowering"),
            region(24, 32, Some(0x2000), "guest_lowering"),
        ],
        vec![region(12, 16, Some(0x1000), "dispatch_link")],
    );
    assert_eq!(
        actual,
        vec![
            region(0, 8, None, "generated_scaffolding"),
            region(8, 12, Some(0x1000), "guest_lowering"),
            region(12, 16, Some(0x1000), "dispatch_link"),
            region(16, 20, Some(0x1000), "guest_lowering"),
            region(20, 24, None, "generated_scaffolding"),
            region(24, 32, Some(0x2000), "guest_lowering"),
            region(32, 48, None, "entry_exit"),
        ]
    );
    assert_eq!(
        actual
            .iter()
            .map(|region| region.end - region.start)
            .sum::<u32>(),
        48
    );
}

fn directory() -> std::path::PathBuf {
    let directory = std::env::temp_dir().join(format!(
        "nixe-jitdump-test-{}-{}",
        std::process::id(),
        now()
    ));
    std::fs::create_dir(&directory).unwrap();
    directory
}
fn load(index: u64, address: usize) -> Load {
    Load {
        index,
        address,
        name: format!("nixe_LCQ_p1_as2_u{index}_v{index}_pc1000"),
        bytes: vec![0x90, 0xc3].into(),
        regions: vec![
            region(0, 1, Some(0x1000), "guest_lowering"),
            region(1, 2, None, "entry_exit"),
        ],
        _permit: Permit {
            status: Arc::new(Status {
                queued: AtomicUsize::new(0),
                dropped: AtomicU64::new(0),
                failed: AtomicBool::new(false),
            }),
            bytes: 0,
        },
    }
}
fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_ne_bytes(bytes[offset..offset + 4].try_into().unwrap())
}
fn u64_at(bytes: &[u8], offset: usize) -> u64 {
    u64::from_ne_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

#[test]
fn binary_records_preserve_unique_loads_at_reused_addresses_and_debug_order() {
    let directory = directory();
    let pid = std::process::id();
    {
        let mut writer = Writer::new(&directory).unwrap();
        writer.load(100, 7, &load(1, 0x10000)).unwrap();
        writer.load(200, 8, &load(2, 0x10000)).unwrap();
        writer.flush().unwrap();
    }
    let bytes = std::fs::read(directory.join(format!("jit-{pid}.dump"))).unwrap();
    assert_eq!(u32_at(&bytes, 0), 0x4a695444);
    assert_eq!(u32_at(&bytes, 8), 40);
    assert_eq!(u32_at(&bytes, 20), pid);
    let mut offset = 40;
    let mut records = Vec::new();
    let mut indices = Vec::new();
    while offset < bytes.len() {
        let kind = u32_at(&bytes, offset);
        let size = u32_at(&bytes, offset + 4) as usize;
        assert!(size >= 16 && offset + size <= bytes.len());
        records.push((kind, u64_at(&bytes, offset + 8)));
        if kind == 0 {
            assert_eq!(u64_at(&bytes, offset + 32), 0x10000); // code_addr
            indices.push(u64_at(&bytes, offset + 48));
            assert_eq!(&bytes[offset + size - 2..offset + size], &[0x90, 0xc3]);
        }
        offset += size;
    }
    assert_eq!(records, [(2, 100), (0, 100), (2, 200), (0, 200)]);
    assert_eq!(indices, [1, 2]);
    let csv = std::fs::read_to_string(directory.join(format!("jit-{pid}.regions.csv"))).unwrap();
    assert!(csv.lines().all(|line| line.split(',').count() == 10));
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn bounded_diagnostics_drop_excess_work_and_release_rejected_payloads() {
    let status = Arc::new(Status {
        queued: AtomicUsize::new(0),
        dropped: AtomicU64::new(0),
        failed: AtomicBool::new(false),
    });
    let (sender, receiver) = mpsc::sync_channel(1);
    let profile = Profiler {
        sender,
        status: Arc::clone(&status),
        next: AtomicU64::new(1),
        order: Mutex::new(()),
    };
    let permit = profile.reserve(QUEUED_BYTES).unwrap();
    assert!(profile.reserve(1).is_none());
    drop(permit);
    assert_eq!(status.queued.load(Ordering::Relaxed), 0);
    profile.send(Event::Retire {
        timestamp: 0,
        address: 1,
        size: 1,
    });
    profile.send(Event::Retire {
        timestamp: 0,
        address: 2,
        size: 1,
    });
    assert_eq!(status.dropped.load(Ordering::Relaxed), 2);
    assert!(matches!(receiver.recv().unwrap(), Event::Retire { timestamp, .. } if timestamp > 0));
    status.failed.store(true, Ordering::Release);
    assert!(profile.reserve(1).is_none());
}
