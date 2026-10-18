//! Single-threaded compiler phase and allocation measurements, emitted as CSV.
use nixe_shader_recompiler::{
    Compiler, CompilerSettings, Environment, HostTranslateInfo, Profile, Stage,
    artifact::Bindings,
    environment::{EnvironmentMetadata, Observations, SnapshotEnvironment},
    maxwell::{bits::insert, decode::Opcode, header::ProgramHeader},
    passes::GraphicsRuntimeState,
    types::ProgramOffset,
};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    hint::black_box,
    sync::atomic::{AtomicUsize, Ordering::Relaxed},
    time::Instant,
};

struct MeasuredAllocator;
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
fn allocated(bytes: usize) {
    ALLOCATIONS.fetch_add(1, Relaxed);
    BYTES.fetch_add(bytes, Relaxed);
    let live = LIVE.fetch_add(bytes, Relaxed) + bytes;
    PEAK.fetch_max(live, Relaxed);
}
// Each operation forwards the allocator's original pointer, layout and size.
unsafe impl GlobalAlloc for MeasuredAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            allocated(layout.size());
        }
        pointer
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Relaxed);
        unsafe { System.dealloc(pointer, layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let pointer = unsafe { System.realloc(pointer, layout, size) };
        if !pointer.is_null() {
            LIVE.fetch_sub(layout.size(), Relaxed);
            allocated(size);
        }
        pointer
    }
}
#[global_allocator]
static ALLOCATOR: MeasuredAllocator = MeasuredAllocator;

fn word(op: Opcode) -> u64 {
    op.encoding().value | 7 << 16
}
fn environment(stage: Stage, code: Vec<u64>, header: [u32; 20]) -> SnapshotEnvironment {
    let mut pc = if stage == Stage::Compute { 8 } else { 0x50 };
    let instructions = code
        .into_iter()
        .map(|word| {
            if pc % 32 == 0 {
                pc += 8;
            }
            let entry = (ProgramOffset(pc), word);
            pc += 8;
            entry
        })
        .collect();
    SnapshotEnvironment::new(
        EnvironmentMetadata {
            stage,
            sph: ProgramHeader::from_bytes(&std::array::from_fn(|i| {
                header[i / 4].to_le_bytes()[i % 4]
            })),
            workgroup_size: [32, 1, 1],
            shared_memory_size: if stage == Stage::Compute { 4096 } else { 0 },
            ..Default::default()
        },
        Observations {
            instructions,
            viewport_transform_state: Some(1),
            ..Default::default()
        },
    )
}
fn corpus() -> Vec<(String, Vec<SnapshotEnvironment>)> {
    let exit = word(Opcode::EXIT) | 15;
    let mut cases = vec![(
        "compute-exit".to_owned(),
        vec![environment(Stage::Compute, vec![exit], [0; 20])],
    )];
    let vertex = environment(
        Stage::VertexB,
        vec![
            word(Opcode::ALD) | 3 << 47 | 255 << 39 | 0x80 << 20 | 255 << 8,
            word(Opcode::AST) | 3 << 47 | 255 << 39 | 0x70 << 20 | 255 << 8,
            exit,
        ],
        [0; 20],
    );
    for length in [1, 64, 512] {
        let mut code = vec![word(Opcode::LDC) | 4 << 48 | 255 << 8];
        code.extend((0..length).map(|_| word(Opcode::FADD32I) | (0.125f32.to_bits() as u64) << 20));
        code.push(exit);
        let mut header = [0; 20];
        header[18] = 1;
        cases.push((
            format!("linked-fadd-{length}"),
            vec![vertex.clone(), environment(Stage::Fragment, code, header)],
        ));
    }
    let mut code = Vec::new();
    for index in 0..64 {
        code.push(insert::<20, 32>(word(Opcode::MOV32I) | 15 << 12, index));
        code.push(word(Opcode::STS) | 4 << 48 | (index * 4) << 20 | 255 << 8);
    }
    code.push(exit);
    cases.push((
        "compute-shared-64".to_owned(),
        vec![environment(Stage::Compute, code, [0; 20])],
    ));
    cases
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let samples: usize = std::env::var("SAMPLES")
        .unwrap_or_else(|_| "50".into())
        .parse()?;
    let output = std::env::var_os("SPIRV_OUTPUT").map(std::path::PathBuf::from);
    if let Some(path) = &output {
        std::fs::create_dir_all(path)?;
    }
    let host = HostTranslateInfo::default();
    let profile = Profile {
        supported_spirv: 0x10500,
        unified_descriptor_binding: true,
        max_user_clip_distances: 8,
        ..Default::default()
    };
    let settings = CompilerSettings::default();
    eprintln!(
        "host={host:?}\nprofile={profile:?}\nsettings={settings:?}\nallocator=System with atomic counters; allocation totals include ephemeral frontend/backend objects"
    );
    println!(
        "case,scratch,sample,translate_ns,prepare_ns,emit_ns,allocations,allocated_bytes,peak_extra_bytes,retained_scratch_bytes,spirv_words"
    );
    for (name, mut environments) in corpus() {
        for reuse in [false, true] {
            let baseline = LIVE.load(Relaxed);
            let mut compiler = Compiler::new()?;
            for sample in 0..samples {
                if !reuse {
                    compiler = Compiler::new()?;
                }
                compiler.reset()?;
                let live_before = LIVE.load(Relaxed);
                PEAK.store(live_before, Relaxed);
                let allocations = ALLOCATIONS.load(Relaxed);
                let bytes = BYTES.load(Relaxed);
                let started = Instant::now();
                let ids = environments
                    .iter_mut()
                    .map(|env| compiler.translate(env, &host, &settings))
                    .collect::<Result<Vec<_>, _>>()?;
                let translated = Instant::now();
                let prepared = if environments[0].metadata().stage == Stage::Compute {
                    vec![compiler.prepare_stage(ids[0], Default::default())?]
                } else {
                    compiler.prepare_graphics(&ids, &GraphicsRuntimeState::default())?
                };
                let linked = Instant::now();
                let mut bindings = Bindings::default();
                let artifacts = prepared
                    .into_iter()
                    .map(|stage| compiler.emit_prepared(stage, &profile, &settings, &mut bindings))
                    .collect::<Result<Vec<_>, _>>()?;
                let ended = Instant::now();
                let count = ALLOCATIONS.load(Relaxed) - allocations;
                let allocated = BYTES.load(Relaxed) - bytes;
                let peak = PEAK.load(Relaxed).saturating_sub(live_before);
                let words: usize = artifacts.iter().map(|a| a.words.len()).sum();
                if sample == 0
                    && let Some(path) = &output
                {
                    for artifact in &artifacts {
                        let bytes: Vec<_> = artifact
                            .words
                            .iter()
                            .flat_map(|word| word.to_le_bytes())
                            .collect();
                        std::fs::write(
                            path.join(format!("{name}-{:?}.spv", artifact.stage)),
                            bytes,
                        )?;
                    }
                }
                black_box(&artifacts);
                drop(artifacts);
                compiler.reset()?;
                let retained = LIVE.load(Relaxed).saturating_sub(baseline);
                println!(
                    "{name},{},{sample},{},{},{},{count},{allocated},{peak},{retained},{words}",
                    if reuse { "reused" } else { "fresh" },
                    translated.duration_since(started).as_nanos(),
                    linked.duration_since(translated).as_nanos(),
                    ended.duration_since(linked).as_nanos()
                );
            }
        }
    }
    Ok(())
}
