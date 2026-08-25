//! Opt-in, provider-private JIT compilation diagnostics.

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, PoisonError};

use nixe_cpu::decode::{DecodeResult, decode, disassemble};
use nixe_cpu::ir::print::{IrDumpStage, IrPrintOptions, print_ir_dump};
use nixe_cpu::ir::region::IrRegion;
use nixe_cpu::location::{ExecutionState, InstructionEncoding};
use nixe_cpu::profile::GuestCpuProfile;
use nixe_cpu_engine::EngineDomainId;

use crate::compiler::CompiledRegion;

static NEXT_SESSION: AtomicU64 = AtomicU64::new(0);

pub(crate) struct JitDiagnostics {
    directory: PathBuf,
    state: Mutex<DumpState>,
}

struct DumpState {
    next_region: u64,
    slots: Box<[Option<PathBuf>]>,
}

impl JitDiagnostics {
    pub(crate) fn new(
        root: &Path,
        domain: EngineDomainId,
        max_regions: usize,
    ) -> Result<Self, Box<str>> {
        if max_regions == 0 {
            return Err("JIT dump capacity must contain at least one region".into());
        }
        let session = NEXT_SESSION.fetch_add(1, Ordering::Relaxed);
        let directory = root.join(format!(
            "session-{}-{session}-domain-{}",
            std::process::id(),
            domain.get()
        ));
        fs::create_dir_all(&directory).map_err(|error| {
            format!(
                "cannot create JIT dump directory {}: {error}",
                directory.display()
            )
            .into_boxed_str()
        })?;
        Ok(Self {
            directory,
            state: Mutex::new(DumpState {
                next_region: 0,
                slots: vec![None; max_regions].into_boxed_slice(),
            }),
        })
    }

    pub(crate) fn directory(&self) -> &Path {
        &self.directory
    }

    pub(crate) fn dump_region(
        &self,
        profile: &GuestCpuProfile,
        region: &IrRegion,
        compiled: &CompiledRegion,
    ) -> Result<PathBuf, Box<str>> {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let sequence = state.next_region;
        state.next_region = state
            .next_region
            .checked_add(1)
            .ok_or_else(|| Box::<str>::from("JIT dump sequence exhausted"))?;
        let slot = sequence as usize % state.slots.len();
        if let Some(previous) = state.slots[slot].take() {
            fs::remove_dir_all(&previous).map_err(|error| {
                format!(
                    "cannot recycle JIT region dump {}: {error}",
                    previous.display()
                )
                .into_boxed_str()
            })?;
        }
        let start = region.metadata.start;
        let directory = self.directory.join(format!(
            "region-{sequence:08}-slot-{slot:08}-{}-{:016x}",
            state_name(start.execution_state),
            start.pc.get()
        ));
        create_directory(&directory)?;

        let metadata_path = directory.join("metadata.txt");
        write_file(&metadata_path, region_metadata(region, compiled).as_bytes())?;
        let assembly_path = directory.join("guest.asm");
        write_file(&assembly_path, guest_assembly(profile, region).as_bytes())?;
        let ir_path = directory.join("nixe-ir.txt");
        let ir = print_ir_dump(
            region,
            IrDumpStage::PreOptimization,
            IrPrintOptions {
                raw_encoding_comments: true,
                disassembly_comments: false,
            },
        );
        write_file(&ir_path, ir.as_bytes())?;

        for (index, block) in region.blocks.iter().enumerate() {
            let path = directory.join(format!(
                "block-{index:03}-{:016x}.bin",
                block.metadata.start.pc.get()
            ));
            let mut bytes = Vec::with_capacity(block.metadata.guest_byte_count as usize);
            for source in &block.metadata.sources {
                append_encoding(&mut bytes, source.encoding);
            }
            write_file(&path, &bytes)?;
        }
        write_file(&directory.join("complete"), &[])?;
        state.slots[slot] = Some(directory.clone());
        Ok(directory)
    }
}

fn create_directory(path: &Path) -> Result<(), Box<str>> {
    fs::create_dir(path).map_err(|error| {
        format!("cannot create JIT region dump {}: {error}", path.display()).into_boxed_str()
    })
}

fn write_file(path: &Path, bytes: &[u8]) -> Result<(), Box<str>> {
    fs::write(path, bytes).map_err(|error| {
        format!("cannot write JIT diagnostic {}: {error}", path.display()).into_boxed_str()
    })
}

fn region_metadata(region: &IrRegion, compiled: &CompiledRegion) -> String {
    let mut output = String::new();
    writeln!(output, "start={}", region.metadata.start).unwrap();
    writeln!(output, "blocks={}", region.blocks.len()).unwrap();
    writeln!(output, "entries={}", region.metadata.entries.len()).unwrap();
    writeln!(output, "exits={}", region.metadata.exits.len()).unwrap();
    writeln!(
        output,
        "guest_instructions={}",
        region.metadata.guest_instruction_count
    )
    .unwrap();
    writeln!(output, "guest_bytes={}", region.metadata.guest_byte_count).unwrap();
    writeln!(
        output,
        "ir_operations={}",
        region.metadata.ir_operation_count
    )
    .unwrap();
    writeln!(
        output,
        "code_dependencies={}",
        region.metadata.code_dependencies.len()
    )
    .unwrap();
    writeln!(output, "native_mapped_bytes={}", compiled.mapped_len()).unwrap();
    output
}

fn guest_assembly(profile: &GuestCpuProfile, region: &IrRegion) -> String {
    let mut output = String::new();
    for (index, block) in region.blocks.iter().enumerate() {
        writeln!(
            output,
            "block {index} start={:#018x}",
            block.metadata.start.pc.get()
        )
        .unwrap();
        for source in &block.metadata.sources {
            writeln!(
                output,
                "  {:#018x}: {}  {}",
                source.location.pc.get(),
                source.encoding,
                source_disassembly(profile, source.location, source.encoding)
            )
            .unwrap();
        }
    }
    output
}

fn source_disassembly(
    profile: &GuestCpuProfile,
    location: nixe_cpu::location::LocationDescriptor,
    encoding: InstructionEncoding,
) -> String {
    match decode(profile, location, encoding) {
        DecodeResult::Decoded(decoded) | DecodeResult::RecognizedUnimplemented(decoded) => {
            disassemble(&decoded.instruction).to_string()
        }
        DecodeResult::Unallocated { reason, .. } => format!("<unallocated: {reason}>"),
        DecodeResult::Reserved { name, reason, .. } => {
            format!("<{name}: reserved: {reason}>")
        }
        DecodeResult::ProfileDisabled {
            name, rejection, ..
        } => format!("<{name}: profile-disabled: {rejection}>"),
    }
}

fn append_encoding(bytes: &mut Vec<u8>, encoding: InstructionEncoding) {
    let raw = encoding.bits().to_le_bytes();
    bytes.extend_from_slice(&raw[..usize::from(encoding.size().bytes())]);
}

const fn state_name(state: ExecutionState) -> &'static str {
    match state {
        ExecutionState::A64 => "a64",
        ExecutionState::A32 => "a32",
        ExecutionState::T32 => "t32",
    }
}
