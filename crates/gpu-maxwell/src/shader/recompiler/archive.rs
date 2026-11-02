//! Content-indexed frontend shader observations. Prefixes locate candidates;
//! code, headers and observed mutable compilation inputs validate admission.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/shader_environment.cpp
use super::*;
use nixe_gpu::{
    persistent_cache::Store,
    shader_compilation::{Configuration, ShaderInputs},
    shader_encoding,
    shader_identity::ContentId,
};
use nixe_shader_recompiler::maxwell::location::Location;
use std::sync::Weak;

const MAXIMUM_ENTRIES: usize = 2048;
const PREFIX_INSTRUCTIONS: u8 = 16;

struct Candidates {
    entries: Vec<Weak<ShaderInputs>>,
    // Every candidate actually observed these consecutive executable slots.
    // Scheduling words are included in the read but never interpreted as code.
    words: Box<[u8]>,
}
impl Candidates {
    fn prefix_words(environment: &SnapshotEnvironment) -> u8 {
        let Ok(entry) = environment
            .metadata()
            .entry_address()
            .and_then(Location::new)
        else {
            return 0;
        };
        let observed = &environment.observations().instructions;
        (0..PREFIX_INSTRUCTIONS)
            .take_while(|&index| observed.contains_key(&entry.advance(i32::from(index)).address()))
            .count() as u8
    }
}

pub(crate) struct CaptureArchive {
    store: Store,
    configuration: Configuration,
    // This index borrows the existing canonical input interner's lifetime.
    candidates: rustc_hash::FxHashMap<ContentId, Candidates>,
    indexed: usize,
}
impl CaptureArchive {
    pub fn new(
        directory: &std::path::Path,
        namespace: ContentId,
        configuration: Configuration,
        budget: u64,
    ) -> Self {
        Self {
            store: Store::new(directory.join("captures-v1").join(namespace.hex()), budget),
            configuration,
            candidates: Default::default(),
            indexed: 0,
        }
    }

    pub fn load(&mut self, inputs: &mut nixe_gpu::WorkingSet<ContentId, Arc<ShaderInputs>>) {
        let keys = match self.store.keys(MAXIMUM_ENTRIES) {
            Ok(keys) => keys,
            Err(error) => {
                log::warn!("frontend shader capture directory: {error}");
                return;
            }
        };
        let mut loaded = 0;
        for key in keys {
            let Some(bytes) = self.store.read(key) else {
                continue;
            };
            let Some(input) = shader_encoding::decode_inputs(&bytes, &self.configuration) else {
                continue;
            };
            let input = Arc::new(input);
            self.remember(&input);
            inputs.insert(input.identity, Arc::clone(&input), input.retained_bytes());
            loaded += 1;
        }
        if loaded != 0 {
            log::info!("loaded {loaded} frontend shader captures before guest execution");
        }
    }

    fn append_prefix(
        bytes: &mut Vec<u8>,
        metadata: &EnvironmentMetadata,
        group: Option<u8>,
        instruction: u64,
    ) {
        use shader_encoding::Encode;
        metadata.encode(bytes);
        group.encode(bytes);
        instruction.encode(bytes);
    }

    fn prefix(inputs: &ShaderInputs) -> Option<ContentId> {
        let mut bytes = Vec::new();
        for input in &inputs.stages {
            let metadata = input.environment.metadata();
            let entry = nixe_shader_recompiler::maxwell::location::Location::new(
                metadata.entry_address().ok()?,
            )
            .ok()?
            .address();
            let instruction = *input.environment.observations().instructions.get(&entry)?;
            Self::append_prefix(&mut bytes, metadata, input.binding_group, instruction);
        }
        Some(ContentId::of(b"nixe-capture-code-prefix-v1", &[&bytes]))
    }

    pub fn remember(&mut self, inputs: &Arc<ShaderInputs>) {
        let Some(prefix) = Self::prefix(inputs) else {
            return;
        };
        if self.indexed >= MAXIMUM_ENTRIES {
            self.indexed = 0;
            self.candidates.retain(|_, candidates| {
                candidates.entries.retain(|entry| entry.strong_count() != 0);
                self.indexed += candidates.entries.len();
                !candidates.entries.is_empty()
            });
            if self.indexed >= MAXIMUM_ENTRIES {
                return;
            }
        }
        let candidates = self.candidates.entry(prefix).or_insert_with(|| Candidates {
            entries: Vec::new(),
            words: vec![PREFIX_INSTRUCTIONS; inputs.stages.len()].into(),
        });
        let entries = &mut candidates.entries;
        self.indexed -= entries.len();
        entries.retain(|entry| entry.strong_count() != 0);
        self.indexed += entries.len();
        if entries
            .iter()
            .any(|entry| entry.as_ptr() == Arc::as_ptr(inputs))
        {
            return;
        }
        for (words, stage) in candidates.words.iter_mut().zip(&inputs.stages) {
            *words = (*words).min(Candidates::prefix_words(&stage.environment));
        }
        entries.push(Arc::downgrade(inputs));
        self.indexed += 1;
    }

    pub fn restore(
        &self,
        requests: &[(Option<u8>, CaptureRequest)],
        space: &MaxwellGpuMemory,
        memory: &mut MemoryCache,
        access: &mut nixe_gpu::MemoryAccess<'_>,
    ) -> Option<CapturedProgram> {
        let mut bytes = Vec::new();
        let mut locations = Vec::with_capacity(requests.len());
        for (group, request) in requests {
            let mut source = CaptureEnvironment {
                memory: MaxwellShaderMemoryView::new(space, memory, access),
                address: request.address,
                metadata: request.metadata.clone(),
                bindings: request.bindings.clone(),
                code_dependencies: Vec::new(),
                textures: BTreeMap::new(),
            };
            source.initialize_metadata().ok()?;
            let entry = Location::new(source.metadata.entry_address().ok()?).ok()?;
            let instruction = source.read_instruction(entry.address()).ok()?;
            Self::append_prefix(&mut bytes, &source.metadata, *group, instruction);
            locations.push((source.shader_stage(), source.metadata.start_address, entry));
        }
        let prefix = ContentId::of(b"nixe-capture-code-prefix-v1", &[&bytes]);
        let candidates = self.candidates.get(&prefix)?;
        // Read the small common code prefix once, before installing freshness
        // proofs for a candidate. Rejected programs must not repeatedly protect
        // overlapping code or clone their full instruction observations.
        let mut probes = Vec::with_capacity(requests.len());
        for (((_, request), &(stage, start, entry)), &words) in
            requests.iter().zip(&locations).zip(&candidates.words)
        {
            let first = entry.address().0;
            let last = entry.advance(i32::from(words.checked_sub(1)?)).address().0;
            let size = last.checked_sub(first)?.checked_add(8)?;
            let address = request
                .address
                .checked_add(u64::from(first.checked_sub(start.0)?))?;
            let bytes = MaxwellShaderMemoryView::new(space, memory, access)
                .read_data(stage, address, size as usize)
                .ok()?;
            let mut probe = [0; PREFIX_INSTRUCTIONS as usize];
            for index in 0..words {
                let at = entry
                    .advance(i32::from(index))
                    .address()
                    .0
                    .checked_sub(first)? as usize;
                probe[usize::from(index)] = u64::from_le_bytes(bytes[at..at + 8].try_into().ok()?);
            }
            probes.push(probe);
        }
        let mut tested = 0;
        for candidate in &candidates.entries {
            let Some(inputs) = candidate.upgrade() else {
                continue;
            };
            if !inputs
                .stages
                .iter()
                .zip(&locations)
                .zip(&probes)
                .zip(&candidates.words)
                .all(|(((input, &(_, _, entry)), probe), &words)| {
                    (0..words).all(|index| {
                        input
                            .environment
                            .observations()
                            .instructions
                            .get(&entry.advance(i32::from(index)).address())
                            == Some(&probe[usize::from(index)])
                    })
                })
            {
                continue;
            }
            tested += 1;
            if let Some(capture) = Self::restore_candidate(inputs, requests, space, memory, access)
            {
                nixe_gpu::trace::event(
                    "shader_capture_restore",
                    requests.len() as u64,
                    tested,
                    candidates.entries.len() as u64,
                );
                return Some(capture);
            }
        }
        None
    }

    fn restore_candidate(
        inputs: Arc<ShaderInputs>,
        requests: &[(Option<u8>, CaptureRequest)],
        space: &MaxwellGpuMemory,
        memory: &mut MemoryCache,
        access: &mut nixe_gpu::MemoryAccess<'_>,
    ) -> Option<CapturedProgram> {
        if inputs.stages.len() != requests.len() {
            return None;
        }
        let mut stages = Vec::with_capacity(requests.len());
        for (input, (group, request)) in inputs.stages.iter().zip(requests) {
            if input.binding_group != *group {
                return None;
            }
            let mut source = CaptureEnvironment {
                memory: MaxwellShaderMemoryView::new(space, memory, access),
                address: request.address,
                metadata: request.metadata.clone(),
                bindings: request.bindings.clone(),
                code_dependencies: Vec::new(),
                textures: BTreeMap::new(),
            };
            // Stale cached instruction spans can be unmapped; normal capture
            // determines the actual guest-visible outcome after a cache miss.
            if !source.matches_snapshot(&input.environment).ok()? {
                return None;
            }
            stages.push(CapturedStage {
                address: request.address,
                request_metadata: request.metadata.clone(),
                environment: Arc::clone(&input.environment),
                info: Arc::clone(&input.info),
                code_dependencies: source.code_dependencies.into(),
            });
        }
        Some(CapturedProgram {
            inputs,
            stages: stages.into(),
        })
    }

    /// Flush at owner shutdown, outside frame generation. The canonical
    /// identity includes compiler specialization and survives VA relocation.
    pub fn persist(&self, inputs: &ShaderInputs) {
        let key = inputs.identity;
        let bytes = shader_encoding::encode_inputs(inputs);
        if self
            .store
            .read(key)
            .is_some_and(|previous| previous == bytes)
        {
            return;
        }
        if let Err(error) = self.store.write(key, &bytes) {
            log::warn!("frontend shader capture cache: {error}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejected_code_candidates_do_not_install_full_shader_observations() {
        use nixe_shader_recompiler::maxwell::decode::Opcode;
        let (allocation, space, base) = crate::shader::test_support::mapped_memory();
        let configuration = Configuration::default();
        let mut archive = CaptureArchive::new(
            &std::env::temp_dir(),
            ContentId::of(b"prefix-observations", &[]),
            configuration.clone(),
            1 << 20,
        );
        let request = || {
            [(
                None,
                CaptureRequest {
                    address: base,
                    metadata: EnvironmentMetadata {
                        stage: Stage::Compute,
                        ..Default::default()
                    },
                    bindings: CaptureBindings::default(),
                },
            )]
        };
        let mut compiler = Compiler::new().unwrap();
        let mut memory = MemoryCache::default();
        let mut access = crate::test_memory::access(space.memory().machine());
        let mut captured = Vec::new();
        let mov = Opcode::MOV32I.encoding().value | 7 << 16 | 15 << 12;
        for variant in 1..=24 {
            let entry = Location::new(nixe_shader_recompiler::types::ProgramOffset(8)).unwrap();
            for index in 0..variant + 3 {
                let instruction = if index == variant + 2 {
                    0xe300_0000_0007_000f
                } else {
                    mov | if index == 1 { variant as u64 } else { 0 }
                };
                crate::test_memory::write(
                    &allocation,
                    entry.advance(index).address().0 as u64,
                    &instruction.to_le_bytes(),
                )
                .unwrap();
            }
            let capture = capture_program(
                &mut compiler,
                space.memory(),
                request(),
                &configuration,
                &mut memory,
                &mut access,
            )
            .unwrap();
            archive.remember(&capture.inputs);
            captured.push(capture);
        }
        let before = space.memory().machine().tracking_statistics();
        let restored = archive
            .restore(
                &request(),
                space.memory(),
                &mut MemoryCache::default(),
                &mut access,
            )
            .unwrap();
        let after = space.memory().machine().tracking_statistics();
        assert_eq!(
            restored.inputs.identity,
            captured.last().unwrap().inputs.identity
        );
        assert!(after.observer_registrations - before.observer_registrations <= 3);
        assert!(Arc::ptr_eq(
            &restored.stages[0].environment,
            &restored.inputs.stages[0].environment
        ));
        assert!(Arc::ptr_eq(
            &restored.stages[0].info,
            &restored.inputs.stages[0].info
        ));
    }

    #[test]
    fn restored_code_uses_the_current_mapping_after_a_gpu_va_remap() {
        let (allocation, mut space, base) = crate::shader::test_support::mapped_memory();
        let machine = space.memory().machine().clone();
        let exit = 0xe300_0000_0007_000fu64;
        crate::test_memory::write(&allocation, 8, &exit.to_le_bytes()).unwrap();
        let request = || {
            [(
                None,
                CaptureRequest {
                    address: base,
                    metadata: EnvironmentMetadata {
                        stage: Stage::Compute,
                        ..Default::default()
                    },
                    bindings: CaptureBindings::default(),
                },
            )]
        };
        let configuration = Configuration::default();
        let directory = std::env::temp_dir().join(format!(
            "nixe-capture-remap-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut archive = CaptureArchive::new(
            &directory,
            ContentId::of(b"remap", &[]),
            configuration.clone(),
            1 << 20,
        );
        let mut memory = MemoryCache::default();
        let mut access = crate::test_memory::access(&machine);
        let original = capture_program(
            &mut Compiler::new().unwrap(),
            space.memory(),
            request(),
            &configuration,
            &mut memory,
            &mut access,
        )
        .unwrap();
        archive.persist(&original.inputs);
        let mut inputs = nixe_gpu::WorkingSet::new(32, 1 << 20);
        archive.load(&mut inputs);
        let first = archive
            .restore(&request(), space.memory(), &mut memory, &mut access)
            .unwrap();
        let replacement = crate::test_memory::allocate(&machine, 0x1000);
        crate::test_memory::write(&replacement, 8, &exit.to_le_bytes()).unwrap();
        let address = space.address(base).unwrap();
        space.unmap(address).unwrap();
        space.reserve(1, 0x1000, 1, base).unwrap();
        space
            .map(crate::MaxwellMapRequest {
                allocation: crate::MaxwellAllocationId::new(2),
                backing: crate::test_memory::ranges(&replacement),
                backing_offset: 0,
                size: 0x1000,
                allocation_alignment: 0x1000,
                page_size: 0,
                kind: 0,
                cacheable: false,
                permissions: nixe_memory::MemoryPermissions::READ_WRITE,
                fixed_offset: Some(address),
            })
            .unwrap();
        assert!(!first.stages[0].is_current(space.memory()));
        let remapped = archive
            .restore(&request(), space.memory(), &mut memory, &mut access)
            .unwrap();
        assert_eq!(remapped.inputs.identity, first.inputs.identity);
        assert!(remapped.stages[0].is_current(space.memory()));
        let alias = space
            .map(crate::MaxwellMapRequest {
                allocation: crate::MaxwellAllocationId::new(2),
                backing: crate::test_memory::ranges(&replacement),
                backing_offset: 0,
                size: 0x1000,
                allocation_alignment: 0x1000,
                page_size: 0,
                kind: 0,
                cacheable: false,
                permissions: nixe_memory::MemoryPermissions::READ_WRITE,
                fixed_offset: None,
            })
            .unwrap();
        let mut relocated = request();
        relocated[0].1.address = alias.offset().get();
        assert_ne!(relocated[0].1.address, base);
        let reused = archive
            .restore(&relocated, space.memory(), &mut memory, &mut access)
            .unwrap();
        assert_eq!(reused.inputs.identity, first.inputs.identity);
        assert!(reused.stages[0].request_is_current(&relocated[0].1, space.memory()));
        crate::test_memory::write(&replacement, 8, &0u64.to_le_bytes()).unwrap();
        assert!(
            archive
                .restore(&request(), space.memory(), &mut memory, &mut access)
                .is_none()
        );
        std::fs::remove_dir_all(directory).unwrap();
    }
}
