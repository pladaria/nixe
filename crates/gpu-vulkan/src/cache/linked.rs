//! Content-addressed, linked shader stages. The environment remains the recovery source.
use super::{CompiledStage, GuestStageMetadata, ShaderArtifact};
use bincode::Options;
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{self, Read, Write},
    path::{Path, PathBuf},
    sync::Arc,
};
type Hash = [u8; 32];
const LIMIT: u64 = 64 << 20;
fn codec() -> impl Options {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_limit(LIMIT)
        .reject_trailing_bytes()
}
fn path(dir: &Path, hash: &Hash) -> PathBuf {
    use std::fmt::Write;
    let mut name = String::with_capacity(64);
    for byte in hash {
        write!(&mut name, "{byte:02x}").unwrap();
    }
    dir.join(name)
}
fn save(path: &Path, data: &[u8]) -> io::Result<()> {
    let mut f = tempfile::NamedTempFile::new_in(path.parent().unwrap())?;
    f.write_all(data)?;
    f.persist(path)?;
    Ok(())
}
fn matches_file(path: &Path, bytes: &[u8]) -> io::Result<bool> {
    let mut file = fs::File::open(path)?;
    if file.metadata()?.len() != bytes.len() as u64 {
        return Ok(false);
    }
    let mut buffer = [0; 8192];
    for chunk in bytes.chunks(buffer.len()) {
        let block = &mut buffer[..chunk.len()];
        file.read_exact(block)?;
        if block != chunk {
            return Ok(false);
        }
    }
    Ok(true)
}
#[derive(serde::Serialize)]
struct StageBytes<'a> {
    stage: u8,
    push: u32,
    words: &'a [u32],
    metadata: &'a GuestStageMetadata,
}
#[derive(serde::Deserialize)]
struct StoredStage {
    stage: u8,
    push: u32,
    words: Box<[u32]>,
    metadata: GuestStageMetadata,
}
impl StoredStage {
    fn restore(self) -> Option<CompiledStage> {
        use nixe_shader_recompiler::{Stage, StageArtifact};
        let stage = *[
            Stage::VertexB,
            Stage::TessellationControl,
            Stage::TessellationEval,
            Stage::Geometry,
            Stage::Fragment,
            Stage::Compute,
        ]
        .get(self.stage as usize)?;
        if self.words.len() < 5 || self.words[0] != 0x07230203 {
            return None;
        }
        let GuestStageMetadata {
            bindings,
            info,
            interface,
            execution,
        } = self.metadata;
        CompiledStage::from_guest(StageArtifact {
            stage,
            words: self.words,
            info,
            bindings,
            interface,
            execution,
            push_constant_bytes: self.push,
        })
        .ok()
    }
}
#[derive(Clone)]
pub(super) struct Store {
    root: Arc<PathBuf>,
}
impl Store {
    pub(super) fn new(dir: &Path, profile: &crate::DeviceProfile) -> Self {
        let mut h = Sha256::new();
        h.update(b"NX linked shaders 1");
        h.update(nixe_shader_recompiler::artifact::ARTIFACT_ABI_VERSION.to_le_bytes());
        h.update(format!("{:?}", super::disk_compiler_profile(profile)));
        Self {
            root: Arc::new(path(&dir.join("spirv"), &h.finalize().into())),
        }
    }
    pub(super) fn entry(&self, key: &[u8], inputs: &[u8], scale: [u32; 2]) -> LinkedEntry {
        let mut h = Sha256::new();
        for value in scale {
            h.update(value.to_le_bytes());
        }
        h.update((key.len() as u64).to_le_bytes());
        h.update(key);
        h.update(inputs);
        LinkedEntry {
            root: self.root.clone(),
            id: h.finalize().into(),
        }
    }
}
type StageCell = std::sync::OnceLock<Option<Arc<CompiledStage>>>;
/// One load session may queue many pipelines sharing the same immutable stage.
#[derive(Default)]
pub(super) struct LoadedStages(std::sync::Mutex<std::collections::HashMap<Hash, Arc<StageCell>>>);
pub(crate) struct LinkedEntry {
    root: Arc<PathBuf>,
    id: Hash,
}
impl LinkedEntry {
    pub(super) fn load(&self, loaded: &LoadedStages) -> Option<Arc<ShaderArtifact>> {
        let index = path(&self.root, &self.id).with_extension("stages");
        let size = fs::metadata(&index).ok()?.len();
        if !(64..=7 * 32).contains(&size) || !size.is_multiple_of(32) {
            return None;
        }
        let index = fs::read(index).ok()?;
        let (checksum, refs) = index.split_at_checked(32)?;
        if index.len() != size as usize || self.checksum(refs).as_slice() != checksum {
            return None;
        }
        let stages = refs
            .chunks_exact(32)
            .map(|key| {
                let hash: Hash = key.try_into().ok()?;
                let entry = loaded.0.lock().unwrap().entry(hash).or_default().clone();
                entry
                    .get_or_init(|| {
                        let p = path(&self.root, &hash);
                        if fs::metadata(&p).ok()?.len() > LIMIT {
                            return None;
                        }
                        let bytes = fs::read(p).ok()?;
                        if Sha256::digest(&bytes).as_slice() != key {
                            return None;
                        }
                        Some(Arc::new(
                            codec().deserialize::<StoredStage>(&bytes).ok()?.restore()?,
                        ))
                    })
                    .clone()
            })
            .collect::<Option<Box<[_]>>>()?;
        Some(Arc::new(ShaderArtifact::new(stages)))
    }
    pub(crate) fn save(&self, shader: &ShaderArtifact) -> io::Result<()> {
        fs::create_dir_all(&*self.root)?;
        let mut index = Vec::with_capacity((1 + shader.stages.len()) * 32);
        index.resize(32, 0);
        for stage in &shader.stages {
            let metadata = stage.guest.as_deref().ok_or(io::ErrorKind::InvalidData)?;
            let bytes = codec()
                .serialize(&StageBytes {
                    stage: stage.stage as u8,
                    push: stage.module.push_constant_bytes,
                    words: &stage.module.words,
                    metadata,
                })
                .map_err(io::Error::other)?;
            let key: Hash = Sha256::digest(&bytes).into();
            let target = path(&self.root, &key);
            if !matches_file(&target, &bytes).unwrap_or(false) {
                save(&target, &bytes)?;
            }
            index.extend(key);
        }
        let checksum = self.checksum(&index[32..]);
        index[..32].copy_from_slice(&checksum);
        save(&path(&self.root, &self.id).with_extension("stages"), &index)
    }
    fn checksum(&self, refs: &[u8]) -> Hash {
        Sha256::new()
            .chain_update(self.id)
            .chain_update(refs)
            .finalize()
            .into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nixe_shader_recompiler::{Stage, StageArtifact};
    #[test]
    fn restores_reflection_and_recovers_a_damaged_stage() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store {
            root: Arc::new(dir.path().to_path_buf()),
        };
        let entry = store.entry(b"key", b"code", [1, 0]);
        let mut source = StageArtifact {
            stage: Stage::Compute,
            words: Box::new([0x07230203, 0x10000, 0, 1, 0]),
            info: Default::default(),
            bindings: Box::new([]),
            push_constant_bytes: 4,
            interface: Default::default(),
            execution: Default::default(),
        };
        source.execution.workgroup_size = [8, 4, 1];
        source.info.uses_local_memory = true;
        let artifact = ShaderArtifact::new(Box::new([CompiledStage::from_guest(source).unwrap()]));
        entry.save(&artifact).unwrap();
        let loaded = entry.load(&LoadedStages::default()).unwrap();
        assert_eq!(loaded.stages[0].workgroup_size, Some([8, 4, 1]));
        assert!(
            loaded.stages[0]
                .guest
                .as_ref()
                .unwrap()
                .info
                .uses_local_memory
        );
        assert!(
            store
                .entry(b"key", b"code", [2, 0])
                .load(&LoadedStages::default())
                .is_none()
        );
        let module = fs::read_dir(dir.path())
            .unwrap()
            .flatten()
            .map(|f| f.path())
            .find(|p| p.extension().is_none())
            .unwrap();
        fs::write(module, b"broken").unwrap();
        assert!(entry.load(&LoadedStages::default()).is_none());
        entry.save(&artifact).unwrap();
        assert!(entry.load(&LoadedStages::default()).is_some());
    }
}
