//! Serialized guest environments and pipeline keys, never translated SPIR-V.
//! Load and queue one record at a time, as VideoCommon::LoadPipelines.
//! https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/shader_environment.cpp
use super::{CompilePool, PERSISTENCE_VERSION};
use nixe_gpu::{shader_compilation::ShaderInputs, shader_encoding};
use std::{
    io::{BufReader, Read, Write},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};
const MAGIC: &[u8; 8] = b"NXVKENV1";
pub(crate) struct FileEnvironment {
    pub key: Vec<u8>,
    pub inputs: ShaderInputs,
}
pub(crate) struct EnvironmentArchive {
    path: PathBuf,
    serialization: Arc<CompilePool>,
}
impl EnvironmentArchive {
    pub fn open(directory: &Path, serialization: Arc<CompilePool>) -> std::io::Result<Self> {
        std::fs::create_dir_all(directory)?;
        Ok(Self {
            path: directory.join("vulkan.bin"),
            serialization,
        })
    }
    pub fn load(&self, stop: &AtomicBool, mut load: impl FnMut(FileEnvironment)) {
        let mut result = || -> std::io::Result<()> {
            let file = match std::fs::File::open(&self.path) {
                Ok(file) => file,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
                Err(error) => return Err(error),
            };
            let invalid = || {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "incompatible or truncated shader environment cache",
                )
            };
            let mut remaining = file.metadata()?.len().checked_sub(12).ok_or_else(invalid)?;
            let mut reader = BufReader::new(file);
            let mut header = [0; 12];
            reader.read_exact(&mut header)?;
            if &header[..8] != MAGIC || header[8..] != PERSISTENCE_VERSION.to_le_bytes() {
                return Err(invalid());
            }
            while remaining != 0 && !stop.load(Ordering::Relaxed) {
                remaining = remaining.checked_sub(12).ok_or_else(invalid)?;
                let mut lengths = [0; 12];
                reader.read_exact(&mut lengths)?;
                let key_len = u32::from_le_bytes(lengths[..4].try_into().unwrap()) as usize;
                let input_len = u64::from_le_bytes(lengths[4..].try_into().unwrap());
                let bytes = (key_len as u64)
                    .checked_add(input_len)
                    .ok_or_else(invalid)?;
                remaining = remaining.checked_sub(bytes).ok_or_else(invalid)?;
                let mut bytes = vec![0; usize::try_from(bytes).map_err(|_| invalid())?];
                reader.read_exact(&mut bytes)?;
                let inputs =
                    shader_encoding::decode_inputs(&bytes[key_len..]).ok_or_else(invalid)?;
                load(FileEnvironment {
                    key: bytes[..key_len].to_vec(),
                    inputs,
                });
            }
            Ok(())
        };
        if let Err(error) = result() {
            log::warn!(
                "shader environment cache load: {error}; discarding {}",
                self.path.display()
            );
            remove_failed_file(&self.path);
        }
    }
    pub fn serialize(&self, key: Vec<u8>, inputs: Arc<ShaderInputs>) {
        if !inputs.serializable {
            return;
        }
        let path = self.path.clone();
        if let Err(error) = self.serialization.submit(move |_| {
            let write = || -> std::io::Result<()> {
                let bytes = shader_encoding::encode_inputs(&inputs);
                let mut file = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&path)?;
                if file.metadata()?.len() == 0 {
                    file.write_all(MAGIC)?;
                    file.write_all(&PERSISTENCE_VERSION.to_le_bytes())?;
                }
                file.write_all(&(key.len() as u32).to_le_bytes())?;
                file.write_all(&(bytes.len() as u64).to_le_bytes())?;
                file.write_all(&key)?;
                file.write_all(&bytes)
            };
            if let Err(error) = write() {
                log::warn!("shader environment cache write: {error}");
                remove_failed_file(&path);
            }
        }) {
            log::warn!("shader environment serialization: {error}");
        }
    }
}
fn remove_failed_file(path: &Path) {
    if let Err(error) = std::fs::remove_file(path)
        && error.kind() != std::io::ErrorKind::NotFound
    {
        log::warn!(
            "cannot remove invalid shader environment cache {}: {error}",
            path.display()
        );
    }
}
