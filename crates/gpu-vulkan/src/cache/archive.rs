//! Guest environments retain the data needed to rebuild a missing linked artifact.
//! Load and queue one record at a time, as VideoCommon::LoadPipelines.
//! https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/shader_environment.cpp
use super::{CompilePool, PERSISTENCE_VERSION, PipelineCompiler};
use nixe_gpu::{shader_compilation::ShaderInputs, shader_encoding};
use std::{
    io::{BufReader, Read, Write},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};
/// Let the graphics owner present UI updates while its compiler workers are active.
pub(crate) struct WarmupUi<F> {
    service: F,
    next: Instant,
}
impl<F: FnMut()> WarmupUi<F> {
    pub(crate) fn new(service: F) -> Self {
        Self {
            service,
            next: Instant::now(),
        }
    }
    pub(crate) fn poll(&mut self) {
        let now = Instant::now();
        if now >= self.next {
            (self.service)();
            self.next = now + Duration::from_millis(16);
        }
    }
    pub(crate) fn receive<T>(&mut self, rx: &std::sync::mpsc::Receiver<T>) -> Option<T> {
        loop {
            self.poll();
            match rx.recv_timeout(self.next.saturating_duration_since(Instant::now())) {
                Ok(value) => return Some(value),
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => return None,
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            }
        }
    }
}
const MAGIC: &[u8; 8] = b"NXVKENV1";
pub(crate) struct FileEnvironment {
    bytes: Vec<u8>,
    key_len: usize,
    linked: super::linked::LinkedEntry,
    stages: Arc<super::linked::LoadedStages>,
    restored: Arc<AtomicUsize>,
}
impl FileEnvironment {
    pub fn key(&self) -> &[u8] {
        &self.bytes[..self.key_len]
    }
    pub fn inputs(&self) -> Result<ShaderInputs, super::CompileError> {
        shader_encoding::decode_inputs(&self.bytes[self.key_len..])
            .ok_or_else(|| super::CompileError::Failed("invalid shader environment".into()))
    }
    pub fn shader(
        self,
        compiler: &PipelineCompiler,
        configuration: &nixe_gpu::shader_compilation::Configuration,
        runtime: Option<&nixe_shader_recompiler::passes::GraphicsRuntimeState>,
        decoded: Option<ShaderInputs>,
    ) -> Result<Arc<super::ShaderArtifact>, super::CompileError> {
        if let Some(shader) = self.linked.load(&self.stages) {
            self.restored.fetch_add(1, Ordering::Relaxed);
            return Ok(shader);
        }
        let inputs = match decoded {
            Some(inputs) => inputs,
            None => self.inputs()?,
        };
        let linked_entry = self.linked;
        drop(self.bytes);
        let _job = compiler.compiling_shaders(inputs.stages.len());
        let mut pools = nixe_shader_recompiler::Compiler::new()
            .map_err(|e| super::CompileError::Failed(e.to_string().into()))?;
        let ids = inputs
            .translate(&mut pools, configuration)
            .map_err(|e| super::CompileError::Failed(e.to_string().into()))?;
        let shader = super::compile_guest(&mut pools, &ids, configuration, runtime)?;
        let linked = shader.clone();
        if let Err(error) = compiler.serialization().submit(move |_| {
            if let Err(e) = linked_entry.save(&linked) {
                log::warn!("linked shader cache write: {e}");
            }
        }) {
            log::warn!("linked shader serialization: {error}");
        }
        Ok(shader)
    }
}
#[derive(Default)]
struct FlushStatus {
    in_flight: AtomicBool,
    bytes: AtomicUsize,
}
pub(crate) struct EnvironmentArchive {
    store: super::linked::Store,
    restored: Arc<AtomicUsize>,
    path: PathBuf,
    serialization: Arc<CompilePool>,
    compiler: Arc<PipelineCompiler>,
    flush: Arc<FlushStatus>,
    pending: usize,
    last_flush: Instant,
}
impl EnvironmentArchive {
    pub fn open(
        directory: &Path,
        compiler: Arc<PipelineCompiler>,
        profile: &crate::DeviceProfile,
    ) -> std::io::Result<Self> {
        std::fs::create_dir_all(directory)?;
        Ok(Self {
            store: super::linked::Store::new(directory, profile),
            restored: Arc::default(),
            path: directory.join("vulkan.bin"),
            serialization: compiler.serialization(),
            compiler,
            flush: Arc::default(),
            pending: 0,
            last_flush: Instant::now(),
        })
    }
    pub fn load(&self, stop: &AtomicBool, scale: [u32; 2], mut load: impl FnMut(FileEnvironment)) {
        let stages = Arc::<super::linked::LoadedStages>::default();
        let mut original_len = 0;
        let mut valid_len = 0;
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
            file.lock_shared()?;
            original_len = file.metadata()?.len();
            let mut remaining = original_len.checked_sub(12).ok_or_else(invalid)?;
            let mut reader = BufReader::new(file);
            let mut header = [0; 12];
            reader.read_exact(&mut header)?;
            if &header[..8] != MAGIC || header[8..] != PERSISTENCE_VERSION.to_le_bytes() {
                return Err(invalid());
            }
            valid_len = 12;
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
                valid_len = original_len - remaining;
                load(FileEnvironment {
                    stages: stages.clone(),
                    restored: self.restored.clone(),
                    key_len,
                    linked: self
                        .store
                        .entry(&bytes[..key_len], &bytes[key_len..], scale),
                    bytes,
                });
            }
            Ok(())
        };
        if let Err(error) = result() {
            log::warn!(
                "shader environment cache load ({}): {error}",
                self.path.display()
            );
            if matches!(
                error.kind(),
                std::io::ErrorKind::InvalidData | std::io::ErrorKind::UnexpectedEof
            ) {
                let repair = || -> std::io::Result<()> {
                    let file = std::fs::OpenOptions::new().write(true).open(&self.path)?;
                    file.lock()?;
                    if file.metadata()?.len() == original_len {
                        file.set_len(valid_len)?;
                    }
                    Ok(())
                };
                if let Err(e) = repair() {
                    log::warn!("cannot truncate damaged shader cache tail: {e}");
                }
            }
        }
    }
    pub fn serialize(
        &mut self,
        key: Vec<u8>,
        inputs: Arc<ShaderInputs>,
        scale: [u32; 2],
        shader: Arc<super::ShaderArtifact>,
        new_environment: bool,
    ) {
        if !inputs.serializable {
            return;
        }
        let path = self.path.clone();
        let store = self.store.clone();
        if let Err(error) = self.serialization.submit(move |_| {
            let write = || -> std::io::Result<()> {
                let bytes = shader_encoding::encode_inputs(&inputs);
                let entry = store.entry(&key, &bytes, scale);
                if let Err(error) = entry.save(&shader) {
                    log::warn!("linked shader cache write: {error}");
                }
                if !new_environment {
                    return Ok(());
                }
                let mut file = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&path)?;
                file.lock()?;
                let end = file.metadata()?.len();
                let mut append = || -> std::io::Result<()> {
                    if end == 0 {
                        file.write_all(MAGIC)?;
                        file.write_all(&PERSISTENCE_VERSION.to_le_bytes())?;
                    }
                    file.write_all(&(key.len() as u32).to_le_bytes())?;
                    file.write_all(&(bytes.len() as u64).to_le_bytes())?;
                    file.write_all(&key)?;
                    file.write_all(&bytes)
                };
                if let Err(error) = append() {
                    file.set_len(end)?;
                    return Err(error);
                }
                Ok(())
            };
            if let Err(error) = write() {
                log::warn!("shader environment cache write: {error}");
            }
        }) {
            log::warn!("shader environment serialization: {error}");
        }
        self.pending += 1;
        if self.pending < 128 {
            return;
        }
        let interval = Duration::from_secs(
            (self.flush.bytes.load(Ordering::Relaxed) / (1024 * 1024)).max(30) as u64,
        );
        if self.last_flush.elapsed() < interval || self.flush.in_flight.swap(true, Ordering::AcqRel)
        {
            return;
        }
        self.pending = 0;
        self.last_flush = Instant::now();
        let flush = Arc::clone(&self.flush);
        if let Err(error) = self.compiler.queue_checkpoint(move |bytes| {
            flush.bytes.store(bytes, Ordering::Relaxed);
            flush.in_flight.store(false, Ordering::Release);
        }) {
            self.flush.in_flight.store(false, Ordering::Release);
            log::warn!("Vulkan driver cache serialization: {error}");
        }
    }
    pub fn restored(&self) -> usize {
        self.restored.load(Ordering::Relaxed)
    }
    pub fn checkpoint(&mut self) {
        match self.compiler.checkpoint() {
            Ok(bytes) => {
                self.flush.bytes.store(bytes, Ordering::Relaxed);
                self.last_flush = Instant::now();
            }
            Err(error) => log::warn!("Vulkan driver cache write: {error}"),
        }
    }
}
