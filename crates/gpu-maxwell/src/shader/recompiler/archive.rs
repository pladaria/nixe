//! FileEnvironment serialization and pregame loading by native pipeline key.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/vk_pipeline_cache.cpp#L928-L968
use nixe_gpu::{
    persistent_cache::Store,
    shader_compilation::{Configuration, ShaderInputs},
    shader_encoding,
    shader_identity::ContentId,
};
use std::sync::{Arc, mpsc};
use std::thread::{self, JoinHandle};
const MAXIMUM_ENTRIES: usize = 2048;

pub struct FileEnvironment {
    pub key: Vec<u8>,
    pub inputs: Arc<ShaderInputs>,
}
pub struct EnvironmentArchive {
    store: Arc<Store>,
    configuration: Configuration,
    sender: Option<mpsc::Sender<FileEnvironment>>,
    worker: Option<JoinHandle<()>>,
}
impl EnvironmentArchive {
    pub fn new(
        directory: &std::path::Path,
        namespace: ContentId,
        configuration: Configuration,
        budget: u64,
    ) -> std::io::Result<Self> {
        let store = Arc::new(Store::new(
            directory.join("environments-v2").join(namespace.hex()),
            budget,
        ));
        let writer = Arc::clone(&store);
        let (sender, receiver) = mpsc::channel::<FileEnvironment>();
        let worker = thread::Builder::new()
            .name("nixe-vk-serialize".into())
            .spawn(move || {
                while let Ok(environment) = receiver.recv() {
                    let key = ContentId::of(b"nixe-pipeline-environment-v2", &[&environment.key]);
                    let mut bytes = (environment.key.len() as u32).to_le_bytes().to_vec();
                    bytes.extend_from_slice(&environment.key);
                    bytes.extend_from_slice(&shader_encoding::encode_inputs(&environment.inputs));
                    if let Err(error) = writer.write(key, &bytes) {
                        log::warn!("shader environment cache write: {error}");
                    }
                }
            })?;
        Ok(Self {
            store,
            configuration,
            sender: Some(sender),
            worker: Some(worker),
        })
    }
    /// Loaded environments populate the actual native pipeline cache before
    /// gameplay; there is no second file/observation cache consulted by draws.
    pub fn load(&self) -> Vec<FileEnvironment> {
        let keys = match self.store.keys(MAXIMUM_ENTRIES) {
            Ok(keys) => keys,
            Err(error) => {
                log::warn!("shader environment directory: {error}");
                return Vec::new();
            }
        };
        keys.into_iter()
            .filter_map(|key| {
                let bytes = self.store.read(key)?;
                let size = u32::from_le_bytes(bytes.get(..4)?.try_into().ok()?) as usize;
                let data = bytes.get(4..4usize.checked_add(size)?)?;
                if ContentId::of(b"nixe-pipeline-environment-v2", &[data]) != key {
                    return None;
                }
                let inputs =
                    shader_encoding::decode_inputs(bytes.get(4 + size..)?, &self.configuration)?;
                Some(FileEnvironment {
                    key: data.to_vec(),
                    inputs: Arc::new(inputs),
                })
            })
            .collect()
    }
    pub fn serialize(&self, key: Vec<u8>, inputs: Arc<ShaderInputs>) {
        if self
            .sender
            .as_ref()
            .unwrap()
            .send(FileEnvironment { key, inputs })
            .is_err()
        {
            log::warn!("shader environment serialization worker stopped");
        }
    }
}
impl Drop for EnvironmentArchive {
    fn drop(&mut self) {
        self.sender.take();
        if let Some(worker) = self.worker.take()
            && worker.join().is_err()
        {
            log::warn!("shader environment serialization worker panicked");
        }
    }
}
