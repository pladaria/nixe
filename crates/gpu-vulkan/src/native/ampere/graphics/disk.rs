//! Persistent native environments use the same replay and worker pool as live misses.
use super::*;
use crate::cache::{
    archive::EnvironmentArchive,
    encoding::{Reader, wide, word},
};
use nixe_gpu::shader_compilation::Configuration;
use std::{path::Path, sync::atomic::AtomicBool, time::Instant};

// Disjoint from the Maxwell key tags; revise when the native pipeline contract changes.
const GRAPHICS: u32 = 0x0100_000a;
const COMPUTE: u32 = 0x0100_000b;
fn header(tag: u32, profile: &DeviceProfile) -> Vec<u8> {
    let mut bytes = Vec::new();
    for value in [
        tag,
        profile.driver_id.as_raw() as u32,
        profile.driver_version,
    ] {
        word(&mut bytes, value);
    }
    bytes
}
pub(in crate::native::ampere) fn compute_key(hash: u64, profile: &DeviceProfile) -> Vec<u8> {
    let mut bytes = header(COMPUTE, profile);
    wide(&mut bytes, hash);
    bytes
}
impl Key {
    pub(super) fn encode(&self, profile: &DeviceProfile) -> Vec<u8> {
        let mut bytes = header(GRAPHICS, profile);
        for hash in self.shaders {
            wide(&mut bytes, hash);
        }
        for mask in self.gp_passthrough_mask {
            word(&mut bytes, mask);
        }
        self.graphics.encode(&mut bytes);
        for kind in self.types {
            word(&mut bytes, kind as u32);
        }
        for value in [
            self.topology as u32,
            self.ndc as u32,
            self.viewport,
            self.point_size,
        ] {
            word(&mut bytes, value);
        }
        bytes
    }
    fn decode(r: &mut Reader<'_>, scale: [u32; 2]) -> Option<Self> {
        let mut shaders = [0; 6];
        for hash in &mut shaders {
            *hash = r.u64()?;
        }
        let mut gp_passthrough_mask = [0; 8];
        for mask in &mut gp_passthrough_mask {
            *mask = r.u32()?;
        }
        let graphics = crate::cache::GraphicsState::decode(r)?;
        let mut types = [AttributeType::Float; 32];
        for kind in &mut types {
            *kind = AttributeType::try_from(r.u32()?).ok()?;
        }
        let topology = InputTopology::try_from(r.u32()?).ok()?;
        let ndc = match r.u32()? {
            0 => false,
            1 => true,
            _ => return None,
        };
        Some(Self {
            shaders,
            gp_passthrough_mask,
            graphics,
            types,
            topology,
            ndc,
            viewport: r.u32()?,
            point_size: r.u32()?,
            scale,
        })
    }
}
enum DiskKey {
    Graphics(Box<Key>),
    Compute(u64),
}
impl DiskKey {
    fn decode(bytes: &[u8], profile: &DeviceProfile, scale: [u32; 2]) -> Option<Self> {
        let mut r = Reader(bytes);
        let tag = r.u32()?;
        if r.u32()? != profile.driver_id.as_raw() as u32 || r.u32()? != profile.driver_version {
            return None;
        }
        let result = match tag {
            GRAPHICS => Self::Graphics(Box::new(Key::decode(&mut r, scale)?)),
            COMPUTE => Self::Compute(r.u64()?),
            _ => return None,
        };
        r.0.is_empty().then_some(result)
    }
}
enum Loaded {
    Graphics(Box<Key>, Box<Pipeline>),
    Compute(u64, Arc<super::super::compute::Pipeline>),
}
impl Cache {
    pub(in crate::native) fn load(
        &mut self,
        directory: &Path,
        compiler: &Arc<crate::cache::PipelineCompiler>,
        profile: &DeviceProfile,
        configuration: Configuration,
        stop: &AtomicBool,
    ) {
        let start = Instant::now();
        log::info!(
            "loading Ampere shader and pipeline cache: {}",
            directory.display()
        );
        let mut archive = match EnvironmentArchive::open(directory, Arc::clone(compiler)) {
            Ok(archive) => archive,
            Err(error) => {
                log::warn!("Ampere shader environment cache load: {error}");
                return;
            }
        };
        let scale = [
            configuration.settings.resolution_info.up_scale,
            configuration.settings.resolution_info.down_shift,
        ];
        let configuration = Arc::new(configuration);
        let (sender, receiver) = std::sync::mpsc::channel();
        let mut total = 0;
        archive.load(stop, |mut record| {
            let Some(key) = DiskKey::decode(&record.key, profile, scale) else {
                return;
            };
            if let DiskKey::Graphics(key) = &key
                && compiler.normalize_graphics(key.graphics.clone()) != key.graphics
            {
                return;
            }
            total += 1;
            let compiler_job = Arc::clone(compiler);
            let configuration = Arc::clone(&configuration);
            let sender = sender.clone();
            if let Err(error) = compiler.queue(move |worker| {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let mut pools = compiler::Compiler::new().map_err(compilation)?;
                    let mut observations = Vec::with_capacity(record.inputs.stages.len());
                    for snapshot in &mut record.inputs.stages {
                        if snapshot.metadata().isa != ShaderIsa::Ampere {
                            return Err(compilation("non-native environment in native cache"));
                        }
                        // Flow constants participate in the program identity and are checked
                        // with that capture; do not check them twice on every pipeline lookup.
                        let mut recorded = RecordingEnvironment::new(snapshot);
                        compiler::ampere::flow::analyze(&mut recorded).map_err(compilation)?;
                        let flow = recorded.into_snapshot().into_observations().cbuf_words;
                        let mut dependency = shader::dependencies(snapshot);
                        dependency
                            .cbuf_words
                            .retain(|word, _| !flow.contains_key(word));
                        observations.push((
                            nixe_gpu::shader_compilation::neutral_stage(snapshot.metadata().stage)
                                as u8,
                            dependency,
                        ));
                    }
                    let ids = record
                        .inputs
                        .translate(&mut pools, &configuration)
                        .map_err(compilation)?;
                    match key {
                        DiskKey::Graphics(key) => {
                            let pipeline = Pipeline::build(
                                &compiler_job,
                                &mut pools,
                                &configuration,
                                &key,
                                &ids,
                                observations,
                                Some(worker),
                            )?;
                            Ok(Loaded::Graphics(key, Box::new(pipeline)))
                        }
                        DiskKey::Compute(hash)
                            if ids.len() == 1
                                && observations.len() == 1
                                && observations[0].0 == 5 =>
                        {
                            let pipeline = super::super::compute::Pipeline::build(
                                &compiler_job,
                                &mut pools,
                                &configuration,
                                ids[0],
                                observations.remove(0).1,
                                Some(worker),
                            )?;
                            Ok(Loaded::Compute(hash, Arc::new(pipeline)))
                        }
                        DiskKey::Compute(_) => {
                            Err(compilation("invalid native compute environment"))
                        }
                    }
                }))
                .unwrap_or_else(|_| {
                    Err(compilation("cached native pipeline construction panicked"))
                });
                let _ = sender.send(result);
            }) {
                log::warn!("cached native pipeline queue: {error}");
            }
        });
        log::info!(
            "read and queued {total} Ampere pipeline environments in {:?}",
            start.elapsed()
        );
        drop(sender);
        let mut loaded = 0;
        for result in receiver {
            match result {
                Ok(Loaded::Graphics(key, pipeline)) => {
                    self.pipelines
                        .entry(*key)
                        .or_default()
                        .push(Rc::from(pipeline));
                    loaded += 1;
                }
                Ok(Loaded::Compute(hash, pipeline)) => {
                    self.compute
                        .pipelines
                        .entry((hash, scale[0], scale[1]))
                        .or_default()
                        .push(pipeline);
                    loaded += 1;
                }
                Err(error) => log::warn!("cached native pipeline construction: {error}"),
            }
        }
        archive.checkpoint();
        self.archive = Some(archive);
        log::info!(
            "loaded {loaded}/{total} Ampere pipeline environments before guest execution in {:?}",
            start.elapsed()
        );
    }
}
