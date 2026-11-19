//! Per-title linked SPIR-V, submit-time reflection and physical pipeline recipes.
//! Records refer to immutable content. Guest addresses and Vulkan handles are process-local.
use super::*;
use crate::cache::{
    CompiledStage, ShaderArtifact,
    encoding::{Reader, wide, word},
};
use nixe_gpu::{
    ShaderStage,
    shader_module::{Binding, DescriptorKind, Module, ResourceId},
};
use sha2::{Digest as _, Sha256};
use std::{
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};

type Digest = [u8; 32];
type Variants = Vec<(Shapes, Rc<Translation>)>;
pub(in crate::native) struct Archive {
    directory: Arc<PathBuf>,
    writer: Arc<CompilePool>,
    programs: HashMap<Digest, Variants>,
    modules: HashMap<usize, Digest>,
}
const MODULE: u8 = 1;
const PROGRAM: u8 = 2;
const PIPELINE: u8 = 3;
const MAX_RECORD: u64 = 64 << 20;

pub(super) fn source_key(
    sources: &[nixe_nvn::StageSource],
    graphics: Option<&GraphicsRuntimeState>,
    resolution: crate::ResolutionScaling,
) -> Digest {
    let mut hash = Sha256::new();
    hash.update(resolution.up_scale.to_le_bytes());
    hash.update(resolution.down_shift.to_le_bytes());
    for source in sources {
        let shader = nixe_nvn::Shader {
            address: 0,
            ..source.shader
        };
        hash.update(bytemuck::bytes_of(&shader));
        hash.update(source.code_digest);
    }
    if let Some(g) = graphics {
        hash.update(g.vertex_input_types.map(|t| t as u8));
        hash.update(g.color_output_types.map(|t| t as u8));
        hash.update([g.input_topology as u8, g.ndc_minus_one_to_one as u8]);
        hash.update(g.point_size.to_bits().to_le_bytes());
    }
    hash.finalize().into()
}
fn hex(bytes: &Digest) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(64);
    for byte in bytes {
        write!(&mut s, "{byte:02x}").unwrap();
    }
    s
}
fn digest(r: &mut Reader<'_>) -> Option<Digest> {
    let (head, rest) = r.0.split_at_checked(32)?;
    r.0 = rest;
    head.try_into().ok()
}
fn stage_kind(n: u32) -> Option<ShaderStage> {
    use ShaderStage::*;
    [
        Vertex,
        TessellationControl,
        TessellationEvaluation,
        Geometry,
        Fragment,
        Compute,
    ]
    .get(n as usize)
    .copied()
}
fn descriptor_kind(n: u32) -> Option<DescriptorKind> {
    use DescriptorKind::*;
    [
        UniformBuffer,
        StorageBuffer,
        SampledImage,
        StorageImage,
        Sampler,
        CombinedImageSampler,
        UniformTexelBuffer,
        StorageTexelBuffer,
    ]
    .get(n as usize)
    .copied()
}
fn module_bytes(s: &CompiledStage) -> Vec<u8> {
    let mut out = vec![MODULE];
    for n in [
        s.stage as u32,
        s.shared_memory_bytes,
        s.output_control_points.unwrap_or(0),
        s.module.push_constant_bytes,
    ] {
        word(&mut out, n);
    }
    for n in s.workgroup_size.unwrap_or([0; 3]) {
        word(&mut out, n);
    }
    word(&mut out, s.module.bindings.len() as u32);
    for b in &s.module.bindings {
        for n in [
            b.binding,
            b.count,
            b.kind as u32,
            b.resource.index() as u32,
            b.sampler.map_or(u32::MAX, |v| v.index() as u32),
            u32::from(b.readable) | (u32::from(b.writable) << 1),
        ] {
            word(&mut out, n);
        }
    }
    for &w in &s.module.words {
        word(&mut out, w);
    }
    out
}
fn decode_module(r: &mut Reader<'_>) -> Option<Arc<CompiledStage>> {
    let stage = stage_kind(r.u32()?)?;
    let shared = r.u32()?;
    let points = r.u32()?;
    let push_constant_bytes = r.u32()?;
    let group = [r.u32()?, r.u32()?, r.u32()?];
    let bindings = r
        .list(4096, |r| {
            let binding = r.u32()?;
            let count = r.u32()?;
            let kind = descriptor_kind(r.u32()?)?;
            let resource = ResourceId::new(r.u32()?);
            let sampler = r.u32()?;
            let access = r.u32()?;
            Some(Binding {
                resource,
                binding,
                count,
                kind,
                sampler: (sampler != u32::MAX).then_some(ResourceId::new(sampler)),
                readable: access & 1 != 0,
                writable: access & 2 != 0,
            })
        })?
        .into_boxed_slice();
    if r.0.len() < 20 || !r.0.len().is_multiple_of(4) || r.0[..4] != 0x07230203u32.to_le_bytes() {
        return None;
    }
    let words =
        r.0.chunks_exact(4)
            .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
            .collect();
    Some(Arc::new(CompiledStage::new(
        stage,
        Module {
            words,
            bindings,
            push_constant_bytes,
        },
        (stage == ShaderStage::Compute).then_some(group),
        shared,
        (points != 0).then_some(points),
    )))
}
impl Input {
    fn encode(&self, out: &mut Vec<u8>) {
        match *self {
            Input::Driver(s, b, n) => {
                for v in [0, s, b, n as u32] {
                    word(out, v);
                }
            }
            Input::Texture {
                stage,
                slot,
                ty,
                scale_index,
            } => {
                for v in [1, stage, slot, ty as u32, scale_index as u32] {
                    word(out, v);
                }
            }
            Input::Image {
                stage,
                slot,
                ty,
                format,
                scale_index,
            } => {
                for v in [2, stage, slot, ty as u32, format as u32, scale_index as u32] {
                    word(out, v);
                }
            }
            Input::Uniform(s, b, n) => {
                for v in [3, s, b] {
                    word(out, v);
                }
                wide(out, n);
            }
            Input::Storage(s, b) => {
                for v in [4, s, b] {
                    word(out, v);
                }
            }
            Input::Constants(s, n) => {
                word(out, 5);
                word(out, s);
                wide(out, n);
            }
        }
    }
    fn decode(r: &mut Reader<'_>) -> Option<Self> {
        let kind = r.u32()?;
        let stage = r.u32()?;
        Some(match kind {
            0 => Self::Driver(stage, r.u32()?, r.u32()? as usize),
            1 => Self::Texture {
                stage,
                slot: r.u32()?,
                ty: TextureType::try_from(r.u32()?).ok()?,
                scale_index: r.u32()? as usize,
            },
            2 => Self::Image {
                stage,
                slot: r.u32()?,
                ty: TextureType::try_from(r.u32()?).ok()?,
                format: ir::types::ImageFormat::try_from(r.u32()?).ok()?,
                scale_index: r.u32()? as usize,
            },
            3 => Self::Uniform(stage, r.u32()?, r.u64()?),
            4 => Self::Storage(stage, r.u32()?),
            5 => Self::Constants(stage, r.u64()?),
            _ => return None,
        })
    }
}
impl Archive {
    // Atomic replacement also repairs an interrupted or damaged entry. Two emulator
    // instances writing the same path have identical bytes.
    fn store(&self, data: Vec<u8>) -> Digest {
        let id = Sha256::digest(&data).into();
        let directory = self.directory.clone();
        if let Err(e) = self.writer.submit(move |_| {
            let save = || -> io::Result<()> {
                let mut f = tempfile::NamedTempFile::new_in(&*directory)?;
                f.write_all(&data)?;
                f.persist(directory.join(hex(&id)))?;
                Ok(())
            };
            if let Err(e) = save() {
                log::warn!("NVN cache write failed: {e}");
            }
        }) {
            log::warn!("NVN cache writer unavailable: {e}");
        }
        id
    }
    pub(super) fn lookup(
        &self,
        key: &Digest,
        bindings: &nixe_nvn::Bindings,
    ) -> Option<(Rc<Translation>, Shapes)> {
        let (shapes, t) = self.programs.get(key)?.iter().find(|(shapes, _)| {
            shapes.iter().all(|&(s, i, f, k)| {
                bindings
                    .sampled
                    .get(s, i)
                    .map_or((s, i, 0, 0), Sampled::shape)
                    == (s, i, f, k)
            })
        })?;
        Some((t.clone(), shapes.clone()))
    }
    pub(super) fn remember(&mut self, key: Digest, shapes: Shapes, t: Rc<Translation>) {
        let mut out = vec![PROGRAM];
        out.extend_from_slice(&key);
        word(&mut out, shapes.len() as u32);
        for &(s, i, f, k) in shapes.iter() {
            for v in [s, i, u32::from(f), u32::from(k)] {
                word(&mut out, v);
            }
        }
        word(&mut out, t.vertex_mask.unwrap_or(0));
        word(&mut out, t.inputs.len() as u32);
        for input in &t.inputs {
            input.encode(&mut out);
        }
        word(&mut out, t.stages.len() as u32);
        for (&id, s) in t.stages.iter().zip(&t.artifact.stages) {
            let module = if let Some(m) = self.modules.get(&id) {
                *m
            } else {
                let m = self.store(module_bytes(s));
                self.modules.insert(id, m);
                m
            };
            out.extend_from_slice(&module);
        }
        self.store(out);
        self.programs.entry(key).or_default().push((shapes, t));
    }
    pub(super) fn record_pipeline(&self, t: &Translation, state: Option<&GraphicsState>) {
        let mut out = vec![PIPELINE];
        word(&mut out, t.stages.len() as u32);
        for id in &t.stages {
            out.extend_from_slice(&self.modules[id]);
        }
        if let Some(s) = state {
            s.encode(&mut out);
        }
        self.store(out);
    }
    fn load(
        &mut self,
        physical: &mut PhysicalPipelines,
        compiler: &PipelineCompiler,
        stop: &AtomicBool,
    ) -> io::Result<()> {
        let mut modules = HashMap::default();
        let mut records = Vec::new();
        for file in fs::read_dir(&*self.directory)? {
            if stop.load(Ordering::Relaxed) {
                return Ok(());
            }
            let file = file?;
            if file.file_name().len() != 64 || file.metadata()?.len() > MAX_RECORD {
                continue;
            }
            let payload = fs::read(file.path())?;
            let id: Digest = Sha256::digest(&payload).into();
            if hex(&id) != file.file_name().to_string_lossy() {
                log::warn!("ignored damaged NVN cache record");
                continue;
            }
            match payload.split_first() {
                Some((&MODULE, bytes)) => {
                    if let Some(module) = decode_module(&mut Reader(bytes)) {
                        let index = physical.intern_stage(&module);
                        self.modules.insert(index, id);
                        modules.insert(id, (index, module));
                    }
                }
                Some((&(PROGRAM | PIPELINE), _)) => records.push(payload),
                _ => {}
            }
        }
        for payload in records {
            if stop.load(Ordering::Relaxed) {
                break;
            }
            let mut r = Reader(&payload[1..]);
            let read_stages = |r: &mut Reader<'_>| -> Option<_> {
                let stages = r.list(6, |r| modules.get(&digest(r)?).cloned())?;
                if stages.is_empty() {
                    return None;
                }
                let (ids, stages): (StageIds, Vec<_>) = stages.into_iter().unzip();
                Some((
                    Arc::new(ShaderArtifact::new(stages.into_boxed_slice())),
                    ids,
                ))
            };
            let mut load = || -> Option<()> {
                if payload[0] == PROGRAM {
                    let key = digest(&mut r)?;
                    let shapes: Shapes = r
                        .list(4096, |r| {
                            Some((
                                r.u32()?,
                                r.u32()?,
                                r.u32()?.try_into().ok()?,
                                r.u32()?.try_into().ok()?,
                            ))
                        })?
                        .into();
                    let mask = r.u32()?;
                    let inputs = r.list(4096, Input::decode)?;
                    let (artifact, stages) = read_stages(&mut r)?;
                    if !r.0.is_empty() {
                        return None;
                    }
                    let vertex_mask = artifact
                        .stages
                        .iter()
                        .any(|s| s.stage == ShaderStage::Vertex)
                        .then_some(mask);
                    self.programs.entry(key).or_default().push((
                        shapes,
                        Rc::new(Translation {
                            artifact,
                            stages,
                            dependencies: Dependencies::new(&inputs),
                            inputs,
                            vertex_mask,
                        }),
                    ));
                } else {
                    let (shader, stages) = read_stages(&mut r)?;
                    let state = if r.0.is_empty() {
                        None
                    } else {
                        Some(GraphicsState::decode(&mut r)?)
                    };
                    if !r.0.is_empty() {
                        return None;
                    }
                    if let std::collections::hash_map::Entry::Vacant(e) =
                        physical.pipelines.entry((stages, state))
                    {
                        let recipe = match e.key().1.as_ref() {
                            Some(s) => PipelineRecipe::Graphics {
                                shader,
                                state: s.clone(),
                            },
                            None => PipelineRecipe::Compute(shader),
                        };
                        e.insert(compiler.request_with_layout(recipe, true).ok()?);
                    }
                }
                Some(())
            };
            if load().is_none() {
                log::debug!("skipped incomplete NVN cache dependency");
            }
        }
        Ok(())
    }
}
impl NativeGpuExecution {
    pub fn load_nvn_disk_resources(&mut self, directory: &Path, stop: &AtomicBool) {
        let start = Instant::now();
        let profile = self.device_profile();
        let mut h = Sha256::new();
        h.update(b"Nixe NVN artifact 2");
        h.update(ir::artifact::ARTIFACT_ABI_VERSION.to_le_bytes());
        h.update(profile.pipeline_cache_uuid);
        h.update(profile.driver_version.to_le_bytes());
        h.update(format!(
            "{:?}{:?}",
            profile.shader_compiler, profile.dynamic_state
        ));
        let path = directory.join("nvn").join(hex(&h.finalize().into()));
        let mut disk = Archive {
            writer: self.compiler.serialization(),
            directory: Arc::new(path),
            modules: Default::default(),
            programs: Default::default(),
        };
        let outcome = fs::create_dir_all(&*disk.directory)
            .and_then(|()| disk.load(&mut self.state.nvn.physical, &self.compiler, stop));
        if let Err(e) = outcome {
            log::warn!("NVN cache unavailable: {e}");
            return;
        }
        log::info!(
            "loaded {} NVN shader variants and queued {} Vulkan pipelines in {:?}",
            disk.programs.values().map(Vec::len).sum::<usize>(),
            self.state.nvn.physical.pipelines.len(),
            start.elapsed()
        );
        self.state.nvn.archive = Some(disk);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persisted_compute_modules_preserve_layout_and_execution_metadata() {
        let stage = CompiledStage::new(
            ShaderStage::Compute,
            Module {
                words: Box::new([0x07230203, 0x00010000, 0, 1, 0]),
                bindings: Box::new([Binding {
                    resource: ResourceId::new(21),
                    sampler: Some(ResourceId::new(8)),
                    binding: 3,
                    count: 2,
                    kind: DescriptorKind::CombinedImageSampler,
                    readable: true,
                    writable: false,
                }]),
                push_constant_bytes: 16,
            },
            Some([4, 8, 2]),
            128,
            None,
        );
        let payload = module_bytes(&stage);
        let loaded = decode_module(&mut Reader(&payload[1..])).unwrap();
        assert_eq!(loaded.module, stage.module);
        assert_eq!(loaded.workgroup_size, stage.workgroup_size);
        assert_eq!(loaded.shared_memory_bytes, 128);
        assert!(decode_module(&mut Reader(&payload[1..payload.len() - 1])).is_none());
    }
}
