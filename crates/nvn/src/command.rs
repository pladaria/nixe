//! Record values in the guest-provided command memory. EndRecording returns a
//! control-memory handle; a submit reads the command bytes as they stand then.
use super::*;

use smallvec::SmallVec;

#[derive(Copy, Clone, Debug)]
pub struct Dynamic {
    pub depth_bounds: [f32; 2],
    pub depth_bias: [f32; 3],
    pub line_width: f32,
    pub stencil_ref: [u32; 2],
    pub stencil_compare: [u32; 2],
    pub stencil_write: [u32; 2],
}
impl Default for Dynamic {
    fn default() -> Self {
        Self {
            depth_bounds: [0., 1.],
            depth_bias: [0.; 3],
            line_width: 1.,
            stencil_ref: [0; 2],
            stencil_compare: [u32::MAX; 2],
            stencil_write: [u32::MAX; 2],
        }
    }
}
/// Immutable pipeline inputs shared by draws until the queue selects different state.
#[derive(Clone, Default, Debug, PartialEq, Eq, Hash)]
pub struct DrawPipeline {
    pub shaders: SmallVec<[Shader; 2]>,
    pub streams: SmallVec<[VertexStream; 2]>,
    pub attributes: SmallVec<[u32; 4]>,
    pub primitive: u32,
    pub upper_left: bool,
    pub minus_one_to_one: bool,
    pub samples: u32,
    pub color_formats: SmallVec<[Option<u16>; 1]>,
    pub depth_format: Option<u16>,
    pub raster: RasterState,
    pub shapes: Arc<[TextureShape]>,
}
#[derive(Clone, Debug, Default)]
pub struct Attachments {
    pub colors: SmallVec<[Option<Texture>; 1]>,
    pub depth: Option<Texture>,
}
#[derive(Debug)]
pub struct Draw {
    pub pipeline: Arc<DrawPipeline>,
    pub dynamic: Dynamic,
    pub bindings: Bindings,
    pub attachments: Arc<Attachments>,
    pub flip_y: bool,
    pub depth_range: [f32; 2],
    pub blend_color: [f32; 4],
    pub indices: Option<(u32, u64, i32)>,
    pub viewport: [i32; 4],
    pub scissor: [i32; 4],
    pub vertices: Arc<Vec<(u32, Range)>>,

    pub first: u32,
    pub count: u32,
    pub first_instance: u32,
    pub instances: u32,
}
impl std::ops::Deref for Draw {
    type Target = DrawPipeline;
    fn deref(&self) -> &DrawPipeline {
        &self.pipeline
    }
}
impl Draw {
    pub fn extent(&self) -> [u32; 2] {
        self.attachments
            .colors
            .iter()
            .flatten()
            .chain(self.attachments.depth.iter())
            .map(|t| {
                [
                    (t.width >> t.first_level).max(1),
                    (t.height >> t.first_level).max(1),
                ]
            })
            .reduce(|a, b| [a[0].min(b[0]), a[1].min(b[1])])
            .unwrap_or_else(|| {
                [
                    self.viewport[0].saturating_add(self.viewport[2]).max(1) as u32,
                    self.viewport[1].saturating_add(self.viewport[3]).max(1) as u32,
                ]
            })
    }
}
#[derive(Debug)]
pub enum Operation {
    CopyTexture {
        source: u64,
        target: Texture,
        region: [u32; 6],
        flags: u32,
    },
    Dispatch {
        shader: Shader,
        groups: [u32; 3],
        bindings: Bindings,
    },
    Report {
        kind: u32,
        address: u64,
    },
    RenderEnable {
        mode: u32,
        address: u64,
    },
    Update {
        address: u64,
        bytes: Box<[u8]>,
    },
    Barrier(u32),
    Clear {
        target: Texture,
        scissor: [i32; 4],
        color: [f32; 4],
        mask: u32,
    },
    ClearDepth {
        target: Texture,
        scissor: [i32; 4],
        depth: Option<f32>,
        stencil: u8,
        mask: u8,
    },
    Draw(usize),
}
#[derive(Debug, Default)]
pub struct Submission {
    pub update_bytes: usize,
    pub operations: Vec<Operation>,
    pub draws: Vec<Draw>,
}

#[derive(Default)]
pub(crate) struct State {
    attachments: Arc<Attachments>,
    raster: RasterState,
    dynamic: Dynamic,
    bindings: [bindings::BindingState; 2],
    flip_y: bool,
    pub(crate) pipeline: Arc<DrawPipeline>,
    depth_range: Option<[f32; 2]>,
    blend_color: [f32; 4],
    viewport: [i32; 4],
    scissor: [i32; 4],
    shaders: SmallVec<[Shader; 2]>,
    texture_pool: DescriptorPool,
    sampler_pool: DescriptorPool,
    scratch: Vec<u64>,
    vertices: Arc<Vec<(u32, Range)>>,
}
impl State {
    pub fn for_device(c: &Call, dev: u64) -> Result<Self> {
        let [origin, depth]: [u32; 2] = c.read(dev + 4)?;
        Ok(Self {
            pipeline: Arc::new(DrawPipeline {
                upper_left: origin == 1,
                minus_one_to_one: depth == 0,
                samples: 1,
                ..Default::default()
            }),
            ..Default::default()
        })
    }

    pub(crate) fn replay(&mut self, c: &Call, handle: u64, batch: &mut Submission) -> Result<()> {
        let mut chain = SmallVec::<[u64; 8]>::new();
        chain.push(handle);
        while let Some(at) = chain.pop() {
            let Range { address, size } = c.read(at)?;
            if size & END == 0 && size & LINK == 0 {
                chain.push(at + CONTROL_SIZE)
            }
            match size & !END {
                CALL => {
                    if chain.len() > 256 {
                        return Err("excessive command list nesting".into());
                    }
                    chain.push(address);
                }
                LINK => chain.push(address),
                0 => {}
                bytes if bytes.is_multiple_of(8) && bytes <= 64 << 20 => {
                    self.scratch.resize(bytes as usize / 8, 0);
                    c.read_bytes(address, bytemuck::cast_slice_mut(&mut self.scratch))?;
                    self.decode(c, batch)?;
                }
                _ => return Err("invalid control entry".into()),
            }
        }
        Ok(())
    }
    fn decode(&mut self, c: &Call, batch: &mut Submission) -> Result<()> {
        let mut commands = self.scratch.as_slice();
        while let Some((&h, tail)) = commands.split_first() {
            let n = (h >> 32) as usize;
            let (w, next) = tail
                .split_at_checked(n)
                .ok_or("truncated NVN command record")?;
            commands = next;
            let op = h as u32;
            match op {
                op::COPY_TEXTURE if n == 5 + size_of::<Texture>() / 8 => {
                    let end = n - 3;
                    let target = bytemuck::pod_read_unaligned(bytemuck::cast_slice(&w[2..end]));
                    let region = bytemuck::pod_read_unaligned(bytemuck::cast_slice(&w[end..]));
                    batch.operations.push(Operation::CopyTexture {
                        source: w[0],
                        target,
                        region,
                        flags: w[1] as u32,
                    })
                }
                op::DISPATCH if n == 3 => {
                    let shader = *self
                        .shaders
                        .iter()
                        .find(|s| s.stage == 5)
                        .ok_or("dispatch without compute program")?;
                    batch.operations.push(Operation::Dispatch {
                        shader,
                        groups: std::array::from_fn(|i| w[i] as u32),
                        bindings: self.bindings[1].snapshot(
                            c,
                            self.texture_pool,
                            self.sampler_pool,
                        )?,
                    })
                }
                op::REPORT if n == 2 => batch.operations.push(Operation::Report {
                    kind: w[0] as u32,
                    address: w[1],
                }),
                op::CONDITION if n == 2 => batch.operations.push(Operation::RenderEnable {
                    mode: w[0] as u32,
                    address: w[1],
                }),
                op::UPDATE if n >= 2 && w[1].div_ceil(8) == (n - 2) as u64 => {
                    batch.update_bytes = batch.update_bytes.saturating_add(w[1] as usize);
                    batch.operations.push(Operation::Update {
                        address: w[0],
                        bytes: bytemuck::cast_slice(&w[2..])[..w[1] as usize].into(),
                    })
                }
                op::SAMPLER_POOL | op::TEXTURE_POOL if n == 2 => {
                    let pool = if op == op::SAMPLER_POOL {
                        &mut self.sampler_pool
                    } else {
                        &mut self.texture_pool
                    };
                    let next = DescriptorPool::new(w[0], w[1] as u32);
                    if (pool.base, pool.count) != (next.base, next.count) {
                        *pool = next;
                        for bindings in &mut self.bindings {
                            bindings.pool_changed();
                        }
                    }
                }
                op::TARGETS
                    if n != 0
                        && n.is_multiple_of(size_of::<Texture>() / 8)
                        && n <= 9 * size_of::<Texture>() / 8 =>
                {
                    let (depth, colors) = bytemuck::cast_slice::<u64, Texture>(w)
                        .split_first()
                        .unwrap();
                    let depth = (depth.address != 0).then_some(*depth);
                    let n = colors
                        .iter()
                        .rposition(|t| t.address != 0)
                        .map_or(0, |i| i + 1);
                    let colors = colors[..n].iter().map(|t| (t.address != 0).then_some(*t));
                    if depth == self.attachments.depth
                        && colors.clone().eq(self.attachments.colors.iter().copied())
                    {
                        continue;
                    }
                    let targets = Arc::make_mut(&mut self.attachments);
                    targets.depth = depth;
                    targets.colors.clear();
                    targets.colors.extend(colors);
                    let samples = targets
                        .colors
                        .iter()
                        .flatten()
                        .chain(depth.iter())
                        .next()
                        .map_or(1, |t| t.samples.max(1));
                    let depth = depth.map(|t| t.format);
                    let formats = targets.colors.iter().map(|t| t.map(|t| t.format));
                    if self.pipeline.samples != samples
                        || self.pipeline.depth_format != depth
                        || !formats
                            .clone()
                            .eq(self.pipeline.color_formats.iter().copied())
                    {
                        let pipeline = Arc::make_mut(&mut self.pipeline);
                        pipeline.samples = samples;
                        pipeline.depth_format = depth;
                        pipeline.color_formats.clear();
                        pipeline.color_formats.extend(formats);
                    }
                }
                op::BLEND if n == 1 => {
                    let blend = w[0].to_le_bytes();
                    *self
                        .raster
                        .blend
                        .get_mut(usize::from(blend[0]))
                        .ok_or("invalid blend target")? = blend;
                }
                op::COLOR if n == 1 => {
                    self.raster.blend_enable = w[0] as u32 & 255;
                    self.raster.logic = (w[0] >> 8) as u8
                }
                op::MASK if n == 1 => self.raster.channel_mask = w[0] as u32,
                op::POLY if n == 1 => self.raster.polygon = (w[0] as u32).to_le_bytes(),
                op::DEPTH if n == 1 => {
                    self.raster.depth = (w[0] as u32).to_le_bytes();
                    self.raster.stencil = bytemuck::cast((w[0] >> 32) as u32);
                }
                op::MULTISAMPLE if n == 1 => self.raster.multisample = w[0] as u32,
                op::CLAMP if n == 1 => self.raster.depth_clamp = w[0] != 0,
                op::DISCARD if n == 1 => self.raster.raster_discard = w[0] != 0,
                op::BOUNDS if n == 2 => {
                    self.raster.depth_bounds = w[0] != 0;
                    self.dynamic.depth_bounds = bytemuck::cast(w[1]);
                }
                op::LINE if n == 1 => self.dynamic.line_width = f32::from_bits(w[0] as u32),
                op::POINT if n == 1 => self.raster.point_size = w[0] as u32,
                op::OFFSET if n == 2 => {
                    let [units, clamp]: [f32; 2] = bytemuck::cast(w[0]);
                    self.dynamic.depth_bias = [units, clamp, f32::from_bits(w[1] as u32)]
                }
                op::REF | op::WRITE_MASK | op::COMPARE_MASK if n == 2 => {
                    let values = match op {
                        op::REF => &mut self.dynamic.stencil_ref,
                        op::WRITE_MASK => &mut self.dynamic.stencil_write,
                        _ => &mut self.dynamic.stencil_compare,
                    };
                    for (i, v) in values.iter_mut().enumerate() {
                        if w[0] & (1 << i) != 0 {
                            *v = w[1] as u32
                        }
                    }
                }
                op::DEPTH_RANGE if n == 1 => self.depth_range = Some(bytemuck::cast(w[0])),
                op::SWIZZLES if n == 1 => self.flip_y = w[0] != 0,
                op::BLEND_COLOR if n == 2 => self.blend_color = bytemuck::cast([w[0], w[1]]),
                op::CLEAR_DEPTH if n == 4 => {
                    if let Some(target) = self.attachments.depth {
                        batch.operations.push(Operation::ClearDepth {
                            target,
                            scissor: self.scissor,
                            depth: (w[1] != 0).then(|| f32::from_bits(w[0] as u32)),
                            stencil: w[2] as u8,
                            mask: w[3] as u8,
                        });
                    }
                }
                op::VIEWPORT if n == 4 => self.viewport = std::array::from_fn(|i| w[i] as i32),
                op::SCISSOR if n == 4 => self.scissor = std::array::from_fn(|i| w[i] as i32),
                op::BARRIER if n == 1 => batch.operations.push(Operation::Barrier(w[0] as u32)),
                op::CLEAR if n == 4 && w[0] < 8 => {
                    if let Some(&Some(target)) = self.attachments.colors.get(w[0] as usize) {
                        batch.operations.push(Operation::Clear {
                            target,
                            scissor: self.scissor,
                            color: bytemuck::cast([w[1], w[2]]),
                            mask: w[3] as u32,
                        });
                    }
                }
                op::PROGRAM if n >= 1 && (n - 1).is_multiple_of(size_of::<Shader>() / 8) => {
                    self.shaders.retain(|s| w[0] & (1 << s.stage) == 0);
                    self.shaders.extend(
                        bytemuck::cast_slice::<u64, Shader>(&w[1..])
                            .iter()
                            .filter(|s| w[0] & (1 << s.stage) != 0)
                            .copied(),
                    );
                    self.shaders.sort_by_key(|s| s.stage);
                    self.bindings[0]
                        .select_shaders(self.shaders.iter().copied().filter(|s| s.stage != 5));
                    self.bindings[1]
                        .select_shaders(self.shaders.iter().copied().filter(|s| s.stage == 5));
                    let graphics = self.shaders.iter().copied().filter(|s| s.stage != 5);
                    if !graphics.clone().eq(self.pipeline.shaders.iter().copied()) {
                        let pipeline = Arc::make_mut(&mut self.pipeline);
                        pipeline.shaders.clear();
                        pipeline.shaders.extend(graphics);
                    }
                }
                op::ATTRIBUTES if n > 0 && w[0] <= 32 && n == 1 + (w[0] as usize).div_ceil(2) => {
                    let values = &bytemuck::cast_slice(&w[1..n])[..w[0] as usize];
                    if self.pipeline.attributes.as_slice() != values {
                        let pipeline = Arc::make_mut(&mut self.pipeline);
                        pipeline.attributes.clear();
                        pipeline.attributes.extend_from_slice(values);
                    }
                }
                op::STREAMS if n > 0 && w[0] <= 32 && n == 1 + w[0] as usize => {
                    let values = bytemuck::cast_slice(&w[1..n]);
                    if self.pipeline.streams.as_slice() != values {
                        let pipeline = Arc::make_mut(&mut self.pipeline);
                        pipeline.streams.clear();
                        pipeline.streams.extend_from_slice(values);
                    }
                }
                op::VERTEX if n == 3 => {
                    let item = (
                        w[0] as u32,
                        Range {
                            address: w[1],
                            size: w[2],
                        },
                    );
                    match self.vertices.iter().position(|v| v.0 == item.0) {
                        Some(i) if self.vertices[i] == item => {}
                        Some(i) => Arc::make_mut(&mut self.vertices)[i] = item,
                        None => Arc::make_mut(&mut self.vertices).push(item),
                    }
                }
                op::UNIFORM | op::STORAGE if n == 4 => {
                    let stage = w[0] as u32;
                    self.bindings[usize::from(stage == 5)].bind_buffer(
                        op == op::STORAGE,
                        stage,
                        w[1] as u32,
                        Range {
                            address: w[2],
                            size: w[3],
                        },
                    );
                }
                op::DEFERRED if n == 4 => {
                    let stage = w[0] as u32;
                    let first = w[1] as u32;
                    for i in 0..w[2] as u32 {
                        let handle = c.read(w[3] + 8 * u64::from(i))?;
                        self.bindings[usize::from(stage == 5)].bind_texture(
                            stage,
                            first + i,
                            handle,
                        );
                    }
                }
                op::TEXTURE if n == 2 => {
                    let stage = w[0] as u32;
                    self.bindings[usize::from(stage == 5)].bind_texture(
                        stage,
                        (w[0] >> 32) as u32,
                        w[1],
                    );
                }
                op::DRAW | op::INDEXED
                    if (op == op::DRAW && n == 5) || (op == op::INDEXED && n == 7) =>
                {
                    let bindings =
                        self.bindings[0].snapshot(c, self.texture_pool, self.sampler_pool)?;
                    let shapes = &bindings.sampled.shapes;
                    let primitive = w[0] as u32;
                    if self.pipeline.raster != self.raster
                        || self.pipeline.primitive != primitive
                        || !(Arc::ptr_eq(&self.pipeline.shapes, shapes)
                            || self.pipeline.shapes == *shapes)
                    {
                        let pipeline = Arc::make_mut(&mut self.pipeline);
                        pipeline.raster = self.raster;
                        pipeline.primitive = primitive;
                        pipeline.shapes = shapes.clone();
                    }
                    batch.operations.push(Operation::Draw(batch.draws.len()));
                    batch.draws.push(Draw {
                        pipeline: self.pipeline.clone(),
                        attachments: self.attachments.clone(),
                        dynamic: self.dynamic,
                        bindings,
                        flip_y: self.flip_y ^ self.pipeline.upper_left,
                        depth_range: self.depth_range.unwrap_or([0., 1.]),
                        blend_color: self.blend_color,
                        indices: (op == op::INDEXED).then(|| (w[1] as u32, w[3], w[4] as i32)),
                        viewport: self.viewport,
                        scissor: self.scissor,
                        vertices: self.vertices.clone(),

                        first: w[1] as u32,
                        count: w[2] as u32,
                        first_instance: w[n - 2] as u32,
                        instances: w[n - 1] as u32,
                    });
                }
                _ => {
                    return Err(
                        format!("invalid or unsupported NVN command {op} ({n} words)").into(),
                    );
                }
            }
        }
        Ok(())
    }
}
