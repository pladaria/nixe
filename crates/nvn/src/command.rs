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
pub struct Draw<'a> {
    pub pipeline: &'a Arc<DrawPipeline>,
    pub dynamic: Dynamic,
    pub bindings: Bindings<'a>,
    pub attachments: &'a Arc<Attachments>,
    pub flip_y: bool,
    pub depth_range: [f32; 2],
    pub blend_color: [f32; 4],
    pub indices: Option<(u32, u64, i32)>,
    pub viewport: [i32; 4],
    pub scissor: [i32; 4],
    pub vertices: &'a [(u32, Range)],

    pub first: u32,
    pub count: u32,
    pub first_instance: u32,
    pub instances: u32,
}
impl std::ops::Deref for Draw<'_> {
    type Target = DrawPipeline;
    fn deref(&self) -> &DrawPipeline {
        self.pipeline
    }
}
impl Draw<'_> {
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
pub enum Operation<'a> {
    CopyTexture {
        source: u64,
        target: Texture,
        region: [u32; 6],
        flags: u32,
    },
    Dispatch {
        shader: Shader,
        groups: [u32; 3],
        bindings: Bindings<'a>,
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
        bytes: &'a [u8],
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
    Draw(Draw<'a>),
}
#[derive(Debug)]
struct BindingRun {
    bindings: bindings::CapturedBindings,
    uses: usize,
}
/// Immutable words and submit-visible descriptors for the graphics owner.
#[derive(Debug, Default)]
pub struct Submission {
    words: Vec<u64>,
    captures: Vec<BindingRun>,
    upper_left: bool,
    minus_one_to_one: bool,
}
impl Submission {
    pub(crate) fn new(upper_left: bool, minus_one_to_one: bool) -> Self {
        Self {
            upper_left,
            minus_one_to_one,
            ..Default::default()
        }
    }
    pub fn allocation_bytes(&self) -> usize {
        self.words.capacity() * size_of::<u64>()
            + self.captures.capacity() * size_of::<BindingRun>()
    }
    pub fn records(&self) -> impl Iterator<Item = Record<'_>> {
        let mut words = self.words.as_slice();
        let mut captures = self
            .captures
            .iter()
            .flat_map(|run| std::iter::repeat_n(&run.bindings, run.uses));
        std::iter::from_fn(move || {
            let (&h, tail) = words.split_first()?;
            // Capture validated the length before publishing this immutable block.
            let (args, rest) = tail.split_at((h >> 32) as usize);
            words = rest;
            let tag = h as u32;
            let bindings = matches!(tag, op::DRAW | op::INDEXED | op::DISPATCH)
                .then(|| captures.next().unwrap());
            Some(Record {
                tag,
                args,
                bindings,
            })
        })
    }
}
pub struct Record<'a> {
    tag: u32,
    args: &'a [u64],
    bindings: Option<&'a bindings::CapturedBindings>,
}
#[derive(Default)]
pub(crate) struct Capture {
    bindings: [bindings::BindingState; 2],
    shaders: SmallVec<[Shader; 2]>,
    texture_pool: DescriptorPool,
    sampler_pool: DescriptorPool,
}
impl Capture {
    pub(crate) fn list(&mut self, c: &Call, mut at: u64, batch: &mut Submission) -> Result<()> {
        loop {
            let Range { address, size } = c.read(at)?;
            if size == LINK {
                at = address;
                continue;
            }
            let bytes = size & !END;
            if !bytes.is_multiple_of(8) || bytes > 64 << 20 {
                return Err("invalid control entry".into());
            }
            if bytes != 0 {
                let start = batch.words.len();
                batch.words.resize(start + bytes as usize / 8, 0);
                let block = &mut batch.words[start..];
                c.read_bytes(address, bytemuck::cast_slice_mut(block))?;
                self.observe(c, block, &mut batch.captures)?;
            }
            if size & END != 0 {
                return Ok(());
            }
            at += CONTROL_SIZE;
        }
    }
    fn observe(
        &mut self,
        c: &Call,
        mut words: &[u64],
        captures: &mut Vec<BindingRun>,
    ) -> Result<()> {
        while let Some((&header, tail)) = words.split_first() {
            let n = (header >> 32) as usize;
            let (w, next) = tail
                .split_at_checked(n)
                .ok_or("truncated NVN command record")?;
            words = next;
            let op = header as u32;
            match op {
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
                op::PROGRAM if n >= 1 && (n - 1).is_multiple_of(size_of::<Shader>() / 8) => {
                    select_stages(&mut self.shaders, w);
                    self.bindings[0]
                        .select_shaders(self.shaders.iter().copied().filter(|s| s.stage != 5));
                    self.bindings[1]
                        .select_shaders(self.shaders.iter().copied().filter(|s| s.stage == 5));
                }
                op::DRAW | op::INDEXED | op::DISPATCH => {
                    let current = self.bindings[usize::from(op == op::DISPATCH)].snapshot(
                        c,
                        self.texture_pool,
                        self.sampler_pool,
                    )?;
                    if let Some(last) = captures.last_mut()
                        && current.same_selection(&last.bindings)
                    {
                        last.uses += 1;
                    } else {
                        captures.push(BindingRun {
                            bindings: current.clone(),
                            uses: 1,
                        });
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }
}
fn select_stages(shaders: &mut SmallVec<[Shader; 2]>, w: &[u64]) {
    shaders.retain(|s| w[0] & (1 << s.stage) == 0);
    shaders.extend(
        bytemuck::cast_slice::<u64, Shader>(&w[1..])
            .iter()
            .filter(|s| w[0] & (1 << s.stage) != 0)
            .copied(),
    );
    shaders.sort_by_key(|s| s.stage);
}

#[derive(Default)]
pub struct CommandState {
    attachments: Arc<Attachments>,
    raster: RasterState,
    dynamic: Dynamic,
    buffers: [bindings::BufferState; 2],
    flip_y: bool,
    pipeline: Arc<DrawPipeline>,
    depth_range: Option<[f32; 2]>,
    blend_color: [f32; 4],
    viewport: [i32; 4],
    scissor: [i32; 4],
    shaders: SmallVec<[Shader; 2]>,
    vertices: Vec<(u32, Range)>,
}
impl CommandState {
    pub fn new(submission: &Submission) -> Self {
        Self {
            pipeline: Arc::new(DrawPipeline {
                upper_left: submission.upper_left,
                minus_one_to_one: submission.minus_one_to_one,
                samples: 1,
                ..Default::default()
            }),
            ..Default::default()
        }
    }
    pub fn apply<'a>(&'a mut self, record: Record<'a>) -> Result<Option<Operation<'a>>> {
        let Record {
            tag: op,
            args: w,
            bindings,
        } = record;
        let n = w.len();
        match op {
            op::COPY_TEXTURE if n == 5 + size_of::<Texture>() / 8 => {
                let end = n - 3;
                let target = bytemuck::pod_read_unaligned(bytemuck::cast_slice(&w[2..end]));
                let region = bytemuck::pod_read_unaligned(bytemuck::cast_slice(&w[end..]));
                return Ok(Some(Operation::CopyTexture {
                    source: w[0],
                    target,
                    region,
                    flags: w[1] as u32,
                }));
            }
            op::DISPATCH if n == 3 => {
                let shader = *self
                    .shaders
                    .iter()
                    .find(|s| s.stage == 5)
                    .ok_or("dispatch without compute program")?;
                return Ok(Some(Operation::Dispatch {
                    shader,
                    groups: std::array::from_fn(|i| w[i] as u32),
                    bindings: Bindings {
                        textures: bindings.unwrap(),
                        buffers: &self.buffers[1],
                    },
                }));
            }
            op::REPORT if n == 2 => {
                return Ok(Some(Operation::Report {
                    kind: w[0] as u32,
                    address: w[1],
                }));
            }
            op::CONDITION if n == 2 => {
                return Ok(Some(Operation::RenderEnable {
                    mode: w[0] as u32,
                    address: w[1],
                }));
            }
            op::UPDATE if n >= 2 && w[1].div_ceil(8) == (n - 2) as u64 => {
                return Ok(Some(Operation::Update {
                    address: w[0],
                    bytes: &bytemuck::cast_slice(&w[2..])[..w[1] as usize],
                }));
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
                    return Ok(None);
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
                    return Ok(Some(Operation::ClearDepth {
                        target,
                        scissor: self.scissor,
                        depth: (w[1] != 0).then(|| f32::from_bits(w[0] as u32)),
                        stencil: w[2] as u8,
                        mask: w[3] as u8,
                    }));
                }
            }
            op::VIEWPORT if n == 4 => self.viewport = std::array::from_fn(|i| w[i] as i32),
            op::SCISSOR if n == 4 => self.scissor = std::array::from_fn(|i| w[i] as i32),
            op::BARRIER if n == 1 => return Ok(Some(Operation::Barrier(w[0] as u32))),
            op::CLEAR if n == 4 && w[0] < 8 => {
                if let Some(&Some(target)) = self.attachments.colors.get(w[0] as usize) {
                    return Ok(Some(Operation::Clear {
                        target,
                        scissor: self.scissor,
                        color: bytemuck::cast([w[1], w[2]]),
                        mask: w[3] as u32,
                    }));
                }
            }
            op::PROGRAM if n >= 1 && (n - 1).is_multiple_of(size_of::<Shader>() / 8) => {
                select_stages(&mut self.shaders, w);
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
                    Some(i) => self.vertices[i] = item,
                    None => self.vertices.push(item),
                }
            }
            op::UNIFORM | op::STORAGE if n == 4 => {
                let stage = w[0] as u32;
                self.buffers[usize::from(stage == 5)].bind(
                    op == op::STORAGE,
                    stage,
                    w[1] as u32,
                    Range {
                        address: w[2],
                        size: w[3],
                    },
                );
            }
            op::SAMPLER_POOL | op::TEXTURE_POOL if n == 2 => {}
            op::TEXTURE if n == 2 => {}
            op::DEFERRED if n == 4 => {}
            op::DRAW | op::INDEXED
                if (op == op::DRAW && n == 5) || (op == op::INDEXED && n == 7) =>
            {
                let bindings = Bindings {
                    textures: bindings.unwrap(),
                    buffers: &self.buffers[0],
                };
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
                return Ok(Some(Operation::Draw(Draw {
                    pipeline: &self.pipeline,
                    attachments: &self.attachments,
                    dynamic: self.dynamic,
                    bindings,
                    flip_y: self.flip_y ^ self.pipeline.upper_left,
                    depth_range: self.depth_range.unwrap_or([0., 1.]),
                    blend_color: self.blend_color,
                    indices: (op == op::INDEXED).then(|| (w[1] as u32, w[3], w[4] as i32)),
                    viewport: self.viewport,
                    scissor: self.scissor,
                    vertices: &self.vertices,

                    first: w[1] as u32,
                    count: w[2] as u32,
                    first_instance: w[n - 2] as u32,
                    instances: w[n - 1] as u32,
                })));
            }
            _ => {
                return Err(format!("invalid or unsupported NVN command {op} ({n} words)").into());
            }
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nixe_memory::{AddressWidth, GuestVirtualAddress, Machine, MemoryPermissions, PAGE_SIZE};

    #[test]
    fn queued_commands_own_guest_bytes_and_apply_buffer_state_in_draw_order() {
        let machine = Machine::new(PAGE_SIZE).unwrap();
        let allocation = machine
            .account(PAGE_SIZE)
            .unwrap()
            .allocate(PAGE_SIZE)
            .unwrap();
        let memory = AddressSpace::new(machine, AddressWidth::Bits32).unwrap();
        memory
            .map(
                GuestVirtualAddress::new(0x1000),
                allocation,
                0,
                PAGE_SIZE,
                MemoryPermissions::READ_WRITE,
            )
            .unwrap();
        let session = Session::new();
        let call = Call {
            session: &session,
            memory: &memory,
            args: [0; 8],
            pending: Default::default(),
        };
        let header = |tag, n: u64| u64::from(tag) | n << 32;
        let mut capture = Capture::default();
        let mut first = Submission::new(true, false);
        let mut second = Submission::new(true, false);
        let mut recording = [
            header(op::UNIFORM, 4),
            1,
            0,
            0x6000,
            128,
            header(op::DRAW, 5),
            4,
            0,
            3,
            0,
            1,
            header(op::DRAW, 5),
            4,
            0,
            6,
            0,
            1,
        ];
        call.write(
            0x1000,
            &Range {
                address: 0x1100,
                size: size_of_val(&recording) as u64 | END,
            },
        )
        .unwrap();
        call.write(0x1100, &recording).unwrap();
        capture.list(&call, 0x1000, &mut first).unwrap();
        recording[3] = 0x7000;
        call.write(0x1100, &recording).unwrap();
        capture.list(&call, 0x1000, &mut second).unwrap();
        call.write(0x1100, &[0u64; 17]).unwrap();
        let mut state = CommandState::new(&first);
        for (submission, address) in [(&first, 0x6000), (&second, 0x7000)] {
            let mut records = submission.records();
            assert!(state.apply(records.next().unwrap()).unwrap().is_none());
            assert_eq!(submission.captures.len(), 1);
            for count in [3, 6] {
                let Some(Operation::Draw(draw)) = state.apply(records.next().unwrap()).unwrap()
                else {
                    panic!("expected a draw")
                };
                assert_eq!(
                    draw.bindings.buffers.get(false, 1, 0),
                    Some(Range { address, size: 128 })
                );
                assert!(draw.pipeline.upper_left);
                assert_eq!(draw.count, count);
            }
            assert!(records.next().is_none());
        }
    }
}
