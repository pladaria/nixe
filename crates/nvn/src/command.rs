//! Record values in the guest-provided command memory. EndRecording returns a
//! control-memory handle; a submit reads the command bytes as they stand then.
mod block_cache;

use super::*;

use nixe_memory::DirectWordsComparison;
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
    pub viewport_count: u8,
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
#[derive(Debug, Copy, Clone)]
pub struct Viewport {
    pub rect: [f32; 4],
    pub depth: [f32; 2],
    pub scissor: [i32; 4],
}
impl Default for Viewport {
    fn default() -> Self {
        Self {
            rect: [0., 0., 1., 1.],
            depth: [0., 1.],
            scissor: [0; 4],
        }
    }
}
#[derive(Debug)]
pub struct Draw<'a> {
    pub pipeline: &'a Arc<DrawPipeline>,
    pub dynamic: Dynamic,
    pub bindings: Bindings<'a>,
    pub attachments: &'a Arc<Attachments>,
    pub flip_y: bool,
    pub viewports: &'a [Viewport],
    pub blend_color: [f32; 4],
    pub indices: Option<(u32, u64, i32)>,

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
            .map(|t| t.view_extent())
            .reduce(|a, b| [a[0].min(b[0]), a[1].min(b[1])])
            .unwrap_or_else(|| {
                [
                    (self.viewports[0].rect[0] + self.viewports[0].rect[2])
                        .ceil()
                        .max(1.) as u32,
                    (self.viewports[0].rect[1] + self.viewports[0].rect[3])
                        .ceil()
                        .max(1.) as u32,
                ]
            })
    }
}
#[derive(Debug)]
pub enum Operation<'a> {
    ClearImage {
        texture: Texture,
        region: [u32; 6],
        color: [f32; 4],
        mask: u32,
    },
    CopyImages {
        textures: [Texture; 2],
        regions: [[u32; 6]; 2],
        flags: u32,
    },
    CopyTexture {
        source: u64,
        target: Texture,
        region: [u32; 6],
        flags: u32,
        strides: CopyStrides,
    },
    Dispatch {
        shader: Shader,
        groups: [u32; 3],
        bindings: Bindings<'a>,
    },
    ResetCounter(u32),
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
#[derive(Debug)]
struct CommandBlock {
    words: Vec<u64>,
    end: usize,
    capture_offsets: Vec<u32>,
}
impl CommandBlock {
    fn new(words: Vec<u64>, called: bool) -> Result<Self> {
        let mut end = words.len();
        let mut capture_offsets = Vec::new();
        let mut index = 0;
        while index < end {
            let header = words[index];
            let next = index + 1 + record_words(header);
            if next > end {
                if !called {
                    return Err("truncated NVN command record".into());
                }
                // @SIMPLIFICATION A called range ending inside a replacement
                // record executes only its complete prefix.
                // https://github.com/seedds/nxnative/blob/a366365/docs/reference/nvn-objects.md#a-called-recording-recorded-over-before-its-caller-is-submitted
                end = index;
                break;
            }
            if needs_capture(record_tag(header)) {
                capture_offsets.push(index as u32);
            }
            if called {
                // A called range remains a span of bytes if re-recorded. Its new
                // commands can begin at different offsets than the original.
                index = next;
            } else {
                let skip = (header >> 32) as usize;
                if skip == 0 {
                    break;
                }
                index += skip;
                if index < next || index >= end {
                    return Err("invalid NVN capture offset".into());
                }
            }
        }
        Ok(Self {
            words,
            end,
            capture_offsets,
        })
    }
}
/// Immutable words and submit-visible descriptors for the graphics owner.
#[derive(Debug, Default)]
pub struct Submission {
    pub pool_bounds: Arc<PoolBounds>,
    blocks: SmallVec<[Arc<CommandBlock>; 4]>,
    retained_bytes: usize,
    captures: Vec<BindingRun>,
    upper_left: bool,
    minus_one_to_one: bool,
}
impl Submission {
    pub(crate) fn new(
        upper_left: bool,
        minus_one_to_one: bool,
        pool_bounds: Arc<PoolBounds>,
    ) -> Self {
        Self {
            upper_left,
            minus_one_to_one,
            pool_bounds,
            blocks: SmallVec::new(),
            captures: Vec::new(),
            retained_bytes: 0,
        }
    }
    #[cfg(feature = "execution-trace")]
    pub(crate) fn block_count(&self) -> usize {
        self.blocks.len()
    }
    pub fn allocation_bytes(&self) -> usize {
        self.retained_bytes + self.captures.capacity() * size_of::<BindingRun>()
    }
    pub fn records(&self) -> impl Iterator<Item = Result<Record<'_>>> {
        let mut blocks = self.blocks.iter();
        let mut words: &[u64] = &[];
        let mut captures = self
            .captures
            .iter()
            .flat_map(|run| std::iter::repeat_n(&run.bindings, run.uses));
        std::iter::from_fn(move || {
            while words.is_empty() {
                let block = blocks.next()?;
                words = &block.words[..block.end];
            }
            let (&h, tail) = words.split_first()?;
            let Some((args, rest)) = tail.split_at_checked(record_words(h)) else {
                words = &[];
                return Some(Err("truncated NVN command record".into()));
            };
            words = rest;
            let tag = record_tag(h);
            let bindings = if matches!(tag, op::DRAW | op::INDEXED | op::DISPATCH) {
                let Some(capture) = captures.next() else {
                    return Some(Err("missing submit-visible NVN bindings".into()));
                };
                Some(capture)
            } else {
                None
            };
            Some(Ok(Record {
                tag,
                args,
                bindings,
            }))
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
    blocks: block_cache::Blocks,
    #[cfg(feature = "execution-trace")]
    capture_counts: [u64; 4],
    #[cfg(feature = "execution-trace")]
    diagnostics: ListDiagnostics,
    texture_pool: DescriptorPool,
    sampler_pool: DescriptorPool,
}
#[cfg(feature = "execution-trace")]
#[derive(Default)]
struct ListDiagnostics {
    evicted: std::collections::HashSet<Range>,
    // attempted comparison bytes, first-seen misses, modified, evicted, coherent, evictions
    counts: [u64; 6],
}

#[cfg(feature = "execution-trace")]
impl ListDiagnostics {
    fn record_miss(&mut self, key: Range, comparison: Option<DirectWordsComparison>) {
        let index = match comparison {
            Some(DirectWordsComparison::Different) => 2,
            Some(DirectWordsComparison::NeedsCoherentRead) => 4,
            None if self.evicted.remove(&key) => 3,
            None => 1,
            Some(DirectWordsComparison::Equal) => return,
        };
        self.counts[index] += 1;
    }

    fn note_eviction(&mut self, key: Range) {
        self.counts[5] += 1;
        if self.evicted.len() < 131_072 {
            self.evicted.insert(key);
        }
    }

    fn trace(&mut self) {
        let [compared, first, modified, evicted, coherent, removals] =
            std::mem::take(&mut self.counts);
        nixe_memory::trace::detail_event("nvn_list_comparison", compared, coherent, removals);
        nixe_memory::trace::detail_event("nvn_list_misses", first, modified, evicted);
    }
}
impl Capture {
    #[cfg(feature = "execution-trace")]
    pub(crate) fn trace_submit(&mut self, blocks: usize) {
        let [copied, visited, reused, hits] = std::mem::take(&mut self.capture_counts);
        nixe_memory::trace::event("nvn_command_capture", copied, visited, blocks as u64);
        nixe_memory::trace::event("nvn_command_reuse", reused, hits, 0);
        self.diagnostics.trace();
    }
    pub(crate) fn list(&mut self, c: &Call, mut at: u64, batch: &mut Submission) -> Result<()> {
        #[cfg(feature = "execution-trace")]
        let detailed = nixe_memory::trace::detailed_active();
        loop {
            let Range { address, size } = c.read(at)?;
            if size == LINK {
                at = address;
                continue;
            }
            let bytes = size & !(END | CALLED);
            if !bytes.is_multiple_of(8) || bytes > 64 << 20 {
                return Err("invalid control entry".into());
            }
            if bytes != 0 {
                let key = Range {
                    address,
                    size: size & !END,
                };
                let previous = self.blocks.index(&key);
                let comparison = previous
                    .map(|index| {
                        let block = self.blocks.block(index);
                        c.memory.compare_direct_words(
                            nixe_memory::GuestVirtualAddress::new(address),
                            &block.words,
                        )
                    })
                    .transpose()
                    .map_err(|e| e.to_string())?;
                #[cfg(feature = "execution-trace")]
                if detailed && comparison.is_some() {
                    self.diagnostics.counts[0] += bytes;
                }
                let block = if let Some(index) = previous
                    && comparison == Some(DirectWordsComparison::Equal)
                {
                    #[cfg(feature = "execution-trace")]
                    {
                        self.capture_counts[2] += bytes;
                        self.capture_counts[3] += 1;
                    }
                    self.blocks.use_block(index)
                } else {
                    #[cfg(feature = "execution-trace")]
                    {
                        self.capture_counts[0] += bytes;
                        if detailed {
                            self.diagnostics.record_miss(key, comparison);
                        }
                    }
                    let mut words = vec![0; bytes as usize / 8];
                    c.read_bytes(address, bytemuck::cast_slice_mut(&mut words))?;
                    let block = Arc::new(CommandBlock::new(words, size & CALLED != 0)?);
                    self.blocks
                        .insert(key, previous, block.clone(), |_evicted| {
                            #[cfg(feature = "execution-trace")]
                            if detailed {
                                self.diagnostics.note_eviction(_evicted);
                            }
                        });
                    block
                };
                for &index in &block.capture_offsets {
                    let i = index as usize;
                    let header = block.words[i];
                    self.observe_record(
                        c,
                        record_tag(header),
                        &block.words[i + 1..i + 1 + record_words(header)],
                        &mut batch.captures,
                    )?;
                }
                batch.retained_bytes += bytes as usize;
                batch.blocks.push(block);
            }
            if size & END != 0 {
                return Ok(());
            }
            at += CONTROL_SIZE;
        }
    }
    fn observe_record(
        &mut self,
        c: &Call,
        op: u32,
        w: &[u64],
        captures: &mut Vec<BindingRun>,
    ) -> Result<()> {
        #[cfg(feature = "execution-trace")]
        {
            self.capture_counts[1] += 1;
        }
        let n = w.len();
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
                    self.bindings[usize::from(stage == 5)].bind_texture(stage, first + i, handle);
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
    viewports: [Viewport; 16],
    blend_color: [f32; 4],

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
                viewport_count: 1,
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
            op::CLEAR_IMAGE if n == size_of::<Texture>() / 8 + 6 => {
                let (texture, tail) = w.split_at(size_of::<Texture>() / 8);
                return Ok(Some(Operation::ClearImage {
                    texture: bytemuck::pod_read_unaligned(bytemuck::cast_slice(texture)),
                    region: bytemuck::cast([tail[0], tail[1], tail[2]]),
                    color: bytemuck::cast([tail[3], tail[4]]),
                    mask: tail[5] as u32,
                }));
            }
            op::COPY_IMAGES if n == 2 * size_of::<Texture>() / 8 + 7 => {
                let (textures, tail) = w.split_at(2 * size_of::<Texture>() / 8);
                return Ok(Some(Operation::CopyImages {
                    textures: bytemuck::pod_read_unaligned(bytemuck::cast_slice(textures)),
                    regions: bytemuck::pod_read_unaligned(bytemuck::cast_slice(&tail[..6])),
                    flags: tail[6] as u32,
                }));
            }
            op::COPY_TEXTURE
                if matches!(n.checked_sub(5 + size_of::<Texture>() / 8), Some(0 | 1)) =>
            {
                let end = 2 + size_of::<Texture>() / 8;
                let target = bytemuck::pod_read_unaligned(bytemuck::cast_slice(&w[2..end]));
                let region = bytemuck::pod_read_unaligned(bytemuck::cast_slice(&w[end..end + 3]));
                let strides = w
                    .get(end + 3)
                    .copied()
                    .map_or_else(CopyStrides::default, bytemuck::cast);
                return Ok(Some(Operation::CopyTexture {
                    source: w[0],
                    target,
                    region,
                    flags: w[1] as u32,
                    strides,
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
            op::RESET_COUNTER if n == 1 => return Ok(Some(Operation::ResetCounter(w[0] as u32))),
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
            op::CLAMP if n == 1 => self.raster.depth_clamp = w[0] as u8 != 0,
            op::DISCARD if n == 1 => self.raster.raster_discard = w[0] as u8 != 0,
            op::BOUNDS if n == 2 => {
                self.raster.depth_bounds = w[0] as u8 != 0;
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
            op::CONSERVATIVE if n == 1 => self.raster.conservative_raster = w[0] as u8 != 0,
            op::DEPTH_RANGE if n == 1 => self.viewports[0].depth = bytemuck::cast(w[0]),
            op::VIEWPORTS | op::SCISSORS | op::DEPTH_RANGES if n >= 1 => {
                let stride = if op == op::DEPTH_RANGES { 1 } else { 2 };
                let first = w[0] as usize;
                let count = (n - 1) / stride;
                let end = first
                    .checked_add(count)
                    .filter(|&end| end <= 16 && (n - 1).is_multiple_of(stride))
                    .ok_or("invalid viewport range")?;
                if end > usize::from(self.pipeline.viewport_count) {
                    Arc::make_mut(&mut self.pipeline).viewport_count = end as u8;
                }
                for (viewport, words) in self.viewports[first..end]
                    .iter_mut()
                    .zip(w[1..].chunks_exact(stride))
                {
                    match op {
                        op::VIEWPORTS => viewport.rect = bytemuck::cast([words[0], words[1]]),
                        op::SCISSORS => viewport.scissor = bytemuck::cast([words[0], words[1]]),
                        _ => viewport.depth = bytemuck::cast(words[0]),
                    }
                }
            }
            op::SWIZZLES if n == 1 => self.flip_y = w[0] != 0,
            op::BLEND_COLOR if n == 2 => self.blend_color = bytemuck::cast([w[0], w[1]]),
            op::CLEAR_DEPTH if n == 4 => {
                if let Some(target) = self.attachments.depth {
                    return Ok(Some(Operation::ClearDepth {
                        target,
                        scissor: self.viewports[0].scissor,
                        depth: (w[1] as u8 != 0).then(|| f32::from_bits(w[0] as u32)),
                        stencil: w[2] as u8,
                        mask: w[3] as u8,
                    }));
                }
            }
            op::VIEWPORT if n == 2 => {
                self.viewports[0].rect =
                    bytemuck::cast::<_, [i32; 4]>([w[0], w[1]]).map(|v| v as f32)
            }
            op::SCISSOR if n == 2 => self.viewports[0].scissor = bytemuck::cast([w[0], w[1]]),
            op::BARRIER if n == 1 => return Ok(Some(Operation::Barrier(w[0] as u32))),
            op::CLEAR if n == 4 && w[0] < 8 => {
                if let Some(&Some(target)) = self.attachments.colors.get(w[0] as usize) {
                    return Ok(Some(Operation::Clear {
                        target,
                        scissor: self.viewports[0].scissor,
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
            op::UNIFORM | op::STORAGE if n == 3 => {
                let stage = w[0] as u32;
                let size = if op == op::UNIFORM && stage != 5 {
                    // Graphics constant banks encode their extent in 256-byte units.
                    u64::from((w[2] as u32).wrapping_add(255) & !255)
                } else {
                    u64::from(w[2] as u32)
                };
                let range = if op == op::UNIFORM && stage != 5 && size == 0 {
                    Range::default()
                } else {
                    Range {
                        address: w[1],
                        size,
                    }
                };
                self.buffers[usize::from(stage == 5)].bind(
                    op == op::STORAGE,
                    stage,
                    (w[0] >> 32) as u32,
                    range,
                )?;
            }
            op::SAMPLER_POOL | op::TEXTURE_POOL if n == 2 => {}
            op::TEXTURE if n == 2 => {}
            op::DEFERRED if n == 4 => {}
            op::DRAW | op::INDEXED
                if (op == op::DRAW && n == 3) || (op == op::INDEXED && n == 4) =>
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
                    viewports: &self.viewports[..usize::from(self.pipeline.viewport_count)],
                    blend_color: self.blend_color,
                    indices: (op == op::INDEXED)
                        .then(|| ((w[0] >> 32) as u32, w[2], (w[1] >> 32) as i32)),
                    vertices: &self.vertices,

                    first: (w[0] >> 32) as u32,
                    count: w[1] as u32,
                    first_instance: w[n - 1] as u32,
                    instances: (w[n - 1] >> 32) as u32,
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

    #[test]
    fn called_ranges_follow_current_record_boundaries_and_stop_at_the_cut() {
        let h = |tag, n: u32| u64::from(tag | (n << 8));
        // A different recording now occupies this called range. Its stored
        // capture links describe a different control list.
        let block = CommandBlock::new(
            vec![
                h(op::BARRIER, 1),
                0,
                h(op::TEXTURE, 2),
                0,
                4,
                h(op::DRAW, 3),
                1,
                3,
                1 << 32,
                h(op::TARGETS, 24),
                0,
            ],
            true,
        )
        .unwrap();
        assert_eq!(block.capture_offsets, [2, 5]);
        assert_eq!(block.end, 9);
        let empty = CommandBlock::new(vec![h(op::TARGETS, 24)], true).unwrap();
        assert_eq!(empty.end, 0);
        let mut submission = Submission::default();
        submission.blocks.push(Arc::new(empty));
        submission.blocks.push(Arc::new(
            CommandBlock::new(vec![h(op::BARRIER, 1), 1], false).unwrap(),
        ));
        let mut records = submission.records();
        assert_eq!(records.next().unwrap().unwrap().tag, op::BARRIER);
        assert!(records.next().is_none());
    }
}
