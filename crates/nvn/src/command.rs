//! Record values in the guest-provided command memory. EndRecording returns a
//! control-memory handle; a submit reads the command bytes as they stand then.
use super::*;
use bytemuck::{Pod, Zeroable};
use smallvec::SmallVec;
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct Range {
    pub address: u64,
    pub size: u64,
}
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct VertexStream {
    pub stride: u32,
    pub divisor: u32,
}
#[derive(Clone, Copy, Debug)]
pub struct Sampled {
    pub stage: u32,
    pub slot: u32,
    pub texture: Texture,
    pub sampler: Sampler,
}
#[derive(Debug)]
pub struct Draw {
    pub target: Texture,
    pub viewport: [i32; 4],
    pub scissor: [i32; 4],
    pub shaders: SmallVec<[Shader; 2]>,
    pub attributes: SmallVec<[u32; 4]>,
    pub streams: SmallVec<[VertexStream; 2]>,
    pub vertices: SmallVec<[(u32, Range); 2]>,
    pub uniforms: SmallVec<[(u32, u32, Range); 2]>,
    pub textures: SmallVec<[Sampled; 2]>,
    pub primitive: u32,
    pub first: u32,
    pub count: u32,
}
#[derive(Debug)]
pub enum Operation {
    Barrier(u32),
    Clear {
        target: Texture,
        scissor: [i32; 4],
        color: [f32; 4],
        mask: u32,
    },
    Draw(Box<Draw>),
}
#[derive(Clone, Copy, Default, Pod, Zeroable)]
#[repr(C)]
struct Recorder {
    command: u64,
    position: u64,
    inline_flags: u64,
    limit: u64,
    control: u64,
    control_end: u64,
    begin: u64,
    recording: u64,
}
pub fn initialize(c: &Call) -> Result<u64> {
    c.write(c.args[0], &Recorder::default())?;
    Ok(1)
}
pub fn add_commands(c: &Call) -> Result<u64> {
    let pool: pool::Pool = c.read(c.args[1])?;
    let start = pool
        .cpu
        .checked_add(c.args[2])
        .ok_or("command address overflow")?;
    if c.args[2] > pool.bytes || c.args[3] > pool.bytes - c.args[2] {
        return Err("command memory exceeds pool".into());
    }
    let mut r: Recorder = c.read(c.args[0])?;
    if r.recording != 0 {
        return Err("adding another command segment during recording is unsupported".into());
    }
    r.command = start;
    r.position = start;
    r.limit = start + c.args[3];
    c.write(c.args[0], &r)?;
    Ok(0)
}
pub fn add_control(c: &Call) -> Result<u64> {
    let end = c.args[1]
        .checked_add(c.args[2])
        .ok_or("control memory overflow")?;
    c.write(c.args[0] + 32, &[c.args[1], end])?;
    Ok(0)
}
pub fn begin(c: &Call) -> Result<u64> {
    let r: Recorder = c.read(c.args[0])?;
    if r.recording != 0 {
        return Err("already recording".into());
    }
    c.write(c.args[0] + 48, &[r.position, 1])?;
    Ok(0)
}
pub fn end(c: &Call) -> Result<u64> {
    let r: Recorder = c.read(c.args[0])?;
    if r.recording == 0 || r.control_end.saturating_sub(r.control) < 16 {
        return Err("no recording or insufficient control memory".into());
    }
    c.write(
        r.control,
        &Range {
            address: r.begin,
            size: r.position - r.begin,
        },
    )?;
    c.write(c.args[0] + 32, &(r.control + 16))?;
    c.write(c.args[0] + 56, &0u64)?;
    Ok(r.control)
}
// The record tag is a guest word. Match its value without constructing an
// enum from unchecked guest bytes.
mod op {
    pub const SAMPLER_POOL: u32 = 0;
    pub const TEXTURE_POOL: u32 = 1;
    pub const TARGETS: u32 = 2;
    pub const SCISSOR: u32 = 3;
    pub const VIEWPORT: u32 = 4;
    pub const BARRIER: u32 = 5;
    pub const CLEAR: u32 = 6;
    pub const PROGRAM: u32 = 7;
    pub const ATTRIBUTES: u32 = 8;
    pub const STREAMS: u32 = 9;
    pub const VERTEX: u32 = 10;
    pub const UNIFORM: u32 = 11;
    pub const TEXTURE: u32 = 12;
    pub const DRAW: u32 = 13;
}
fn record(c: &Call, op: u32, words: &[u64]) -> Result<u64> {
    let r: Recorder = c.read(c.args[0])?;
    let bytes = 8 * (1 + words.len()) as u64;
    if r.recording == 0 || r.limit.saturating_sub(r.position) < bytes {
        return Err("command memory exhausted or outside recording".into());
    }
    let mut record = [0u64; 34];
    record[0] = (op as u64) | ((words.len() as u64) << 32);
    record[1..=words.len()].copy_from_slice(words);
    c.write_bytes(r.position, bytemuck::cast_slice(&record[..1 + words.len()]))?;
    c.write(c.args[0] + 8, &(r.position + bytes))?;
    Ok(0)
}
macro_rules! simple {
    ($name:ident,$op:ident,$n:literal) => {
        pub fn $name(c: &Call) -> Result<u64> {
            record(c, op::$op, &c.args[1..=$n])
        }
    };
}
fn bind_pool(c: &Call, op: u32) -> Result<u64> {
    let pool: pool::DescriptorPool = c.read(c.args[1])?;
    record(c, op, &[pool.base, pool.count.into()])
}
pub fn sampler_pool(c: &Call) -> Result<u64> {
    bind_pool(c, op::SAMPLER_POOL)
}
pub fn texture_pool(c: &Call) -> Result<u64> {
    bind_pool(c, op::TEXTURE_POOL)
}
pub fn program(c: &Call) -> Result<u64> {
    if c.args[2] != 3 {
        return Err("only a vertex/fragment stage group is supported".into());
    }
    let p: program::Program = c.read(c.args[1])?;
    let stages = p
        .stages
        .get(..p.count as usize)
        .ok_or("invalid program stage count")?;
    record(c, op::PROGRAM, bytemuck::cast_slice(stages))
}
simple!(scissor, SCISSOR, 4);
simple!(viewport, VIEWPORT, 4);
simple!(barrier, BARRIER, 1);
simple!(vertex, VERTEX, 3);
simple!(uniform, UNIFORM, 4);
simple!(draw, DRAW, 3);
pub fn texture(c: &Call) -> Result<u64> {
    record(
        c,
        op::TEXTURE,
        &[(c.args[1] as u32 as u64) | (c.args[2] << 32), c.args[3]],
    )
}
pub fn targets(c: &Call) -> Result<u64> {
    if c.args[1] != 1
        || c.args[4] != 0
        || c.args[5] != 0
        || c.args[3] != 0 && c.read::<u64>(c.args[3])? != 0
    {
        return Err("unsupported attachment count, depth target or texture view".into());
    }
    let target: Texture = c.read(c.read::<u64>(c.args[2])?)?;
    record(
        c,
        op::TARGETS,
        bytemuck::cast_slice(std::slice::from_ref(&target)),
    )
}
pub fn clear(c: &Call) -> Result<u64> {
    let rgba: [u64; 2] = c.read(c.args[2])?;
    record(c, op::CLEAR, &[c.args[1], rgba[0], rgba[1], c.args[3]])
}
fn array(c: &Call, op: u32, stride: usize) -> Result<u64> {
    let n = c.args[1] as u32 as usize;
    if n > 32 {
        return Err("too many vertex inputs".into());
    }
    let mut buf = [0u64; 33];
    buf[0] = n as u64;
    c.read_bytes(
        c.args[2],
        &mut bytemuck::cast_slice_mut(&mut buf[1..])[..n * stride],
    )?;
    record(c, op, &buf[..1 + (n * stride).div_ceil(8)])
}
pub fn attributes(c: &Call) -> Result<u64> {
    array(c, op::ATTRIBUTES, 4)
}
pub fn streams(c: &Call) -> Result<u64> {
    array(c, op::STREAMS, 8)
}
#[derive(Default)]
pub(crate) struct State {
    target: Option<Texture>,
    viewport: [i32; 4],
    scissor: [i32; 4],
    shaders: SmallVec<[Shader; 2]>,
    texture_pool: pool::DescriptorPool,
    sampler_pool: pool::DescriptorPool,
    scratch: Vec<u64>,
    attributes: Vec<u32>,
    streams: Vec<VertexStream>,
    vertices: Vec<(u32, Range)>,
    uniforms: Vec<(u32, u32, Range)>,
    textures: Vec<(u32, u32, u64)>,
}
impl State {
    pub(crate) fn replay(&mut self, c: &Call, handle: u64, ops: &mut Vec<Operation>) -> Result<()> {
        let r: Range = c.read(handle)?;
        if !r.size.is_multiple_of(8) || r.size > 64 << 20 {
            return Err("invalid command list length".into());
        }
        self.scratch.resize(r.size as usize / 8, 0);
        c.read_bytes(r.address, bytemuck::cast_slice_mut(&mut self.scratch))?;
        let mut commands = self.scratch.as_slice();
        while let Some((&h, tail)) = commands.split_first() {
            let n = (h >> 32) as usize;
            let (w, next) = tail
                .split_at_checked(n)
                .ok_or("truncated NVN command record")?;
            commands = next;
            let op = h as u32;
            match op {
                op::SAMPLER_POOL if n == 2 => {
                    self.sampler_pool = pool::DescriptorPool::new(w[0], w[1] as u32)
                }
                op::TEXTURE_POOL if n == 2 => {
                    self.texture_pool = pool::DescriptorPool::new(w[0], w[1] as u32)
                }
                op::TARGETS if n == 4 => {
                    self.target = Some(bytemuck::pod_read_unaligned(bytemuck::cast_slice(w)))
                }
                op::VIEWPORT if n == 4 => self.viewport = std::array::from_fn(|i| w[i] as i32),
                op::SCISSOR if n == 4 => self.scissor = std::array::from_fn(|i| w[i] as i32),
                op::BARRIER if n == 1 => ops.push(Operation::Barrier(w[0] as u32)),
                op::CLEAR if n == 4 && w[0] == 0 => ops.push(Operation::Clear {
                    target: self.target.ok_or("no color attachment bound")?,
                    scissor: self.scissor,
                    color: bytemuck::cast([w[1], w[2]]),
                    mask: w[3] as u32,
                }),
                op::PROGRAM if n.is_multiple_of(3) && n <= 18 => {
                    self.shaders = SmallVec::from_slice(bytemuck::cast_slice(w))
                }
                op::ATTRIBUTES if n > 0 && w[0] <= 32 && n == 1 + (w[0] as usize).div_ceil(2) => {
                    self.attributes.clear();
                    self.attributes
                        .extend_from_slice(&bytemuck::cast_slice(&w[1..n])[..w[0] as usize]);
                }
                op::STREAMS if n > 0 && w[0] <= 32 && n == 1 + w[0] as usize => {
                    self.streams.clear();
                    self.streams
                        .extend_from_slice(bytemuck::cast_slice(&w[1..n]));
                }
                op::VERTEX if n == 3 => {
                    let item = (
                        w[0] as u32,
                        Range {
                            address: w[1],
                            size: w[2],
                        },
                    );
                    put(&mut self.vertices, item, |v| v.0 == item.0);
                }
                op::UNIFORM if n == 4 => {
                    let item = (
                        w[0] as u32,
                        w[1] as u32,
                        Range {
                            address: w[2],
                            size: w[3],
                        },
                    );
                    put(&mut self.uniforms, item, |v| v.0 == item.0 && v.1 == item.1);
                }
                op::TEXTURE if n == 2 => {
                    let item = (w[0] as u32, (w[0] >> 32) as u32, w[1]);
                    put(&mut self.textures, item, |v| v.0 == item.0 && v.1 == item.1);
                }
                op::DRAW if n == 3 => {
                    let shaders = self.shaders.clone();
                    let textures = self
                        .textures
                        .iter()
                        .map(|&(stage, slot, h)| {
                            Ok(Sampled {
                                stage,
                                slot,
                                texture: c.read(self.texture_pool.slot(h as u32 & 0xfffff)?)?,
                                sampler: c.read(self.sampler_pool.slot(h as u32 >> 20)?)?,
                            })
                        })
                        .collect::<Result<_>>()?;
                    ops.push(Operation::Draw(Box::new(Draw {
                        target: self.target.ok_or("no color attachment bound")?,
                        viewport: self.viewport,
                        scissor: self.scissor,
                        shaders,
                        attributes: SmallVec::from_slice(&self.attributes),
                        streams: SmallVec::from_slice(&self.streams),
                        vertices: SmallVec::from_slice(&self.vertices),
                        uniforms: SmallVec::from_slice(&self.uniforms),
                        textures,
                        primitive: w[0] as u32,
                        first: w[1] as u32,
                        count: w[2] as u32,
                    })));
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
fn put<T>(slots: &mut Vec<T>, value: T, matches: impl FnMut(&T) -> bool) {
    if let Some(i) = slots.iter().position(matches) {
        slots[i] = value
    } else {
        slots.push(value)
    }
}
