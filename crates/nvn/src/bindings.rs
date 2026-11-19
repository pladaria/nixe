//! Queue-local binding selections and immutable inputs for submitted work.
use super::{
    Call, DescriptorPool, Range, Result, Sampler, Shader, ShaderUsage, Texture, UsageSlot,
};
use nixe_memory::GuestVirtualAddress;
use smallvec::SmallVec;
use std::sync::Arc;

type Slot = (u32, u32);
pub type TextureShape = (u32, u32, u16, u8);

#[derive(Clone, Copy, Debug)]
pub struct Sampled {
    pub stage: u32,
    pub slot: u32,
    pub handle: u64,
    pub texture: Texture,
    pub sampler: Sampler,
    ids: [u32; 2],
}

impl Sampled {
    pub fn descriptor_ids(&self) -> [u32; 2] {
        self.ids
    }

    fn matches(&self, key: Slot, handle: u64, ids: [u32; 2]) -> bool {
        (self.stage, self.slot) == key && self.handle == handle && self.ids == ids
    }

    fn shape(&self) -> TextureShape {
        (
            self.stage,
            self.slot,
            self.texture.format,
            self.texture.target_kind(),
        )
    }
}

#[derive(Clone, Default, Debug)]
pub struct SampledBindings {
    pub entries: Vec<Sampled>,
    pub shapes: Arc<[TextureShape]>,
}

impl SampledBindings {
    pub fn get(&self, stage: u32, slot: u32) -> Option<&Sampled> {
        let index = self
            .entries
            .binary_search_by_key(&(stage, slot), |s| (s.stage, s.slot))
            .ok()?;
        Some(&self.entries[index])
    }
}

type Ranges = SmallVec<[(u32, u32, Range); 2]>;

#[derive(Clone, Default, Debug)]
pub struct BufferBindings {
    pub uniforms: Ranges,
    pub storage: Ranges,
}

impl BufferBindings {
    fn table(&self, storage: bool) -> &Ranges {
        if storage {
            &self.storage
        } else {
            &self.uniforms
        }
    }
    pub fn get(&self, storage: bool, stage: u32, slot: u32) -> Option<Range> {
        let table = self.table(storage);
        let i = table
            .binary_search_by_key(&(stage, slot), |x| (x.0, x.1))
            .ok()?;
        Some(table[i].2)
    }
}

#[derive(Clone, Default, Debug)]
pub struct Bindings {
    pub sampled: Arc<SampledBindings>,
    pub buffers: Arc<BufferBindings>,
    /// The driver bank exposes storage and compute-uniform addresses only.
    pub driver_buffers: Arc<BufferBindings>,
    pub handles: Arc<Vec<(Slot, u64)>>,
    pub usage_slot: UsageSlot,
    pub captured: Option<Arc<ShaderUsage>>,
}

#[derive(Default)]
pub(crate) struct BindingState {
    current: Bindings,
    programs: std::collections::HashMap<SmallVec<[Shader; 2]>, UsageSlot>,
    shaders: SmallVec<[Shader; 2]>,
    active: Vec<(Slot, u64)>,
    projected: Option<Arc<[TextureShape]>>,
    addresses_dirty: bool,
    words: Vec<(GuestVirtualAddress, u32)>,
}

impl BindingState {
    pub fn select_shaders(&mut self, shaders: impl Iterator<Item = Shader> + Clone) {
        if shaders.clone().eq(self.shaders.iter().copied()) {
            return;
        }
        self.shaders.clear();
        self.shaders.extend(shaders);
        self.current.usage_slot = self
            .programs
            .entry(self.shaders.clone())
            .or_default()
            .clone();
    }

    pub fn bind_buffer(&mut self, storage: bool, stage: u32, slot: u32, range: Range) {
        let table = self.current.buffers.table(storage);
        let pos = table.binary_search_by_key(&(stage, slot), |x| (x.0, x.1));
        if pos.is_ok_and(|i| table[i].2 == range) {
            return;
        }
        let buffers = Arc::make_mut(&mut self.current.buffers);
        let table = if storage {
            &mut buffers.storage
        } else {
            &mut buffers.uniforms
        };
        match pos {
            Ok(i) => table[i].2 = range,
            Err(i) => table.insert(i, (stage, slot, range)),
        }
        if storage || stage == 5 {
            self.current.driver_buffers = self.current.buffers.clone();
        }
    }

    pub fn bind_texture(&mut self, stage: u32, slot: u32, handle: u64) {
        let key = (stage, slot);
        let handles = &mut self.current.handles;
        let pos = handles.binary_search_by_key(&key, |&(k, _)| k);
        match (pos, handle as u32 == 0) {
            (Ok(i), true) => {
                Arc::make_mut(handles).remove(i);
            }
            (Ok(i), false) if handles[i].1 != handle => Arc::make_mut(handles)[i].1 = handle,
            (Err(i), false) => Arc::make_mut(handles).insert(i, (key, handle)),
            _ => return,
        }
        self.addresses_dirty = true;
    }

    pub fn pool_changed(&mut self) {
        self.addresses_dirty = true;
    }

    pub fn snapshot(
        &mut self,
        c: &Call,
        texture_pool: DescriptorPool,
        sampler_pool: DescriptorPool,
    ) -> Result<Bindings> {
        let known = self.current.usage_slot.get().cloned();
        if let Some(usage) = &known
            && usage.current(c.session.graphics()?.heap.device().generation())
        {
            self.capture(c, texture_pool, sampler_pool, Some(&usage.shapes))?;
            if self.current.sampled.shapes == usage.shapes {
                self.current.captured = known;
                return Ok(self.current.clone());
            }
        }
        if known.is_some() {
            let slot = UsageSlot::default();
            self.programs.insert(self.shaders.clone(), slot.clone());
            self.current.usage_slot = slot;
        }
        self.capture(c, texture_pool, sampler_pool, None)?;
        self.current.captured = None;
        Ok(self.current.clone())
    }

    fn capture(
        &mut self,
        c: &Call,
        texture_pool: DescriptorPool,
        sampler_pool: DescriptorPool,
        projection: Option<&Arc<[TextureShape]>>,
    ) -> Result<()> {
        let changed = match (&self.projected, projection) {
            (Some(old), Some(new)) => !Arc::ptr_eq(old, new),
            (None, None) => false,
            _ => true,
        };
        if changed || self.addresses_dirty {
            self.projected = projection.cloned();
            self.active.clear();
            if let Some(shapes) = projection {
                for &(stage, slot, _, _) in shapes.iter() {
                    let key = (stage, slot);
                    if let Ok(i) = self.current.handles.binary_search_by_key(&key, |v| v.0) {
                        self.active.push(self.current.handles[i]);
                    }
                }
            } else {
                self.active.extend_from_slice(&self.current.handles);
            }

            self.words.clear();
            for &(_, handle) in &self.active {
                let texture = texture_pool.slot(handle as u32 & 0xfffff)?;
                let sampler = sampler_pool.slot(handle as u32 >> 20)?;
                self.words.extend([
                    (GuestVirtualAddress::new(texture), 0),
                    (GuestVirtualAddress::new(sampler), 0),
                ]);
            }
            self.addresses_dirty = false;
        }

        // Slots can be overwritten or copied through an alias without any new NVN bind.
        // Read their actual IDs at consumption. Release normal memory admission before
        // entering the coherency owner for a pending GPU write or a split mapping.
        let mut remaining = self.words.as_mut_slice();
        while !remaining.is_empty() {
            let n = c
                .memory
                .read_direct_words(remaining)
                .map_err(|e| e.to_string())?;
            let Some((slow, tail)) = remaining[n..].split_first_mut() else {
                break;
            };
            slow.1 = c.read(slow.0.get())?;
            remaining = tail;
        }

        let mut slots = self
            .active
            .iter()
            .zip(self.words.chunks_exact(2))
            .enumerate();
        let sampled = &self.current.sampled;
        let first_change = slots.find(|&(i, (&(key, handle), words))| {
            sampled
                .entries
                .get(i)
                .is_none_or(|s| !s.matches(key, handle, [words[0].1, words[1].1]))
        });
        if first_change.is_none() && sampled.entries.len() == self.active.len() {
            return Ok(());
        }

        let sampled = Arc::make_mut(&mut self.current.sampled);
        let mut shapes_changed = sampled.entries.len() != self.active.len();
        sampled.entries.truncate(self.active.len());
        let mut textures = None;
        let mut samplers = None;
        for (i, (&((stage, slot), handle), words)) in first_change.into_iter().chain(slots) {
            let ids = [words[0].1, words[1].1];
            let previous = sampled.entries.get(i);
            if previous.is_some_and(|s| s.matches((stage, slot), handle, ids)) {
                continue;
            }
            let texture = match previous {
                Some(s) if s.ids[0] == ids[0] => s.texture,
                _ => textures
                    .get_or_insert_with(|| c.session.textures.lock().unwrap())
                    .get(ids[0])?,
            };
            let sampler = match previous {
                Some(s) if s.ids[1] == ids[1] => s.sampler,
                _ => samplers
                    .get_or_insert_with(|| c.session.samplers.lock().unwrap())
                    .get(ids[1])?,
            };
            let next = Sampled {
                stage,
                slot,
                handle,
                texture,
                sampler,
                ids,
            };
            if let Some(old) = sampled.entries.get_mut(i) {
                shapes_changed |= old.shape() != next.shape();
                *old = next;
            } else {
                sampled.entries.push(next);
            }
        }
        if shapes_changed {
            sampled.shapes = sampled.entries.iter().map(Sampled::shape).collect();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Session;
    use nixe_memory::{AddressSpace, AddressWidth, Machine, MemoryPermissions, PAGE_SIZE};

    #[test]
    fn submitted_bindings_keep_their_values_across_alias_writes_pool_switches_and_unbinds() {
        let machine = Machine::new(PAGE_SIZE * 2).unwrap();
        let allocation = machine
            .account(PAGE_SIZE * 2)
            .unwrap()
            .allocate(PAGE_SIZE * 2)
            .unwrap();
        let space = AddressSpace::new(machine, AddressWidth::Bits32).unwrap();
        for base in [0x1000, 0x4000] {
            space
                .map(
                    GuestVirtualAddress::new(base),
                    allocation.clone(),
                    0,
                    PAGE_SIZE * 2,
                    MemoryPermissions::READ_WRITE,
                )
                .unwrap();
        }
        let session = Session::new();
        let c = Call {
            memory: &space,
            session: &session,
            args: [0; 8],
            pending: Default::default(),
        };
        let texture = Texture {
            address: 0x9000,
            format: 1,
            ..Default::default()
        };
        let new_texture = Texture {
            address: 0xa000,
            ..texture
        };
        let id = session.textures.lock().unwrap().intern(texture);
        let new_id = session.textures.lock().unwrap().intern(new_texture);
        let sid = session.samplers.lock().unwrap().intern(Sampler::default());
        let textures = DescriptorPool::new(0x1000, 2);
        let samplers = DescriptorPool::new(0x1800, 1);
        c.write(0x1020, &id).unwrap();
        c.write(0x1800, &sid).unwrap();
        let mut state = BindingState::default();
        state.bind_texture(1, 0, 1);
        let first = state.snapshot(&c, textures, samplers).unwrap();
        state.bind_texture(1, 0, 1);
        let range = Range {
            address: 0x8000,
            size: 128,
        };
        state.bind_buffer(false, 1, 2, range);
        let buffers = state.snapshot(&c, textures, samplers).unwrap();
        assert!(Arc::ptr_eq(&first.sampled, &buffers.sampled));
        assert!(first.buffers.get(false, 1, 2).is_none());
        state.bind_buffer(false, 1, 2, range);
        c.write(0x4020, &new_id).unwrap();
        let changed = state.snapshot(&c, textures, samplers).unwrap();
        assert_eq!(first.sampled.get(1, 0).unwrap().texture, texture);
        assert_eq!(changed.sampled.get(1, 0).unwrap().texture, new_texture);
        assert!(Arc::ptr_eq(&changed.sampled.shapes, &first.sampled.shapes));
        assert!(Arc::ptr_eq(&changed.buffers, &buffers.buffers));
        // A warm program consumes one of two bound slots. The other raw
        // handle remains visible to ordinary reads of driver constant bank 2.
        state.bind_texture(1, 4, 1);
        let used: Arc<[TextureShape]> = Arc::from([changed.sampled.get(1, 0).unwrap().shape()]);
        state.capture(&c, textures, samplers, Some(&used)).unwrap();
        assert_eq!(state.words.len(), 2);
        assert_eq!(state.current.handles.len(), 2);
        assert!(state.current.sampled.get(1, 4).is_none());
        c.write(0x4020, &id).unwrap();
        state.capture(&c, textures, samplers, Some(&used)).unwrap();
        assert_eq!(state.current.sampled.get(1, 0).unwrap().texture, texture);
        state.capture(&c, textures, samplers, None).unwrap();
        assert_eq!(state.current.sampled.get(1, 4).unwrap().texture, texture);
        state.bind_texture(1, 4, 0);
        // A descriptor word spanning two pages must still use the coherent read path.
        let split = DescriptorPool::new(0x1fde, 2);
        let typed = Texture {
            format: 2,
            ..texture
        };
        c.write(0x1ffe, &session.textures.lock().unwrap().intern(typed))
            .unwrap();
        state.pool_changed();
        let switched = state.snapshot(&c, split, samplers).unwrap();
        assert_eq!(switched.sampled.get(1, 0).unwrap().texture, typed);
        assert_ne!(switched.sampled.shapes, changed.sampled.shapes);
        state.bind_texture(1, 0, 0);
        let unbound = state.snapshot(&c, split, samplers).unwrap();
        assert!(unbound.sampled.entries.is_empty());
        assert!(unbound.sampled.shapes.is_empty());
        assert!(switched.sampled.get(1, 0).is_some());
    }
}
