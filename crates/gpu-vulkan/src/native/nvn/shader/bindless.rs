//! NVN2 fixed shader operands may obtain a texture and sampler from separate
//! words in the current uniform buffers. Resolve their combined descriptor
//! against the active GPU pools without copying the uniform allocation.
use super::*;
use smallvec::SmallVec;

#[derive(Default)]
pub(in crate::native::nvn) struct Plan {
    words: Vec<(u32, CbufWordKey)>,
    handles: Vec<(u32, HandleKey)>,
    sources: Vec<(usize, usize)>,
    pub(super) shapes: Vec<usize>,
    pub(in crate::native::nvn) bindings: Vec<usize>,
}

impl Plan {
    pub(super) fn new(shapes: &[DynamicShape], inputs: &[Input]) -> Self {
        let mut plan = Self::default();
        for shape in shapes {
            let slot = plan.insert(shape.stage, shape.key);
            plan.shapes.push(slot);
        }
        if inputs.iter().any(|input| Self::binding(input).is_some()) {
            for input in inputs {
                let slot = Self::binding(input)
                    .map(|(stage, key)| plan.insert(stage, key))
                    .unwrap_or(usize::MAX);
                plan.bindings.push(slot);
            }
        }
        plan
    }

    fn binding(input: &Input) -> Option<(u32, HandleKey)> {
        match *input {
            Input::Texture {
                stage,
                bindless: Some(key),
                ..
            }
            | Input::Image {
                stage,
                bindless: Some(key),
                ..
            }
            | Input::TexelBuffer {
                stage, handle: key, ..
            } => Some((stage, key)),
            _ => None,
        }
    }

    fn insert(&mut self, stage: u32, key: HandleKey) -> usize {
        if let Some(slot) = self.handles.iter().position(|&entry| entry == (stage, key)) {
            return slot;
        }
        let mut word = |index, offset| {
            let source = (stage, CbufWordKey { index, offset });
            self.words
                .iter()
                .position(|&entry| entry == source)
                .unwrap_or_else(|| {
                    self.words.push(source);
                    self.words.len() - 1
                })
        };
        let texture = word(key.index, key.offset);
        let sampler = if key.has_secondary {
            word(key.sec_index, key.sec_offset)
        } else {
            usize::MAX
        };
        let slot = self.handles.len();
        self.handles.push((stage, key));
        self.sources.push((texture, sampler));
        slot
    }
}

/// Captures one draw or dispatch's GPU-visible selector values. Native resource
/// selection may retry during this operation without reading the words again.
#[derive(Default)]
pub(in crate::native::nvn) struct Descriptors {
    translation: Option<Rc<Translation>>,
    ids: SmallVec<[[u32; 2]; 16]>,
}

impl Descriptors {
    pub(in crate::native::nvn) fn prepare(
        &mut self,
        translation: &Rc<Translation>,
        b: &nixe_nvn::Bindings,
        residency: &mut Residency<'_>,
    ) -> Result<(), VulkanError> {
        let plan = &translation.descriptors;
        if plan.handles.is_empty()
            || self
                .translation
                .as_ref()
                .is_some_and(|old| Rc::ptr_eq(old, translation))
        {
            return Ok(());
        }
        let pools = b
            .bindless
            .ok_or_else(|| error("NVN2 pools are not active"))?;
        let mut words: SmallVec<[_; 24]> = plan
            .words
            .iter()
            .map(|&(stage, key)| cbuf::<4>(b, stage, key))
            .collect::<Result<_, _>>()?;
        read_words(residency, &mut words)?;
        let mut addresses: SmallVec<[Word<4>; 32]> = SmallVec::new();
        for (&(_, key), &(texture, sampler)) in plan.handles.iter().zip(&plan.sources) {
            let texture = u32::from_le_bytes(words[texture].value());
            let sampler = words
                .get(sampler)
                .map_or(0, |w| u32::from_le_bytes(w.value()));
            addresses.extend(descriptor_addresses(pools, key, texture, sampler)?.map_or(
                [Word::Value([0; 4]); 2],
                |[texture, sampler]| {
                    [
                        Word::Address(texture),
                        if sampler == 0 {
                            Word::Value([0; 4])
                        } else {
                            Word::Address(sampler)
                        },
                    ]
                },
            ));
        }
        read_words(residency, &mut addresses)?;
        self.ids.clear();
        self.ids.extend(addresses.chunks_exact(2).map(|pair| {
            [
                u32::from_le_bytes(pair[0].value()),
                u32::from_le_bytes(pair[1].value()),
            ]
        }));
        self.translation = Some(translation.clone());
        Ok(())
    }

    pub(in crate::native::nvn) fn get(&self, index: usize) -> [u32; 2] {
        self.ids[index]
    }
}

fn descriptor_addresses(
    pools: nixe_nvn::BindlessPools<'_>,
    key: HandleKey,
    texture: u32,
    sampler: u32,
) -> Result<Option<[u64; 2]>, VulkanError> {
    let handle = texture
        .checked_shl(key.shift_left)
        .ok_or_else(|| error("invalid NVN2 handle texture shift"))?
        | sampler
            .checked_shl(key.sec_shift_left)
            .ok_or_else(|| error("invalid NVN2 handle sampler shift"))?;
    Ok(pools.addresses(handle))
}

pub(super) fn descriptor_ids(
    residency: &mut Residency<'_>,
    bindings: &nixe_nvn::Bindings,
    stage: u32,
    key: HandleKey,
) -> Result<[u32; 2], VulkanError> {
    let pools = bindings
        .bindless
        .ok_or_else(|| error("NVN2 pools are not active"))?;
    let texture = cbuf_word(residency, bindings, stage, key.index, key.offset)?;
    let sampler = if !key.has_secondary {
        0
    } else if key.index == key.sec_index && key.offset == key.sec_offset {
        texture
    } else {
        cbuf_word(residency, bindings, stage, key.sec_index, key.sec_offset)?
    };
    let Some([texture, sampler]) = descriptor_addresses(pools, key, texture, sampler)? else {
        return Ok([0; 2]);
    };
    Ok([
        device_word(residency, texture)?,
        if sampler == 0 {
            0
        } else {
            device_word(residency, sampler)?
        },
    ])
}

fn cbuf_word(
    residency: &mut Residency<'_>,
    bindings: &nixe_nvn::Bindings,
    stage: u32,
    index: CbufIndex,
    offset: ByteOffset,
) -> Result<u32, VulkanError> {
    read_cbuf(residency, bindings, stage, CbufWordKey { index, offset }).map(u32::from_le_bytes)
}

pub(in crate::native::nvn) fn read_cbuf<const N: usize>(
    residency: &mut Residency<'_>,
    bindings: &nixe_nvn::Bindings,
    stage: u32,
    key: CbufWordKey,
) -> Result<[u8; N], VulkanError> {
    match cbuf::<N>(bindings, stage, key)? {
        Word::Value(value) => Ok(value),
        Word::Address(address) => read_device(residency, address),
    }
}

#[derive(Clone, Copy)]
enum Word<const N: usize> {
    Value([u8; N]),
    Address(u64),
}

impl<const N: usize> Word<N> {
    fn value(self) -> [u8; N] {
        match self {
            Self::Value(value) => value,
            Self::Address(_) => unreachable!("unread NVN operand"),
        }
    }
}

fn read_words<const N: usize>(
    residency: &mut Residency<'_>,
    words: &mut [Word<N>],
) -> Result<(), VulkanError> {
    residency.flush_read_spans(words.iter().filter_map(|word| match *word {
        Word::Address(address) => Some((address, N as u64)),
        Word::Value(_) => None,
    }))?;
    for word in words {
        if let Word::Address(address) = *word {
            let mut value = [0; N];
            residency
                .runtime
                .device_reader
                .read_block_unsafe(address, &mut value)?;
            *word = Word::Value(value);
        }
    }
    Ok(())
}

fn cbuf<const N: usize>(
    bindings: &nixe_nvn::Bindings,
    stage: u32,
    key: CbufWordKey,
) -> Result<Word<N>, VulkanError> {
    let bank = key.index.0;
    let offset = key.offset.0;
    if (24..32).contains(&bank) {
        let Some(bytes) = bindings.buffers.push_block((bank - 24) as usize) else {
            return Ok(Word::Value([0; N]));
        };
        let at = offset as usize;
        return bytes
            .get(at..at + N)
            .map(|word| Word::Value(word.try_into().unwrap()))
            .ok_or_else(|| error("NVN2 operand outside its push block"));
    }
    let slot = bank
        .checked_sub(3)
        .ok_or_else(|| error(format!("NVN2 operand in unsupported bank {bank}")))?;
    let Some(range) = bindings.buffers.get(false, stage, slot) else {
        return Ok(Word::Value([0; N]));
    };
    if u64::from(offset) + N as u64 > range.size {
        return Ok(Word::Value([0; N]));
    }
    Ok(Word::Address(range.address + u64::from(offset)))
}

fn device_word(residency: &mut Residency<'_>, address: u64) -> Result<u32, VulkanError> {
    read_device(residency, address).map(u32::from_le_bytes)
}

fn read_device<const N: usize>(
    residency: &mut Residency<'_>,
    address: u64,
) -> Result<[u8; N], VulkanError> {
    residency.flush_read_span(address, N as u64)?;
    let mut word = [0; N];
    residency
        .runtime
        .device_reader
        .read_block_unsafe(address, &mut word)
        .map_err(error)?;
    Ok(word)
}
