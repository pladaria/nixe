//! Compilation dependencies retain their locations independently of runtime bindings.
//! Reads compare live bytes; rebinding a bank does not invalidate code by itself.
use super::*;
use crate::native::ampere::reads::ReadSlot;

pub(super) struct Inputs {
    consistent: bool,
    words: Vec<(u8, CbufWordKey, u32, ReadSlot<4>)>,
    pointers: Vec<(u8, CbufWordKey, bool, ReadSlot<8>, Option<bool>)>,
}
impl Inputs {
    pub fn new(
        programs: &[(Arc<shader::Program>, u64, u8)],
        pipeline: &Pipeline,
        groups: [Option<u8>; 5],
    ) -> Self {
        let mut words = std::collections::BTreeMap::new();
        let mut consistent = true;
        for (program, _, group) in programs {
            for (&key, &value) in &program.flow_constants {
                words.insert((*group, key), value);
            }
        }
        let mut pointers = Vec::new();
        for (stage, observations) in &pipeline.observations {
            let group = groups[*stage as usize].expect("active compilation stage");
            for (&key, &value) in &observations.cbuf_words {
                if let Some(previous) = words.insert((group, key), value) {
                    consistent &= previous == value;
                }
            }
            for (&key, &expected) in &observations.storage_pointer_types {
                pointers.push((group, key, expected, ReadSlot::default(), None));
            }
        }
        Self {
            consistent,
            words: words
                .into_iter()
                .map(|((group, key), value)| (group, key, value, ReadSlot::default()))
                .collect(),
            pointers,
        }
    }
    pub fn matches(
        &mut self,
        residency: &mut Residency<'_>,
        memory: &mut impl RenderMemory,
        constants: &[[Option<(u64, u32)>; 32]; 5],
    ) -> Result<bool, VulkanError> {
        if !self.consistent {
            return Ok(false);
        }
        let address = |group: u8, key: CbufWordKey, count: u64| {
            let (base, size) = constants[group as usize]
                .get(key.index.0 as usize)
                .copied()
                .flatten()?;
            (u64::from(key.offset.0) + count <= u64::from(size))
                .then(|| base.checked_add(u64::from(key.offset.0)))
                .flatten()
        };
        for (group, key, expected, slot) in &mut self.words {
            let Some(address) = address(*group, *key, 4) else {
                return Ok(false);
            };
            slot.update(residency, memory, address)?;
            if u32::from_le_bytes(slot.bytes) != *expected {
                return Ok(false);
            }
        }
        for (group, key, expected, slot, kind) in &mut self.pointers {
            let actual = if let Some(address) = address(*group, *key, 8) {
                if slot.update(residency, memory, address)? || kind.is_none() {
                    let pointer = u64::from_le_bytes(slot.bytes);
                    *kind = Some(pointer != 0 && memory.image_device_address(pointer)?.is_some());
                }
                kind.unwrap()
            } else {
                false
            };
            if actual != *expected {
                return Ok(false);
            }
        }
        Ok(true)
    }
}
