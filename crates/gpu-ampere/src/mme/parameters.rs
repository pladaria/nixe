//! Complete invocations borrow stable command/ring bytes. Split invocations and
//! DMA appends use retained storage, materializing only an unconsumed prefix.
use crate::Result;

#[derive(Default)]
pub(crate) struct Parameters {
    bytes: Vec<u8>,
    consumed: usize,
}
impl Parameters {
    fn reclaim(&mut self, additional: usize) {
        let consumed = self.consumed * 4;
        if consumed == self.bytes.len() {
            self.bytes.clear();
            self.consumed = 0;
        } else if self.bytes.capacity() - self.bytes.len() < additional
            && consumed >= self.bytes.len().div_ceil(2)
        {
            self.bytes.copy_within(consumed.., 0);
            self.bytes.truncate(self.bytes.len() - consumed);
            self.consumed = 0;
        }
    }
    pub fn append(&mut self, bytes: &[u8]) {
        self.reclaim(bytes.len());
        self.bytes.extend_from_slice(bytes);
    }
    pub fn invocation<'a>(&'a mut self, input: &'a [u8]) -> Fifo<'a> {
        let borrowed = if self.consumed * 4 == self.bytes.len() {
            self.bytes.clear();
            self.consumed = 0;
            Some(input)
        } else {
            self.append(input);
            None
        };
        Fifo {
            storage: self,
            borrowed,
            #[cfg(feature = "performance-counters")]
            materialized_bytes: 0,
        }
    }
}
pub(crate) struct Fifo<'a> {
    storage: &'a mut Parameters,
    borrowed: Option<&'a [u8]>,
    #[cfg(feature = "performance-counters")]
    pub materialized_bytes: u64,
}
impl Fifo<'_> {
    #[cfg(feature = "performance-counters")]
    pub fn borrowed(&self) -> bool {
        self.borrowed.is_some()
    }
    pub fn consumed(&self) -> usize {
        self.storage.consumed
    }
    pub fn set_consumed(&mut self, count: usize) {
        self.storage.consumed = count;
    }
    pub fn data(&self) -> &[u8] {
        self.borrowed.unwrap_or(&self.storage.bytes)
    }
    pub fn remaining(&self) -> &[u8] {
        &self.data()[self.consumed() * 4..]
    }
    pub fn len(&self) -> usize {
        self.data().len() / 4 - self.consumed()
    }
    pub fn pop(&mut self) -> Option<u32> {
        let word = self.remaining().get(..4)?;
        let value = u32::from_le_bytes(word.try_into().unwrap());
        self.storage.consumed += 1;
        Some(value)
    }
    fn materialize(&mut self, additional: usize) {
        if let Some(bytes) = self.borrowed.take() {
            let remaining = &bytes[self.consumed() * 4..];
            #[cfg(feature = "performance-counters")]
            {
                self.materialized_bytes += remaining.len() as u64;
            }
            self.storage.bytes.reserve(remaining.len() + additional);
            self.storage.bytes.extend_from_slice(remaining);
            self.storage.consumed = 0;
        } else {
            self.storage.reclaim(additional);
        }
    }
    pub fn append(&mut self, bytes: &[u8]) {
        self.materialize(bytes.len());
        self.storage.bytes.extend_from_slice(bytes);
    }
    pub fn read_append(
        &mut self,
        count: usize,
        read: impl FnOnce(&mut [u8]) -> Result<()>,
    ) -> Result<()> {
        self.materialize(count * 4);
        let offset = self.storage.bytes.len();
        self.storage.bytes.resize(offset + count * 4, 0);
        if let Err(error) = read(&mut self.storage.bytes[offset..]) {
            self.storage.bytes.truncate(offset);
            return Err(error);
        }
        Ok(())
    }
    pub fn clear(&mut self) {
        self.borrowed = None;
        self.storage.bytes.clear();
        self.storage.consumed = 0;
    }
}
