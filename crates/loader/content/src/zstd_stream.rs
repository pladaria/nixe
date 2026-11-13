//! Incremental decoding of one bounded Zstandard frame, independent of its container.

use std::io::{BufReader, Read};

use nixe_loader_storage::{StorageError, StorageRef};

const CHUNK_SIZE: usize = 1024 * 1024;
pub(crate) const MAX_WINDOW_LOG: u32 = 27;
type Decoder = zstd::stream::read::Decoder<'static, BufReader<StorageReader>>;

pub(crate) struct ProgressiveZstd {
    source: StorageRef,
    size: u64,
    produced: u64,
    decoder: Option<Decoder>,
    scratch: Vec<u8>,
    failure: Option<String>,
}

impl ProgressiveZstd {
    pub(crate) fn new(source: StorageRef, size: u64) -> Self {
        Self {
            source,
            size,
            produced: 0,
            decoder: None,
            scratch: Vec::new(),
            failure: None,
        }
    }

    pub(crate) fn decoded(&self) -> u64 {
        self.produced
    }

    /// Publishes consecutive decoded chunks through the caller's storage/transform policy.
    /// A failed decoder or sink is never resumed from a partially committed chunk.
    pub(crate) fn decode_until(
        &mut self,
        end: u64,
        mut sink: impl FnMut(u64, &mut [u8]) -> Result<(), StorageError>,
    ) -> Result<(), StorageError> {
        if let Some(reason) = &self.failure {
            return Err(StorageError::InvalidData(reason.clone()));
        }
        if end > self.size {
            return Err(StorageError::OutOfBounds);
        }
        if end <= self.produced {
            return Ok(());
        }
        let result = self.decode(end, &mut sink);
        if let Err(error) = &result {
            self.failure = Some(error.to_string());
            self.decoder = None;
            self.scratch = Vec::new();
        }
        result
    }

    fn decode(
        &mut self,
        end: u64,
        sink: &mut impl FnMut(u64, &mut [u8]) -> Result<(), StorageError>,
    ) -> Result<(), StorageError> {
        if self.decoder.is_none() {
            let reader = StorageReader {
                source: self.source.clone(),
                position: 0,
                end: self.source.len()?,
            };
            let mut decoder = zstd::stream::read::Decoder::new(reader)?.single_frame();
            decoder.window_log_max(MAX_WINDOW_LOG)?;
            self.decoder = Some(decoder);
            self.scratch
                .resize((CHUNK_SIZE as u64).min(self.size) as usize, 0);
        }
        while self.produced < end {
            let count = (end - self.produced).min(self.scratch.len() as u64) as usize;
            let output = &mut self.scratch[..count];
            self.decoder
                .as_mut()
                .expect("decoder initialized")
                .read_exact(output)?;
            // Validate the frame boundary before publishing its last bytes.
            if self.produced + count as u64 == self.size {
                let decoder = self.decoder.as_mut().expect("decoder initialized");
                let mut extra = [0];
                if decoder.read(&mut extra)? != 0 {
                    return Err(StorageError::InvalidData(
                        "Zstandard frame exceeds its declared size".into(),
                    ));
                }
                let reader = self.decoder.take().expect("decoder initialized").finish();
                if reader.get_ref().position - reader.buffer().len() as u64 != self.source.len()? {
                    return Err(StorageError::InvalidData(
                        "trailing data after Zstandard frame".into(),
                    ));
                }
            }
            sink(self.produced, output)?;
            self.produced += count as u64;
        }
        if self.produced == self.size {
            self.scratch = Vec::new();
        }
        Ok(())
    }
}

pub(crate) struct StorageReader {
    pub(crate) source: StorageRef,
    pub(crate) position: u64,
    pub(crate) end: u64,
}

impl Read for StorageReader {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        let count = (self.end - self.position).min(buffer.len() as u64) as usize;
        if count != 0 {
            self.source
                .read_at(self.position, &mut buffer[..count])
                .map_err(std::io::Error::other)?;
            self.position += count as u64;
        }
        Ok(count)
    }
}
