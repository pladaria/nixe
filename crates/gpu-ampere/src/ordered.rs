//! Ampere uses the same ordered word ring and method/location records as the
//! Maxwell frontend. Only engine identities and the consumers are ISA-specific.
use crate::{Result, Source};
use nixe_gpu::ordered::{self, ENGINE, LOCATION, METHOD, ORIGIN};
pub struct Writer<T> {
    pub ring: ordered::Sender<T>,
    previous: Option<Source>,
}
impl<T> Writer<T> {
    pub fn new(ring: ordered::Sender<T>) -> Self {
        Self {
            ring,
            previous: None,
        }
    }
    pub fn method(&mut self, source: Source) -> Result<()> {
        let old = self.previous;
        if old.is_none_or(|s| {
            s.entry != source.entry
                || s.pushbuffer != source.pushbuffer
                || s.submission != source.submission
        }) {
            self.ring
                .write(
                    &[ORIGIN | (u64::from(source.entry) << 32), source.pushbuffer],
                    &[],
                )
                .map_err(|_| "preparation owner stopped")?;
        }
        if old.is_none_or(|s| s.offset != source.offset || s.entry != source.entry) {
            self.ring
                .write(&[LOCATION | (u64::from(source.offset) << 32)], &[])
                .map_err(|_| "preparation owner stopped")?;
        }
        if old.is_none_or(|s| s.class != source.class || s.subchannel != source.subchannel) {
            self.ring
                .write(
                    &[ENGINE
                        | (u64::from(source.subchannel) << 3)
                        | (u64::from(source.class) << 32)],
                    &[],
                )
                .map_err(|_| "preparation owner stopped")?;
        }
        self.ring
            .write(
                &[METHOD
                    | (u64::from(source.subchannel) << 3)
                    | (u64::from(source.method) << 8)
                    | (u64::from(source.argument) << 32)],
                &[],
            )
            .map_err(|_| "preparation owner stopped")?;
        self.previous = Some(source);
        Ok(())
    }
    pub fn reset(&mut self) {
        self.previous = None;
    }
}
pub fn consume<T>(
    words: &mut ordered::Words<'_, T>,
    source: &mut Source,
    mut method: impl FnMut(Source) -> Result<()>,
) -> Result<()> {
    while !words.remaining().is_empty() {
        let batch = words.remaining();
        let value = batch[0];
        let mut consumed = 1;
        match value & 7 {
            ORIGIN => {
                source.entry = (value >> 32) as u32;
                source.pushbuffer = batch[1];
                consumed = 2;
            }
            LOCATION => source.offset = (value >> 32) as u32,
            ENGINE => {
                source.class = (value >> 32) as u32;
                source.subchannel = ((value >> 3) & 7) as u8;
            }
            METHOD => {
                source.method = ((value >> 8) & 0xffff) as u32;
                source.argument = (value >> 32) as u32;
                source.subchannel = ((value >> 3) & 7) as u8;
                method(*source).map_err(|e| source.error(e))?;
            }
            _ => unreachable!("admitted ordered method record"),
        }
        words.advance(consumed);
    }
    Ok(())
}
