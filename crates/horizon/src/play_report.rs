//! Local play-report storage. Reports retain the original guest payload bytes.
//! https://github.com/eden-emulator/mirror/blob/master/src/core/reporter.cpp
use std::{
    io::Write,
    path::PathBuf,
    sync::{
        Arc, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
    time::{SystemTime, UNIX_EPOCH},
};

#[derive(Clone, Debug)]
pub struct PlayReportSession {
    pub(crate) domain: crate::object::DomainSession<()>,
    directory: Arc<PathBuf>,
}
impl PlayReportSession {
    pub(crate) fn new(directory: impl Into<PathBuf>) -> Self {
        Self {
            domain: crate::object::DomainSession::new(),
            directory: Arc::new(directory.into()),
        }
    }
    pub(crate) fn system_session_id() -> std::io::Result<u64> {
        static ID: OnceLock<u64> = OnceLock::new();
        if let Some(&id) = ID.get() {
            return Ok(id);
        }
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(std::io::Error::other)?
            .as_nanos() as u64;
        Ok(*ID.get_or_init(|| timestamp))
    }
    pub(crate) fn create(
        &self,
        command: u32,
        process: u64,
        inline: &[u8],
        sizes: [u64; 2],
    ) -> std::io::Result<(std::fs::File, PathBuf)> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        std::fs::create_dir_all(self.directory.as_ref())?;
        let sequence = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = self.directory.join(format!(
            "{:016x}-{:08x}-{sequence:016x}.bin",
            Self::system_session_id()?,
            std::process::id()
        ));
        let mut file = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)?;
        file.write_all(b"NXPR0001")?;
        file.write_all(&command.to_le_bytes())?;
        file.write_all(&process.to_le_bytes())?;
        for size in [inline.len() as u64, sizes[0], sizes[1]] {
            file.write_all(&size.to_le_bytes())?;
        }
        file.write_all(inline)?;
        Ok((file, path))
    }
}
