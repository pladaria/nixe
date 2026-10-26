//! Local play-report storage. Reports retain the original guest payload bytes.
//! https://github.com/eden-emulator/mirror/blob/master/src/core/reporter.cpp
use std::{
    io::Write,
    path::{Path, PathBuf},
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
    pub(crate) fn save(
        &self,
        command: u32,
        process: u64,
        inline: &[u8],
        parts: [&[u8]; 2],
    ) -> std::io::Result<PathBuf> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        std::fs::create_dir_all(self.directory.as_ref())?;
        let sequence = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = self.directory.join(format!(
            "{:016x}-{:08x}-{sequence:016x}.bin",
            Self::system_session_id()?,
            std::process::id()
        ));
        Self::write(&path, command, process, inline, parts)?;
        Ok(path)
    }
    fn write(
        path: &Path,
        command: u32,
        process: u64,
        inline: &[u8],
        parts: [&[u8]; 2],
    ) -> std::io::Result<()> {
        let mut file = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(path)?;
        file.write_all(b"NXPR0001")?;
        file.write_all(&command.to_le_bytes())?;
        file.write_all(&process.to_le_bytes())?;
        for bytes in [inline, parts[0], parts[1]] {
            file.write_all(&(bytes.len() as u64).to_le_bytes())?;
        }
        for bytes in [inline, parts[0], parts[1]] {
            file.write_all(bytes)?;
        }
        file.flush()
    }
}
