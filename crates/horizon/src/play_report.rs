//! Play-report IPC session and system-wide session identifier.
use std::{
    sync::OnceLock,
    time::{SystemTime, UNIX_EPOCH},
};

#[derive(Clone, Debug)]
pub struct PlayReportSession {
    pub(crate) domain: crate::object::DomainSession<()>,
}
impl PlayReportSession {
    pub(crate) fn new() -> Self {
        Self {
            domain: crate::object::DomainSession::new(),
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
}
