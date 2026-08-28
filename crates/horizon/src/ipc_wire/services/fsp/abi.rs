//! Verified filesystem ABI sizes shared by request and response codecs.

pub(in crate::ipc_wire::services) const FS_MAX_PATH: usize = 0x301;
pub(in crate::ipc_wire::services) const FS_DIRECTORY_ENTRY_SIZE: usize = 0x310;
pub(in crate::ipc_wire::services) const FS_DIRECTORY_ENTRY_FILE: u8 = 1;
