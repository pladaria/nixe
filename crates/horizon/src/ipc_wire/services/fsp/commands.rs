//! Semantic command identities for `fsp-srv` and its child interfaces.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum FileSystemProxyCommand {
    SetCurrentProcess,
    OpenDataFileSystemByCurrentProcess,
    OpenSdCardFileSystem,
    OpenDataStorageByCurrentProcess,
    GetGlobalAccessLogMode,
    GetProgramIndexForAccessLog,
}

impl FileSystemProxyCommand {
    pub(super) const fn decode(command_id: u32) -> Option<Self> {
        match command_id {
            1 => Some(Self::SetCurrentProcess),
            2 => Some(Self::OpenDataFileSystemByCurrentProcess),
            18 => Some(Self::OpenSdCardFileSystem),
            200 => Some(Self::OpenDataStorageByCurrentProcess),
            1005 => Some(Self::GetGlobalAccessLogMode),
            1011 => Some(Self::GetProgramIndexForAccessLog),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum FileSystemCommand {
    CreateFile,
    DeleteFile,
    CreateDirectory,
    Commit,
    GetEntryType,
    OpenFile,
    OpenDirectory,
    GetFileSystemAttribute,
}

impl FileSystemCommand {
    pub(super) const fn decode(command_id: u32) -> Option<Self> {
        match command_id {
            0 => Some(Self::CreateFile),
            1 => Some(Self::DeleteFile),
            2 => Some(Self::CreateDirectory),
            7 => Some(Self::GetEntryType),
            8 => Some(Self::OpenFile),
            9 => Some(Self::OpenDirectory),
            10 => Some(Self::Commit),
            16 => Some(Self::GetFileSystemAttribute),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum FileCommand {
    Read,
    Write,
    Flush,
    SetSize,
    GetSize,
}

impl FileCommand {
    pub(super) const fn decode(command_id: u32) -> Option<Self> {
        match command_id {
            0 => Some(Self::Read),
            1 => Some(Self::Write),
            2 => Some(Self::Flush),
            3 => Some(Self::SetSize),
            4 => Some(Self::GetSize),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum StorageCommand {
    Read,
    GetSize,
}

impl StorageCommand {
    pub(super) const fn decode(command_id: u32) -> Option<Self> {
        match command_id {
            0 => Some(Self::Read),
            4 => Some(Self::GetSize),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum DirectoryCommand {
    Read,
    GetEntryCount,
}

impl DirectoryCommand {
    pub(super) const fn decode(command_id: u32) -> Option<Self> {
        match command_id {
            0 => Some(Self::Read),
            1 => Some(Self::GetEntryCount),
            _ => None,
        }
    }
}
