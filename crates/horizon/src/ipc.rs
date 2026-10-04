//! Typed semantic IPC dispatcher for the first read-only Horizon services.
//!
//! This module deliberately sits below the HIPC/CMIF wire codec. Its
//! requests and responses have bounded, validated semantics that can be called
//! directly from tests and from Horizon SVC dispatch without depending on
//! guest message layouts.

use std::collections::BTreeMap;
use std::fmt::{Display, Formatter};
use std::fs::OpenOptions;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::sync::Arc;

use nixe_loader_title::TitleId;

use nixe_runtime::{EventObject, HandleTable, ProcessMountNamespace, RunnableProcess};

use crate::{
    DirectoryEntry, DirectoryEntryKind, FileSystemAccessLogMode, HorizonIpcObject,
    HostDirectoryFileSystem, HostFile, IpcSession, ReadOnlyDirectory, ReadOnlyFile,
    ReadOnlyFileSystem, ReadOnlyStorage, SemanticIpcObject,
};

/// Largest path accepted by the semantic filesystem boundary.
pub const MAX_IPC_PATH_BYTES: usize = 0x300;
/// Largest file payload returned by one request.
pub const MAX_IPC_READ_BYTES: usize = 1024 * 1024;
/// Largest raw storage payload returned by one request.
pub const MAX_IPC_STORAGE_READ_BYTES: usize = 256 * 1024 * 1024;
/// Largest number of directory or add-on entries returned by one request.
pub const MAX_IPC_LIST_ENTRIES: usize = 1024;
// Guest-visible file and directory mode bits follow libnx's pinned FsOpenMode
// and FsDirOpenMode definitions:
// https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/services/fs.h#L156-L173
const FILE_OPEN_READ: u32 = 1;
const FILE_OPEN_WRITE: u32 = 2;
const FILE_OPEN_APPEND: u32 = 4;
const DIRECTORY_OPEN_DIRECTORIES: u32 = 1;
const DIRECTORY_OPEN_FILES: u32 = 2;
const DIRECTORY_OPEN_NO_FILE_SIZE: u32 = 1 << 31;

/// Stable service identity used by the Horizon service registry.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum IpcService {
    FileSystem,
    AddOnContent,
}

impl IpcService {
    #[must_use]
    pub const fn name(self) -> &'static [u8] {
        match self {
            Self::FileSystem => b"fsp-srv",
            Self::AddOnContent => b"aoc:u",
        }
    }

    #[must_use]
    pub(crate) fn from_name(name: &[u8]) -> Option<Self> {
        [Self::FileSystem, Self::AddOnContent]
            .into_iter()
            .find(|service| service.name() == name)
    }
}

/// Stable semantic result code, deliberately distinct from guest-visible
/// Horizon values. [`crate::HorizonIpcResult::from_semantic`] performs the
/// contextual conversion at the wire boundary.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct IpcResultCode(u32);

impl IpcResultCode {
    pub const SUCCESS: Self = Self(0);
    pub const INVALID_HANDLE: Self = Self(1);
    pub const ACCESS_DENIED: Self = Self(2);
    pub const INVALID_COMMAND: Self = Self(3);
    pub const INVALID_ARGUMENT: Self = Self(4);
    pub const PATH_NOT_FOUND: Self = Self(5);
    pub const NOT_A_FILE: Self = Self(6);
    pub const NOT_A_DIRECTORY: Self = Self(7);
    pub const OUT_OF_RANGE: Self = Self(8);
    pub const RESOURCE_LIMIT: Self = Self(9);
    pub const STORAGE_FAILURE: Self = Self(10);
    pub const INTERNAL_STATE: Self = Self(11);
    pub const NO_SPACE: Self = Self(12);
    pub const WRITE_FILE_NOT_CLOSED: Self = Self(13);

    pub(crate) const fn semantic_id(self) -> u32 {
        self.0
    }
}

impl Display for IpcResultCode {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "IPC result {:#x}", self.0)
    }
}

/// One authorized add-on reported to the guest.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct AddOnContentEntry {
    pub title_id: TitleId,
    pub version: u32,
    pub horizon_index: Option<u32>,
    pub mount_count: u32,
}

/// Bounded semantic requests accepted by [`IpcDispatcher`].
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum IpcRequest {
    SetCurrentProcess,
    OpenPrimaryFileSystem,
    OpenPrimaryStorage,
    OpenSdCardFileSystem,
    GetGlobalAccessLogMode,
    GetProgramIndexForAccessLog,
    CreateFile {
        path: String,
        size: u64,
        option: u32,
    },
    CreateDirectory {
        path: String,
    },
    OpenFile {
        path: String,
        mode: u32,
    },
    OpenDirectory {
        path: String,
        mode: u32,
    },
    GetEntryType {
        path: String,
    },
    CommitFileSystem,
    GetFileSize,
    ReadFile {
        offset: u64,
        size: usize,
    },
    GetStorageSize,
    ReadStorage {
        offset: u64,
        size: usize,
    },
    WriteFile {
        offset: u64,
        data: Vec<u8>,
        flush: bool,
    },
    FlushFile,
    SetFileSize {
        size: u64,
    },
    GetDirectoryEntryCount,
    ReadDirectory {
        max_entries: usize,
    },
    GetAddOnContentCount,
    GetIndexedAddOnContentCount,
    ListAddOnContent {
        offset: usize,
        max_entries: usize,
    },
    ListIndexedAddOnContent {
        offset: usize,
        max_entries: usize,
    },
    PrepareAddOnContent {
        horizon_index: u32,
    },
    GetAddOnContentListChangedEvent,
    CheckAddOnContentMountStatus,
    OpenAddOnContent {
        title_id: TitleId,
        mount_index: usize,
    },
}

/// Typed response returned on a successful semantic dispatch.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum IpcResponse {
    None,
    Handle(u32),
    Event(u32),
    Size(u64),
    EntryType(DirectoryEntryKind),
    FileSystemAccessLogMode(FileSystemAccessLogMode),
    AccessLogProgramIndex { version: u32, program_index: u32 },
    Data(Vec<u8>),
    StorageRead { offset: u64, size: usize },
    DirectoryEntries(Vec<DirectoryEntry>),
    AddOnContentEntries(Vec<AddOnContentEntry>),
}

/// Stateless service dispatcher. All guest-owned state remains in handles.
#[derive(Clone, Copy, Debug, Default)]
pub struct IpcDispatcher;

impl IpcDispatcher {
    /// Connects to one built-in service after applying the effective NPDM SAC.
    pub fn connect(
        mounts: &ProcessMountNamespace,
        handles: &mut HandleTable,
        service: IpcService,
    ) -> Result<u32, IpcResultCode> {
        if !mounts.allows_service(service.name()) {
            return Err(IpcResultCode::ACCESS_DENIED);
        }
        handles
            .insert(HorizonIpcObject::SemanticService(IpcSession::new(service)))
            .map_err(|_| IpcResultCode::RESOURCE_LIMIT)
    }

    /// Dispatches a validated request against the type of its target handle.
    pub fn dispatch(
        mounts: &ProcessMountNamespace,
        handles: &mut HandleTable,
        target: u32,
        request: IpcRequest,
    ) -> Result<IpcResponse, IpcResultCode> {
        let object = handles
            .get_as::<HorizonIpcObject>(target)
            .cloned()
            .ok_or(IpcResultCode::INVALID_HANDLE)?;
        match object {
            HorizonIpcObject::SemanticService(session) => dispatch_session(
                mounts,
                handles,
                &session,
                request,
                FileSystemAccessLogMode::None,
            ),
            HorizonIpcObject::SemanticObject(object) => {
                dispatch_semantic_object(mounts, handles, &object, request)
            }
            _ => Err(IpcResultCode::INVALID_COMMAND),
        }
    }

    pub(crate) fn dispatch_session(
        mounts: &ProcessMountNamespace,
        handles: &mut HandleTable,
        session: &IpcSession,
        request: IpcRequest,
        file_system_access_log_mode: FileSystemAccessLogMode,
    ) -> Result<IpcResponse, IpcResultCode> {
        dispatch_session(
            mounts,
            handles,
            session,
            request,
            file_system_access_log_mode,
        )
    }

    pub(crate) fn dispatch_semantic_object(
        mounts: &ProcessMountNamespace,
        handles: &mut HandleTable,
        object: &SemanticIpcObject,
        request: IpcRequest,
    ) -> Result<IpcResponse, IpcResultCode> {
        dispatch_semantic_object(mounts, handles, object, request)
    }
}

fn dispatch_semantic_object(
    mounts: &ProcessMountNamespace,
    handles: &mut HandleTable,
    object: &SemanticIpcObject,
    request: IpcRequest,
) -> Result<IpcResponse, IpcResultCode> {
    match object {
        SemanticIpcObject::ReadOnlyFileSystem(filesystem) => {
            dispatch_filesystem(handles, filesystem, request)
        }
        SemanticIpcObject::ReadOnlyStorage(storage) => dispatch_storage(storage, request),
        SemanticIpcObject::HostDirectoryFileSystem(filesystem) => {
            dispatch_host_filesystem(mounts, handles, filesystem, request)
        }
        SemanticIpcObject::ReadOnlyFile(file) => dispatch_file(file, request),
        SemanticIpcObject::HostFile(file) => dispatch_host_file(mounts, file, request),
        SemanticIpcObject::ReadOnlyDirectory(directory) => dispatch_directory(directory, request),
    }
}

fn dispatch_session(
    mounts: &ProcessMountNamespace,
    handles: &mut HandleTable,
    session: &IpcSession,
    request: IpcRequest,
    file_system_access_log_mode: FileSystemAccessLogMode,
) -> Result<IpcResponse, IpcResultCode> {
    if !mounts.allows_service(session.service().name()) {
        return Err(IpcResultCode::ACCESS_DENIED);
    }
    match (session.service(), request) {
        (IpcService::FileSystem, IpcRequest::SetCurrentProcess) => Ok(IpcResponse::None),
        (IpcService::FileSystem, IpcRequest::GetGlobalAccessLogMode) => Ok(
            IpcResponse::FileSystemAccessLogMode(file_system_access_log_mode),
        ),
        (IpcService::FileSystem, IpcRequest::GetProgramIndexForAccessLog) => {
            // The current launcher requires exactly one Program content and
            // verifies NPDM's program ID against the base application ID.
            // Multi-program launches and RegisterProgramIndexMapInfo are not
            // supported; the registered application is therefore program zero.
            // The wire ABI is two u32 values: access-log version, program index.
            // https://switchbrew.org/wiki/Filesystem_services#GetProgramIndexForAccessLog
            // https://github.com/eden-emulator/mirror/blob/master/src/core/hle/service/filesystem/fsp/fsp_srv.h
            Ok(IpcResponse::AccessLogProgramIndex {
                version: 2,
                program_index: 0,
            })
        }
        // The `ByCurrentProcess` commands resolve content from the process
        // registration established by SetCurrentProcess. `CanMountContentData`
        // applies to the generic content mount commands, not to these commands:
        // https://switchbrew.org/w/index.php?title=Filesystem_services&oldid=14757#Permissions
        (IpcService::FileSystem, IpcRequest::OpenPrimaryFileSystem) => {
            let mount = mounts
                .primary()
                .cloned()
                .ok_or(IpcResultCode::PATH_NOT_FOUND)?;
            insert_handle(
                handles,
                SemanticIpcObject::ReadOnlyFileSystem(ReadOnlyFileSystem::new(mount)),
            )
        }
        (IpcService::FileSystem, IpcRequest::OpenPrimaryStorage) => {
            let storage = mounts
                .primary()
                .map(|mount| mount.romfs().storage())
                .ok_or(IpcResultCode::PATH_NOT_FOUND)?;
            insert_handle(
                handles,
                SemanticIpcObject::ReadOnlyStorage(ReadOnlyStorage::new(storage)),
            )
        }
        (IpcService::FileSystem, IpcRequest::OpenSdCardFileSystem) => {
            require_sd_card_access(mounts)?;
            let root = mounts.sd_card_root().map(ToOwned::to_owned);
            if root.is_none() && mounts.homebrew_executable().is_none() {
                return Err(IpcResultCode::PATH_NOT_FOUND);
            }
            insert_handle(
                handles,
                SemanticIpcObject::HostDirectoryFileSystem(HostDirectoryFileSystem::new(root)),
            )
        }
        (IpcService::AddOnContent, IpcRequest::GetAddOnContentCount) => Ok(IpcResponse::Size(
            u64::try_from(mounts.add_ons().len()).map_err(|_| IpcResultCode::OUT_OF_RANGE)?,
        )),
        (IpcService::AddOnContent, IpcRequest::GetIndexedAddOnContentCount) => {
            let count = mounts
                .add_ons()
                .iter()
                .filter(|add_on| add_on.horizon_index().is_some())
                .count();
            Ok(IpcResponse::Size(
                u64::try_from(count).map_err(|_| IpcResultCode::OUT_OF_RANGE)?,
            ))
        }
        (
            IpcService::AddOnContent,
            IpcRequest::ListAddOnContent {
                offset,
                max_entries,
            },
        ) => {
            validate_list_limit(max_entries)?;
            let entries = mounts
                .add_ons()
                .iter()
                .skip(offset)
                .take(max_entries)
                .map(|add_on| {
                    Ok(AddOnContentEntry {
                        title_id: add_on.title_id(),
                        version: add_on.version().raw(),
                        horizon_index: add_on.horizon_index(),
                        mount_count: u32::try_from(add_on.mounts().len())
                            .map_err(|_| IpcResultCode::OUT_OF_RANGE)?,
                    })
                })
                .collect::<Result<Vec<_>, IpcResultCode>>()?;
            Ok(IpcResponse::AddOnContentEntries(entries))
        }
        (
            IpcService::AddOnContent,
            IpcRequest::ListIndexedAddOnContent {
                offset,
                max_entries,
            },
        ) => {
            validate_list_limit(max_entries)?;
            let entries = mounts
                .add_ons()
                .iter()
                .filter(|add_on| add_on.horizon_index().is_some())
                .skip(offset)
                .take(max_entries)
                .map(add_on_entry)
                .collect::<Result<Vec<_>, IpcResultCode>>()?;
            Ok(IpcResponse::AddOnContentEntries(entries))
        }
        (IpcService::AddOnContent, IpcRequest::PrepareAddOnContent { horizon_index }) => {
            if mounts
                .add_ons()
                .iter()
                .any(|add_on| add_on.horizon_index() == Some(horizon_index))
            {
                Ok(IpcResponse::None)
            } else {
                Err(IpcResultCode::PATH_NOT_FOUND)
            }
        }
        (IpcService::AddOnContent, IpcRequest::CheckAddOnContentMountStatus) => {
            // This checks for loss of content mounted by the caller, rather
            // than requiring any DLC to be installed. The process namespace
            // owns immutable, resolved backing views for its entire lifetime;
            // it cannot lose a mount. Mount/unmount notifications and live
            // removal remain separate, unsupported commands.
            // https://github.com/alula/Ryujinx/blob/master/src/Ryujinx.HLE/HOS/Services/Ns/Aoc/IAddOnContentManager.cs
            Ok(IpcResponse::None)
        }
        (IpcService::AddOnContent, IpcRequest::GetAddOnContentListChangedEvent) => {
            let (_writable, readable) = EventObject::create_pair();
            handles
                .insert(readable)
                .map(IpcResponse::Event)
                .map_err(|_| IpcResultCode::RESOURCE_LIMIT)
        }
        (
            IpcService::AddOnContent,
            IpcRequest::OpenAddOnContent {
                title_id,
                mount_index,
            },
        ) => {
            let add_on = mounts
                .add_on(title_id)
                .ok_or(IpcResultCode::PATH_NOT_FOUND)?;
            let mount = add_on
                .mounts()
                .get(mount_index)
                .cloned()
                .ok_or(IpcResultCode::OUT_OF_RANGE)?;
            insert_handle(
                handles,
                SemanticIpcObject::ReadOnlyFileSystem(ReadOnlyFileSystem::new(mount)),
            )
        }
        _ => Err(IpcResultCode::INVALID_COMMAND),
    }
}

fn add_on_entry(add_on: &nixe_runtime::AddOnContent) -> Result<AddOnContentEntry, IpcResultCode> {
    Ok(AddOnContentEntry {
        title_id: add_on.title_id(),
        version: add_on.version().raw(),
        horizon_index: add_on.horizon_index(),
        mount_count: u32::try_from(add_on.mounts().len())
            .map_err(|_| IpcResultCode::OUT_OF_RANGE)?,
    })
}

fn dispatch_filesystem(
    handles: &mut HandleTable,
    filesystem: &ReadOnlyFileSystem,
    request: IpcRequest,
) -> Result<IpcResponse, IpcResultCode> {
    match request {
        IpcRequest::OpenFile { path, mode } => {
            if mode != FILE_OPEN_READ {
                return Err(IpcResultCode::ACCESS_DENIED);
            }
            let path = normalize_path(&path)?;
            let file = filesystem
                .mount()
                .romfs()
                .file(&path)
                .ok_or(IpcResultCode::PATH_NOT_FOUND)?;
            let storage = filesystem
                .mount()
                .romfs()
                .open_file(file)
                .map_err(|_| IpcResultCode::STORAGE_FAILURE)?;
            insert_handle(
                handles,
                SemanticIpcObject::ReadOnlyFile(ReadOnlyFile::new(
                    Arc::from(path),
                    file.size(),
                    storage,
                )),
            )
        }
        IpcRequest::OpenDirectory { path, .. } => {
            let path = normalize_path(&path)?;
            let entries = directory_entries(filesystem, &path)?;
            insert_handle(
                handles,
                SemanticIpcObject::ReadOnlyDirectory(ReadOnlyDirectory::new(
                    Arc::from(path),
                    entries.into(),
                )),
            )
        }
        _ => Err(IpcResultCode::INVALID_COMMAND),
    }
}

fn dispatch_file(file: &ReadOnlyFile, request: IpcRequest) -> Result<IpcResponse, IpcResultCode> {
    match request {
        IpcRequest::GetFileSize => Ok(IpcResponse::Size(file.size())),
        IpcRequest::ReadFile { offset, size } => {
            if size > MAX_IPC_READ_BYTES {
                return Err(IpcResultCode::RESOURCE_LIMIT);
            }
            if offset >= file.size() {
                return Ok(IpcResponse::Data(Vec::new()));
            }
            let remaining = file.size() - offset;
            let read_size = usize::try_from(remaining.min(size as u64))
                .map_err(|_| IpcResultCode::OUT_OF_RANGE)?;
            let mut bytes = vec![0; read_size];
            file.storage()
                .read_at(offset, &mut bytes)
                .map_err(|_| IpcResultCode::STORAGE_FAILURE)?;
            Ok(IpcResponse::Data(bytes))
        }
        _ => Err(IpcResultCode::INVALID_COMMAND),
    }
}

fn dispatch_storage(
    storage: &ReadOnlyStorage,
    request: IpcRequest,
) -> Result<IpcResponse, IpcResultCode> {
    match request {
        IpcRequest::GetStorageSize => {
            let size = storage
                .storage()
                .len()
                .map_err(|_| IpcResultCode::STORAGE_FAILURE)?;
            if size > i64::MAX as u64 {
                return Err(IpcResultCode::OUT_OF_RANGE);
            }
            Ok(IpcResponse::Size(size))
        }
        IpcRequest::ReadStorage { offset, size } => {
            if size > MAX_IPC_STORAGE_READ_BYTES {
                return Err(IpcResultCode::RESOURCE_LIMIT);
            }
            if offset > i64::MAX as u64 {
                return Err(IpcResultCode::OUT_OF_RANGE);
            }
            let size_u64 = u64::try_from(size).map_err(|_| IpcResultCode::OUT_OF_RANGE)?;
            let end = offset
                .checked_add(size_u64)
                .ok_or(IpcResultCode::OUT_OF_RANGE)?;
            let storage_size = storage
                .storage()
                .len()
                .map_err(|_| IpcResultCode::STORAGE_FAILURE)?;
            if end > storage_size {
                return Err(IpcResultCode::OUT_OF_RANGE);
            }
            Ok(IpcResponse::StorageRead { offset, size })
        }
        _ => Err(IpcResultCode::INVALID_COMMAND),
    }
}

fn dispatch_host_filesystem(
    mounts: &ProcessMountNamespace,
    handles: &mut HandleTable,
    filesystem: &HostDirectoryFileSystem,
    request: IpcRequest,
) -> Result<IpcResponse, IpcResultCode> {
    let mut save = filesystem
        .save_volume()
        .map(|volume| volume.lock().map_err(|_| IpcResultCode::INTERNAL_STATE))
        .transpose()?;
    if save.is_none() {
        require_sd_card_access(mounts)?;
    }
    match request {
        IpcRequest::GetEntryType { path } => {
            let path = normalize_path(&path)?;
            if save.is_none()
                && let Some(identity) = mounts.homebrew_executable()
            {
                if path == identity.guest_path() {
                    return Ok(IpcResponse::EntryType(DirectoryEntryKind::File));
                }
                if is_homebrew_virtual_directory(identity.guest_path(), &path) {
                    return Ok(IpcResponse::EntryType(DirectoryEntryKind::Directory));
                }
            }
            let metadata = std::fs::metadata(
                filesystem
                    .resolve_existing(&path)
                    .map_err(map_host_io_error)?,
            )
            .map_err(map_host_io_error)?;
            let kind = if metadata.is_dir() {
                DirectoryEntryKind::Directory
            } else if metadata.is_file() {
                DirectoryEntryKind::File
            } else {
                return Err(IpcResultCode::PATH_NOT_FOUND);
            };
            Ok(IpcResponse::EntryType(kind))
        }
        IpcRequest::CreateFile { path, size, option } => {
            if option != 0 {
                return Err(IpcResultCode::INVALID_ARGUMENT);
            }
            let path = normalize_path(&path)?;
            if is_reserved_homebrew_path(mounts, &path) {
                return Err(IpcResultCode::ACCESS_DENIED);
            }
            let host_path = filesystem.resolve_new(&path).map_err(map_host_io_error)?;
            if save
                .as_ref()
                .is_some_and(|volume| !volume.can_resize(0, size))
            {
                return Err(IpcResultCode::NO_SPACE);
            }
            let file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&host_path)
                .map_err(map_host_io_error)?;
            if let Err(error) = file.set_len(size) {
                drop(file);
                let _ = std::fs::remove_file(host_path);
                return Err(map_host_io_error(error));
            }
            if let Some(volume) = save.as_mut() {
                volume.record_resize(&path, 0, size);
            }
            Ok(IpcResponse::None)
        }
        IpcRequest::CreateDirectory { path } => {
            let path = normalize_path(&path)?;
            if is_reserved_homebrew_path(mounts, &path) {
                return Err(IpcResultCode::ACCESS_DENIED);
            }
            let host_path = filesystem.resolve_new(&path).map_err(map_host_io_error)?;
            std::fs::create_dir(host_path).map_err(map_host_io_error)?;
            if let Some(volume) = save.as_mut() {
                volume.mark_changed(&path);
            }
            Ok(IpcResponse::None)
        }
        IpcRequest::OpenFile { path, mode } => {
            if mode & (FILE_OPEN_READ | FILE_OPEN_WRITE) == 0
                || mode & !(FILE_OPEN_READ | FILE_OPEN_WRITE | FILE_OPEN_APPEND) != 0
            {
                return Err(IpcResultCode::INVALID_ARGUMENT);
            }
            let path = normalize_path(&path)?;
            if let Some(identity) = mounts.homebrew_executable() {
                if path == identity.guest_path() {
                    if mode & (FILE_OPEN_WRITE | FILE_OPEN_APPEND) != 0 {
                        return Err(IpcResultCode::ACCESS_DENIED);
                    }
                    return insert_handle(
                        handles,
                        SemanticIpcObject::ReadOnlyFile(ReadOnlyFile::new(
                            Arc::from(path),
                            identity.size(),
                            identity.source().clone(),
                        )),
                    );
                }
                if is_homebrew_virtual_directory(identity.guest_path(), &path) {
                    return Err(IpcResultCode::NOT_A_FILE);
                }
                if is_reserved_homebrew_path(mounts, &path) {
                    return Err(IpcResultCode::PATH_NOT_FOUND);
                }
            }
            let host_path = filesystem
                .resolve_existing(&path)
                .map_err(map_host_io_error)?;
            let metadata = std::fs::metadata(&host_path).map_err(map_host_io_error)?;
            if !metadata.is_file() {
                return Err(IpcResultCode::NOT_A_FILE);
            }
            let readable = mode & FILE_OPEN_READ != 0;
            let writable = mode & FILE_OPEN_WRITE != 0;
            let allow_append = mode & FILE_OPEN_APPEND != 0;
            let file = OpenOptions::new()
                .read(readable)
                .write(writable)
                .open(host_path)
                .map_err(map_host_io_error)?;
            insert_handle(
                handles,
                SemanticIpcObject::HostFile(
                    HostFile::new(Arc::from(path), file, readable, writable, allow_append)
                        .with_save(
                            filesystem.save_volume().cloned(),
                            if writable {
                                save.as_ref().map(|volume| volume.open_writer())
                            } else {
                                None
                            },
                        ),
                ),
            )
        }
        IpcRequest::OpenDirectory { path, mode } => {
            if mode & (DIRECTORY_OPEN_DIRECTORIES | DIRECTORY_OPEN_FILES) == 0
                || mode
                    & !(DIRECTORY_OPEN_DIRECTORIES
                        | DIRECTORY_OPEN_FILES
                        | DIRECTORY_OPEN_NO_FILE_SIZE)
                    != 0
            {
                return Err(IpcResultCode::INVALID_ARGUMENT);
            }
            let path = normalize_path(&path)?;
            let entries = host_directory_entries(mounts, filesystem, &path, mode)?;
            insert_handle(
                handles,
                SemanticIpcObject::ReadOnlyDirectory(ReadOnlyDirectory::new(
                    Arc::from(path),
                    entries.into(),
                )),
            )
        }
        IpcRequest::CommitFileSystem => {
            if let Some(volume) = save.as_mut() {
                if volume.has_open_writers() {
                    return Err(IpcResultCode::WRITE_FILE_NOT_CLOSED);
                }
                volume.commit().map_err(map_host_io_error)?;
            }
            Ok(IpcResponse::None)
        }
        _ => Err(IpcResultCode::INVALID_COMMAND),
    }
}

fn dispatch_host_file(
    mounts: &ProcessMountNamespace,
    file: &HostFile,
    request: IpcRequest,
) -> Result<IpcResponse, IpcResultCode> {
    let mut save = file
        .save_volume()
        .map(|volume| volume.lock().map_err(|_| IpcResultCode::INTERNAL_STATE))
        .transpose()?;
    if save.is_none() {
        require_sd_card_access(mounts)?;
    }
    match request {
        IpcRequest::GetFileSize => {
            let file = file
                .file()
                .lock()
                .map_err(|_| IpcResultCode::INTERNAL_STATE)?;
            Ok(IpcResponse::Size(
                file.metadata().map_err(map_host_io_error)?.len(),
            ))
        }
        IpcRequest::ReadFile { offset, size } => {
            if !file.readable() {
                return Err(IpcResultCode::ACCESS_DENIED);
            }
            if size > MAX_IPC_READ_BYTES {
                return Err(IpcResultCode::RESOURCE_LIMIT);
            }
            let mut file = file
                .file()
                .lock()
                .map_err(|_| IpcResultCode::INTERNAL_STATE)?;
            let file_size = file.metadata().map_err(map_host_io_error)?.len();
            if offset >= file_size {
                return Ok(IpcResponse::Data(Vec::new()));
            }
            let read_size = usize::try_from((file_size - offset).min(size as u64))
                .map_err(|_| IpcResultCode::OUT_OF_RANGE)?;
            let mut data = vec![0; read_size];
            file.seek(SeekFrom::Start(offset))
                .map_err(map_host_io_error)?;
            file.read_exact(&mut data).map_err(map_host_io_error)?;
            Ok(IpcResponse::Data(data))
        }
        IpcRequest::WriteFile {
            offset,
            data,
            flush,
        } => {
            if !file.writable() {
                return Err(IpcResultCode::ACCESS_DENIED);
            }
            if data.len() > MAX_IPC_READ_BYTES {
                return Err(IpcResultCode::RESOURCE_LIMIT);
            }
            let allow_append = file.allows_append();
            let path = file.path();
            let mut file = file
                .file()
                .lock()
                .map_err(|_| IpcResultCode::INTERNAL_STATE)?;
            let end = offset
                .checked_add(u64::try_from(data.len()).map_err(|_| IpcResultCode::OUT_OF_RANGE)?)
                .ok_or(IpcResultCode::OUT_OF_RANGE)?;
            let old_size = file.metadata().map_err(map_host_io_error)?.len();
            if end > old_size && !allow_append {
                return Err(IpcResultCode::OUT_OF_RANGE);
            }
            file.seek(SeekFrom::Start(offset))
                .map_err(map_host_io_error)?;
            if save
                .as_ref()
                .is_some_and(|volume| !volume.can_resize(old_size, old_size.max(end)))
            {
                return Err(IpcResultCode::NO_SPACE);
            }
            let result = file.write_all(&data);
            if let Some(volume) = save.as_mut() {
                volume.record_resize(
                    path,
                    old_size,
                    file.metadata().map_err(map_host_io_error)?.len(),
                );
            }
            result.map_err(map_host_io_error)?;
            if flush {
                file.sync_data().map_err(map_host_io_error)?;
            }
            Ok(IpcResponse::None)
        }
        IpcRequest::FlushFile => {
            if !file.writable() {
                return Err(IpcResultCode::ACCESS_DENIED);
            }
            file.file()
                .lock()
                .map_err(|_| IpcResultCode::INTERNAL_STATE)?
                .sync_data()
                .map_err(map_host_io_error)?;
            Ok(IpcResponse::None)
        }
        IpcRequest::SetFileSize { size } => {
            if !file.writable() {
                return Err(IpcResultCode::ACCESS_DENIED);
            }
            let handle = file
                .file()
                .lock()
                .map_err(|_| IpcResultCode::INTERNAL_STATE)?;
            let old_size = handle.metadata().map_err(map_host_io_error)?.len();
            if save
                .as_ref()
                .is_some_and(|volume| !volume.can_resize(old_size, size))
            {
                return Err(IpcResultCode::NO_SPACE);
            }
            let result = handle.set_len(size);
            if let Some(volume) = save.as_mut() {
                volume.record_resize(
                    file.path(),
                    old_size,
                    handle.metadata().map_err(map_host_io_error)?.len(),
                );
            }
            result.map_err(map_host_io_error)?;
            Ok(IpcResponse::None)
        }
        _ => Err(IpcResultCode::INVALID_COMMAND),
    }
}

fn dispatch_directory(
    directory: &ReadOnlyDirectory,
    request: IpcRequest,
) -> Result<IpcResponse, IpcResultCode> {
    match request {
        IpcRequest::GetDirectoryEntryCount => Ok(IpcResponse::Size(
            u64::try_from(directory.entries().len()).map_err(|_| IpcResultCode::OUT_OF_RANGE)?,
        )),
        IpcRequest::ReadDirectory { max_entries } => {
            validate_list_limit(max_entries)?;
            let mut cursor = directory
                .cursor()
                .lock()
                .map_err(|_| IpcResultCode::INTERNAL_STATE)?;
            let end = cursor
                .saturating_add(max_entries)
                .min(directory.entries().len());
            let result = directory.entries()[*cursor..end].to_vec();
            *cursor = end;
            Ok(IpcResponse::DirectoryEntries(result))
        }
        _ => Err(IpcResultCode::INVALID_COMMAND),
    }
}

fn require_sd_card_access(mounts: &ProcessMountNamespace) -> Result<(), IpcResultCode> {
    if mounts.allows_sd_card_access() {
        Ok(())
    } else {
        Err(IpcResultCode::ACCESS_DENIED)
    }
}

fn insert_handle(
    handles: &mut HandleTable,
    object: SemanticIpcObject,
) -> Result<IpcResponse, IpcResultCode> {
    handles
        .insert(HorizonIpcObject::SemanticObject(object))
        .map(IpcResponse::Handle)
        .map_err(|_| IpcResultCode::RESOURCE_LIMIT)
}

/// Horizon service access implemented for a runnable process without making
/// the generic runtime crate depend on Horizon.
pub trait HorizonProcess {
    fn connect_ipc_service(&mut self, service: IpcService) -> Result<u32, IpcResultCode>;

    fn dispatch_ipc(
        &mut self,
        target: u32,
        request: IpcRequest,
    ) -> Result<IpcResponse, IpcResultCode>;
}

impl HorizonProcess for RunnableProcess {
    fn connect_ipc_service(&mut self, service: IpcService) -> Result<u32, IpcResultCode> {
        let (mounts, handles) = self.mounts_and_handles_mut();
        IpcDispatcher::connect(mounts, handles, service)
    }

    fn dispatch_ipc(
        &mut self,
        target: u32,
        request: IpcRequest,
    ) -> Result<IpcResponse, IpcResultCode> {
        let (mounts, handles) = self.mounts_and_handles_mut();
        IpcDispatcher::dispatch(mounts, handles, target, request)
    }
}

fn normalize_path(path: &str) -> Result<String, IpcResultCode> {
    if path.is_empty()
        || path.len() > MAX_IPC_PATH_BYTES
        || !path.starts_with('/')
        || path.as_bytes().contains(&0)
    {
        return Err(IpcResultCode::INVALID_ARGUMENT);
    }
    if path == "/" {
        return Ok(path.to_owned());
    }
    if path.ends_with('/')
        || path
            .split('/')
            .skip(1)
            .any(|component| component.is_empty() || component == "." || component == "..")
    {
        return Err(IpcResultCode::INVALID_ARGUMENT);
    }
    Ok(path.to_owned())
}

fn directory_entries(
    filesystem: &ReadOnlyFileSystem,
    path: &str,
) -> Result<Vec<DirectoryEntry>, IpcResultCode> {
    if filesystem.mount().romfs().file(path).is_some() {
        return Err(IpcResultCode::NOT_A_DIRECTORY);
    }
    let prefix = if path == "/" {
        "/".to_owned()
    } else {
        format!("{path}/")
    };
    let mut entries = BTreeMap::<&str, (DirectoryEntryKind, u64)>::new();
    let mut directory_exists = path == "/";
    for file in filesystem.mount().romfs().files() {
        let Some(remainder) = file.path().strip_prefix(&prefix) else {
            continue;
        };
        directory_exists = true;
        if let Some((name, _)) = remainder.split_once('/') {
            entries.insert(name, (DirectoryEntryKind::Directory, 0));
        } else {
            entries.insert(remainder, (DirectoryEntryKind::File, file.size()));
        }
    }
    if !directory_exists {
        return Err(IpcResultCode::PATH_NOT_FOUND);
    }
    Ok(entries
        .into_iter()
        .map(|(name, (kind, size))| DirectoryEntry::new(Arc::from(name), kind, size))
        .collect())
}

fn host_directory_entries(
    mounts: &ProcessMountNamespace,
    filesystem: &HostDirectoryFileSystem,
    path: &str,
    mode: u32,
) -> Result<Vec<DirectoryEntry>, IpcResultCode> {
    if let Some(identity) = mounts.homebrew_executable() {
        if path == identity.guest_path() {
            return Err(IpcResultCode::NOT_A_DIRECTORY);
        }
        if is_reserved_homebrew_path(mounts, path) {
            if !is_homebrew_virtual_directory(identity.guest_path(), path) {
                return Err(IpcResultCode::PATH_NOT_FOUND);
            }
            let mut entries = BTreeMap::new();
            insert_homebrew_directory_child(&mut entries, identity, path, mode);
            return collect_directory_entries(entries);
        }
    }
    let mut entries = BTreeMap::new();
    if filesystem.has_host_root() {
        let host_path = filesystem
            .resolve_existing(path)
            .map_err(map_host_io_error)?;
        let metadata = std::fs::metadata(&host_path).map_err(map_host_io_error)?;
        if !metadata.is_dir() {
            return Err(IpcResultCode::NOT_A_DIRECTORY);
        }
        for entry in std::fs::read_dir(host_path).map_err(map_host_io_error)? {
            let entry = entry.map_err(map_host_io_error)?;
            let file_type = entry.file_type().map_err(map_host_io_error)?;
            if file_type.is_symlink() {
                continue;
            }
            let name = entry
                .file_name()
                .into_string()
                .map_err(|_| IpcResultCode::STORAGE_FAILURE)?;
            if name.len() > MAX_IPC_PATH_BYTES {
                continue;
            }
            let (kind, size) = if file_type.is_dir() && mode & DIRECTORY_OPEN_DIRECTORIES != 0 {
                (DirectoryEntryKind::Directory, 0)
            } else if file_type.is_file() && mode & DIRECTORY_OPEN_FILES != 0 {
                let size = if mode & DIRECTORY_OPEN_NO_FILE_SIZE != 0 {
                    0
                } else {
                    entry.metadata().map_err(map_host_io_error)?.len()
                };
                (DirectoryEntryKind::File, size)
            } else {
                continue;
            };
            entries.insert(name, (kind, size));
            if entries.len() > MAX_IPC_LIST_ENTRIES {
                return Err(IpcResultCode::RESOURCE_LIMIT);
            }
        }
    }
    if let Some(identity) = mounts.homebrew_executable() {
        insert_homebrew_directory_child(&mut entries, identity, path, mode);
    }
    if !filesystem.has_host_root()
        && !mounts
            .homebrew_executable()
            .is_some_and(|identity| is_homebrew_virtual_directory(identity.guest_path(), path))
    {
        return Err(IpcResultCode::PATH_NOT_FOUND);
    }
    collect_directory_entries(entries)
}

fn is_reserved_homebrew_path(mounts: &ProcessMountNamespace, path: &str) -> bool {
    let Some(identity) = mounts.homebrew_executable() else {
        return false;
    };
    let Some(component) = identity
        .guest_path()
        .strip_prefix('/')
        .and_then(|path| path.split('/').next())
    else {
        return false;
    };
    let root = format!("/{component}");
    path == root
        || path
            .strip_prefix(&root)
            .is_some_and(|suffix| suffix.starts_with('/'))
}

fn is_homebrew_virtual_directory(file_path: &str, path: &str) -> bool {
    path == "/"
        || file_path
            .strip_prefix(path)
            .is_some_and(|suffix| suffix.starts_with('/'))
}

fn insert_homebrew_directory_child(
    entries: &mut BTreeMap<String, (DirectoryEntryKind, u64)>,
    identity: &nixe_runtime::HomebrewIdentity,
    path: &str,
    mode: u32,
) {
    let prefix = if path == "/" {
        "/".to_owned()
    } else {
        format!("{path}/")
    };
    let Some(remainder) = identity.guest_path().strip_prefix(&prefix) else {
        return;
    };
    if let Some((name, _)) = remainder.split_once('/') {
        if mode & DIRECTORY_OPEN_DIRECTORIES != 0 {
            entries.insert(name.to_owned(), (DirectoryEntryKind::Directory, 0));
        }
    } else if !remainder.is_empty() && mode & DIRECTORY_OPEN_FILES != 0 {
        let size = if mode & DIRECTORY_OPEN_NO_FILE_SIZE != 0 {
            0
        } else {
            identity.size()
        };
        entries.insert(remainder.to_owned(), (DirectoryEntryKind::File, size));
    }
}

fn collect_directory_entries(
    entries: BTreeMap<String, (DirectoryEntryKind, u64)>,
) -> Result<Vec<DirectoryEntry>, IpcResultCode> {
    if entries.len() > MAX_IPC_LIST_ENTRIES {
        return Err(IpcResultCode::RESOURCE_LIMIT);
    }
    Ok(entries
        .into_iter()
        .map(|(name, (kind, size))| DirectoryEntry::new(Arc::from(name), kind, size))
        .collect())
}

fn map_host_io_error(error: io::Error) -> IpcResultCode {
    match error.kind() {
        io::ErrorKind::NotFound => IpcResultCode::PATH_NOT_FOUND,
        io::ErrorKind::PermissionDenied => IpcResultCode::ACCESS_DENIED,
        _ => IpcResultCode::STORAGE_FAILURE,
    }
}

fn validate_list_limit(limit: usize) -> Result<(), IpcResultCode> {
    if limit > MAX_IPC_LIST_ENTRIES {
        Err(IpcResultCode::RESOURCE_LIMIT)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use nixe_loader_storage::{Storage, StorageError};

    use super::*;

    #[test]
    fn save_files_enforce_quota_and_publish_only_after_writers_close() {
        use nixe_runtime::{Launcher, LauncherInput, ProcessBuilder, TransactionalDirectory};
        let directory = tempfile::tempdir().unwrap();
        let mut nro = vec![0; 0x2800];
        nro[0x10..0x14].copy_from_slice(b"NRO0");
        for (offset, value) in [
            (0x18, 0x2800_u32),
            (0x24, 0x1000),
            (0x28, 0x1000),
            (0x2c, 0x1000),
            (0x30, 0x2000),
            (0x34, 0x800),
            (0x38, 0x800),
        ] {
            nro[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
        let path = directory.path().join("test.nro");
        std::fs::write(&path, nro).unwrap();
        let plan = Launcher::build(LauncherInput::new(path)).unwrap();
        let mut process = ProcessBuilder::new().build(&plan).unwrap();
        let base = directory.path().join("volume");
        std::fs::create_dir_all(base.join("data")).unwrap();
        let volume = Arc::new(std::sync::Mutex::new(
            TransactionalDirectory::open(base.clone(), 4).unwrap(),
        ));
        let filesystem = HostDirectoryFileSystem::from_save(volume.clone());
        let (mounts, handles) = process.mounts_and_handles_mut();
        assert_eq!(
            dispatch_host_filesystem(
                mounts,
                handles,
                &filesystem,
                IpcRequest::GetEntryType {
                    path: "/progress".into()
                }
            ),
            Err(IpcResultCode::PATH_NOT_FOUND)
        );
        dispatch_host_filesystem(
            mounts,
            handles,
            &filesystem,
            IpcRequest::CreateFile {
                path: "/progress".into(),
                size: 4,
                option: 0,
            },
        )
        .unwrap();
        assert_eq!(
            dispatch_host_filesystem(
                mounts,
                handles,
                &filesystem,
                IpcRequest::CreateFile {
                    path: "/extra".into(),
                    size: 1,
                    option: 0
                }
            ),
            Err(IpcResultCode::NO_SPACE)
        );
        let IpcResponse::Handle(handle) = dispatch_host_filesystem(
            mounts,
            handles,
            &filesystem,
            IpcRequest::OpenFile {
                path: "/progress".into(),
                mode: FILE_OPEN_WRITE | FILE_OPEN_APPEND,
            },
        )
        .unwrap() else {
            panic!()
        };
        let object = handles.get_as::<HorizonIpcObject>(handle).unwrap().clone();
        let HorizonIpcObject::SemanticObject(SemanticIpcObject::HostFile(file)) = object else {
            panic!()
        };
        assert_eq!(
            dispatch_host_file(mounts, &file, IpcRequest::SetFileSize { size: 5 }),
            Err(IpcResultCode::NO_SPACE)
        );
        dispatch_host_file(
            mounts,
            &file,
            IpcRequest::WriteFile {
                offset: 0,
                data: b"save".to_vec(),
                flush: true,
            },
        )
        .unwrap();
        assert_eq!(
            dispatch_host_filesystem(mounts, handles, &filesystem, IpcRequest::CommitFileSystem),
            Err(IpcResultCode::WRITE_FILE_NOT_CLOSED)
        );
        handles.close(handle).unwrap();
        drop(file);
        dispatch_host_filesystem(mounts, handles, &filesystem, IpcRequest::CommitFileSystem)
            .unwrap();
        drop(filesystem);
        drop(volume);
        let reopened = TransactionalDirectory::open(base, 4).unwrap();
        assert_eq!(
            std::fs::read(reopened.working_directory().join("progress")).unwrap(),
            b"save"
        );
    }

    struct SizedStorage(u64);

    impl Storage for SizedStorage {
        fn len(&self) -> Result<u64, StorageError> {
            Ok(self.0)
        }

        fn read_at(&self, _offset: u64, _buffer: &mut [u8]) -> Result<(), StorageError> {
            panic!("semantic storage dispatch must defer payload transfer to the wire boundary")
        }
    }

    #[test]
    fn storage_reads_use_the_dedicated_limit_without_materializing_the_payload() {
        let storage =
            ReadOnlyStorage::new(Arc::new(SizedStorage(MAX_IPC_STORAGE_READ_BYTES as u64)));

        assert_eq!(
            dispatch_storage(
                &storage,
                IpcRequest::ReadStorage {
                    offset: 0,
                    size: MAX_IPC_STORAGE_READ_BYTES,
                },
            ),
            Ok(IpcResponse::StorageRead {
                offset: 0,
                size: MAX_IPC_STORAGE_READ_BYTES,
            })
        );
        assert_eq!(
            dispatch_storage(
                &storage,
                IpcRequest::ReadStorage {
                    offset: 0,
                    size: MAX_IPC_STORAGE_READ_BYTES + 1,
                },
            ),
            Err(IpcResultCode::RESOURCE_LIMIT)
        );
    }
}
