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

use nixe_runtime::{
    EventObject, HandleObject, HandleTable, ProcessMountNamespace, RunnableProcess,
};

use crate::{
    DirectoryEntry, DirectoryEntryKind, HostDirectoryFileSystem, HostFile, IpcSession,
    ReadOnlyDirectory, ReadOnlyFile, ReadOnlyFileSystem,
};

/// Largest path accepted by the semantic filesystem boundary.
pub const MAX_IPC_PATH_BYTES: usize = 0x300;
/// Largest file payload returned by one request.
pub const MAX_IPC_READ_BYTES: usize = 1024 * 1024;
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
    OpenSdCardFileSystem,
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
    GetFileSize,
    ReadFile {
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
    Data(Vec<u8>),
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
            .insert(IpcSession::new(service))
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
            .get(target)
            .cloned()
            .ok_or(IpcResultCode::INVALID_HANDLE)?;
        Self::dispatch_object(mounts, handles, &object, request)
    }

    pub(crate) fn dispatch_session(
        mounts: &ProcessMountNamespace,
        handles: &mut HandleTable,
        session: &IpcSession,
        request: IpcRequest,
    ) -> Result<IpcResponse, IpcResultCode> {
        dispatch_session(mounts, handles, session, request)
    }

    pub(crate) fn dispatch_object(
        mounts: &ProcessMountNamespace,
        handles: &mut HandleTable,
        object: &HandleObject,
        request: IpcRequest,
    ) -> Result<IpcResponse, IpcResultCode> {
        if let Some(session) = object.downcast_ref::<IpcSession>() {
            dispatch_session(mounts, handles, session, request)
        } else if let Some(filesystem) = object.downcast_ref::<ReadOnlyFileSystem>().cloned() {
            dispatch_filesystem(mounts, handles, &filesystem, request)
        } else if let Some(filesystem) = object.downcast_ref::<HostDirectoryFileSystem>().cloned() {
            dispatch_host_filesystem(mounts, handles, &filesystem, request)
        } else if let Some(file) = object.downcast_ref::<ReadOnlyFile>().cloned() {
            dispatch_file(mounts, &file, request)
        } else if let Some(file) = object.downcast_ref::<HostFile>() {
            dispatch_host_file(mounts, file, request)
        } else if let Some(directory) = object.downcast_ref::<ReadOnlyDirectory>().cloned() {
            dispatch_directory(mounts, &directory, request)
        } else {
            Err(IpcResultCode::INVALID_COMMAND)
        }
    }
}

fn dispatch_session(
    mounts: &ProcessMountNamespace,
    handles: &mut HandleTable,
    session: &IpcSession,
    request: IpcRequest,
) -> Result<IpcResponse, IpcResultCode> {
    if !mounts.allows_service(session.service().name()) {
        return Err(IpcResultCode::ACCESS_DENIED);
    }
    match (session.service(), request) {
        (IpcService::FileSystem, IpcRequest::SetCurrentProcess) => Ok(IpcResponse::None),
        (IpcService::FileSystem, IpcRequest::OpenPrimaryFileSystem) => {
            require_content_data_read(mounts)?;
            let mount = mounts
                .primary()
                .cloned()
                .ok_or(IpcResultCode::PATH_NOT_FOUND)?;
            insert_handle(handles, ReadOnlyFileSystem::new(mount))
        }
        (IpcService::FileSystem, IpcRequest::OpenSdCardFileSystem) => {
            require_sd_card_access(mounts)?;
            let root = mounts
                .sd_card_root()
                .map(ToOwned::to_owned)
                .ok_or(IpcResultCode::PATH_NOT_FOUND)?;
            insert_handle(handles, HostDirectoryFileSystem::new(root))
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
            require_content_data_read(mounts)?;
            let add_on = mounts
                .add_on(title_id)
                .ok_or(IpcResultCode::PATH_NOT_FOUND)?;
            let mount = add_on
                .mounts()
                .get(mount_index)
                .cloned()
                .ok_or(IpcResultCode::OUT_OF_RANGE)?;
            insert_handle(handles, ReadOnlyFileSystem::new(mount))
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
    mounts: &ProcessMountNamespace,
    handles: &mut HandleTable,
    filesystem: &ReadOnlyFileSystem,
    request: IpcRequest,
) -> Result<IpcResponse, IpcResultCode> {
    require_content_data_read(mounts)?;
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
                ReadOnlyFile::new(Arc::from(path), file.size(), storage),
            )
        }
        IpcRequest::OpenDirectory { path, .. } => {
            let path = normalize_path(&path)?;
            let entries = directory_entries(filesystem, &path)?;
            insert_handle(
                handles,
                ReadOnlyDirectory::new(Arc::from(path), entries.into()),
            )
        }
        _ => Err(IpcResultCode::INVALID_COMMAND),
    }
}

fn dispatch_file(
    mounts: &ProcessMountNamespace,
    file: &ReadOnlyFile,
    request: IpcRequest,
) -> Result<IpcResponse, IpcResultCode> {
    require_content_data_read(mounts)?;
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

fn dispatch_host_filesystem(
    mounts: &ProcessMountNamespace,
    handles: &mut HandleTable,
    filesystem: &HostDirectoryFileSystem,
    request: IpcRequest,
) -> Result<IpcResponse, IpcResultCode> {
    require_sd_card_access(mounts)?;
    match request {
        IpcRequest::CreateFile { path, size, option } => {
            if option != 0 {
                return Err(IpcResultCode::INVALID_ARGUMENT);
            }
            let path = normalize_path(&path)?;
            let host_path = filesystem.resolve_new(&path).map_err(map_host_io_error)?;
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
            Ok(IpcResponse::None)
        }
        IpcRequest::CreateDirectory { path } => {
            let path = normalize_path(&path)?;
            let host_path = filesystem.resolve_new(&path).map_err(map_host_io_error)?;
            std::fs::create_dir(host_path).map_err(map_host_io_error)?;
            Ok(IpcResponse::None)
        }
        IpcRequest::OpenFile { path, mode } => {
            if mode & (FILE_OPEN_READ | FILE_OPEN_WRITE) == 0
                || mode & !(FILE_OPEN_READ | FILE_OPEN_WRITE | FILE_OPEN_APPEND) != 0
            {
                return Err(IpcResultCode::INVALID_ARGUMENT);
            }
            let path = normalize_path(&path)?;
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
                HostFile::new(Arc::from(path), file, readable, writable, allow_append),
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
            let entries = host_directory_entries(filesystem, &path, mode)?;
            insert_handle(
                handles,
                ReadOnlyDirectory::new(Arc::from(path), entries.into()),
            )
        }
        _ => Err(IpcResultCode::INVALID_COMMAND),
    }
}

fn dispatch_host_file(
    mounts: &ProcessMountNamespace,
    file: &HostFile,
    request: IpcRequest,
) -> Result<IpcResponse, IpcResultCode> {
    require_sd_card_access(mounts)?;
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
            let mut file = file
                .file()
                .lock()
                .map_err(|_| IpcResultCode::INTERNAL_STATE)?;
            let end = offset
                .checked_add(u64::try_from(data.len()).map_err(|_| IpcResultCode::OUT_OF_RANGE)?)
                .ok_or(IpcResultCode::OUT_OF_RANGE)?;
            if end > file.metadata().map_err(map_host_io_error)?.len() && !allow_append {
                return Err(IpcResultCode::OUT_OF_RANGE);
            }
            file.seek(SeekFrom::Start(offset))
                .map_err(map_host_io_error)?;
            file.write_all(&data).map_err(map_host_io_error)?;
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
            file.file()
                .lock()
                .map_err(|_| IpcResultCode::INTERNAL_STATE)?
                .set_len(size)
                .map_err(map_host_io_error)?;
            Ok(IpcResponse::None)
        }
        _ => Err(IpcResultCode::INVALID_COMMAND),
    }
}

fn dispatch_directory(
    mounts: &ProcessMountNamespace,
    directory: &ReadOnlyDirectory,
    request: IpcRequest,
) -> Result<IpcResponse, IpcResultCode> {
    require_content_data_read(mounts)?;
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

fn require_content_data_read(mounts: &ProcessMountNamespace) -> Result<(), IpcResultCode> {
    if mounts.allows_content_data_read() {
        Ok(())
    } else {
        Err(IpcResultCode::ACCESS_DENIED)
    }
}

fn require_sd_card_access(mounts: &ProcessMountNamespace) -> Result<(), IpcResultCode> {
    if mounts.allows_sd_card_access() {
        Ok(())
    } else {
        Err(IpcResultCode::ACCESS_DENIED)
    }
}

fn insert_handle<T>(handles: &mut HandleTable, object: T) -> Result<IpcResponse, IpcResultCode>
where
    T: nixe_runtime::HandleValue,
{
    handles
        .insert(object)
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
    filesystem: &HostDirectoryFileSystem,
    path: &str,
    mode: u32,
) -> Result<Vec<DirectoryEntry>, IpcResultCode> {
    let host_path = filesystem
        .resolve_existing(path)
        .map_err(map_host_io_error)?;
    let metadata = std::fs::metadata(&host_path).map_err(map_host_io_error)?;
    if !metadata.is_dir() {
        return Err(IpcResultCode::NOT_A_DIRECTORY);
    }
    let mut entries = BTreeMap::new();
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
