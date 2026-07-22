//! Typed semantic IPC dispatcher for the first read-only Horizon services.
//!
//! This module deliberately sits below the future HIPC/CMIF wire decoder. Its
//! requests and responses have bounded, validated semantics that can be called
//! from tests today and from Horizon SVC dispatch later.

use std::collections::BTreeMap;
use std::fmt::{Display, Formatter};
use std::sync::Arc;

use nixe_loader_title::TitleId;

use nixe_runtime::{HandleTable, ProcessMountNamespace, RunnableProcess};

use crate::{
    DirectoryEntry, DirectoryEntryKind, IpcSession, ReadOnlyDirectory, ReadOnlyFile,
    ReadOnlyFileSystem,
};

/// Largest path accepted by the semantic filesystem boundary.
pub const MAX_IPC_PATH_BYTES: usize = 0x300;
/// Largest file payload returned by one request.
pub const MAX_IPC_READ_BYTES: usize = 1024 * 1024;
/// Largest number of directory or add-on entries returned by one request.
pub const MAX_IPC_LIST_ENTRIES: usize = 1024;

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
}

/// Stable semantic result code. A future CMIF adapter maps these to verified
/// Horizon module/description values without changing semantic service logic.
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

    #[must_use]
    pub const fn raw(self) -> u32 {
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
    OpenPrimaryFileSystem,
    OpenFile {
        path: String,
    },
    OpenDirectory {
        path: String,
    },
    GetFileSize,
    ReadFile {
        offset: u64,
        size: usize,
    },
    GetDirectoryEntryCount,
    ReadDirectory {
        max_entries: usize,
    },
    GetAddOnContentCount,
    ListAddOnContent {
        offset: usize,
        max_entries: usize,
    },
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
        let object = handles.get(target).ok_or(IpcResultCode::INVALID_HANDLE)?;
        if let Some(session) = object.downcast_ref::<IpcSession>().copied() {
            dispatch_session(mounts, handles, session, request)
        } else if let Some(filesystem) = object.downcast_ref::<ReadOnlyFileSystem>().cloned() {
            dispatch_filesystem(handles, &filesystem, request)
        } else if let Some(file) = object.downcast_ref::<ReadOnlyFile>().cloned() {
            dispatch_file(&file, request)
        } else if let Some(directory) = object.downcast_ref::<ReadOnlyDirectory>().cloned() {
            dispatch_directory(&directory, request)
        } else {
            Err(IpcResultCode::INVALID_COMMAND)
        }
    }
}

fn dispatch_session(
    mounts: &ProcessMountNamespace,
    handles: &mut HandleTable,
    session: IpcSession,
    request: IpcRequest,
) -> Result<IpcResponse, IpcResultCode> {
    match (session.service(), request) {
        (IpcService::FileSystem, IpcRequest::OpenPrimaryFileSystem) => {
            let mount = mounts
                .primary()
                .cloned()
                .ok_or(IpcResultCode::PATH_NOT_FOUND)?;
            insert_handle(handles, ReadOnlyFileSystem::new(mount))
        }
        (IpcService::AddOnContent, IpcRequest::GetAddOnContentCount) => Ok(IpcResponse::Size(
            u64::try_from(mounts.add_ons().len()).map_err(|_| IpcResultCode::OUT_OF_RANGE)?,
        )),
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
            insert_handle(handles, ReadOnlyFileSystem::new(mount))
        }
        _ => Err(IpcResultCode::INVALID_COMMAND),
    }
}

fn dispatch_filesystem(
    handles: &mut HandleTable,
    filesystem: &ReadOnlyFileSystem,
    request: IpcRequest,
) -> Result<IpcResponse, IpcResultCode> {
    match request {
        IpcRequest::OpenFile { path } => {
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
        IpcRequest::OpenDirectory { path } => {
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

fn validate_list_limit(limit: usize) -> Result<(), IpcResultCode> {
    if limit > MAX_IPC_LIST_ENTRIES {
        Err(IpcResultCode::RESOURCE_LIMIT)
    } else {
        Ok(())
    }
}
