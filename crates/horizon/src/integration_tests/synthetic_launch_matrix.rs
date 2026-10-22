use crate::integration_test_support as support;

use std::fs;

use nixe_cpu::state::a64::A64Register;
use nixe_horizon::{DirectoryEntryKind, IpcRequest, IpcResponse, IpcResultCode, IpcService};
use nixe_runtime::{
    CpuBackendConfig, LaunchKind, Launcher, LauncherInput, ModuleRole, MountProvenance,
    ProcessBuilder, RelocationState,
};

fn reference_process_builder() -> ProcessBuilder {
    support::process_builder().with_cpu_backend(CpuBackendConfig::Jit)
}

use support::synthetic_packages::{
    APPLICATION_ID, FIRST_DLC_ID, MetaKind, PATCH_ID, Package, SECOND_DLC_ID, bktr_data_content,
    build_nsp, build_romfs, build_xci, content_id, data_content, program_content,
    program_content_with_fs_permissions, program_content_without_services, public_data_content,
};

#[test]
fn effective_npdm_service_policy_denies_unlisted_runtime_services() {
    let directory = tempfile::tempdir().unwrap();
    let package = Package {
        title_id: APPLICATION_ID,
        version: 0,
        kind: MetaKind::Application,
        contents: vec![
            program_content_without_services(content_id(1), &[("main", 1)]),
            data_content(
                content_id(2),
                APPLICATION_ID,
                1,
                build_romfs(&[("file", b"bytes")]),
            ),
        ],
    };
    fs::write(directory.path().join("restricted.nsp"), build_nsp(&package)).unwrap();
    let plan = Launcher::build(LauncherInput::new(directory.path())).unwrap();
    let process = reference_process_builder().build(&plan).unwrap();
    for service in [IpcService::FileSystem, IpcService::AddOnContent] {
        let session = crate::IpcSession::new(service);
        assert!(matches!(
            crate::ipc::dispatch_session(
                process.mounts(),
                &session,
                IpcRequest::SetCurrentProcess,
                crate::FileSystemAccessLogMode::None
            ),
            Err(IpcResultCode::ACCESS_DENIED)
        ));
    }
    let permitted_directory = tempfile::tempdir().unwrap();
    let permitted_package = Package {
        title_id: APPLICATION_ID,
        version: 0,
        kind: MetaKind::Application,
        contents: vec![program_content(content_id(3), &[("main", 3)])],
    };
    fs::write(
        permitted_directory.path().join("permitted.nsp"),
        build_nsp(&permitted_package),
    )
    .unwrap();
    let permitted_plan = Launcher::build(LauncherInput::new(permitted_directory.path())).unwrap();
    let permitted = reference_process_builder().build(&permitted_plan).unwrap();
    let session = permitted
        .handles()
        .insert(crate::HorizonIpcObject::SemanticService(
            crate::IpcSession::new(IpcService::FileSystem),
        ))
        .unwrap();
    let transferred = permitted
        .handles()
        .transfer_to(process.handles(), session)
        .unwrap();
    let object = process
        .handles()
        .lookup_as::<crate::HorizonIpcObject>(transferred)
        .unwrap();
    let crate::HorizonIpcObject::SemanticService(session) = object.as_ref() else {
        panic!("transferred service")
    };
    assert!(matches!(
        crate::ipc::dispatch_session(
            process.mounts(),
            session,
            IpcRequest::SetCurrentProcess,
            crate::FileSystemAccessLogMode::None
        ),
        Err(IpcResultCode::ACCESS_DENIED)
    ));
}

#[test]
fn current_process_content_does_not_require_general_mount_permission() {
    fn build_process(
        directory: &tempfile::TempDir,
        filesystem_permissions: u64,
    ) -> nixe_runtime::RunnableProcess {
        let package = Package {
            title_id: APPLICATION_ID,
            version: 0,
            kind: MetaKind::Application,
            contents: vec![
                program_content_with_fs_permissions(
                    content_id(1),
                    &[("main", 1)],
                    filesystem_permissions,
                ),
                data_content(
                    content_id(2),
                    APPLICATION_ID,
                    1,
                    build_romfs(&[("file", b"bytes")]),
                ),
            ],
        };
        fs::write(directory.path().join("title.nsp"), build_nsp(&package)).unwrap();
        let plan = Launcher::build(LauncherInput::new(directory.path())).unwrap();
        reference_process_builder().build(&plan).unwrap()
    }

    let directory = tempfile::tempdir().unwrap();
    let process = build_process(&directory, 0);
    let session = crate::IpcSession::new(IpcService::FileSystem);
    let IpcResponse::Object(filesystem) = crate::ipc::dispatch_session(
        process.mounts(),
        &session,
        IpcRequest::OpenPrimaryFileSystem,
        crate::FileSystemAccessLogMode::None,
    )
    .unwrap() else {
        panic!("the process must receive its registered content filesystem");
    };
    let IpcResponse::Object(file) = crate::ipc::dispatch_semantic_object(
        process.mounts(),
        &filesystem,
        IpcRequest::OpenFile {
            path: "/file".into(),
            mode: 1,
        },
    )
    .unwrap() else {
        panic!("the content filesystem must return a file handle");
    };
    {
        let IpcResponse::Data(value) = crate::ipc::dispatch_semantic_object(
            process.mounts(),
            &file,
            IpcRequest::ReadFile { offset: 0, size: 5 },
        )
        .unwrap() else {
            panic!("response payload")
        };
        assert_eq!(value, b"bytes".to_vec());
    };

    let IpcResponse::Object(storage) = crate::ipc::dispatch_session(
        process.mounts(),
        &session,
        IpcRequest::OpenPrimaryStorage,
        crate::FileSystemAccessLogMode::None,
    )
    .unwrap() else {
        panic!("the process must receive its registered content storage");
    };
    assert!(matches!(
        crate::ipc::dispatch_semantic_object(process.mounts(), &storage, IpcRequest::GetStorageSize),
        Ok(IpcResponse::Size(size)) if size > 0
    ));
}

#[test]
fn public_add_ons_remain_visible_and_readable_without_content_owner_permissions() {
    for permissions in [0, 1_u64 << 62] {
        let directory = tempfile::tempdir().unwrap();
        let base = Package {
            title_id: APPLICATION_ID,
            version: 0,
            kind: MetaKind::Application,
            contents: vec![program_content_with_fs_permissions(
                content_id(1),
                &[("main", 1)],
                permissions,
            )],
        };
        let addon = Package {
            title_id: FIRST_DLC_ID,
            version: 0,
            kind: MetaKind::AddOnContent {
                required_application_version: 0,
            },
            contents: vec![public_data_content(
                content_id(2),
                FIRST_DLC_ID,
                0,
                build_romfs(&[("addon_key", b"public-content")]),
            )],
        };
        fs::write(directory.path().join("base.nsp"), build_nsp(&base)).unwrap();
        fs::write(directory.path().join("addon.nsp"), build_nsp(&addon)).unwrap();
        let plan = Launcher::build(LauncherInput::new(directory.path())).unwrap();
        let process = reference_process_builder().build(&plan).unwrap();
        let policy = process.mounts().effective_policy().unwrap().filesystem();
        assert!(policy.content_owner_ids().is_empty());
        assert_eq!(policy.permissions().raw(), permissions);
        assert_eq!(process.mounts().add_ons().len(), 1);

        let service = crate::IpcSession::new(IpcService::AddOnContent);
        {
            let IpcResponse::Size(value) = crate::ipc::dispatch_session(
                process.mounts(),
                &service,
                IpcRequest::GetAddOnContentCount,
                crate::FileSystemAccessLogMode::None,
            )
            .unwrap() else {
                panic!("response payload")
            };
            assert_eq!(value, 1);
        };
        let IpcResponse::AddOnContentEntries(entries) = crate::ipc::dispatch_session(
            process.mounts(),
            &service,
            IpcRequest::ListAddOnContent {
                offset: 0,
                max_entries: 1,
            },
            crate::FileSystemAccessLogMode::None,
        )
        .unwrap() else {
            panic!("add-on metadata must remain visible")
        };
        assert_eq!(entries[0].title_id.get(), FIRST_DLC_ID);
        let IpcResponse::Object(filesystem) = crate::ipc::dispatch_session(
            process.mounts(),
            &service,
            IpcRequest::OpenAddOnContent {
                title_id: entries[0].title_id,
                mount_index: 0,
            },
            crate::FileSystemAccessLogMode::None,
        )
        .unwrap() else {
            panic!("PublicData must not require content-owner permission")
        };
        let IpcResponse::Object(file) = crate::ipc::dispatch_semantic_object(
            process.mounts(),
            &filesystem,
            IpcRequest::OpenFile {
                path: "/addon_key".into(),
                mode: 1,
            },
        )
        .unwrap() else {
            panic!("PublicData must be readable")
        };
        {
            let IpcResponse::Data(value) = crate::ipc::dispatch_semantic_object(
                process.mounts(),
                &file,
                IpcRequest::ReadFile {
                    offset: 0,
                    size: 14,
                },
            )
            .unwrap() else {
                panic!("response payload")
            };
            assert_eq!(value, b"public-content".to_vec());
        };
        assert!(matches!(
            crate::ipc::dispatch_session(
                process.mounts(),
                &service,
                IpcRequest::OpenAddOnContent {
                    title_id: nixe_loader_title::TitleId::new(SECOND_DLC_ID),
                    mount_index: 0
                },
                crate::FileSystemAccessLogMode::None
            ),
            Err(IpcResultCode::PATH_NOT_FOUND)
        ));
    }
}

#[test]
fn builds_complete_launch_plan_from_redistributable_nsp_xci_matrix() {
    let directory = tempfile::tempdir().unwrap();
    let base_romfs = build_romfs(&[("keep", b"same"), ("replace", b"old!")]);
    let effective_romfs = build_romfs(&[("keep", b"same"), ("replace", b"new!")]);
    assert_eq!(base_romfs.len(), effective_romfs.len());

    let base = Package {
        title_id: APPLICATION_ID,
        version: 0,
        kind: MetaKind::Application,
        contents: vec![
            program_content(content_id(0x10), &[("main", 0x10)]),
            data_content(content_id(0x20), APPLICATION_ID, 1, base_romfs.clone()),
        ],
    };
    let first_dlc_old = Package {
        title_id: FIRST_DLC_ID,
        version: 1,
        kind: MetaKind::AddOnContent {
            required_application_version: 0,
        },
        contents: vec![public_data_content(
            content_id(0x30),
            FIRST_DLC_ID,
            0,
            build_romfs(&[("revision", b"old")]),
        )],
    };
    let second_dlc = Package {
        title_id: SECOND_DLC_ID,
        version: 4,
        kind: MetaKind::AddOnContent {
            required_application_version: 3,
        },
        contents: vec![public_data_content(
            content_id(0x40),
            SECOND_DLC_ID,
            0,
            build_romfs(&[("content", b"second")]),
        )],
    };
    fs::write(
        directory.path().join("base-and-dlc.xci"),
        build_xci(&[base, first_dlc_old, second_dlc]),
    )
    .unwrap();

    let patch = Package {
        title_id: PATCH_ID,
        version: 7,
        kind: MetaKind::Patch,
        contents: vec![
            program_content(
                content_id(0x50),
                &[
                    ("rtld", 0x50),
                    ("main", 0x60),
                    ("subsdk0", 0x70),
                    ("sdk", 0x80),
                ],
            ),
            bktr_data_content(content_id(0x60), 1, &base_romfs, &effective_romfs),
        ],
    };
    fs::write(directory.path().join("update.nsp"), build_nsp(&patch)).unwrap();

    let first_dlc_new = Package {
        title_id: FIRST_DLC_ID,
        version: 9,
        kind: MetaKind::AddOnContent {
            required_application_version: 7,
        },
        contents: vec![public_data_content(
            content_id(0x70),
            FIRST_DLC_ID,
            0,
            build_romfs(&[("revision", b"new")]),
        )],
    };
    fs::write(
        directory.path().join("dlc-revision.nsp"),
        build_nsp(&first_dlc_new),
    )
    .unwrap();

    let plan = Launcher::build(LauncherInput::new(directory.path())).unwrap();
    assert!(matches!(plan.kind(), LaunchKind::Packaged(_)));
    let identity = plan.packaged_identity().unwrap();
    assert_eq!(identity.application_id().get(), APPLICATION_ID);
    assert_eq!(identity.effective_title_id().get(), PATCH_ID);
    assert_eq!(identity.effective_version().raw(), 7);
    assert_eq!(identity.program_content_id(), &content_id(0x50));
    assert!(
        plan.effective_policy()
            .unwrap()
            .filesystem()
            .permissions()
            .raw()
            & (1 << 63)
            != 0
    );
    assert_eq!(
        plan.effective_policy().unwrap().handle_table_size(),
        Some(0x40)
    );
    assert!(plan.control_metadata().is_none());
    assert_eq!(
        plan.modules()
            .iter()
            .map(|module| (module.name(), module.role()))
            .collect::<Vec<_>>(),
        vec![
            ("rtld", ModuleRole::RuntimeLoader),
            ("main", ModuleRole::Main),
            ("subsdk0", ModuleRole::SubSdk(0)),
            ("sdk", ModuleRole::Sdk),
        ]
    );
    assert_eq!(plan.entry_module().name(), "rtld");

    let primary = plan.primary_file_system().unwrap();
    assert_eq!(primary.provenance(), MountProvenance::BaseAndPatch);
    let replacement = primary.romfs().open("/replace").unwrap().unwrap();
    let mut replacement_bytes = [0_u8; 4];
    replacement.read_at(0, &mut replacement_bytes).unwrap();
    assert_eq!(&replacement_bytes, b"new!");

    assert_eq!(plan.add_ons().len(), 2);
    assert_eq!(plan.add_ons()[0].title_id().get(), FIRST_DLC_ID);
    assert_eq!(plan.add_ons()[0].version().raw(), 9);
    assert_eq!(plan.add_ons()[1].title_id().get(), SECOND_DLC_ID);
    assert_eq!(plan.add_ons()[1].version().raw(), 4);
    assert!(plan.add_ons().iter().all(|add_on| {
        add_on
            .mounts()
            .iter()
            .all(|mount| mount.provenance() == MountProvenance::AddOn)
    }));
    let first_revision = plan.add_ons()[0].mounts()[0]
        .romfs()
        .open("/revision")
        .unwrap()
        .unwrap();
    let mut first_revision_bytes = [0_u8; 3];
    first_revision
        .read_at(0, &mut first_revision_bytes)
        .unwrap();
    assert_eq!(&first_revision_bytes, b"new");
    let second_content = plan.add_ons()[1].mounts()[0]
        .romfs()
        .open("/content")
        .unwrap()
        .unwrap();
    let mut second_content_bytes = [0_u8; 6];
    second_content
        .read_at(0, &mut second_content_bytes)
        .unwrap();
    assert_eq!(&second_content_bytes, b"second");

    let mut process = reference_process_builder().build(&plan).unwrap();
    assert_eq!(process.handles().capacity_limit(), 0x40);
    assert_eq!(process.mounts().add_ons().len(), 2);
    assert_eq!(process.modules().len(), 4);
    assert!(
        process
            .modules()
            .iter()
            .all(|module| module.relocation_state() == RelocationState::PendingGuestRuntime)
    );
    let entry_mapping = process
        .memory()
        .query(nixe_memory::GuestVirtualAddress::new(
            process.entry_module().entry_address(),
        ))
        .unwrap()
        .mapping
        .unwrap();
    assert_eq!(entry_mapping.kind, nixe_runtime::memory::MemoryKind::Code);
    let state = process.main_thread().state();
    assert_eq!(state.read_x(A64Register::General(x(0))), 0);
    assert_eq!(
        state.read_x(A64Register::General(x(1))),
        u64::from(process.main_thread().handle)
    );
    exercise_read_only_ipc(&mut process);
    let _ = process.try_teardown().unwrap();
}

fn x(index: u8) -> nixe_cpu::state::a64::A64GeneralRegister {
    nixe_cpu::state::a64::A64GeneralRegister::new(index).unwrap()
}

fn exercise_read_only_ipc(process: &mut nixe_runtime::RunnableProcess) {
    let filesystem_session = crate::IpcSession::new(IpcService::FileSystem);
    let IpcResponse::Object(primary) = crate::ipc::dispatch_session(
        process.mounts(),
        &filesystem_session,
        IpcRequest::OpenPrimaryFileSystem,
        crate::FileSystemAccessLogMode::None,
    )
    .unwrap() else {
        panic!("filesystem service must return a filesystem handle");
    };
    let IpcResponse::Object(root) = crate::ipc::dispatch_semantic_object(
        process.mounts(),
        &primary,
        IpcRequest::OpenDirectory {
            path: "/".into(),
            mode: 3,
        },
    )
    .unwrap() else {
        panic!("opening the root must return a directory handle");
    };
    {
        let IpcResponse::Size(value) = crate::ipc::dispatch_semantic_object(
            process.mounts(),
            &root,
            IpcRequest::GetDirectoryEntryCount,
        )
        .unwrap() else {
            panic!("response payload")
        };
        assert_eq!(value, 2);
    };
    let duplicate_root = root.clone();
    let IpcResponse::DirectoryEntries(first) = crate::ipc::dispatch_semantic_object(
        process.mounts(),
        &root,
        IpcRequest::ReadDirectory { max_entries: 1 },
    )
    .unwrap() else {
        panic!("directory read must return entries");
    };
    assert_eq!(first[0].name(), "keep");
    assert_eq!(first[0].kind(), DirectoryEntryKind::File);
    let IpcResponse::DirectoryEntries(second) = crate::ipc::dispatch_semantic_object(
        process.mounts(),
        &duplicate_root,
        IpcRequest::ReadDirectory { max_entries: 1 },
    )
    .unwrap() else {
        panic!("duplicated directory handle must preserve its shared cursor");
    };
    assert_eq!(second[0].name(), "replace");

    let IpcResponse::Object(file) = crate::ipc::dispatch_semantic_object(
        process.mounts(),
        &primary,
        IpcRequest::OpenFile {
            path: "/replace".into(),
            mode: 1,
        },
    )
    .unwrap() else {
        panic!("opening a file must return a file handle");
    };
    {
        let IpcResponse::Size(value) =
            crate::ipc::dispatch_semantic_object(process.mounts(), &file, IpcRequest::GetFileSize)
                .unwrap()
        else {
            panic!("response payload")
        };
        assert_eq!(value, 4);
    };
    {
        let IpcResponse::Data(value) = crate::ipc::dispatch_semantic_object(
            process.mounts(),
            &file,
            IpcRequest::ReadFile { offset: 1, size: 8 },
        )
        .unwrap() else {
            panic!("response payload")
        };
        assert_eq!(value, b"ew!".to_vec());
    };

    let add_on_session = crate::IpcSession::new(IpcService::AddOnContent);
    {
        let IpcResponse::Size(value) = crate::ipc::dispatch_session(
            process.mounts(),
            &add_on_session,
            IpcRequest::GetAddOnContentCount,
            crate::FileSystemAccessLogMode::None,
        )
        .unwrap() else {
            panic!("response payload")
        };
        assert_eq!(value, 2);
    };
    let IpcResponse::AddOnContentEntries(entries) = crate::ipc::dispatch_session(
        process.mounts(),
        &add_on_session,
        IpcRequest::ListAddOnContent {
            offset: 0,
            max_entries: 10,
        },
        crate::FileSystemAccessLogMode::None,
    )
    .unwrap() else {
        panic!("add-on listing must return metadata entries");
    };
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].title_id.get(), FIRST_DLC_ID);
    assert_eq!(entries[0].version, 9);
    assert_eq!(entries[0].mount_count, 1);
    assert!(matches!(
        crate::ipc::dispatch_session(
            process.mounts(),
            &add_on_session,
            IpcRequest::PrepareAddOnContent {
                horizon_index: u32::MAX,
            },
            crate::FileSystemAccessLogMode::None
        ),
        Err(IpcResultCode::PATH_NOT_FOUND)
    ));
    let IpcResponse::Event(add_on_event) = crate::ipc::dispatch_session(
        process.mounts(),
        &add_on_session,
        IpcRequest::GetAddOnContentListChangedEvent,
        crate::FileSystemAccessLogMode::None,
    )
    .unwrap() else {
        panic!("add-on event request must return a readable event handle");
    };
    assert!(!add_on_event.is_signalled());

    let IpcResponse::Object(add_on_filesystem) = crate::ipc::dispatch_session(
        process.mounts(),
        &add_on_session,
        IpcRequest::OpenAddOnContent {
            title_id: entries[0].title_id,
            mount_index: 0,
        },
        crate::FileSystemAccessLogMode::None,
    )
    .unwrap() else {
        panic!("authorized add-on must return a filesystem handle");
    };
    let IpcResponse::Object(revision) = crate::ipc::dispatch_semantic_object(
        process.mounts(),
        &add_on_filesystem,
        IpcRequest::OpenFile {
            path: "/revision".into(),
            mode: 1,
        },
    )
    .unwrap() else {
        panic!("add-on filesystem must open its own files");
    };
    {
        let IpcResponse::Data(value) = crate::ipc::dispatch_semantic_object(
            process.mounts(),
            &revision,
            IpcRequest::ReadFile { offset: 0, size: 3 },
        )
        .unwrap() else {
            panic!("response payload")
        };
        assert_eq!(value, b"new".to_vec());
    };
}
