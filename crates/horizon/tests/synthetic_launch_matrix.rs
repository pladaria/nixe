mod support;

use std::fs;

use nixe_cpu::memory::MemoryMappingPurpose;
use nixe_cpu::state::{ThreadCpuState, a64::A64Register};
use nixe_horizon::{
    DirectoryEntryKind, HorizonProcess, IpcRequest, IpcResponse, IpcResultCode, IpcService,
};
use nixe_runtime::{
    LaunchKind, Launcher, LauncherInput, ModuleRole, MountProvenance, ProcessBuilder,
    RelocationState,
};

use support::synthetic_packages::{
    APPLICATION_ID, FIRST_DLC_ID, MetaKind, PATCH_ID, Package, SECOND_DLC_ID, bktr_data_content,
    build_nsp, build_romfs, build_xci, content_id, data_content, program_content,
    program_content_with_fs_permissions, program_content_without_services,
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
    let mut process = ProcessBuilder::new().build(&plan).unwrap();
    assert_eq!(
        process.connect_ipc_service(IpcService::FileSystem),
        Err(IpcResultCode::ACCESS_DENIED)
    );
    assert_eq!(
        process.connect_ipc_service(IpcService::AddOnContent),
        Err(IpcResultCode::ACCESS_DENIED)
    );

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
    let mut permitted = ProcessBuilder::new().build(&permitted_plan).unwrap();
    let session = permitted
        .connect_ipc_service(IpcService::FileSystem)
        .unwrap();
    let transferred = permitted
        .handles_mut()
        .transfer_to(process.handles_mut(), session)
        .unwrap();
    assert_eq!(
        process.dispatch_ipc(transferred, IpcRequest::SetCurrentProcess),
        Err(IpcResultCode::ACCESS_DENIED)
    );
}

#[test]
fn filesystem_operations_require_effective_content_data_read_permission() {
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
        ProcessBuilder::new().build(&plan).unwrap()
    }

    let denied_directory = tempfile::tempdir().unwrap();
    let mut denied = build_process(&denied_directory, 0);
    let denied_session = denied.connect_ipc_service(IpcService::FileSystem).unwrap();
    assert_eq!(
        denied.dispatch_ipc(denied_session, IpcRequest::OpenPrimaryFileSystem),
        Err(IpcResultCode::ACCESS_DENIED)
    );

    let allowed_directory = tempfile::tempdir().unwrap();
    let mut allowed = build_process(&allowed_directory, 1);
    let allowed_session = allowed.connect_ipc_service(IpcService::FileSystem).unwrap();
    let IpcResponse::Handle(filesystem) = allowed
        .dispatch_ipc(allowed_session, IpcRequest::OpenPrimaryFileSystem)
        .unwrap()
    else {
        panic!("ApplicationInfo must permit mounting content data");
    };

    let transferred = allowed
        .handles_mut()
        .transfer_to(denied.handles_mut(), filesystem)
        .unwrap();
    assert_eq!(
        denied.dispatch_ipc(transferred, IpcRequest::OpenDirectory { path: "/".into() }),
        Err(IpcResultCode::ACCESS_DENIED)
    );

    let manager_directory = tempfile::tempdir().unwrap();
    let mut manager = build_process(&manager_directory, 1 << 11);
    let manager_session = manager.connect_ipc_service(IpcService::FileSystem).unwrap();
    assert!(matches!(
        manager.dispatch_ipc(manager_session, IpcRequest::OpenPrimaryFileSystem),
        Ok(IpcResponse::Handle(_))
    ));
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
        contents: vec![data_content(
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
        contents: vec![data_content(
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
        contents: vec![data_content(
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

    let mut process = ProcessBuilder::new().build(&plan).unwrap();
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
        .mapping_info(
            process.cpu_context().address_space_id(),
            nixe_cpu::address::GuestVirtualAddress::new(process.entry_module().entry_address()),
        )
        .unwrap();
    assert_eq!(entry_mapping.purpose, MemoryMappingPurpose::CodeStatic);
    let ThreadCpuState::A64(state) = &process.main_thread().state else {
        panic!("synthetic packaged program must initialize AArch64 state");
    };
    assert_eq!(state.read_x(A64Register::General(x(0))), 0);
    assert_eq!(
        state.read_x(A64Register::General(x(1))),
        u64::from(process.main_thread().handle)
    );
    exercise_read_only_ipc(&mut process);
    let _ = process.teardown();
}

fn x(index: u8) -> nixe_cpu::state::a64::A64GeneralRegister {
    nixe_cpu::state::a64::A64GeneralRegister::new(index).unwrap()
}

fn exercise_read_only_ipc(process: &mut nixe_runtime::RunnableProcess) {
    let filesystem_session = process.connect_ipc_service(IpcService::FileSystem).unwrap();
    let IpcResponse::Handle(primary) = process
        .dispatch_ipc(filesystem_session, IpcRequest::OpenPrimaryFileSystem)
        .unwrap()
    else {
        panic!("filesystem service must return a filesystem handle");
    };
    let IpcResponse::Handle(root) = process
        .dispatch_ipc(primary, IpcRequest::OpenDirectory { path: "/".into() })
        .unwrap()
    else {
        panic!("opening the root must return a directory handle");
    };
    assert_eq!(
        process
            .dispatch_ipc(root, IpcRequest::GetDirectoryEntryCount)
            .unwrap(),
        IpcResponse::Size(2)
    );
    let duplicate_root = process.handles_mut().duplicate(root).unwrap();
    let IpcResponse::DirectoryEntries(first) = process
        .dispatch_ipc(root, IpcRequest::ReadDirectory { max_entries: 1 })
        .unwrap()
    else {
        panic!("directory read must return entries");
    };
    assert_eq!(first[0].name(), "keep");
    assert_eq!(first[0].kind(), DirectoryEntryKind::File);
    let IpcResponse::DirectoryEntries(second) = process
        .dispatch_ipc(duplicate_root, IpcRequest::ReadDirectory { max_entries: 1 })
        .unwrap()
    else {
        panic!("duplicated directory handle must preserve its shared cursor");
    };
    assert_eq!(second[0].name(), "replace");

    let IpcResponse::Handle(file) = process
        .dispatch_ipc(
            primary,
            IpcRequest::OpenFile {
                path: "/replace".into(),
            },
        )
        .unwrap()
    else {
        panic!("opening a file must return a file handle");
    };
    assert_eq!(
        process.dispatch_ipc(file, IpcRequest::GetFileSize).unwrap(),
        IpcResponse::Size(4)
    );
    assert_eq!(
        process
            .dispatch_ipc(file, IpcRequest::ReadFile { offset: 1, size: 8 })
            .unwrap(),
        IpcResponse::Data(b"ew!".to_vec())
    );

    let add_on_session = process
        .connect_ipc_service(IpcService::AddOnContent)
        .unwrap();
    assert_eq!(
        process
            .dispatch_ipc(add_on_session, IpcRequest::GetAddOnContentCount)
            .unwrap(),
        IpcResponse::Size(2)
    );
    let IpcResponse::AddOnContentEntries(entries) = process
        .dispatch_ipc(
            add_on_session,
            IpcRequest::ListAddOnContent {
                offset: 0,
                max_entries: 10,
            },
        )
        .unwrap()
    else {
        panic!("add-on listing must return metadata entries");
    };
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].title_id.get(), FIRST_DLC_ID);
    assert_eq!(entries[0].version, 9);
    assert_eq!(entries[0].mount_count, 1);
    assert_eq!(
        process.dispatch_ipc(
            add_on_session,
            IpcRequest::PrepareAddOnContent {
                horizon_index: u32::MAX,
            },
        ),
        Err(IpcResultCode::PATH_NOT_FOUND)
    );
    let IpcResponse::Event(add_on_event) = process
        .dispatch_ipc(add_on_session, IpcRequest::GetAddOnContentListChangedEvent)
        .unwrap()
    else {
        panic!("add-on event request must return a readable event handle");
    };
    assert!(
        process
            .handles()
            .get_as::<nixe_runtime::ReadableEventObject>(add_on_event)
            .is_some()
    );

    let IpcResponse::Handle(add_on_filesystem) = process
        .dispatch_ipc(
            add_on_session,
            IpcRequest::OpenAddOnContent {
                title_id: entries[0].title_id,
                mount_index: 0,
            },
        )
        .unwrap()
    else {
        panic!("authorized add-on must return a filesystem handle");
    };
    let IpcResponse::Handle(revision) = process
        .dispatch_ipc(
            add_on_filesystem,
            IpcRequest::OpenFile {
                path: "/revision".into(),
            },
        )
        .unwrap()
    else {
        panic!("add-on filesystem must open its own files");
    };
    assert_eq!(
        process
            .dispatch_ipc(revision, IpcRequest::ReadFile { offset: 0, size: 3 })
            .unwrap(),
        IpcResponse::Data(b"new".to_vec())
    );

    process.handles_mut().close(file).unwrap();
    assert_eq!(
        process.dispatch_ipc(file, IpcRequest::GetFileSize),
        Err(IpcResultCode::INVALID_HANDLE)
    );
}
