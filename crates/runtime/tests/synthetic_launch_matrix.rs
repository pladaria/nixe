mod support;

use std::fs;

use swiitx_runtime::{
    LaunchKind, Launcher, LauncherInput, ModuleRole, MountProvenance, ProcessBuilder,
};

use support::synthetic_packages::{
    APPLICATION_ID, FIRST_DLC_ID, MetaKind, PATCH_ID, Package, SECOND_DLC_ID, bktr_data_content,
    build_nsp, build_romfs, build_xci, content_id, data_content, program_content,
};

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
    assert_eq!(plan.entry_module().name(), "main");
    assert_eq!(plan.symbol_scope(), &[0, 1, 2, 3]);

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

    let process = ProcessBuilder::new().build(&plan).unwrap();
    assert_eq!(process.mounts().add_ons().len(), 2);
    assert_eq!(process.modules().len(), 4);
    let _ = process.teardown();
}
