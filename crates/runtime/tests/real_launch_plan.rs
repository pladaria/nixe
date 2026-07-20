//! Opt-in launch-plan integration against caller-owned title content.

use std::env;
use std::path::PathBuf;

use swiitx_loader_content::NcaKeySet;
use swiitx_runtime::{LaunchKind, Launcher, LauncherInput, ModuleRole};

#[test]
#[ignore = "requires caller-owned title content and keys"]
fn constructs_a_complete_plan_from_a_real_package() {
    let package = PathBuf::from(env::var_os("SWIITX_REAL_PACKAGE").expect("SWIITX_REAL_PACKAGE"));
    let keys_dir = PathBuf::from(env::var_os("SWIITX_KEYS_DIR").expect("SWIITX_KEYS_DIR"));
    let title_keys = keys_dir.join("title.keys");
    let keys = NcaKeySet::from_files(
        keys_dir.join("prod.keys"),
        title_keys.is_file().then_some(title_keys.as_path()),
    )
    .expect("load caller-owned keys");
    let plan = Launcher::build(LauncherInput::new(package).with_keys(keys))
        .expect("construct complete launch plan");
    assert!(matches!(plan.kind(), LaunchKind::Packaged(_)));
    assert_eq!(plan.entry_module().role(), ModuleRole::Main);
    assert!(plan.effective_policy().is_some());
    assert_eq!(plan.symbol_scope().len(), plan.modules().len());
    let mut scope = plan.symbol_scope().to_vec();
    scope.sort_unstable();
    assert_eq!(scope, (0..plan.modules().len()).collect::<Vec<_>>());
    if let Some(mount) = plan.primary_file_system() {
        let _ = mount.romfs().files().first();
    }
    for add_on in plan.add_ons() {
        assert!(!add_on.mounts().is_empty());
        let _ = add_on.mounts()[0].romfs().files().first();
    }
}
