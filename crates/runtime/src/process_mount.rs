//! Process-local read-only filesystem namespace derived from a launch plan.

use std::path::{Path, PathBuf};

use nixe_loader_executable::{EffectiveNpdmPolicy, FileSystemPermissions};
use nixe_loader_title::TitleId;

use crate::{AddOnContent, HomebrewIdentity, LaunchPlan, ReadOnlyMount};

/// Immutable filesystem views visible to one process before Horizon IPC objects exist.
#[derive(Clone, Debug)]
pub struct ProcessMountNamespace {
    primary: Option<ReadOnlyMount>,
    add_ons: Box<[AddOnContent]>,
    sd_card_root: Option<PathBuf>,
    homebrew: Option<HomebrewIdentity>,
    policy: Option<EffectiveNpdmPolicy>,
}

impl ProcessMountNamespace {
    pub(crate) fn from_launch_plan(plan: &LaunchPlan, sd_card_root: Option<PathBuf>) -> Self {
        let policy = plan.effective_policy().cloned();
        // The title resolver selects only this application's add-ons, and the
        // package loader verifies that their payload NCAs are PublicData.
        // PublicData is exempt from FS content permissions; NPDM owner lists
        // must not hide installed add-ons from aoc:u or prevent their access.
        // https://switchbrew.org/wiki/Filesystem_services#OpenDataStorageByDataId
        let add_ons = plan.add_ons().to_vec().into_boxed_slice();
        Self {
            primary: plan.primary_file_system().cloned(),
            add_ons,
            sd_card_root,
            homebrew: plan.homebrew_identity().cloned(),
            policy,
        }
    }

    /// Returns the effective base/update RomFS view, when one exists.
    pub const fn primary(&self) -> Option<&ReadOnlyMount> {
        self.primary.as_ref()
    }

    /// Returns the installed PublicData add-ons resolved for this application.
    pub fn add_ons(&self) -> &[AddOnContent] {
        &self.add_ons
    }

    /// Returns the canonical host directory exposed as `sdmc:`, when present.
    pub fn sd_card_root(&self) -> Option<&Path> {
        self.sd_card_root.as_deref()
    }

    /// Returns the launched NRO overlaid into this process's SD-card view.
    pub const fn homebrew_executable(&self) -> Option<&HomebrewIdentity> {
        self.homebrew.as_ref()
    }

    /// Returns the immutable authorization policy associated with these mounts.
    pub const fn effective_policy(&self) -> Option<&EffectiveNpdmPolicy> {
        self.policy.as_ref()
    }

    /// Looks up one application add-on without exposing unrelated installed content.
    pub fn add_on(&self, title_id: TitleId) -> Option<&AddOnContent> {
        self.add_ons
            .iter()
            .find(|add_on| add_on.title_id() == title_id)
    }

    /// Applies the effective NPDM service access-control list. Homebrew has no
    /// NPDM and is allowed to reach a platform service registry.
    pub fn allows_service(&self, name: &[u8]) -> bool {
        self.policy
            .as_ref()
            .is_none_or(|policy| policy.allows_client(name))
    }

    /// Returns whether the process may access the removable SD-card filesystem.
    ///
    /// Permission identity follows Atmosphère's pinned filesystem access bits:
    /// https://github.com/Atmosphere-NX/Atmosphere/blob/e468f59c9d369b8ebbffa040f4c9fc201b9f75a8/libraries/libstratosphere/include/stratosphere/fssrv/impl/fssrv_access_control_bits.hpp
    pub fn allows_sd_card_access(&self) -> bool {
        self.policy.as_ref().is_none_or(|policy| {
            let permissions = policy.filesystem().permissions();
            permissions.contains(FileSystemPermissions::SD_CARD)
                || permissions.contains(FileSystemPermissions::FULL_PERMISSION)
        })
    }

    pub(crate) fn mount_count(&self) -> usize {
        usize::from(self.primary.is_some())
            + usize::from(self.sd_card_root.is_some())
            + self
                .add_ons
                .iter()
                .map(|add_on| add_on.mounts().len())
                .sum::<usize>()
    }
}
