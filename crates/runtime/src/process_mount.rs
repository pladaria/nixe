//! Process-local read-only filesystem namespace derived from a launch plan.

use std::path::{Path, PathBuf};

use nixe_loader_executable::{EffectiveNpdmPolicy, FileSystemPermissions};
use nixe_loader_title::TitleId;

use crate::{AddOnContent, LaunchPlan, ReadOnlyMount};

/// Immutable filesystem views visible to one process before Horizon IPC objects exist.
#[derive(Clone, Debug)]
pub struct ProcessMountNamespace {
    primary: Option<ReadOnlyMount>,
    add_ons: Box<[AddOnContent]>,
    sd_card_root: Option<PathBuf>,
    policy: Option<EffectiveNpdmPolicy>,
}

impl ProcessMountNamespace {
    pub(crate) fn from_launch_plan(plan: &LaunchPlan, sd_card_root: Option<PathBuf>) -> Self {
        let policy = plan.effective_policy().cloned();
        let add_ons = plan
            .add_ons()
            .iter()
            .filter(|add_on| content_owner_allowed(policy.as_ref(), add_on.title_id()))
            .cloned()
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self {
            primary: plan.primary_file_system().cloned(),
            add_ons,
            sd_card_root,
            policy,
        }
    }

    /// Returns the effective base/update RomFS view, when one exists.
    pub const fn primary(&self) -> Option<&ReadOnlyMount> {
        self.primary.as_ref()
    }

    /// Returns only add-on views authorized by the effective NPDM policy.
    pub fn add_ons(&self) -> &[AddOnContent] {
        &self.add_ons
    }

    /// Returns the canonical host directory exposed as `sdmc:`, when present.
    pub fn sd_card_root(&self) -> Option<&Path> {
        self.sd_card_root.as_deref()
    }

    /// Returns the immutable authorization policy associated with these mounts.
    pub const fn effective_policy(&self) -> Option<&EffectiveNpdmPolicy> {
        self.policy.as_ref()
    }

    /// Looks up one authorized add-on without exposing unrelated installed content.
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

    /// Returns whether the process may mount and read program content data.
    ///
    /// Horizon derives `CanMountContentDataRead` from `ApplicationInfo`,
    /// `ContentManager`, or `FullPermission`. Keep this operation-level check
    /// separate from service access: permission to connect to `fsp-srv` does
    /// not itself grant access to a content filesystem.
    ///
    /// Reference (Atmosphère, commit e468f59):
    /// https://github.com/Atmosphere-NX/Atmosphere/blob/e468f59c9d369b8ebbffa040f4c9fc201b9f75a8/libraries/libstratosphere/include/stratosphere/fssrv/impl/fssrv_access_control_bits.hpp#L75-L79
    pub fn allows_content_data_read(&self) -> bool {
        self.policy.as_ref().is_none_or(|policy| {
            let permissions = policy.filesystem().permissions();
            permissions.contains(FileSystemPermissions::APPLICATION_INFO)
                || permissions.contains(FileSystemPermissions::CONTENT_MANAGER)
                || permissions.contains(FileSystemPermissions::FULL_PERMISSION)
        })
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

fn content_owner_allowed(policy: Option<&EffectiveNpdmPolicy>, title_id: TitleId) -> bool {
    let Some(policy) = policy else {
        return true;
    };
    let filesystem = policy.filesystem();
    if filesystem
        .permissions()
        .contains(FileSystemPermissions::FULL_PERMISSION)
    {
        return true;
    }
    let owner = title_id.get();
    filesystem.content_owner_ids().contains(&owner)
        || filesystem
            .content_owner_range()
            .is_some_and(|(start, end)| (start..=end).contains(&owner))
}
