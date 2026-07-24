//! Small implementation helpers shared by memory backends.

use crate::address::GuestVirtualAddress;

use super::{SYNTHETIC_PAGE_SIZE, SyntheticInstallError, SyntheticInstallStage};

pub(super) fn install_error(
    stage: SyntheticInstallStage,
    address: Option<GuestVirtualAddress>,
    reason: impl Into<Box<str>>,
) -> SyntheticInstallError {
    SyntheticInstallError {
        stage,
        address,
        reason: reason.into(),
    }
}

pub(super) fn page_offset(address: GuestVirtualAddress) -> usize {
    address.get() as usize % SYNTHETIC_PAGE_SIZE
}
