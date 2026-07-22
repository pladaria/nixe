//! Verified guest-visible Horizon result values for IPC responses.
//!
//! Module IDs, description IDs, and the 9/13-bit layout are taken from
//! Atmosphere commit `e468f59c9d369b8ebbffa040f4c9fc201b9f75a8`:
//! - `libraries/libvapours/include/vapours/results/results_common.hpp`
//! - `libraries/libvapours/include/vapours/results/{fs,lr,sf,sm}_results.hpp`

use crate::{IpcResultCode, IpcService};

const MODULE_FS: u32 = 2;
const MODULE_LR: u32 = 8;
const MODULE_SF: u32 = 10;
const MODULE_SM: u32 = 21;
const MODULE_MASK: u32 = 0x1ff;
const DESCRIPTION_MASK: u32 = 0x1fff;
const DESCRIPTION_SHIFT: u32 = 9;

/// A result value encoded exactly as Horizon exposes it to IPC clients.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct HorizonIpcResult(u32);

impl HorizonIpcResult {
    pub const SUCCESS: Self = Self(0);

    pub(crate) const SM_INVALID_CLIENT: Self = Self::new(MODULE_SM, 2);
    pub(crate) const SM_OUT_OF_SESSIONS: Self = Self::new(MODULE_SM, 3);
    pub(crate) const SM_INVALID_SERVICE_NAME: Self = Self::new(MODULE_SM, 6);
    pub(crate) const SM_NOT_REGISTERED: Self = Self::new(MODULE_SM, 7);
    pub(crate) const SM_NOT_ALLOWED: Self = Self::new(MODULE_SM, 8);

    pub const CMIF_NOT_SUPPORTED: Self = Self::new(MODULE_SF, 1);
    pub const CMIF_UNKNOWN_COMMAND_ID: Self = Self::new(MODULE_SF, 221);
    pub const CMIF_TARGET_NOT_FOUND: Self = Self::new(MODULE_SF, 261);
    pub const FS_PATH_NOT_FOUND: Self = Self::new(MODULE_FS, 1);
    pub const FS_OUT_OF_RANGE: Self = Self::new(MODULE_FS, 3005);
    pub const FS_ALLOCATION_MEMORY_FAILED: Self = Self::new(MODULE_FS, 3420);
    pub const FS_UNEXPECTED: Self = Self::new(MODULE_FS, 5000);
    pub const FS_INVALID_ARGUMENT: Self = Self::new(MODULE_FS, 6001);
    pub const FS_PERMISSION_DENIED: Self = Self::new(MODULE_FS, 6400);
    pub const LR_ADD_ON_CONTENT_NOT_FOUND: Self = Self::new(MODULE_LR, 7);

    const fn new(module: u32, description: u32) -> Self {
        assert!(module <= MODULE_MASK);
        assert!(description <= DESCRIPTION_MASK);
        Self(module | (description << DESCRIPTION_SHIFT))
    }

    /// Maps a semantic service failure to an official guest-visible result.
    ///
    /// CMIF framing/object errors use module `sf`. Filesystem-backed failures
    /// use module `fs`; an absent add-on uses the verified `lr` result that the
    /// Horizon content-location path reports for that condition.
    #[must_use]
    pub const fn from_semantic(service: IpcService, result: IpcResultCode) -> Self {
        match result.semantic_id() {
            0 => Self::SUCCESS,
            1 => Self::CMIF_TARGET_NOT_FOUND,
            2 => Self::FS_PERMISSION_DENIED,
            3 => Self::CMIF_UNKNOWN_COMMAND_ID,
            4 => Self::FS_INVALID_ARGUMENT,
            5 if matches!(service, IpcService::AddOnContent) => Self::LR_ADD_ON_CONTENT_NOT_FOUND,
            5..=7 => Self::FS_PATH_NOT_FOUND,
            8 => Self::FS_OUT_OF_RANGE,
            9 => Self::FS_ALLOCATION_MEMORY_FAILED,
            10 | 11 => Self::FS_UNEXPECTED,
            _ => unreachable!(),
        }
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }

    #[must_use]
    pub const fn module(self) -> u32 {
        self.0 & MODULE_MASK
    }

    #[must_use]
    pub const fn description(self) -> u32 {
        (self.0 >> DESCRIPTION_SHIFT) & DESCRIPTION_MASK
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verified_named_results_have_the_expected_wire_values() {
        let cases = [
            (HorizonIpcResult::SM_INVALID_CLIENT, 21, 2, 0x415),
            (HorizonIpcResult::SM_OUT_OF_SESSIONS, 21, 3, 0x615),
            (HorizonIpcResult::SM_INVALID_SERVICE_NAME, 21, 6, 0xc15),
            (HorizonIpcResult::SM_NOT_REGISTERED, 21, 7, 0xe15),
            (HorizonIpcResult::SM_NOT_ALLOWED, 21, 8, 0x1015),
            (HorizonIpcResult::CMIF_NOT_SUPPORTED, 10, 1, 0x20a),
            (HorizonIpcResult::CMIF_UNKNOWN_COMMAND_ID, 10, 221, 0x1ba0a),
            (HorizonIpcResult::CMIF_TARGET_NOT_FOUND, 10, 261, 0x20a0a),
            (HorizonIpcResult::FS_PATH_NOT_FOUND, 2, 1, 0x202),
            (HorizonIpcResult::FS_OUT_OF_RANGE, 2, 3005, 0x177a02),
            (
                HorizonIpcResult::FS_ALLOCATION_MEMORY_FAILED,
                2,
                3420,
                0x1ab802,
            ),
            (HorizonIpcResult::FS_UNEXPECTED, 2, 5000, 0x271002),
            (HorizonIpcResult::FS_INVALID_ARGUMENT, 2, 6001, 0x2ee202),
            (HorizonIpcResult::FS_PERMISSION_DENIED, 2, 6400, 0x320002),
            (HorizonIpcResult::LR_ADD_ON_CONTENT_NOT_FOUND, 8, 7, 0xe08),
        ];

        for (result, module, description, raw) in cases {
            assert_eq!(result.module(), module);
            assert_eq!(result.description(), description);
            assert_eq!(result.raw(), raw);
        }
    }

    #[test]
    fn every_semantic_failure_has_a_contextual_official_mapping() {
        let filesystem_cases = [
            (IpcResultCode::SUCCESS, HorizonIpcResult::SUCCESS),
            (
                IpcResultCode::INVALID_HANDLE,
                HorizonIpcResult::CMIF_TARGET_NOT_FOUND,
            ),
            (
                IpcResultCode::ACCESS_DENIED,
                HorizonIpcResult::FS_PERMISSION_DENIED,
            ),
            (
                IpcResultCode::INVALID_COMMAND,
                HorizonIpcResult::CMIF_UNKNOWN_COMMAND_ID,
            ),
            (
                IpcResultCode::INVALID_ARGUMENT,
                HorizonIpcResult::FS_INVALID_ARGUMENT,
            ),
            (
                IpcResultCode::PATH_NOT_FOUND,
                HorizonIpcResult::FS_PATH_NOT_FOUND,
            ),
            (
                IpcResultCode::NOT_A_FILE,
                HorizonIpcResult::FS_PATH_NOT_FOUND,
            ),
            (
                IpcResultCode::NOT_A_DIRECTORY,
                HorizonIpcResult::FS_PATH_NOT_FOUND,
            ),
            (
                IpcResultCode::OUT_OF_RANGE,
                HorizonIpcResult::FS_OUT_OF_RANGE,
            ),
            (
                IpcResultCode::RESOURCE_LIMIT,
                HorizonIpcResult::FS_ALLOCATION_MEMORY_FAILED,
            ),
            (
                IpcResultCode::STORAGE_FAILURE,
                HorizonIpcResult::FS_UNEXPECTED,
            ),
            (
                IpcResultCode::INTERNAL_STATE,
                HorizonIpcResult::FS_UNEXPECTED,
            ),
        ];
        for (semantic, expected) in filesystem_cases {
            assert_eq!(
                HorizonIpcResult::from_semantic(IpcService::FileSystem, semantic),
                expected
            );
        }

        assert_eq!(
            HorizonIpcResult::from_semantic(
                IpcService::AddOnContent,
                IpcResultCode::PATH_NOT_FOUND,
            ),
            HorizonIpcResult::LR_ADD_ON_CONTENT_NOT_FOUND
        );
        for (semantic, expected) in filesystem_cases {
            if semantic != IpcResultCode::PATH_NOT_FOUND {
                assert_eq!(
                    HorizonIpcResult::from_semantic(IpcService::AddOnContent, semantic),
                    expected
                );
            }
        }
    }
}
