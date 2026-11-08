//! Stable identities for users exposed by Horizon account-facing services.

use std::fmt::{Display, Formatter};
use std::sync::Mutex;

// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
// JPEG exposed by the built-in local account, following Eden's backup-profile path.
// https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/core/constants.cpp
const PROFILE_IMAGE: &[u8] = include_bytes!("../../../assets/account-profile.jpg");

/// Account-module state shared by service sessions and per-user managers.
/// Stored context starts empty and changes only through StoreOpenContext.
/// https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/core/hle/service/acc/profile_manager.cpp
#[derive(Debug)]
pub(crate) struct ProfileManager {
    pub user: UserIdentity,
    stored_opened_users: Mutex<[u8; 128]>,
}
impl Default for ProfileManager {
    fn default() -> Self {
        Self {
            user: DEFAULT_USER,
            stored_opened_users: Mutex::new([0; 128]),
        }
    }
}
impl ProfileManager {
    pub fn stored_opened_users(&self) -> [u8; 128] {
        *self.stored_opened_users.lock().unwrap()
    }
    pub fn store_opened_users(&self) {
        let mut users = self.stored_opened_users.lock().unwrap();
        users.fill(0);
        users[..16].copy_from_slice(&self.user.id().encode());
    }
}

/// The single user available while Nixe has no configurable account database.
///
/// Keep this identifier stable: save-data ownership will use the same identity,
/// so changing it later requires an explicit migration rather than a new
/// arbitrary value.
pub(crate) const DEFAULT_USER: UserIdentity = UserIdentity {
    id: UserId::from_u128(1),
    name: "Nixe",
};

/// Horizon's 128-bit `AccountUid`, represented in its two-word wire order.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct UserId([u64; 2]);

impl UserId {
    pub(crate) const fn from_u128(value: u128) -> Self {
        Self([value as u64, (value >> 64) as u64])
    }

    pub(crate) const fn encode(self) -> [u8; 16] {
        let low = self.0[0].to_le_bytes();
        let high = self.0[1].to_le_bytes();
        [
            low[0], low[1], low[2], low[3], low[4], low[5], low[6], low[7], high[0], high[1],
            high[2], high[3], high[4], high[5], high[6], high[7],
        ]
    }
}

impl Display for UserId {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{:016x}{:016x}", self.0[1], self.0[0])
    }
}

/// User information shared by applet, account and future save-data services.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct UserIdentity {
    id: UserId,
    name: &'static str,
}

impl UserIdentity {
    pub(crate) const fn image(self) -> &'static [u8] {
        PROFILE_IMAGE
    }
    pub(crate) const fn id(self) -> UserId {
        self.id
    }

    pub(crate) const fn name(self) -> &'static str {
        self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_user_has_a_stable_horizon_uid() {
        assert_eq!(DEFAULT_USER.id().encode(), 1_u128.to_le_bytes());
        assert_eq!(DEFAULT_USER.name(), "Nixe");
        assert_eq!(
            DEFAULT_USER.id().to_string(),
            "00000000000000000000000000000001"
        );
    }
}
