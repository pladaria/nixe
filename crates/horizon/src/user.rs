//! Stable identities for users exposed by Horizon account-facing services.

use std::fmt::{Display, Formatter};

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
