//! Typed state for the undocumented GM20B tiled-cache configuration block.

use super::MaxwellThreeDRegister;
use crate::MaxwellMethodSource;

/// Tile dimensions programmed for the GM20B tiled cache.
///
/// deko3d identifies the low and high halfwords as width and height and emits
/// this register together with four still-undocumented cache controls:
/// <https://github.com/devkitPro/deko3d/blob/350f2b00a3e76ecd4f00191f8c5d6544ffbcb9db/source/maxwell/engine_3d.def#L234-L241>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellThreeDTiledCacheTileSize {
    width: u16,
    height: u16,
}

impl MaxwellThreeDTiledCacheTileSize {
    #[must_use]
    pub const fn parse(raw: u32) -> Self {
        Self {
            width: raw as u16,
            height: (raw >> 16) as u16,
        }
    }

    #[must_use]
    pub const fn width(self) -> u16 {
        self.width
    }

    #[must_use]
    pub const fn height(self) -> u16 {
        self.height
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self.width as u32 | ((self.height as u32) << 16)
    }
}

/// Exact bits of one still-undocumented tiled-cache configuration register.
///
/// Keeping this distinct from a generic raw register records the known family
/// and source without inventing fields or guest-visible rendering semantics.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct MaxwellThreeDTiledCacheUnknownConfig(u32);

impl MaxwellThreeDTiledCacheUnknownConfig {
    #[must_use]
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }
}

/// Tiled-cache state that is orthogonal to neutral render-pipeline identity.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellThreeDTiledCacheState {
    enabled: MaxwellThreeDRegister<bool>,
    tile_size: MaxwellThreeDRegister<MaxwellThreeDTiledCacheTileSize>,
    unknown_config: [MaxwellThreeDRegister<MaxwellThreeDTiledCacheUnknownConfig>; 4],
}

impl MaxwellThreeDTiledCacheState {
    #[must_use]
    pub const fn enabled(&self) -> &MaxwellThreeDRegister<bool> {
        &self.enabled
    }

    #[must_use]
    pub const fn tile_size(&self) -> &MaxwellThreeDRegister<MaxwellThreeDTiledCacheTileSize> {
        &self.tile_size
    }

    #[must_use]
    pub fn unknown_config(
        &self,
        index: u8,
    ) -> Option<&MaxwellThreeDRegister<MaxwellThreeDTiledCacheUnknownConfig>> {
        self.unknown_config.get(usize::from(index))
    }

    pub(super) fn apply(&mut self, write: MaxwellThreeDTiledCacheStateWrite) {
        match write {
            MaxwellThreeDTiledCacheStateWrite::Enable { value, source } => {
                self.enabled = MaxwellThreeDRegister::programmed(u32::from(value), value, source);
            }
            MaxwellThreeDTiledCacheStateWrite::TileSize { value, source } => {
                self.tile_size = MaxwellThreeDRegister::programmed(value.raw(), value, source);
            }
            MaxwellThreeDTiledCacheStateWrite::UnknownConfig {
                index,
                value,
                source,
            } => {
                self.unknown_config[usize::from(index)] =
                    MaxwellThreeDRegister::programmed(value.bits(), value, source);
            }
        }
    }
}

/// One validated tiled-cache state transition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellThreeDTiledCacheStateWrite {
    Enable {
        value: bool,
        source: MaxwellMethodSource,
    },
    TileSize {
        value: MaxwellThreeDTiledCacheTileSize,
        source: MaxwellMethodSource,
    },
    UnknownConfig {
        index: u8,
        value: MaxwellThreeDTiledCacheUnknownConfig,
        source: MaxwellMethodSource,
    },
}
