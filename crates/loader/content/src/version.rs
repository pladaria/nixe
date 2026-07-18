use std::fmt::{Display, Formatter};

use crate::CnmtMetaType;

/// Packed version used by Nintendo Switch system content metadata.
///
/// The component layout is documented by Switchbrew's system title list:
/// <https://switchbrew.org/wiki/Title_list#System_Modules>.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SystemVersion(u32);

impl SystemVersion {
    /// Creates a system version from its canonical packed value.
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// Returns the canonical packed value without losing any components.
    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Returns bits 26 through 31.
    pub const fn major(self) -> u8 {
        ((self.0 >> 26) & 0x3F) as u8
    }

    /// Returns bits 20 through 25.
    pub const fn minor(self) -> u8 {
        ((self.0 >> 20) & 0x3F) as u8
    }

    /// Returns bits 16 through 19.
    pub const fn micro(self) -> u8 {
        ((self.0 >> 16) & 0x0F) as u8
    }

    /// Returns the release-step value stored in bits 0 through 15.
    pub const fn release_step(self) -> u16 {
        (self.0 & 0xFFFF) as u16
    }
}

impl From<u32> for SystemVersion {
    fn from(raw: u32) -> Self {
        Self::from_raw(raw)
    }
}

impl From<SystemVersion> for u32 {
    fn from(version: SystemVersion) -> Self {
        version.raw()
    }
}

impl Display for SystemVersion {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{}.{}.{}.{}",
            self.major(),
            self.minor(),
            self.micro(),
            self.release_step()
        )
    }
}

/// Packed version used by application-family content metadata.
///
/// The release/private representation follows the NCM application-version
/// layout documented at <https://switchbrew.org/wiki/NCM_services>.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ApplicationVersion(u32);

impl ApplicationVersion {
    /// Creates an application version from its canonical packed value.
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// Returns the canonical packed value without losing any components.
    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Returns the public release component stored in bits 16 through 31.
    pub const fn release(self) -> u16 {
        (self.0 >> 16) as u16
    }

    /// Returns the private component stored in bits 0 through 15.
    pub const fn private(self) -> u16 {
        (self.0 & 0xFFFF) as u16
    }
}

impl From<u32> for ApplicationVersion {
    fn from(raw: u32) -> Self {
        Self::from_raw(raw)
    }
}

impl From<ApplicationVersion> for u32 {
    fn from(version: ApplicationVersion) -> Self {
        version.raw()
    }
}

impl Display for ApplicationVersion {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}.{}", self.release(), self.private())
    }
}

/// Raw content-meta version whose encoding depends on its content-meta type.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ContentMetaVersion(u32);

impl ContentMetaVersion {
    /// Creates a neutral content-meta version from its canonical raw value.
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// Returns the canonical raw value.
    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Interprets this value using the role declared by the same CNMT record.
    pub const fn decode(self, content_meta_type: CnmtMetaType) -> DecodedContentMetaVersion {
        match content_meta_type {
            CnmtMetaType::SystemProgram
            | CnmtMetaType::SystemData
            | CnmtMetaType::SystemUpdate
            | CnmtMetaType::BootImagePackage
            | CnmtMetaType::BootImagePackageSafe => {
                DecodedContentMetaVersion::System(SystemVersion::from_raw(self.0))
            }
            CnmtMetaType::Application
            | CnmtMetaType::Patch
            | CnmtMetaType::AddOnContent
            | CnmtMetaType::Delta
            | CnmtMetaType::DataPatch => {
                DecodedContentMetaVersion::Application(ApplicationVersion::from_raw(self.0))
            }
            CnmtMetaType::Unknown | CnmtMetaType::Unrecognized(_) => {
                DecodedContentMetaVersion::Unknown(self)
            }
        }
    }
}

impl From<u32> for ContentMetaVersion {
    fn from(raw: u32) -> Self {
        Self::from_raw(raw)
    }
}

impl From<ContentMetaVersion> for u32 {
    fn from(version: ContentMetaVersion) -> Self {
        version.raw()
    }
}

/// A content-meta version interpreted using its record's content-meta type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DecodedContentMetaVersion {
    /// Version of a system content-meta record.
    System(SystemVersion),
    /// Version of an application-family content-meta record.
    Application(ApplicationVersion),
    /// Raw value from a record whose encoding is not known.
    Unknown(ContentMetaVersion),
}

impl Display for DecodedContentMetaVersion {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::System(version) => Display::fmt(version, formatter),
            Self::Application(version) => Display::fmt(version, formatter),
            Self::Unknown(version) => write!(formatter, "raw {}", version.raw()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_and_displays_system_version_components() {
        let version = SystemVersion::from_raw(0x2803_0014);

        assert_eq!(version.major(), 10);
        assert_eq!(version.minor(), 0);
        assert_eq!(version.micro(), 3);
        assert_eq!(version.release_step(), 20);
        assert_eq!(version.raw(), 671_285_268);
        assert_eq!(version.to_string(), "10.0.3.20");
    }

    #[test]
    fn decodes_and_displays_application_version_components() {
        let version = ApplicationVersion::from_raw(0x000C_0000);

        assert_eq!(version.release(), 12);
        assert_eq!(version.private(), 0);
        assert_eq!(version.raw(), 786_432);
        assert_eq!(version.to_string(), "12.0");
    }

    #[test]
    fn preserves_zero_and_component_boundaries() {
        let zero_system = SystemVersion::from_raw(0);
        let zero_application = ApplicationVersion::from_raw(0);
        let maximum_system = SystemVersion::from_raw(u32::MAX);
        let maximum_application = ApplicationVersion::from_raw(u32::MAX);

        assert_eq!(zero_system.to_string(), "0.0.0.0");
        assert_eq!(zero_application.to_string(), "0.0");
        assert_eq!(maximum_system.major(), 63);
        assert_eq!(maximum_system.minor(), 63);
        assert_eq!(maximum_system.micro(), 15);
        assert_eq!(maximum_system.release_step(), u16::MAX);
        assert_eq!(maximum_application.release(), u16::MAX);
        assert_eq!(maximum_application.private(), u16::MAX);
    }

    #[test]
    fn raw_values_round_trip_and_preserve_numeric_ordering() {
        assert_eq!(u32::from(SystemVersion::from_raw(0x1234_5678)), 0x1234_5678);
        assert_eq!(
            u32::from(ApplicationVersion::from_raw(0x89AB_CDEF)),
            0x89AB_CDEF
        );
        assert!(ApplicationVersion::from_raw(1) > ApplicationVersion::from_raw(0));
        assert!(
            ApplicationVersion::from_raw(0x0002_0000) > ApplicationVersion::from_raw(0x0001_FFFF)
        );
        assert!(SystemVersion::from_raw(1) > SystemVersion::from_raw(0));
        assert!(SystemVersion::from_raw(0x0001_0000) > SystemVersion::from_raw(0x0000_FFFF));
        assert!(SystemVersion::from_raw(0x0010_0000) > SystemVersion::from_raw(0x000F_FFFF));
        assert!(SystemVersion::from_raw(0x0400_0000) > SystemVersion::from_raw(0x03FF_FFFF));
    }

    #[test]
    fn decodes_every_known_content_meta_family() {
        let version = ContentMetaVersion::from_raw(0x2803_0014);
        let system_types = [
            CnmtMetaType::SystemProgram,
            CnmtMetaType::SystemData,
            CnmtMetaType::SystemUpdate,
            CnmtMetaType::BootImagePackage,
            CnmtMetaType::BootImagePackageSafe,
        ];
        for content_meta_type in system_types {
            assert!(matches!(
                version.decode(content_meta_type),
                DecodedContentMetaVersion::System(_)
            ));
        }

        let application_types = [
            CnmtMetaType::Application,
            CnmtMetaType::Patch,
            CnmtMetaType::AddOnContent,
            CnmtMetaType::Delta,
            CnmtMetaType::DataPatch,
        ];
        for content_meta_type in application_types {
            assert!(matches!(
                version.decode(content_meta_type),
                DecodedContentMetaVersion::Application(_)
            ));
        }
    }

    #[test]
    fn leaves_unknown_content_meta_versions_raw() {
        let version = ContentMetaVersion::from_raw(42);

        assert_eq!(
            version.decode(CnmtMetaType::Unknown),
            DecodedContentMetaVersion::Unknown(version)
        );
        assert_eq!(
            version.decode(CnmtMetaType::Unrecognized(0xF0)),
            DecodedContentMetaVersion::Unknown(version)
        );
    }
}
