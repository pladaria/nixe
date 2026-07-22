use std::fmt::{Display, Formatter};

use nixe_loader_storage::{FormatLoader, LoadError, StorageRef};

pub const NACP_SIZE: usize = 0x4000;
const LANGUAGE_COUNT: usize = 16;

/// Loads Nintendo Switch application control properties (`control.nacp`).
#[derive(Debug)]
pub struct NacpLoader;

impl FormatLoader for NacpLoader {
    type Output = ApplicationControlProperty;

    const FORMAT_NAME: &'static str = "NACP";

    fn load(storage: StorageRef) -> Result<Self::Output, LoadError> {
        if storage.len()? != NACP_SIZE as u64 {
            return Err(LoadError::invalid(
                Self::FORMAT_NAME,
                format!("expected {NACP_SIZE:#x} bytes"),
            ));
        }
        let mut bytes = Box::new([0_u8; NACP_SIZE]);
        storage.read_at(0, bytes.as_mut())?;
        parse_nacp(&bytes)
    }
}

/// Language slots used by NACP titles, language flags, and localized icons.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum NacpLanguage {
    AmericanEnglish,
    BritishEnglish,
    Japanese,
    French,
    German,
    LatinAmericanSpanish,
    Spanish,
    Italian,
    Dutch,
    CanadianFrench,
    Portuguese,
    Russian,
    Korean,
    TraditionalChinese,
    SimplifiedChinese,
    BrazilianPortuguese,
}

impl NacpLanguage {
    pub const ALL: [Self; LANGUAGE_COUNT] = [
        Self::AmericanEnglish,
        Self::BritishEnglish,
        Self::Japanese,
        Self::French,
        Self::German,
        Self::LatinAmericanSpanish,
        Self::Spanish,
        Self::Italian,
        Self::Dutch,
        Self::CanadianFrench,
        Self::Portuguese,
        Self::Russian,
        Self::Korean,
        Self::TraditionalChinese,
        Self::SimplifiedChinese,
        Self::BrazilianPortuguese,
    ];

    pub const fn index(self) -> usize {
        self as usize
    }

    pub const fn bit(self) -> u32 {
        1 << self.index()
    }

    pub const fn icon_suffix(self) -> &'static str {
        match self {
            Self::AmericanEnglish => "AmericanEnglish",
            Self::BritishEnglish => "BritishEnglish",
            Self::Japanese => "Japanese",
            Self::French => "French",
            Self::German => "German",
            Self::LatinAmericanSpanish => "LatinAmericanSpanish",
            Self::Spanish => "Spanish",
            Self::Italian => "Italian",
            Self::Dutch => "Dutch",
            Self::CanadianFrench => "CanadianFrench",
            Self::Portuguese => "Portuguese",
            Self::Russian => "Russian",
            Self::Korean => "Korean",
            Self::TraditionalChinese => "TraditionalChinese",
            Self::SimplifiedChinese => "SimplifiedChinese",
            Self::BrazilianPortuguese => "BrazilianPortuguese",
        }
    }
}

impl Display for NacpLanguage {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.icon_suffix())
    }
}

/// Raw supported-language mask with helpers for the known language slots.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SupportedLanguages(u32);

impl SupportedLanguages {
    pub const KNOWN_MASK: u32 = (1 << LANGUAGE_COUNT) - 1;

    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    pub const fn raw(self) -> u32 {
        self.0
    }

    pub const fn contains(self, language: NacpLanguage) -> bool {
        self.0 & language.bit() != 0
    }

    pub const fn unknown_bits(self) -> u32 {
        self.0 & !Self::KNOWN_MASK
    }

    pub fn iter(self) -> impl Iterator<Item = NacpLanguage> {
        NacpLanguage::ALL
            .into_iter()
            .filter(move |language| self.contains(*language))
    }
}

macro_rules! byte_enum {
    ($name:ident { $($value:literal => $variant:ident),+ $(,)? }) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum $name {
            $($variant,)+
            Unknown(u8),
        }

        impl From<u8> for $name {
            fn from(value: u8) -> Self {
                match value {
                    $($value => Self::$variant,)+
                    value => Self::Unknown(value),
                }
            }
        }
    };
}

byte_enum!(StartupUserAccount {
    0 => None,
    1 => Required,
    2 => RequiredWithNetworkServiceAccountAvailable,
});
byte_enum!(UserAccountSwitchLock { 0 => Disable, 1 => Enable });
byte_enum!(AddOnContentRegistrationType { 0 => AllOnLaunch, 1 => OnDemand });
byte_enum!(ScreenshotPolicy { 0 => Allow, 1 => Deny });
byte_enum!(VideoCapturePolicy { 0 => Disable, 1 => Manual, 2 => Enable });
byte_enum!(DataLossConfirmation { 0 => None, 1 => Required });
byte_enum!(PlayLogPolicy { 0 => Open, 1 => LogOnly, 2 => None, 3 => Closed });
byte_enum!(LogoType {
    0 => LicensedByNintendo,
    1 => DistributedByNintendo,
    2 => Nintendo,
});
byte_enum!(LogoHandling { 0 => Auto, 1 => Manual });
byte_enum!(RuntimeAddOnContentInstall {
    0 => Deny,
    1 => AllowAppend,
    2 => AllowAppendWithoutNetworkDownload,
});
byte_enum!(RuntimeParameterDelivery {
    0 => Always,
    1 => AlwaysIfUserStateMatched,
    2 => OnRestart,
});
byte_enum!(AppropriateAgeForChina { 0 => None, 1 => Age8, 2 => Age12, 3 => Age16 });
byte_enum!(CrashReportPolicy { 0 => Deny, 1 => Allow });
byte_enum!(HdcpPolicy { 0 => None, 1 => Required });
byte_enum!(PlayLogQueryCapability { 0 => None, 1 => WhiteList, 2 => All });

/// One localized application-title slot.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ApplicationTitle {
    pub name: String,
    pub publisher: String,
}

impl ApplicationTitle {
    pub fn is_empty(&self) -> bool {
        self.name.is_empty() && self.publisher.is_empty()
    }
}

/// Parsed application control property with raw preservation for flags and
/// newer format regions whose semantics are not yet fully verified.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ApplicationControlProperty {
    pub titles: [ApplicationTitle; LANGUAGE_COUNT],
    pub isbn: String,
    pub startup_user_account: StartupUserAccount,
    pub user_account_switch_lock: UserAccountSwitchLock,
    pub add_on_content_registration_type: AddOnContentRegistrationType,
    pub attribute_flag: u32,
    pub supported_languages: SupportedLanguages,
    pub parental_control_flag: u32,
    pub screenshot: ScreenshotPolicy,
    pub video_capture: VideoCapturePolicy,
    pub data_loss_confirmation: DataLossConfirmation,
    pub play_log_policy: PlayLogPolicy,
    pub presence_group_id: u64,
    pub rating_ages: [i8; 0x20],
    pub display_version: String,
    pub add_on_content_base_id: u64,
    pub save_data_owner_id: u64,
    pub user_account_save_data_size: i64,
    pub user_account_save_data_journal_size: i64,
    pub device_save_data_size: i64,
    pub device_save_data_journal_size: i64,
    pub bcat_delivery_cache_storage_size: i64,
    pub application_error_code_category: String,
    pub local_communication_ids: [u64; 8],
    pub logo_type: LogoType,
    pub logo_handling: LogoHandling,
    pub runtime_add_on_content_install: RuntimeAddOnContentInstall,
    pub runtime_parameter_delivery: RuntimeParameterDelivery,
    pub appropriate_age_for_china: AppropriateAgeForChina,
    pub crash_report: CrashReportPolicy,
    pub hdcp: HdcpPolicy,
    pub seed_for_pseudo_device_id: u64,
    pub bcat_passphrase: String,
    pub startup_user_account_option: u8,
    pub user_account_save_data_size_max: i64,
    pub user_account_save_data_journal_size_max: i64,
    pub device_save_data_size_max: i64,
    pub device_save_data_journal_size_max: i64,
    pub temporary_storage_size: i64,
    pub cache_storage_size: i64,
    pub cache_storage_journal_size: i64,
    pub cache_storage_data_and_journal_size_max: i64,
    pub cache_storage_index_max: u16,
    pub runtime_upgrade: u8,
    pub supporting_limited_application_licenses: u32,
    pub play_log_queryable_application_ids: [u64; 16],
    pub play_log_query_capability: PlayLogQueryCapability,
    pub repair_flag: u8,
    pub program_index: u8,
    pub required_network_service_license_on_launch_flag: u8,
    pub application_error_code_prefix: u8,
    pub titles_data_format: u8,
    pub acd_index: u8,
    pub apparent_platform: u8,
    pub neighbor_detection_client_configuration: [u8; 0x198],
    pub jit_configuration_flags: u64,
    pub jit_configuration_memory_size: u64,
    pub required_add_on_contents: [u16; 32],
    pub play_report_permission: u8,
    pub crash_screenshot_for_prod: u8,
    pub crash_screenshot_for_dev: u8,
    pub contents_availability_transition_policy: u8,
    pub supported_language_flag_for_nx_addon: u32,
    pub accessible_launch_required_versions: [u64; 8],
    /// Raw bytes from firmware-dependent fields at 0x3448 through the end.
    pub firmware_dependent_region: Vec<u8>,
}

impl ApplicationControlProperty {
    pub fn title(&self, language: NacpLanguage) -> &ApplicationTitle {
        &self.titles[language.index()]
    }

    pub fn localized_titles(&self) -> impl Iterator<Item = (NacpLanguage, &ApplicationTitle)> {
        NacpLanguage::ALL.into_iter().filter_map(|language| {
            let title = self.title(language);
            (!title.is_empty()).then_some((language, title))
        })
    }

    /// Chooses the first non-empty requested title, then American English,
    /// then the first populated canonical language slot.
    pub fn preferred_title(
        &self,
        preferences: &[NacpLanguage],
    ) -> Option<(NacpLanguage, &ApplicationTitle)> {
        preferences
            .iter()
            .copied()
            .chain([NacpLanguage::AmericanEnglish])
            .chain(NacpLanguage::ALL)
            .find_map(|language| {
                let title = self.title(language);
                (!title.is_empty()).then_some((language, title))
            })
    }
}

fn parse_nacp(bytes: &[u8; NACP_SIZE]) -> Result<ApplicationControlProperty, LoadError> {
    let mut titles = std::array::from_fn(|_| ApplicationTitle::default());
    for (index, title) in titles.iter_mut().enumerate() {
        let offset = index * 0x300;
        title.name = fixed_string(bytes, offset, 0x200, "application name")?;
        title.publisher = fixed_string(bytes, offset + 0x200, 0x100, "publisher")?;
    }

    Ok(ApplicationControlProperty {
        titles,
        isbn: fixed_string(bytes, 0x3000, 0x25, "ISBN")?,
        startup_user_account: bytes[0x3025].into(),
        user_account_switch_lock: bytes[0x3026].into(),
        add_on_content_registration_type: bytes[0x3027].into(),
        attribute_flag: read_u32(bytes, 0x3028),
        supported_languages: SupportedLanguages::from_raw(read_u32(bytes, 0x302C)),
        parental_control_flag: read_u32(bytes, 0x3030),
        screenshot: bytes[0x3034].into(),
        video_capture: bytes[0x3035].into(),
        data_loss_confirmation: bytes[0x3036].into(),
        play_log_policy: bytes[0x3037].into(),
        presence_group_id: read_u64(bytes, 0x3038),
        rating_ages: std::array::from_fn(|index| bytes[0x3040 + index] as i8),
        display_version: fixed_string(bytes, 0x3060, 0x10, "display version")?,
        add_on_content_base_id: read_u64(bytes, 0x3070),
        save_data_owner_id: read_u64(bytes, 0x3078),
        user_account_save_data_size: read_i64(bytes, 0x3080),
        user_account_save_data_journal_size: read_i64(bytes, 0x3088),
        device_save_data_size: read_i64(bytes, 0x3090),
        device_save_data_journal_size: read_i64(bytes, 0x3098),
        bcat_delivery_cache_storage_size: read_i64(bytes, 0x30A0),
        application_error_code_category: fixed_string(
            bytes,
            0x30A8,
            8,
            "application error code category",
        )?,
        local_communication_ids: read_u64_array(bytes, 0x30B0),
        logo_type: bytes[0x30F0].into(),
        logo_handling: bytes[0x30F1].into(),
        runtime_add_on_content_install: bytes[0x30F2].into(),
        runtime_parameter_delivery: bytes[0x30F3].into(),
        appropriate_age_for_china: bytes[0x30F4].into(),
        crash_report: bytes[0x30F6].into(),
        hdcp: bytes[0x30F7].into(),
        seed_for_pseudo_device_id: read_u64(bytes, 0x30F8),
        bcat_passphrase: fixed_string(bytes, 0x3100, 0x41, "BCAT passphrase")?,
        startup_user_account_option: bytes[0x3141],
        user_account_save_data_size_max: read_i64(bytes, 0x3148),
        user_account_save_data_journal_size_max: read_i64(bytes, 0x3150),
        device_save_data_size_max: read_i64(bytes, 0x3158),
        device_save_data_journal_size_max: read_i64(bytes, 0x3160),
        temporary_storage_size: read_i64(bytes, 0x3168),
        cache_storage_size: read_i64(bytes, 0x3170),
        cache_storage_journal_size: read_i64(bytes, 0x3178),
        cache_storage_data_and_journal_size_max: read_i64(bytes, 0x3180),
        cache_storage_index_max: read_u16(bytes, 0x3188),
        runtime_upgrade: bytes[0x318B],
        supporting_limited_application_licenses: read_u32(bytes, 0x318C),
        play_log_queryable_application_ids: read_u64_array(bytes, 0x3190),
        play_log_query_capability: bytes[0x3210].into(),
        repair_flag: bytes[0x3211],
        program_index: bytes[0x3212],
        required_network_service_license_on_launch_flag: bytes[0x3213],
        application_error_code_prefix: bytes[0x3214],
        titles_data_format: bytes[0x3215],
        acd_index: bytes[0x3216],
        apparent_platform: bytes[0x3217],
        neighbor_detection_client_configuration: bytes[0x3218..0x33B0]
            .try_into()
            .expect("fixed NACP range"),
        jit_configuration_flags: read_u64(bytes, 0x33B0),
        jit_configuration_memory_size: read_u64(bytes, 0x33B8),
        required_add_on_contents: std::array::from_fn(|index| read_u16(bytes, 0x33C0 + index * 2)),
        play_report_permission: bytes[0x3400],
        crash_screenshot_for_prod: bytes[0x3401],
        crash_screenshot_for_dev: bytes[0x3402],
        contents_availability_transition_policy: bytes[0x3403],
        supported_language_flag_for_nx_addon: read_u32(bytes, 0x3404),
        accessible_launch_required_versions: read_u64_array(bytes, 0x3408),
        firmware_dependent_region: bytes[0x3448..].to_vec(),
    })
}

fn fixed_string(bytes: &[u8], offset: usize, size: usize, name: &str) -> Result<String, LoadError> {
    let field = &bytes[offset..offset + size];
    let used = field.iter().position(|byte| *byte == 0).unwrap_or(size);
    std::str::from_utf8(&field[..used])
        .map(str::to_owned)
        .map_err(|_| LoadError::invalid("NACP", format!("{name} is not valid UTF-8")))
}

fn read_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(bytes[offset..offset + 2].try_into().expect("fixed range"))
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("fixed range"))
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().expect("fixed range"))
}

fn read_i64(bytes: &[u8], offset: usize) -> i64 {
    i64::from_le_bytes(bytes[offset..offset + 8].try_into().expect("fixed range"))
}

fn read_u64_array<const N: usize>(bytes: &[u8], offset: usize) -> [u64; N] {
    std::array::from_fn(|index| read_u64(bytes, offset + index * 8))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use nixe_loader_storage::{Storage, StorageError};

    use super::*;

    #[derive(Debug)]
    struct VecStorage(Vec<u8>);

    impl Storage for VecStorage {
        fn len(&self) -> Result<u64, StorageError> {
            Ok(self.0.len() as u64)
        }

        fn read_at(&self, offset: u64, buffer: &mut [u8]) -> Result<(), StorageError> {
            let start = usize::try_from(offset).map_err(|_| StorageError::OutOfBounds)?;
            let end = start + buffer.len();
            buffer.copy_from_slice(self.0.get(start..end).ok_or(StorageError::OutOfBounds)?);
            Ok(())
        }
    }

    fn load(bytes: Vec<u8>) -> Result<ApplicationControlProperty, LoadError> {
        NacpLoader::load(Arc::new(VecStorage(bytes)))
    }

    fn put_string(bytes: &mut [u8], offset: usize, size: usize, value: &str) {
        assert!(value.len() < size);
        bytes[offset..offset + value.len()].copy_from_slice(value.as_bytes());
    }

    #[test]
    fn parses_localized_titles_languages_and_properties() {
        let mut bytes = vec![0_u8; NACP_SIZE];
        put_string(&mut bytes, 0, 0x200, "Example");
        put_string(&mut bytes, 0x200, 0x100, "Publisher");
        let japanese = NacpLanguage::Japanese.index() * 0x300;
        put_string(&mut bytes, japanese, 0x200, "例");
        put_string(&mut bytes, 0x3060, 0x10, "1.2.3");
        bytes[0x302C..0x3030].copy_from_slice(
            &(NacpLanguage::AmericanEnglish.bit() | NacpLanguage::Japanese.bit() | 0x8000_0000)
                .to_le_bytes(),
        );
        bytes[0x3025] = 1;
        bytes[0x3035] = 2;
        bytes[0x3212] = 7;
        bytes[0x3080..0x3088].copy_from_slice(&123_i64.to_le_bytes());

        let nacp = load(bytes).unwrap();
        assert_eq!(nacp.title(NacpLanguage::AmericanEnglish).name, "Example");
        assert_eq!(nacp.title(NacpLanguage::Japanese).name, "例");
        assert_eq!(nacp.display_version, "1.2.3");
        assert_eq!(nacp.startup_user_account, StartupUserAccount::Required);
        assert_eq!(nacp.video_capture, VideoCapturePolicy::Enable);
        assert_eq!(nacp.program_index, 7);
        assert_eq!(nacp.user_account_save_data_size, 123);
        assert_eq!(nacp.supported_languages.iter().count(), 2);
        assert_eq!(nacp.supported_languages.unknown_bits(), 0x8000_0000);
    }

    #[test]
    fn chooses_titles_with_deterministic_fallback() {
        let mut bytes = vec![0_u8; NACP_SIZE];
        put_string(&mut bytes, 0, 0x200, "English");
        let french = NacpLanguage::French.index() * 0x300;
        put_string(&mut bytes, french, 0x200, "Français");
        let nacp = load(bytes).unwrap();

        assert_eq!(
            nacp.preferred_title(&[NacpLanguage::French])
                .unwrap()
                .1
                .name,
            "Français"
        );
        assert_eq!(
            nacp.preferred_title(&[NacpLanguage::Japanese])
                .unwrap()
                .1
                .name,
            "English"
        );
    }

    #[test]
    fn rejects_wrong_size_and_invalid_utf8() {
        assert!(matches!(
            load(vec![0; NACP_SIZE - 1]),
            Err(LoadError::InvalidFormat { .. })
        ));
        let mut bytes = vec![0_u8; NACP_SIZE];
        bytes[0] = 0xFF;
        assert!(matches!(load(bytes), Err(LoadError::InvalidFormat { .. })));
    }

    #[test]
    fn maps_every_language_to_a_distinct_bit_and_suffix() {
        let mut bits = 0_u32;
        let mut suffixes = std::collections::BTreeSet::new();
        for language in NacpLanguage::ALL {
            assert_eq!(bits & language.bit(), 0);
            bits |= language.bit();
            assert!(suffixes.insert(language.icon_suffix()));
        }
        assert_eq!(bits, SupportedLanguages::KNOWN_MASK);
    }
}
