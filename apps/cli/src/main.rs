use std::collections::BTreeMap;
use std::env;
use std::ffi::OsString;
use std::path::PathBuf;
use std::process::ExitCode;

use swiitx_config::SwiitxConfig;
use swiitx_loader_content::{NcaFormatVersion, NcaKeySet};
use swiitx_loader_title::{
    CnmtExtendedHeader, EntryKind, NacpLanguage, NcaInspection, TitleInspection, TitleInspector,
};

struct CliArguments {
    config_path: Option<PathBuf>,
}

fn main() -> ExitCode {
    let mut raw_arguments = env::args_os();
    let program = raw_arguments
        .next()
        .unwrap_or_else(|| OsString::from("swiitx-cli"));
    let arguments = match parse_arguments(raw_arguments) {
        Ok(Some(arguments)) => arguments,
        Ok(None) => {
            print_usage(&program);
            return ExitCode::SUCCESS;
        }
        Err(error) => {
            eprintln!("error: {error}");
            print_usage(&program);
            return ExitCode::from(2);
        }
    };

    match inspect_titles(arguments) {
        Ok((inspections, preferred_languages)) => {
            for (index, inspection) in inspections.iter().enumerate() {
                if index != 0 {
                    println!();
                }
                print_inspection(inspection, &preferred_languages);
            }
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn print_usage(program: &OsString) {
    eprintln!(
        "Usage: {} [--config <file>]\n\n\
         Inspect the title-library paths from the shared configuration.\n\
         Pass --config to select a TOML file explicitly. Otherwise the CLI uses\n\
         SWIITX_CONFIG, ./swiitx.toml, or the platform user configuration.",
        program.to_string_lossy()
    );
}

fn parse_arguments(
    arguments: impl Iterator<Item = OsString>,
) -> Result<Option<CliArguments>, String> {
    let mut config_path = None;
    let mut arguments = arguments;

    while let Some(argument) = arguments.next() {
        if argument == "-h" || argument == "--help" {
            return Ok(None);
        }
        if argument == "--config" {
            if config_path.is_some() {
                return Err("--config may only be specified once".to_owned());
            }
            let path = arguments
                .next()
                .ok_or_else(|| "--config requires a file path".to_owned())?;
            config_path = Some(PathBuf::from(path));
            continue;
        }
        return Err(format!("unknown argument: {}", argument.to_string_lossy()));
    }

    Ok(Some(CliArguments { config_path }))
}

fn inspect_titles(
    arguments: CliArguments,
) -> Result<(Vec<TitleInspection>, Vec<NacpLanguage>), String> {
    let config = match arguments.config_path {
        Some(path) => SwiitxConfig::load(path).map_err(|error| error.to_string())?,
        None => SwiitxConfig::load_discovered()
            .map_err(|error| error.to_string())?
            .ok_or_else(|| {
                "no configuration found; pass --config or create swiitx.toml".to_owned()
            })?,
    };

    let paths = config.library.paths.clone();
    if paths.is_empty() {
        return Err("the configuration has no library paths".to_owned());
    }

    let options = config.library.scan_options();
    let preferred_languages = config.system.preferred_languages.clone();
    let keys_dir = config.system.keys;

    let prod_keys = keys_dir.join("prod.keys");
    let title_keys_path = keys_dir.join("title.keys");
    let title_keys = title_keys_path.is_file().then_some(title_keys_path);
    let mut keys = NcaKeySet::from_files(&prod_keys, title_keys.as_deref())
        .map_err(|error| error.to_string())?;
    let inspections = paths
        .into_iter()
        .map(|path| {
            TitleInspector::inspect_with_key_set_and_options(path, &mut keys, options)
                .map_err(|error| error.to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((inspections, preferred_languages))
}

fn print_inspection(inspection: &TitleInspection, preferred_languages: &[NacpLanguage]) {
    println!("Title: {}", inspection.path.display());
    println!("Packages: {}", inspection.packages.len());

    for (index, package) in inspection.packages.iter().enumerate() {
        println!();
        println!("Package {}: {}", index + 1, package.path.display());
        println!("  Format: {}", package.format);
        println!("  Size: {}", format_size(package.size));
        println!("  PFS0 data offset: {:#X}", package.data_offset);
        println!(
            "  Container overhead: {}",
            format_size(package.container_overhead())
        );
        println!("  Entries: {}", package.entries.len());

        let mut counts = BTreeMap::<EntryKind, usize>::new();
        for entry in &package.entries {
            *counts.entry(entry.kind).or_default() += 1;
            println!(
                "    {:<12} {:>12} bytes {:<14} offset {:#X}  {}",
                entry.kind,
                entry.size,
                format_binary_size(entry.size),
                entry.offset,
                entry.name
            );
            if let Some(nca) = &entry.nca {
                print_nca(nca);
            }
            if let Some(warning) = &entry.nca_warning {
                println!("      NCA inspection warning: {warning}");
            }
        }

        println!("  Entry summary:");
        for (kind, count) in counts {
            println!("    {kind}: {count}");
        }
        for warning in &package.ticket_warnings {
            println!("  Ticket warning: {warning}");
        }

        if let Some(metadata) = &package.canonical_content_meta {
            println!("  Canonical content metadata:");
            println!("    Type: {}", metadata.content_meta_type);
            println!("    Title ID: {:016X}", metadata.title_id);
            println!(
                "    Version: {} (raw {})",
                metadata.decoded_version(),
                metadata.version.raw()
            );
            println!("    Platform: {}", metadata.platform);
            println!("    Attributes: {:#04X}", metadata.attributes);
            println!("    Storage ID: {:#04X}", metadata.storage_id);
            println!("    Install type: {}", metadata.install_type);
            println!("    Committed: {}", metadata.committed);
            println!(
                "    Required download system version: {} (raw {})",
                metadata.required_download_system_version,
                metadata.required_download_system_version.raw()
            );
            println!(
                "    Extended header size: {:#X}",
                metadata.extended_header_size
            );
            print_extended_cnmt_header(&metadata.extended_header);
            println!("    Declared contents: {}", metadata.contents.len());
            for content in &metadata.contents {
                println!(
                    "      {:<18} {:>18}  {}",
                    content.content_type,
                    format_size(content.size),
                    format_hex(&content.content_id)
                );
                println!("        SHA-256: {}", format_hex(&content.hash));
                println!("        Attributes: {:#04X}", content.attributes);
                println!("        ID offset: {}", content.id_offset);
            }
            println!(
                "    Content-meta references: {}",
                metadata.content_meta.len()
            );
            for reference in &metadata.content_meta {
                println!(
                    "      {} {:016X} version {} (raw {}) attributes {:#04X}",
                    reference.content_meta_type,
                    reference.title_id,
                    reference.decoded_version(),
                    reference.version.raw(),
                    reference.attributes
                );
            }
            if metadata.extended_data_size != 0 {
                println!(
                    "    Extended data size: {}",
                    format_size(metadata.extended_data_size)
                );
            }
            println!("    Digest: {}", format_hex(&metadata.digest));
        }
        if let Some(warning) = &package.canonical_metadata_warning {
            println!("  Canonical metadata warning: {warning}");
        }

        if let Some(control) = &package.control_metadata {
            println!("  Control metadata:");
            if let Some((language, title)) = control.nacp.preferred_title(preferred_languages) {
                println!("    Preferred title ({language}): {}", title.name);
                println!("    Preferred publisher: {}", title.publisher);
            }
            if let Some(icon) = control.preferred_icon(preferred_languages) {
                println!("    Preferred icon: {} ({})", icon.filename, icon.language);
            }
            for (language, title) in control.nacp.localized_titles() {
                println!("    {language}:");
                println!("      Name: {}", title.name);
                println!("      Publisher: {}", title.publisher);
            }
            let languages: Vec<_> = control
                .supported_languages()
                .iter()
                .map(|language| language.to_string())
                .collect();
            println!(
                "    Supported languages: {}",
                if languages.is_empty() {
                    "none".to_owned()
                } else {
                    languages.join(", ")
                }
            );
            let unknown_language_bits = control.supported_languages().unknown_bits();
            if unknown_language_bits != 0 {
                println!("    Unknown supported-language bits: {unknown_language_bits:#010X}");
            }
            println!("    Display version: {}", control.nacp.display_version);
            println!("    Program index: {}", control.nacp.program_index);
            println!(
                "    Startup user account: {:?}",
                control.nacp.startup_user_account
            );
            println!("    Screenshot policy: {:?}", control.nacp.screenshot);
            println!("    Video capture policy: {:?}", control.nacp.video_capture);
            println!("    Play-log policy: {:?}", control.nacp.play_log_policy);
            println!(
                "    Save-data owner ID: {:016X}",
                control.nacp.save_data_owner_id
            );
            println!("    Icons: {}", control.icons().len());
            for icon in control.icons() {
                println!(
                    "      {}: {} ({}, JPEG)",
                    icon.language,
                    icon.filename,
                    format_size(icon.size)
                );
            }
        }
        if let Some(warning) = &package.control_metadata_warning {
            println!("  Control metadata warning: {warning}");
        }

        if let Some(metadata) = &package.content_meta {
            println!("  Auxiliary content metadata:");
            println!("    Type: {}", metadata.content_type);
            println!("    Title ID: {:016X}", metadata.title_id);
            println!(
                "    Version: {} (raw {})",
                metadata.decoded_version(),
                metadata.version.raw()
            );
            if let Some(original_id) = metadata.original_id {
                println!("    Original/base ID: {original_id:016X}");
            }
            if let Some(application_id) = metadata.application_id {
                println!("    Application ID: {application_id:016X}");
            }
            if let Some(key_generation) = metadata.minimum_key_generation {
                println!("    Minimum key generation: {key_generation}");
            }
            if let Some(version) = metadata.required_system_version {
                println!(
                    "    Required system version: {version} (raw {})",
                    version.raw()
                );
            }
            if let Some(version) = metadata.required_application_version {
                println!(
                    "    Required application version: {version} (raw {})",
                    version.raw()
                );
            }
            if let Some(digest) = &metadata.digest {
                println!("    Digest: {digest}");
            }
            println!("    Declared contents: {}", metadata.contents.len());
            for content in &metadata.contents {
                println!(
                    "      {:<18} {:>18}  {}",
                    content.content_type,
                    format_size(content.size),
                    content.id
                );
                if let Some(key_generation) = content.key_generation {
                    println!("        Key generation: {key_generation}");
                }
                if let Some(hash) = &content.hash {
                    println!("        SHA-256: {hash}");
                }
            }
        }
        if let Some(warning) = &package.metadata_warning {
            println!("  Auxiliary metadata warning: {warning}");
        }
    }

    if !inspection.ignored_files.is_empty() {
        println!();
        println!("Ignored unsupported files:");
        for path in &inspection.ignored_files {
            println!("  {}", path.display());
        }
    }

    println!();
    println!(
        "Note: binary CNMT is canonical package metadata; auxiliary XML remains informational."
    );
    println!("Canonical metadata does not by itself establish full package authenticity.");
}

fn print_extended_cnmt_header(header: &CnmtExtendedHeader) {
    match header {
        CnmtExtendedHeader::None => {}
        CnmtExtendedHeader::Application {
            patch_id,
            required_system_version,
            required_application_version,
        } => {
            println!("    Patch ID: {patch_id:016X}");
            println!(
                "    Required system version: {required_system_version} (raw {})",
                required_system_version.raw()
            );
            println!(
                "    Required application version: {required_application_version} (raw {})",
                required_application_version.raw()
            );
        }
        CnmtExtendedHeader::Patch {
            application_id,
            required_system_version,
            ..
        } => {
            println!("    Original/base application ID: {application_id:016X}");
            println!(
                "    Required system version: {required_system_version} (raw {})",
                required_system_version.raw()
            );
        }
        CnmtExtendedHeader::AddOnContent {
            application_id,
            required_application_version,
            content_accessibilities,
            data_patch_id,
            ..
        } => {
            println!("    Application ID: {application_id:016X}");
            println!(
                "    Required application version: {required_application_version} (raw {})",
                required_application_version.raw()
            );
            println!("    Content accessibilities: {content_accessibilities:#04X}");
            println!("    Data patch ID: {data_patch_id:016X}");
        }
        CnmtExtendedHeader::LegacyAddOnContent {
            application_id,
            required_application_version,
            ..
        } => {
            println!("    Application ID: {application_id:016X}");
            println!(
                "    Required application version: {required_application_version} (raw {})",
                required_application_version.raw()
            );
        }
        CnmtExtendedHeader::Delta { application_id, .. } => {
            println!("    Application ID: {application_id:016X}");
        }
        CnmtExtendedHeader::SystemUpdate { .. } => {}
        CnmtExtendedHeader::Unknown(bytes) => {
            println!("    Unknown extended-header bytes: {}", bytes.len());
        }
    }
}

fn print_nca(nca: &NcaInspection) {
    let version = match nca.format_version {
        NcaFormatVersion::Nca2 => "NCA2",
        NcaFormatVersion::Nca3 => "NCA3",
    };
    println!("      NCA header:");
    println!("        Format: {version}");
    println!("        Distribution: {:?}", nca.distribution_type);
    println!("        Content type: {:?}", nca.content_type);
    println!("        Title ID: {:016X}", nca.title_id);
    println!("        Declared size: {}", format_size(nca.size));
    println!("        SDK version (raw): {:#010X}", nca.sdk_version);
    println!("        Key generation: {}", nca.key_generation);
    println!("        Key-area index: {}", nca.key_area_key_index);
    match nca.rights_id {
        Some(rights_id) => println!("        Rights ID: {}", format_hex(&rights_id)),
        None => println!("        Rights ID: none"),
    }
    println!(
        "        Source: {}",
        if nca.source_is_decrypted {
            "decrypted"
        } else {
            "encrypted"
        }
    );
    println!("        Sections: {}", nca.sections.len());
    for section in &nca.sections {
        println!(
            "          {}: {:?}, {:?}, offset {:#X}, {}, FS header hash {}",
            section.index,
            section.section_type,
            section.encryption_type,
            section.offset,
            format_size(section.size),
            if section.fs_header_hash_valid {
                "valid"
            } else {
                "INVALID"
            }
        );
    }
}

fn format_hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut result = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        result.push(char::from(DIGITS[usize::from(byte >> 4)]));
        result.push(char::from(DIGITS[usize::from(byte & 0x0f)]));
    }
    result
}

fn format_size(bytes: u64) -> String {
    let binary_size = format_binary_size(bytes);
    if binary_size.is_empty() {
        format!("{bytes} bytes")
    } else {
        format!("{bytes} bytes {binary_size}")
    }
}

fn format_binary_size(bytes: u64) -> String {
    const KIB: u64 = 1024;
    const MIB: u64 = KIB * 1024;
    const GIB: u64 = MIB * 1024;

    if bytes >= GIB {
        format!("({:.2} GiB)", bytes as f64 / GIB as f64)
    } else if bytes >= MIB {
        format!("({:.2} MiB)", bytes as f64 / MIB as f64)
    } else if bytes >= KIB {
        format!("({:.2} KiB)", bytes as f64 / KIB as f64)
    } else {
        String::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn os_arguments(values: &[&str]) -> impl Iterator<Item = OsString> {
        values
            .iter()
            .map(|value| OsString::from(*value))
            .collect::<Vec<_>>()
            .into_iter()
    }

    #[test]
    fn parses_config() {
        let arguments = parse_arguments(os_arguments(&["--config", "custom.toml"]))
            .unwrap()
            .unwrap();

        assert_eq!(arguments.config_path, Some(PathBuf::from("custom.toml")));
    }

    #[test]
    fn accepts_help() {
        assert!(
            parse_arguments(os_arguments(&["--help"]))
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn accepts_no_arguments_for_config_discovery() {
        let arguments = parse_arguments(os_arguments(&[])).unwrap().unwrap();

        assert_eq!(arguments.config_path, None);
    }

    #[test]
    fn rejects_every_other_argument() {
        for arguments in [
            &["title.nsp"][..],
            &["--keys-dir", "keys"][..],
            &["--"][..],
            &["--unknown"][..],
        ] {
            assert!(parse_arguments(os_arguments(arguments)).is_err());
        }
    }

    #[test]
    fn rejects_missing_or_duplicate_config_values() {
        assert!(parse_arguments(os_arguments(&["--config"])).is_err());
        assert!(
            parse_arguments(os_arguments(&[
                "--config", "one.toml", "--config", "two.toml"
            ]))
            .is_err()
        );
    }
}
