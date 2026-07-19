use std::collections::BTreeMap;
use std::env;
use std::ffi::OsString;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use sanitize_filename::{Options, sanitize_with_options};
use swiitx_config::SwiitxConfig;
use swiitx_loader_content::{NcaFormatVersion, NcaKeySet};
use swiitx_loader_title::{
    CnmtExtendedHeader, EntryKind, NacpLanguage, NcaInspection, ResolvedTitle, TitleCatalog,
    TitleInspection, TitleInspector, TitleResolver,
};

struct CliArguments {
    config_path: Option<PathBuf>,
    title_path: Option<PathBuf>,
}

enum CliOutput {
    Inspection {
        inspection: Box<TitleInspection>,
        preferred_languages: Vec<NacpLanguage>,
    },
    LibrarySummary {
        titles: Vec<ResolvedTitle>,
        preferred_languages: Vec<NacpLanguage>,
    },
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

    match run(arguments) {
        Ok(CliOutput::Inspection {
            inspection,
            preferred_languages,
        }) => {
            print_inspection(&inspection, &preferred_languages);
            ExitCode::SUCCESS
        }
        Ok(CliOutput::LibrarySummary {
            titles,
            preferred_languages,
        }) => {
            print_library_summary(&titles, &preferred_languages);
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
        "Usage: {} [--config <file>] [<title-path>]\n\n\
         Without title-path, scan the configured library and summarize resolved titles.\n\
         With title-path, show a detailed inspection of that file or directory.\n\
         Pass --config to select a TOML file explicitly. Otherwise the CLI uses\n\
         SWIITX_CONFIG, ./swiitx.toml, or the platform user configuration.",
        program.to_string_lossy()
    );
}

fn parse_arguments(
    arguments: impl Iterator<Item = OsString>,
) -> Result<Option<CliArguments>, String> {
    let mut config_path = None;
    let mut title_path = None;
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
        if argument.to_string_lossy().starts_with('-') {
            return Err(format!("unknown option: {}", argument.to_string_lossy()));
        }
        if title_path.replace(PathBuf::from(argument)).is_some() {
            return Err("expected at most one title path".to_owned());
        }
    }

    Ok(Some(CliArguments {
        config_path,
        title_path,
    }))
}

fn run(arguments: CliArguments) -> Result<CliOutput, String> {
    let config = match arguments.config_path {
        Some(path) => SwiitxConfig::load(path).map_err(|error| error.to_string())?,
        None => SwiitxConfig::load_discovered()
            .map_err(|error| error.to_string())?
            .ok_or_else(|| {
                "no configuration found; pass --config or create swiitx.toml".to_owned()
            })?,
    };

    let options = config.library.scan_options();
    let preferred_languages = config.system.preferred_languages.clone();
    let prod_keys = config.system.keys.join("prod.keys");
    let title_keys_path = config.system.keys.join("title.keys");
    let title_keys = title_keys_path.is_file().then_some(title_keys_path);
    let mut keys = NcaKeySet::from_files(&prod_keys, title_keys.as_deref())
        .map_err(|error| error.to_string())?;

    if let Some(path) = arguments.title_path {
        let inspection = TitleInspector::inspect_with_key_set_and_options(path, &mut keys, options)
            .map_err(|error| error.to_string())?;
        return Ok(CliOutput::Inspection {
            inspection: Box::new(inspection),
            preferred_languages,
        });
    }

    if config.library.paths.is_empty() {
        return Err("the configuration has no library paths".to_owned());
    }

    let mut catalog = TitleCatalog::new();
    for path in config.library.paths {
        let discovered =
            TitleCatalog::scan_directory_with_key_set_and_options(path, &mut keys, options)
                .map_err(|error| error.to_string())?;
        for package in discovered.packages() {
            catalog.add(package.clone());
        }
    }
    let titles = TitleResolver::resolve_all(&catalog).map_err(|error| error.to_string())?;
    cache_preferred_icons(&titles, &preferred_languages)?;
    Ok(CliOutput::LibrarySummary {
        titles,
        preferred_languages,
    })
}

fn cache_preferred_icons(
    titles: &[ResolvedTitle],
    preferred_languages: &[NacpLanguage],
) -> Result<(), String> {
    let cache_root = PathBuf::from("cache");
    for title in titles {
        let Some(control) = title.control_metadata() else {
            continue;
        };
        let Some(icon) = control.preferred_icon(preferred_languages) else {
            continue;
        };
        let directory_name = control
            .nacp
            .preferred_title(preferred_languages)
            .map(|(_, metadata)| sanitize_directory_name(&metadata.name))
            .filter(|name| !name.is_empty())
            .unwrap_or_else(|| title.application_id.to_string());
        let directory = cache_root.join(directory_name);
        fs::create_dir_all(&directory).map_err(|source| {
            format!(
                "cannot create icon cache directory {}: {source}",
                directory.display()
            )
        })?;
        let output_path = directory.join(format!("{}.jpg", icon.filename));
        let bytes = icon.bytes().map_err(|error| {
            format!(
                "cannot read icon {} for {}: {error}",
                icon.filename, title.application_id
            )
        })?;
        fs::write(&output_path, bytes).map_err(|source| {
            format!(
                "cannot write cached icon {}: {source}",
                output_path.display()
            )
        })?;
    }
    Ok(())
}

fn sanitize_directory_name(name: &str) -> String {
    sanitize_with_options(
        name,
        Options {
            windows: true,
            truncate: false,
            replacement: "_",
        },
    )
}

fn print_library_summary(titles: &[ResolvedTitle], preferred_languages: &[NacpLanguage]) {
    println!("Titles: {}", titles.len());

    for (index, title) in titles.iter().enumerate() {
        println!();
        let preferred_title = title
            .control_metadata()
            .and_then(|control| control.nacp.preferred_title(preferred_languages));
        match preferred_title {
            Some((language, metadata)) => {
                println!("Title {}: {}", index + 1, metadata.name);
                println!("  Publisher: {}", metadata.publisher);
                println!("  Language: {language}");
            }
            None => println!("Title {}: {}", index + 1, title.application_id),
        }
        println!("  Application ID: {}", title.application_id);
        println!(
            "  Base: {} version {}",
            title.base.title_id, title.base.version
        );
        match &title.patch {
            Some(patch) => println!("  Patch: {} version {}", patch.title_id, patch.version),
            None => println!("  Patch: none"),
        }
        let effective_version = title
            .patch
            .as_ref()
            .map_or(title.base.version, |patch| patch.version);
        println!("  Effective version: {effective_version}");
        println!("  DLC: {}", title.add_ons.len());
        for add_on in &title.add_ons {
            match add_on.required_application_version() {
                Some(required_version) => println!(
                    "    {} version {} (requires application version {})",
                    add_on.title_id, add_on.version, required_version
                ),
                None => println!("    {} version {}", add_on.title_id, add_on.version),
            }
        }

        if let Some(control) = title.control_metadata() {
            println!("  Display version: {}", control.nacp.display_version);
            let languages = control
                .supported_languages()
                .iter()
                .map(|language| language.to_string())
                .collect::<Vec<_>>();
            println!(
                "  Supported languages: {}",
                if languages.is_empty() {
                    "none".to_owned()
                } else {
                    languages.join(", ")
                }
            );
            if let Some(icon) = control.preferred_icon(preferred_languages) {
                println!("  Preferred icon: {} ({})", icon.filename, icon.language);
            }
        }
    }
}

fn print_inspection(inspection: &TitleInspection, preferred_languages: &[NacpLanguage]) {
    println!("Title: {}", inspection.path.display());
    if let Some(standalone) = &inspection.standalone_ncz {
        println!("Format: standalone NCZ");
        println!("Stored size: {}", format_size(standalone.stored_size));
        print_ncz(&standalone.ncz, "  ");
        if let Some(nca) = &standalone.nca {
            print_nca(nca);
        }
        if let Some(warning) = &standalone.nca_warning {
            println!("NCA inspection warning: {warning}");
        }
        return;
    }
    println!("Packages: {}", inspection.packages.len());

    for (index, package) in inspection.packages.iter().enumerate() {
        println!();
        println!("Package {}: {}", index + 1, package.path.display());
        println!("  Format: {}", package.format);
        println!("  Size: {}", format_size(package.size));
        println!("  Content data offset: {:#X}", package.data_offset);
        if let Some(xci) = &package.xci {
            println!("  XCI header:");
            println!("    Package ID: {:016X}", xci.header.package_id);
            println!(
                "    Root HFS0: offset {:#X}, header size {:#X}, hash {}",
                xci.header.root_hfs0_offset,
                xci.header.root_hfs0_header_size,
                match xci.root_header_hash_valid {
                    Some(true) => "valid",
                    Some(false) => "invalid",
                    None => "not applicable",
                }
            );
            println!(
                "    Card size code: {:#04X}; header version: {:#04X}; flags: {:#04X}",
                xci.header.card_size_code, xci.header.header_version, xci.header.flags
            );
            println!("  HFS0 partitions: {}", xci.partitions.len());
            for partition in &xci.partitions {
                println!(
                    "    {} ({:?}): offset {:#X}, size {}, hashed {}, hash {}",
                    partition.name,
                    partition.kind,
                    partition.offset,
                    format_size(partition.size),
                    format_size(partition.hashed_region_size),
                    match partition.hash_valid {
                        Some(true) => "valid",
                        Some(false) => "invalid",
                        None => "not advertised",
                    }
                );
                println!(
                    "      data offset {:#X}; entries {}",
                    partition.data_offset,
                    partition.entries.len()
                );
            }
        }
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
            if let Some(stored_name) = &entry.stored_name {
                println!(
                    "      Stored as: {} ({} bytes)",
                    stored_name,
                    entry.stored_size.unwrap_or(entry.size)
                );
            }
            if let Some(ncz) = &entry.ncz {
                print_ncz(ncz, "      ");
            }
            if let (Some(hashed_size), Some(valid)) = (entry.hashed_region_size, entry.hash_valid) {
                println!(
                    "      HFS0 hashed prefix: {} ({})",
                    format_size(hashed_size),
                    if valid { "valid" } else { "invalid" }
                );
            }
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

        for (metadata_index, metadata) in package.canonical_content_metas.iter().enumerate() {
            println!(
                "  Canonical content metadata {} of {}:",
                metadata_index + 1,
                package.canonical_content_metas.len()
            );
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

        for (control_index, control) in package.control_metadatas.iter().enumerate() {
            println!(
                "  Control metadata {} of {}:",
                control_index + 1,
                package.control_metadatas.len()
            );
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

fn print_ncz(ncz: &swiitx_loader_title::NczInspection, indent: &str) {
    println!("{indent}NCZ compression: {:?}", ncz.compression);
    println!(
        "{indent}Logical NCA size: {}",
        format_size(ncz.logical_size)
    );
    if let (Some(block_size), Some(block_count)) = (ncz.block_size, ncz.block_count) {
        println!(
            "{indent}Blocks: {block_count}, block size {}",
            format_size(block_size)
        );
    }
    println!("{indent}Reconstruction sections: {}", ncz.sections.len());
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
        assert_eq!(arguments.title_path, None);
    }

    #[test]
    fn parses_one_title_path_with_config_in_either_order() {
        for values in [
            &["--config", "custom.toml", "title.nsp"][..],
            &["title.nsp", "--config", "custom.toml"][..],
        ] {
            let arguments = parse_arguments(os_arguments(values)).unwrap().unwrap();

            assert_eq!(arguments.config_path, Some(PathBuf::from("custom.toml")));
            assert_eq!(arguments.title_path, Some(PathBuf::from("title.nsp")));
        }
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
        assert_eq!(arguments.title_path, None);
    }

    #[test]
    fn rejects_unsupported_options_and_multiple_title_paths() {
        for arguments in [
            &["--keys-dir", "keys"][..],
            &["--"][..],
            &["--unknown"][..],
            &["first.nsp", "second.nsp"][..],
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

    #[test]
    fn sanitizes_title_names_for_cache_directories() {
        assert_eq!(sanitize_directory_name("Mario: A/B?"), "Mario_ A_B_");
        assert_eq!(
            sanitize_directory_name("Mario vs. Donkey Kong"),
            "Mario vs. Donkey Kong"
        );
        assert_eq!(sanitize_directory_name("CON.txt"), "_");
        assert_eq!(sanitize_directory_name("LPT9"), "_");
        assert_eq!(
            sanitize_directory_name("Pokémon™ (Deluxe) #1!."),
            "Pokémon™ (Deluxe) #1!_"
        );
        assert_eq!(sanitize_directory_name("line\nfeed"), "line_feed");
    }
}
