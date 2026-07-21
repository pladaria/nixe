use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::ffi::OsString;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use sanitize_filename::{Options, sanitize_with_options};
use swiitx_config::SwiitxConfig;
use swiitx_loader_content::{NacpLoader, NcaFormatVersion, NcaKeySet};
use swiitx_loader_executable::NroLoader;
use swiitx_loader_storage::{FileStorage, FormatLoader, StorageRef};
use swiitx_loader_title::{
    CnmtExtendedHeader, ControlMetadata, EntryKind, NacpLanguage, NcaInspection, PackageInspection,
    ResolvedTitle, TitleCatalog, TitleError, TitleInspection, TitleInspector, TitleResolver,
};

const MAX_CACHED_ICON_SIZE: u64 = 16 * 1024 * 1024;

struct CliArguments {
    config_path: Option<PathBuf>,
    paths: Vec<PathBuf>,
}

struct CliOutput {
    file_inspections: Vec<TitleInspection>,
    resolved_titles: Vec<ResolvedTitle>,
    directory_nros: Vec<swiitx_loader_title::NroInspection>,
    inspected_directories: bool,
    preferred_languages: Vec<NacpLanguage>,
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
        Ok(output) => {
            let mut wrote_output = false;
            if output.inspected_directories {
                print_resolved_titles(&output.resolved_titles, &output.preferred_languages);
                println!();
                print_nros(&output.directory_nros);
                wrote_output = true;
            }
            for inspection in &output.file_inspections {
                if wrote_output {
                    println!();
                }
                print_inspection(inspection, &output.preferred_languages);
                wrote_output = true;
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
        "Usage: {} [--config <file>] [<path> ...]\n\n\
         Resolve titles found below supplied directories; inspect supplied files in\n\
         detail. Without paths, process library.paths under the same rules.\n\
         Pass --config to select a TOML file explicitly. Otherwise the CLI uses\n\
         SWIITX_CONFIG, ./swiitx.toml, or the platform user configuration.",
        program.to_string_lossy()
    );
}

fn parse_arguments(
    arguments: impl Iterator<Item = OsString>,
) -> Result<Option<CliArguments>, String> {
    let mut config_path = None;
    let mut paths = Vec::new();
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
        paths.push(PathBuf::from(argument));
    }

    Ok(Some(CliArguments { config_path, paths }))
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

    let paths = select_inspection_paths(arguments.paths, config.library.paths);
    if paths.is_empty() {
        return Err("no inspection paths were supplied".to_owned());
    }
    let mut catalog = TitleCatalog::new();
    let mut file_inspections = Vec::new();
    let mut directory_nros = Vec::new();
    let mut seen_nros = BTreeSet::new();
    let mut inspected_directories = false;
    for path in paths {
        let metadata = fs::metadata(&path)
            .map_err(|error| format!("cannot access {}: {error}", path.display()))?;
        if metadata.is_dir() {
            inspected_directories = true;
            let discovered_nros = TitleInspector::inspect_nros_with_options(&path, options)
                .map_err(|error| error.to_string())?;
            let directory_has_nros = !discovered_nros.is_empty();
            for nro in discovered_nros {
                let identity = fs::canonicalize(&nro.path).map_err(|error| {
                    format!("cannot resolve NRO path {}: {error}", nro.path.display())
                })?;
                if seen_nros.insert(identity) {
                    directory_nros.push(nro);
                }
            }
            match TitleCatalog::scan_directory_with_key_set_and_options(&path, &mut keys, options) {
                Ok(discovered) => {
                    for package in discovered.packages() {
                        catalog.add(package.clone());
                    }
                }
                Err(TitleError::NoSupportedPackages { .. }) if directory_has_nros => {}
                Err(error) => return Err(error.to_string()),
            }
        } else if metadata.is_file() {
            file_inspections.push(
                TitleInspector::inspect_with_key_set_and_options(&path, &mut keys, options)
                    .map_err(|error| error.to_string())?,
            );
        } else {
            return Err(format!("unsupported path type: {}", path.display()));
        }
    }
    let resolved_titles =
        TitleResolver::resolve_all(&catalog).map_err(|error| error.to_string())?;
    let cache_root = PathBuf::from("cache");
    cache_resolved_title_icons(&resolved_titles, &preferred_languages, &cache_root)?;
    cache_inspection_icons(&file_inspections, &preferred_languages, &cache_root)?;
    for nro in &directory_nros {
        cache_nro_icon(&nro.path, &preferred_languages, &cache_root)?;
    }
    Ok(CliOutput {
        file_inspections,
        resolved_titles,
        directory_nros,
        inspected_directories,
        preferred_languages,
    })
}

fn select_inspection_paths(explicit: Vec<PathBuf>, configured: Vec<PathBuf>) -> Vec<PathBuf> {
    if explicit.is_empty() {
        configured
    } else {
        explicit
    }
}

fn cache_resolved_title_icons(
    titles: &[ResolvedTitle],
    preferred_languages: &[NacpLanguage],
    cache_root: &std::path::Path,
) -> Result<(), String> {
    for title in titles {
        let Some(control) = title.control_metadata() else {
            continue;
        };
        let Some(icon) = control.preferred_icon(preferred_languages) else {
            continue;
        };
        let fallback = title.application_id.to_string();
        let title_name = control
            .nacp
            .preferred_title(preferred_languages)
            .map(|(_, title)| title.name.as_str());
        let directory = cache_directory(cache_root, title_name, &fallback)?;
        let bytes = icon.bytes().map_err(|error| {
            format!(
                "cannot read effective icon {} for {}: {error}",
                icon.filename, title.application_id
            )
        })?;
        write_cached_icon(
            &directory.join(cached_icon_filename(&icon.filename)),
            &bytes,
        )?;
    }
    Ok(())
}

fn cache_inspection_icons(
    inspections: &[TitleInspection],
    preferred_languages: &[NacpLanguage],
    cache_root: &std::path::Path,
) -> Result<(), String> {
    for inspection in inspections {
        for package in &inspection.packages {
            if package.control_metadatas.is_empty() {
                if let Some(control) = &package.control_metadata {
                    cache_control_icon(package, control, preferred_languages, cache_root)?;
                }
            } else {
                for control in &package.control_metadatas {
                    cache_control_icon(package, control, preferred_languages, cache_root)?;
                }
            }
        }
        for nro in &inspection.nros {
            cache_nro_icon(&nro.path, preferred_languages, cache_root)?;
        }
    }
    Ok(())
}

fn cache_control_icon(
    package: &PackageInspection,
    control: &ControlMetadata,
    preferred_languages: &[NacpLanguage],
    cache_root: &std::path::Path,
) -> Result<(), String> {
    let Some(icon) = control.preferred_icon(preferred_languages) else {
        return Ok(());
    };
    let fallback = package
        .canonical_content_metas
        .first()
        .map(|metadata| format!("{:016X}", metadata.title_id))
        .unwrap_or_else(|| path_fallback_name(&package.path));
    let title_name = control
        .nacp
        .preferred_title(preferred_languages)
        .map(|(_, title)| title.name.as_str());
    let directory = cache_directory(cache_root, title_name, &fallback)?;
    let filename = cached_icon_filename(&icon.filename);
    let bytes = icon.bytes().map_err(|error| {
        format!(
            "cannot read icon {} from {}: {error}",
            icon.filename,
            package.path.display()
        )
    })?;
    write_cached_icon(&directory.join(filename), &bytes)
}

fn cache_nro_icon(
    path: &std::path::Path,
    preferred_languages: &[NacpLanguage],
    cache_root: &std::path::Path,
) -> Result<(), String> {
    let storage = FileStorage::open(path).map_err(|error| {
        format!(
            "cannot open NRO {} for icon caching: {error}",
            path.display()
        )
    })?;
    let image = NroLoader::load(Arc::new(storage)).map_err(|error| {
        format!(
            "cannot load NRO {} for icon caching: {error}",
            path.display()
        )
    })?;
    let Some(assets) = image.assets() else {
        return Ok(());
    };
    let Some(icon) = assets.icon() else {
        return Ok(());
    };
    let title_name = assets
        .nacp()
        .map(|storage| NacpLoader::load(storage.clone()))
        .transpose()
        .map_err(|error| format!("cannot read NRO NACP from {}: {error}", path.display()))?
        .and_then(|nacp| {
            nacp.preferred_title(preferred_languages)
                .map(|(_, title)| title.name.clone())
        });
    let fallback = path_fallback_name(path);
    let directory = cache_directory(cache_root, title_name.as_deref(), &fallback)?;
    let bytes = read_nro_icon(icon, path)?;
    write_cached_icon(&directory.join("icon.jpg"), &bytes)
}

fn cache_directory(
    cache_root: &std::path::Path,
    title_name: Option<&str>,
    fallback: &str,
) -> Result<PathBuf, String> {
    let directory_name = title_name
        .map(sanitize_directory_name)
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| sanitize_directory_name(fallback));
    let directory = cache_root.join(directory_name);
    fs::create_dir_all(&directory).map_err(|error| {
        format!(
            "cannot create icon cache directory {}: {error}",
            directory.display()
        )
    })?;
    Ok(directory)
}

fn read_nro_icon(icon: &StorageRef, path: &std::path::Path) -> Result<Vec<u8>, String> {
    let size = icon
        .len()
        .map_err(|error| format!("cannot read NRO icon size from {}: {error}", path.display()))?;
    if !(3..=MAX_CACHED_ICON_SIZE).contains(&size) {
        return Err(format!(
            "NRO icon in {} has invalid size {size}",
            path.display()
        ));
    }
    let mut bytes =
        vec![
            0_u8;
            usize::try_from(size)
                .map_err(|_| format!("NRO icon in {} is too large to cache", path.display()))?
        ];
    icon.read_at(0, &mut bytes)
        .map_err(|error| format!("cannot read NRO icon from {}: {error}", path.display()))?;
    if bytes[..3] != [0xff, 0xd8, 0xff] {
        return Err(format!(
            "NRO icon in {} is not a JPEG image",
            path.display()
        ));
    }
    Ok(bytes)
}

fn write_cached_icon(path: &std::path::Path, bytes: &[u8]) -> Result<(), String> {
    fs::write(path, bytes)
        .map_err(|error| format!("cannot write cached icon {}: {error}", path.display()))
}

fn cached_icon_filename(filename: &str) -> String {
    let stem = std::path::Path::new(filename)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("icon");
    let stem = sanitize_directory_name(stem);
    format!("{}.jpg", if stem.is_empty() { "icon" } else { &stem })
}

fn path_fallback_name(path: &std::path::Path) -> String {
    path.file_stem()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .unwrap_or("unknown-title")
        .to_owned()
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

fn print_resolved_titles(titles: &[ResolvedTitle], preferred_languages: &[NacpLanguage]) {
    println!("Resolved titles: {}", titles.len());
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
            Some(patch) => println!(
                "  Selected patch: {} version {}",
                patch.title_id, patch.version
            ),
            None => println!("  Selected patch: none"),
        }
        let effective_version = title
            .patch
            .as_ref()
            .map_or(title.base.version, |patch| patch.version);
        println!("  Effective version: {effective_version}");
        println!("  Selected DLC: {}", title.add_ons.len());
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
                println!("  Effective icon: {} ({})", icon.filename, icon.language);
            }
        }
    }
}

fn print_inspection(inspection: &TitleInspection, preferred_languages: &[NacpLanguage]) {
    println!("Inspection path: {}", inspection.path.display());
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
    print_nros(&inspection.nros);
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

fn print_nros(nros: &[swiitx_loader_title::NroInspection]) {
    println!("NRO executables: {}", nros.len());
    for (index, nro) in nros.iter().enumerate() {
        println!();
        println!("NRO {}: {}", index + 1, nro.path.display());
        println!("  Size: {}", format_size(nro.size));
        println!("  Executable size: {}", format_size(nro.executable_size));
        println!("  Version: {}; flags: {:#010X}", nro.version, nro.flags);
        println!("  Entry offset: {:#X}", nro.entry_offset);
        println!("  Module ID: {}", format_hex(&nro.module_id));
        println!("  MOD0 offset: {:#X}", nro.module_header_offset);
        println!("  DSO handle offset: {:#X}", nro.dso_handle_offset);
        print_nro_range("Embedded API info", nro.embedded_api_info);
        print_nro_range("Dynamic string table", nro.dynamic_string_table);
        print_nro_range("Dynamic symbol table", nro.dynamic_symbol_table);
        println!("  Segments: {}", nro.segments.len());
        for segment in &nro.segments {
            let permissions = [
                if segment.readable { 'r' } else { '-' },
                if segment.writable { 'w' } else { '-' },
                if segment.executable { 'x' } else { '-' },
            ];
            println!(
                "    {:<9} offset {:#X}, file {}, memory {}, mapping {}, {}",
                segment.kind,
                segment.memory_offset,
                format_size(segment.file_size),
                format_size(segment.memory_size),
                format_size(segment.mapping_size),
                permissions.iter().collect::<String>()
            );
        }
        match &nro.assets {
            Some(assets) => {
                println!("  ASET version: {}", assets.version);
                print_optional_asset("Icon", assets.icon_size);
                print_optional_asset("NACP", assets.nacp_size);
                print_optional_asset("RomFS", assets.romfs_size);
            }
            None => println!("  ASET assets: none"),
        }
    }
}

fn print_nro_range(name: &str, range: (u64, u64)) {
    println!(
        "  {name}: offset {:#X}, size {}",
        range.0,
        format_size(range.1)
    );
}

fn print_optional_asset(name: &str, size: Option<u64>) {
    match size {
        Some(size) => println!("    {name}: {}", format_size(size)),
        None => println!("    {name}: none"),
    }
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

    fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
        bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }

    fn synthetic_nro_with_icon(title: &str, icon: &[u8]) -> Vec<u8> {
        const EXECUTABLE_SIZE: usize = 0x2800;
        const ICON_OFFSET: usize = 0x100;
        const NACP_OFFSET: usize = 0x200;
        const NACP_SIZE: usize = 0x4000;

        let mut bytes = vec![0_u8; EXECUTABLE_SIZE + NACP_OFFSET + NACP_SIZE];
        bytes[0x10..0x14].copy_from_slice(b"NRO0");
        put_u32(&mut bytes, 0x04, 0x100);
        put_u32(&mut bytes, 0x18, EXECUTABLE_SIZE as u32);
        put_u32(&mut bytes, 0x20, 0);
        put_u32(&mut bytes, 0x24, 0x1000);
        put_u32(&mut bytes, 0x28, 0x1000);
        put_u32(&mut bytes, 0x2c, 0x1000);
        put_u32(&mut bytes, 0x30, 0x2000);
        put_u32(&mut bytes, 0x34, 0x800);
        put_u32(&mut bytes, 0x38, 0x800);

        let asset = EXECUTABLE_SIZE;
        bytes[asset..asset + 4].copy_from_slice(b"ASET");
        put_u64(&mut bytes, asset + 0x08, ICON_OFFSET as u64);
        put_u64(&mut bytes, asset + 0x10, icon.len() as u64);
        put_u64(&mut bytes, asset + 0x18, NACP_OFFSET as u64);
        put_u64(&mut bytes, asset + 0x20, NACP_SIZE as u64);
        bytes[asset + ICON_OFFSET..asset + ICON_OFFSET + icon.len()].copy_from_slice(icon);
        bytes[asset + NACP_OFFSET..asset + NACP_OFFSET + title.len()]
            .copy_from_slice(title.as_bytes());
        bytes
    }

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
        assert!(arguments.paths.is_empty());
    }

    #[test]
    fn parses_paths_with_config_in_any_order() {
        for values in [
            &["--config", "custom.toml", "first.nsp", "homebrew.nro"][..],
            &["first.nsp", "--config", "custom.toml", "homebrew.nro"][..],
        ] {
            let arguments = parse_arguments(os_arguments(values)).unwrap().unwrap();

            assert_eq!(arguments.config_path, Some(PathBuf::from("custom.toml")));
            assert_eq!(
                arguments.paths,
                vec![PathBuf::from("first.nsp"), PathBuf::from("homebrew.nro")]
            );
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
        assert!(arguments.paths.is_empty());
    }

    #[test]
    fn explicit_paths_override_configured_paths_and_empty_arguments_use_them() {
        let configured = vec![PathBuf::from("configured-a"), PathBuf::from("configured-b")];
        assert_eq!(
            select_inspection_paths(Vec::new(), configured.clone()),
            configured
        );
        assert_eq!(
            select_inspection_paths(
                vec![PathBuf::from("explicit-a"), PathBuf::from("explicit-b")],
                vec![PathBuf::from("configured")],
            ),
            vec![PathBuf::from("explicit-a"), PathBuf::from("explicit-b")]
        );
    }

    #[test]
    fn rejects_unsupported_options() {
        for arguments in [&["--keys-dir", "keys"][..], &["--"][..], &["--unknown"][..]] {
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
    fn uses_the_original_directory_sanitization_and_a_jpeg_output_name() {
        assert_eq!(sanitize_directory_name("Mario: A/B?"), "Mario_ A_B_");
        assert_eq!(sanitize_directory_name("CON.txt"), "_");
        assert_eq!(sanitize_directory_name("line\nfeed"), "line_feed");
        assert_eq!(
            cached_icon_filename("icon_AmericanEnglish.dat"),
            "icon_AmericanEnglish.jpg"
        );
    }

    #[test]
    fn caches_an_nro_aset_icon_under_its_preferred_nacp_title() {
        let directory = tempfile::tempdir().unwrap();
        let nro_path = directory.path().join("fallback.nro");
        let cache_root = directory.path().join("cache");
        let icon = [0xff, 0xd8, 0xff, 0xd9];
        fs::write(&nro_path, synthetic_nro_with_icon("Homebrew: Demo", &icon)).unwrap();

        cache_nro_icon(&nro_path, &[NacpLanguage::AmericanEnglish], &cache_root).unwrap();

        assert_eq!(
            fs::read(cache_root.join("Homebrew_ Demo").join("icon.jpg")).unwrap(),
            icon
        );
    }
}
