use std::collections::BTreeMap;
use std::env;
use std::ffi::OsString;
use std::process::ExitCode;

use swiitx_loader_title::{EntryKind, TitleInspection, TitleInspector};

fn main() -> ExitCode {
    let mut arguments = env::args_os();
    let program = arguments
        .next()
        .unwrap_or_else(|| OsString::from("swiitx-cli"));
    let Some(path) = arguments.next() else {
        print_usage(&program);
        return ExitCode::from(2);
    };
    if path == "-h" || path == "--help" {
        print_usage(&program);
        return ExitCode::SUCCESS;
    }
    if arguments.next().is_some() {
        eprintln!("error: expected exactly one title path");
        print_usage(&program);
        return ExitCode::from(2);
    }

    match TitleInspector::inspect(path) {
        Ok(inspection) => {
            print_inspection(&inspection);
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
        "Usage: {} <title-path>\n\nInspect an NSP file or a directory containing title packages.",
        program.to_string_lossy()
    );
}

fn print_inspection(inspection: &TitleInspection) {
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
                "    {:<12} {:>18}  offset {:#X}  {}",
                entry.kind,
                format_size(entry.size),
                entry.offset,
                entry.name
            );
        }

        println!("  Entry summary:");
        for (kind, count) in counts {
            println!("    {kind}: {count}");
        }

        if let Some(metadata) = &package.content_meta {
            println!("  Auxiliary content metadata:");
            println!("    Type: {}", metadata.content_type);
            println!("    Title ID: {:016X}", metadata.title_id);
            println!("    Version: {}", metadata.version);
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
                println!("    Required system version (raw): {version}");
            }
            if let Some(version) = metadata.required_application_version {
                println!("    Required application version: {version}");
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
    println!("Note: auxiliary XML is informational; canonical CNMT validation is not implemented.");
}

fn format_size(bytes: u64) -> String {
    const KIB: u64 = 1024;
    const MIB: u64 = KIB * 1024;
    const GIB: u64 = MIB * 1024;

    if bytes >= GIB {
        format!("{bytes} bytes ({:.2} GiB)", bytes as f64 / GIB as f64)
    } else if bytes >= MIB {
        format!("{bytes} bytes ({:.2} MiB)", bytes as f64 / MIB as f64)
    } else if bytes >= KIB {
        format!("{bytes} bytes ({:.2} KiB)", bytes as f64 / KIB as f64)
    } else {
        format!("{bytes} bytes")
    }
}
