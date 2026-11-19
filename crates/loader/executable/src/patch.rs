//! ExeFS patches address a decompressed NSO with its original 0x100-byte header.
//! https://github.com/3096/ipswitch/blob/master/example/README.md
//! https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/core/file_sys/patch_manager.cpp
use crate::{ExecutableFormat, PreparedModule};
use nixe_loader_storage::LoadError;
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Debug)]
pub struct NsoPatch {
    path: PathBuf,
    pub build_id: [u8; 32],
    records: Vec<Record>,
}

#[derive(Debug)]
struct Record {
    offset: u32,
    data: Data,
    print: bool,
}

#[derive(Debug)]
enum Data {
    Bytes(Vec<u8>),
    Fill(u16, u8),
}

impl NsoPatch {
    /// Match the full module ID, padding shorter IDs on the right as the NSO header does.
    pub fn load(path: &Path, modules: &[[u8; 32]]) -> Result<Option<Self>, LoadError> {
        let invalid = |msg: String| LoadError::invalid("ExeFS patch", msg);
        let binary = path
            .extension()
            .is_some_and(|s| s.eq_ignore_ascii_case("ips"));
        let text = if binary {
            None
        } else {
            Some(fs::read_to_string(path).map_err(nixe_loader_storage::StorageError::from)?)
        };
        let id = if let Some(text) = text.as_deref() {
            text.trim_start_matches('\u{feff}')
                .lines()
                .map(uncomment)
                .take_while(|line| !line.eq_ignore_ascii_case("@stop"))
                .find_map(|line| {
                    let (tag, id) = line.split_once('-')?;
                    tag.eq_ignore_ascii_case("@nsobid").then_some(id)
                })
                .ok_or_else(|| invalid("missing @nsobid header".into()))?
        } else {
            path.file_stem()
                .and_then(|s| s.to_str())
                .and_then(|s| s.split('.').next())
                .ok_or_else(|| invalid("missing build ID".into()))?
        };
        if id.is_empty() || id.len() > 64 || !id.bytes().all(|c| c.is_ascii_hexdigit()) {
            return Err(invalid("invalid NSO build ID".into()));
        }
        let mut build_id = [0; 32];
        for (i, byte) in format!("{id:0<64}").as_bytes().chunks_exact(2).enumerate() {
            build_id[i] = (hex_digit(byte[0]) << 4) | hex_digit(byte[1]);
        }
        if !modules.contains(&build_id) {
            log::info!(
                "skipping ExeFS patch {}: no loaded module has build ID {id}",
                path.display()
            );
            return Ok(None);
        }
        let records = if let Some(text) = text {
            parse_text(&text).map_err(invalid)?
        } else {
            let data = fs::read(path).map_err(nixe_loader_storage::StorageError::from)?;
            parse_ips(&data).map_err(|e| invalid(e.into()))?
        };
        Ok(Some(Self {
            path: path.to_owned(),
            build_id,
            records,
        }))
    }

    pub fn apply(&self, module: &mut PreparedModule, name: &str) -> Result<(), LoadError> {
        if module.format() != ExecutableFormat::Nso || module.module_id() != &self.build_id {
            return Err(LoadError::invalid(
                "ExeFS patch",
                "module build ID mismatch",
            ));
        }
        let mut writes = 0;
        for record in &self.records {
            let len = match &record.data {
                Data::Bytes(b) => b.len(),
                Data::Fill(n, _) => usize::from(*n),
            };
            // The loader preserves the header. Map the rest directly into the prepared segments.
            let address = u64::from(record.offset);
            let mut used = (crate::nso::HEADER_SIZE.saturating_sub(address) as usize).min(len);
            let mut offset = address.saturating_sub(crate::nso::HEADER_SIZE);
            if used == len {
                continue;
            }
            if record.print {
                match &record.data {
                    Data::Bytes(bytes) => {
                        log::debug!("{name}+{offset:#x} <- {:02X?}", &bytes[used..])
                    }
                    Data::Fill(_, value) => {
                        log::debug!("{name}+{offset:#x} <- {value:02X} x {}", len - used)
                    }
                }
            }
            while used < len {
                let dest = module.patch_region(offset).ok_or_else(|| {
                    LoadError::invalid(
                        "ExeFS patch",
                        format!(
                            "{}: {name}+{offset:#x} outside mapped image",
                            self.path.display()
                        ),
                    )
                })?;
                let n = dest.len().min(len - used);
                match &record.data {
                    Data::Bytes(bytes) => dest[..n].copy_from_slice(&bytes[used..used + n]),
                    Data::Fill(_, value) => dest[..n].fill(*value),
                }
                offset += n as u64;
                used += n;
            }
            writes += 1;
        }
        log::info!(
            "applied ExeFS patch {} to {name}: {writes} write(s)",
            self.path.display()
        );
        Ok(())
    }
}

// Callers have validated ASCII hexadecimal digits.
fn hex_digit(c: u8) -> u8 {
    if c.is_ascii_digit() {
        c - b'0'
    } else {
        c.to_ascii_lowercase() - b'a' + 10
    }
}

fn hex_bytes(s: &str) -> Option<Vec<u8>> {
    if !s.len().is_multiple_of(2) || !s.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    Some(
        s.as_bytes()
            .chunks_exact(2)
            .map(|p| hex_digit(p[0]) << 4 | hex_digit(p[1]))
            .collect(),
    )
}

fn uncomment(line: &str) -> &str {
    let mut quote = None;
    let mut chars = line.char_indices();
    while let Some((i, ch)) = chars.next() {
        match ch {
            '\\' if quote.is_some() => {
                chars.next();
            }
            '/' | '#' if quote.is_none() => return line[..i].trim(),
            '"' | '\'' if quote == Some(ch) => quote = None,
            '"' | '\'' if quote.is_none() => quote = Some(ch),
            _ => {}
        }
    }
    line.trim()
}

fn literal(s: &str) -> Option<Vec<u8>> {
    let Some(quote) = s.chars().next().filter(|&ch| ch == '"' || ch == '\'') else {
        return hex_bytes(s).filter(|b| !b.is_empty());
    };
    let mut chars = s[1..].chars();
    let mut out = String::new();
    while let Some(ch) = chars.next() {
        if ch == quote {
            return chars.as_str().is_empty().then(|| out.into_bytes());
        }
        out.push(if ch == '\\' {
            match chars.next()? {
                'n' => '\n',
                'r' => '\r',
                't' => '\t',
                'a' => '\x07',
                'b' => '\x08',
                'e' => '\x1b',
                'f' => '\x0c',
                'v' => '\x0b',
                c => c,
            }
        } else {
            ch
        });
    }
    None
}

fn shift_value(s: &str) -> Option<i64> {
    let unsigned = s
        .strip_prefix('-')
        .or_else(|| s.strip_prefix('+'))
        .unwrap_or(s);
    if unsigned.starts_with(['-', '+']) {
        return None;
    }
    let n = if let Some(digits) = unsigned
        .strip_prefix("0x")
        .or_else(|| unsigned.strip_prefix("0X"))
    {
        i64::from_str_radix(digits, 16).ok()?
    } else {
        unsigned.parse().ok()?
    };
    Some(if s.starts_with('-') { -n } else { n })
}

fn parse_text(text: &str) -> Result<Vec<Record>, String> {
    let mut records = Vec::new();
    let mut enabled = false;
    let mut print = false;
    let mut shift = 0i64;
    for (i, line) in text.lines().enumerate() {
        let line = uncomment(line);
        let error = |s| format!("line {}: {s}", i + 1);
        let (head, rest) = line.split_once(char::is_whitespace).unwrap_or((line, ""));
        let rest = rest.trim();
        if head.eq_ignore_ascii_case("@stop") {
            break;
        }
        if head.eq_ignore_ascii_case("@enabled") || head.eq_ignore_ascii_case("@disabled") {
            enabled = head.eq_ignore_ascii_case("@enabled");
        } else if head.eq_ignore_ascii_case("@flag") {
            let mut args = rest.split_whitespace();
            match args.next().unwrap_or("").to_ascii_lowercase().as_str() {
                "offset_shift" => {
                    shift = args
                        .next()
                        .and_then(shift_value)
                        .ok_or_else(|| error("invalid offset_shift"))?
                }
                "print_values" => print = true,
                _ => return Err(error("unsupported IPSwitch flag")),
            }
            if args.next().is_some() {
                return Err(error("unexpected flag argument"));
            }
        } else if enabled
            && !head.is_empty()
            && (head.as_bytes()[0].is_ascii_digit() || head.bytes().all(|c| c.is_ascii_hexdigit()))
        {
            let digits = head
                .strip_prefix("0x")
                .or_else(|| head.strip_prefix("0X"))
                .unwrap_or(head);
            let offset = u32::from_str_radix(digits, 16)
                .ok()
                .and_then(|v| i64::from(v).checked_add(shift))
                .and_then(|v| u32::try_from(v).ok())
                .ok_or_else(|| error("offset out of range"))?;
            let bytes =
                literal(rest).ok_or_else(|| error("invalid replacement bytes or string"))?;
            records.push(Record {
                offset,
                data: Data::Bytes(bytes),
                print,
            });
        }
    }
    Ok(records)
}

fn parse_ips(data: &[u8]) -> Result<Vec<Record>, &'static str> {
    let (mut src, width, end): (&[u8], usize, &[u8]) =
        if let Some(rest) = data.strip_prefix(b"PATCH") {
            (rest, 3, b"EOF")
        } else if let Some(rest) = data.strip_prefix(b"IPS32") {
            (rest, 4, b"EEOF")
        } else {
            return Err("invalid IPS signature");
        };
    let mut records = Vec::new();
    loop {
        let addr = src.split_off(..width).ok_or("missing IPS terminator")?;
        if addr == end {
            break;
        }
        let offset = addr.iter().fold(0u32, |v, b| (v << 8) | u32::from(*b));
        let size = src.split_off(..2).ok_or("truncated IPS size")?;
        let size = u16::from_be_bytes([size[0], size[1]]);
        let data = if size == 0 {
            let run = src.split_off(..3).ok_or("truncated IPS fill")?;
            Data::Fill(u16::from_be_bytes([run[0], run[1]]), run[2])
        } else {
            Data::Bytes(
                src.split_off(..usize::from(size))
                    .ok_or("truncated IPS record")?
                    .to_vec(),
            )
        };
        records.push(Record {
            offset,
            data,
            print: false,
        });
    }
    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clips_sections_at_stop_and_checks_shifted_offsets() {
        let records =
            parse_text("@flag offset_shift -0x100\n@enabled\n0x200 0F\n@stop\n0x201 33").unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].offset, 0x100);
        assert!(parse_text("@flag offset_shift -1\n@enabled\n00000000 42").is_err());
        assert!(parse_text("@flag offset_shift --9223372036854775808").is_err());
        assert!(parse_text("@enabled\n0000000z ab").is_err());
        assert!(parse_text("@enabled\n00000000 abc").is_err());
    }

    #[test]
    fn truncated_binary_records_do_not_produce_partial_patches() {
        for data in [
            &b"PATCH\x00\x01\x00\x00\x02\x01"[..],
            b"IPS32\x00\x00\x01\x00\x00\x00\x00\x01",
            b"IPS32\x00\x00\x01\x00\x00\x01\xcc",
        ] {
            assert!(parse_ips(data).is_err());
        }
    }
}
