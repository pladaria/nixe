//! Diagnostic decoding for the Switch error library applet.

use std::fmt::{Display, Formatter};

/// Decoded input supplied to the Switch error library applet.
///
/// Nixe currently has no graphical implementation of this system applet. The
/// diagnostic preserves every verified field that would select or populate
/// the real dialog, while retaining bounded raw evidence for unknown layouts.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ErrorAppletDiagnostic {
    launch_mode: Option<u32>,
    common: CommonArguments,
    header: Option<ErrorHeader>,
    payload: ErrorPayload,
    extras: Vec<ErrorExtra>,
    storage_sizes: Vec<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CommonArguments {
    version: Option<u32>,
    size: Option<u32>,
    library_applet_version: Option<u32>,
    expected_theme_color: Option<i32>,
    play_startup_sound: Option<bool>,
    tick: Option<u64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ErrorHeader {
    kind: u8,
    jump: bool,
    context: bool,
    result_flag: bool,
    context2: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum ErrorPayload {
    Normal {
        identity: ErrorIdentity,
    },
    ParentalControls {
        result: ResultIdentity,
    },
    System {
        code: ErrorCode,
        language: String,
        dialog_message: String,
        fullscreen_message: String,
    },
    Application {
        error_number: u32,
        language: String,
        dialog_message: String,
        fullscreen_message: String,
    },
    Eula {
        region: u32,
    },
    Record {
        code: ErrorCode,
        timestamp: u64,
    },
    SystemUpdateEula {
        region: u32,
    },
    Unknown {
        kind: u8,
        raw: String,
    },
    Malformed {
        reason: &'static str,
        raw: String,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ErrorIdentity {
    Result(ResultIdentity),
    Code(ErrorCode),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ResultIdentity {
    raw: u32,
    module: u32,
    description: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ErrorCode {
    low: u32,
    description: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum ErrorExtra {
    ResultBacktrace(Vec<ResultIdentity>),
    Context {
        kind: u8,
        result: ResultIdentity,
        raw_data: String,
    },
    Raw {
        index: usize,
        data: String,
    },
}

impl ErrorAppletDiagnostic {
    /// Decodes the ordered `PushInData` storage sequence emitted by libnx.
    ///
    /// Layout reference:
    /// https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/applets/error.h
    #[must_use]
    pub fn decode(storages: &[Vec<u8>]) -> Self {
        let common = decode_common_arguments(storages.first().map(Vec::as_slice).unwrap_or(&[]));
        let argument = storages.get(1).map(Vec::as_slice).unwrap_or(&[]);
        let header = decode_header(argument);
        let payload = decode_payload(argument, header);
        let extras = storages
            .iter()
            .enumerate()
            .skip(2)
            .map(|(index, storage)| decode_extra(index, storage, header))
            .collect();
        Self {
            launch_mode: None,
            common,
            header,
            payload,
            extras,
            storage_sizes: storages.iter().map(Vec::len).collect(),
        }
    }

    #[must_use]
    pub(crate) const fn with_launch_mode(mut self, mode: u32) -> Self {
        self.launch_mode = Some(mode);
        self
    }
}

impl Display for ErrorAppletDiagnostic {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "mode={} storages={:?} common=[version={:?} size={:?} applet-version={:?} theme={:?} startup-sound={:?} tick={:?}]",
            self.launch_mode
                .map(library_applet_mode_name)
                .unwrap_or("unknown"),
            self.storage_sizes,
            self.common.version,
            self.common.size,
            self.common.library_applet_version,
            self.common.expected_theme_color,
            self.common.play_startup_sound,
            self.common.tick
        )?;
        if let Some(header) = self.header {
            write!(
                formatter,
                " header=[type={} jump={} context={} result-flag={} context2={}]",
                error_kind_name(header.kind),
                header.jump,
                header.context,
                header.result_flag,
                header.context2
            )?;
        } else {
            formatter.write_str(" header=[missing]")?;
        }
        write!(formatter, " payload=[{}]", PayloadDisplay(&self.payload))?;
        if !self.extras.is_empty() {
            formatter.write_str(" extras=[")?;
            for (index, extra) in self.extras.iter().enumerate() {
                if index != 0 {
                    formatter.write_str(", ")?;
                }
                ExtraDisplay(extra).fmt(formatter)?;
            }
            formatter.write_str("]")?;
        }
        Ok(())
    }
}

const fn library_applet_mode_name(mode: u32) -> &'static str {
    match mode {
        0 => "AllForeground",
        1 => "Background",
        2 => "NoUi",
        3 => "BackgroundIndirect",
        4 => "AllForegroundInitiallyHidden",
        _ => "Unknown",
    }
}

struct PayloadDisplay<'a>(&'a ErrorPayload);

impl Display for PayloadDisplay<'_> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            ErrorPayload::Normal { identity } => write!(formatter, "normal {identity}"),
            ErrorPayload::ParentalControls { result } => {
                write!(formatter, "parental-controls {result}")
            }
            ErrorPayload::System {
                code,
                language,
                dialog_message,
                fullscreen_message,
            } => write!(
                formatter,
                "system code={code} language={language:?} dialog={dialog_message:?} details={fullscreen_message:?}"
            ),
            ErrorPayload::Application {
                error_number,
                language,
                dialog_message,
                fullscreen_message,
            } => write!(
                formatter,
                "application number={error_number} language={language:?} dialog={dialog_message:?} details={fullscreen_message:?}"
            ),
            ErrorPayload::Eula { region } => write!(formatter, "eula region={region}"),
            ErrorPayload::Record { code, timestamp } => {
                write!(formatter, "record code={code} timestamp={timestamp}")
            }
            ErrorPayload::SystemUpdateEula { region } => {
                write!(formatter, "system-update-eula region={region}")
            }
            ErrorPayload::Unknown { kind, raw } => {
                write!(formatter, "unknown type={kind} raw={raw}")
            }
            ErrorPayload::Malformed { reason, raw } => {
                write!(formatter, "malformed reason={reason} raw={raw}")
            }
        }
    }
}

impl Display for ErrorIdentity {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Result(result) => write!(formatter, "source=result {result}"),
            Self::Code(code) => write!(formatter, "source=error-code {code}"),
        }
    }
}

impl Display for ResultIdentity {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "result={:#010x} module={} description={} display-code={:04}-{:04}",
            self.raw,
            self.module,
            self.description,
            self.module.saturating_add(2000),
            self.description
        )
    }
}

impl Display for ErrorCode {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{:04}-{:04}", self.low, self.description)
    }
}

struct ExtraDisplay<'a>(&'a ErrorExtra);

impl Display for ExtraDisplay<'_> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            ErrorExtra::ResultBacktrace(results) => {
                formatter.write_str("backtrace=[")?;
                for (index, result) in results.iter().enumerate() {
                    if index != 0 {
                        formatter.write_str(", ")?;
                    }
                    result.fmt(formatter)?;
                }
                formatter.write_str("]")
            }
            ErrorExtra::Context {
                kind,
                result,
                raw_data,
            } => write!(formatter, "context=[type={kind} {result} data={raw_data}]"),
            ErrorExtra::Raw { index, data } => write!(formatter, "storage-{index}={data}"),
        }
    }
}

fn decode_common_arguments(bytes: &[u8]) -> CommonArguments {
    CommonArguments {
        version: read_u32(bytes, 0),
        size: read_u32(bytes, 4),
        library_applet_version: read_u32(bytes, 8),
        expected_theme_color: read_u32(bytes, 12).map(|value| value as i32),
        play_startup_sound: bytes.get(16).map(|value| *value != 0),
        tick: read_u64(bytes, 24),
    }
}

fn decode_header(bytes: &[u8]) -> Option<ErrorHeader> {
    Some(ErrorHeader {
        kind: *bytes.first()?,
        jump: *bytes.get(1)? != 0,
        context: *bytes.get(5)? != 0,
        result_flag: *bytes.get(6)? != 0,
        context2: *bytes.get(7)? != 0,
    })
}

fn decode_payload(bytes: &[u8], header: Option<ErrorHeader>) -> ErrorPayload {
    let Some(header) = header else {
        return ErrorPayload::Malformed {
            reason: "missing 8-byte ErrorCommonHeader",
            raw: hex_preview(bytes),
        };
    };
    match header.kind {
        0 => {
            let (Some(low), Some(description), Some(raw_result)) =
                (read_u32(bytes, 8), read_u32(bytes, 12), read_u32(bytes, 16))
            else {
                return malformed_payload("truncated ErrorCommonArg", bytes);
            };
            let identity = if header.result_flag {
                ErrorIdentity::Code(ErrorCode { low, description })
            } else {
                ErrorIdentity::Result(decode_result(raw_result))
            };
            ErrorPayload::Normal { identity }
        }
        1 => {
            if bytes.len() < 0x1018 {
                return malformed_payload("truncated ErrorSystemArg", bytes);
            }
            ErrorPayload::System {
                code: ErrorCode {
                    low: read_u32(bytes, 8).unwrap(),
                    description: read_u32(bytes, 12).unwrap(),
                },
                language: decode_language(read_u64(bytes, 16).unwrap()),
                dialog_message: decode_c_string(&bytes[24..24 + 0x800]),
                fullscreen_message: decode_c_string(&bytes[24 + 0x800..24 + 0x1000]),
            }
        }
        2 => {
            if bytes.len() < 0x1014 {
                return malformed_payload("truncated ErrorApplicationArg", bytes);
            }
            ErrorPayload::Application {
                error_number: read_u32(bytes, 8).unwrap(),
                language: decode_language(read_u64(bytes, 12).unwrap()),
                dialog_message: decode_c_string(&bytes[20..20 + 0x800]),
                fullscreen_message: decode_c_string(&bytes[20 + 0x800..20 + 0x1000]),
            }
        }
        3 => read_u32(bytes, 8).map_or_else(
            || malformed_payload("truncated ErrorEulaArg", bytes),
            |region| ErrorPayload::Eula { region },
        ),
        4 => read_u32(bytes, 8).map_or_else(
            || malformed_payload("truncated ErrorPctlArg", bytes),
            |result| ErrorPayload::ParentalControls {
                result: decode_result(result),
            },
        ),
        5 => match (read_u32(bytes, 8), read_u32(bytes, 12), read_u64(bytes, 16)) {
            (Some(low), Some(description), Some(timestamp)) => ErrorPayload::Record {
                code: ErrorCode { low, description },
                timestamp,
            },
            _ => malformed_payload("truncated ErrorRecordArg", bytes),
        },
        8 => read_u32(bytes, 8).map_or_else(
            || malformed_payload("truncated ErrorSystemUpdateEulaArg", bytes),
            |region| ErrorPayload::SystemUpdateEula { region },
        ),
        kind => ErrorPayload::Unknown {
            kind,
            raw: hex_preview(bytes),
        },
    }
}

fn decode_extra(index: usize, bytes: &[u8], header: Option<ErrorHeader>) -> ErrorExtra {
    if header.is_some_and(|header| header.kind == 0 && header.context)
        && let Some(count) = read_u32(bytes, 0).and_then(|count| usize::try_from(count).ok())
        && count <= 0x20
        && bytes.len() >= 4 + count * 4
    {
        let results = (0..count)
            .map(|entry| decode_result(read_u32(bytes, 4 + entry * 4).unwrap()))
            .collect();
        return ErrorExtra::ResultBacktrace(results);
    }
    if header.is_some_and(|header| header.context2 || (header.kind != 0 && header.context))
        && bytes.len() >= 0x200
    {
        return ErrorExtra::Context {
            kind: bytes[0],
            result: decode_result(read_u32(bytes, 0x1fc).unwrap()),
            raw_data: hex_preview(&bytes[8..0x1fc]),
        };
    }
    ErrorExtra::Raw {
        index,
        data: hex_preview(bytes),
    }
}

const fn decode_result(raw: u32) -> ResultIdentity {
    ResultIdentity {
        raw,
        module: raw & 0x1ff,
        description: (raw >> 9) & 0x1fff,
    }
}

fn malformed_payload(reason: &'static str, bytes: &[u8]) -> ErrorPayload {
    ErrorPayload::Malformed {
        reason,
        raw: hex_preview(bytes),
    }
}

fn decode_language(raw: u64) -> String {
    let bytes = raw.to_le_bytes();
    decode_c_string(&bytes)
}

fn decode_c_string(bytes: &[u8]) -> String {
    let end = bytes
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

fn hex_preview(bytes: &[u8]) -> String {
    const LIMIT: usize = 64;
    let mut output = String::new();
    for byte in bytes.iter().take(LIMIT) {
        use std::fmt::Write as _;
        let _ = write!(output, "{byte:02x}");
    }
    if bytes.len() > LIMIT {
        use std::fmt::Write as _;
        let _ = write!(output, "…(+{} bytes)", bytes.len() - LIMIT);
    }
    output
}

fn read_u32(bytes: &[u8], offset: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        bytes.get(offset..offset + 4)?.try_into().ok()?,
    ))
}

fn read_u64(bytes: &[u8], offset: usize) -> Option<u64> {
    Some(u64::from_le_bytes(
        bytes.get(offset..offset + 8)?.try_into().ok()?,
    ))
}

const fn error_kind_name(kind: u8) -> &'static str {
    match kind {
        0 => "Normal",
        1 => "System",
        2 => "Application",
        3 => "Eula",
        4 => "ParentalControls",
        5 => "Record",
        8 => "SystemUpdateEula",
        _ => "Unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn common_arguments() -> Vec<u8> {
        let mut bytes = vec![0; 0x20];
        bytes[0..4].copy_from_slice(&1_u32.to_le_bytes());
        bytes[4..8].copy_from_slice(&0x20_u32.to_le_bytes());
        bytes[24..32].copy_from_slice(&0x1234_u64.to_le_bytes());
        bytes
    }

    #[test]
    fn normal_result_reports_raw_module_description_and_display_code() {
        let mut argument = vec![0; 0x14];
        argument[1] = 1;
        let result = 2_u32 | (5 << 9);
        argument[16..20].copy_from_slice(&result.to_le_bytes());

        let diagnostic = ErrorAppletDiagnostic::decode(&[common_arguments(), argument]);
        let text = diagnostic.to_string();
        assert!(text.contains("type=Normal"));
        assert!(text.contains("result=0x00000a02 module=2 description=5"));
        assert!(text.contains("display-code=2002-0005"));
    }

    #[test]
    fn application_error_preserves_guest_supplied_text() {
        let mut argument = vec![0; 0x1014];
        argument[0] = 2;
        argument[1] = 1;
        argument[8..12].copy_from_slice(&42_u32.to_le_bytes());
        argument[12..17].copy_from_slice(b"en-US");
        argument[20..25].copy_from_slice(b"Oops\0");
        argument[20 + 0x800..20 + 0x800 + 8].copy_from_slice(b"Details\0");

        let diagnostic = ErrorAppletDiagnostic::decode(&[common_arguments(), argument]);
        let text = diagnostic.to_string();
        assert!(text.contains("application number=42 language=\"en-US\""));
        assert!(text.contains("dialog=\"Oops\" details=\"Details\""));
    }

    #[test]
    fn malformed_payload_retains_bounded_raw_evidence() {
        let diagnostic = ErrorAppletDiagnostic::decode(&[common_arguments(), vec![0, 1, 2]]);
        let text = diagnostic.to_string();
        assert!(text.contains("missing 8-byte ErrorCommonHeader"));
        assert!(text.contains("raw=000102"));
    }
}
