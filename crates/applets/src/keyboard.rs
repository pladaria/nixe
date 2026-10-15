//! Foreground software keyboard ABI and host request channel.
//! https://switchbrew.org/wiki/Software_Keyboard
use nixe_input::EmulatedControllerState;
use std::collections::VecDeque;
use std::fmt;
use std::sync::{
    Arc, Mutex, Weak,
    atomic::{AtomicBool, Ordering},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KeyboardType {
    Ascii,
    Numeric,
}

#[derive(Clone, Debug)]
pub struct KeyboardConfig {
    pub kind: KeyboardType,
    pub ok: String,
    pub header: String,
    pub subtitle: String,
    pub guide: String,
    pub initial: String,
    pub cursor_at_end: bool,
    pub minimum: usize,
    pub maximum: usize,
    pub password: bool,
    pub multiline: bool,
    pub cancel_disabled: bool,
    pub text_check: bool,
    pub utf8: bool,
    pub disabled: u32,
    pub optional_symbols: [Option<char>; 2],
}

impl KeyboardConfig {
    pub fn decode(common: &[u8], args: &[u8], work: &[u8]) -> Result<Self, &'static str> {
        if common.len() != 32 || u32_at(common, 0)? != 1 || u32_at(common, 4)? != 32 {
            return Err("invalid software keyboard common arguments");
        }
        let version = u32_at(common, 8)?;
        let size = match version {
            5 | 0x10006 => 0x3e0,
            0x30007 | 0x40008 | 0x50009 => 0x400,
            0x6000b | 0x8000d => 0x4c8,
            _ => return Err("unsupported software keyboard API version"),
        };
        // Consume the versioned wire prefix, not the whole IStorage allocation.
        // libnx sends sizeof its C argument struct, including alignment padding.
        // https://github.com/switchbrew/libnx/blob/master/nx/source/applets/swkbd.c
        let args = args
            .get(..size)
            .ok_or("invalid software keyboard configuration size")?;
        let kind = match u32_at(args, 0)? {
            0 | 2 | 3 | 4 | 8 | 9 => KeyboardType::Ascii,
            1 => KeyboardType::Numeric,
            _ => return Err("only ASCII and numeric software keyboards are implemented"),
        };
        let maximum = u32_at(args, 0x3ac)? as usize;
        let minimum = u32_at(args, 0x3b0)? as usize;
        // The foreground output contains 0x7d4 bytes, including the terminator.
        let maximum = if maximum == 0 { 500 } else { maximum.min(500) };
        if minimum > maximum {
            return Err("invalid software keyboard length limits");
        }
        let start = u32_at(args, 0x3c0)? as usize;
        let count = u32_at(args, 0x3c4)? as usize;
        let end = count
            .checked_mul(2)
            .and_then(|n| start.checked_add(n))
            .ok_or("invalid initial keyboard text range")?;
        let initial = if count == 0 {
            String::new()
        } else {
            utf16(
                work.get(start..end)
                    .ok_or("initial keyboard text exceeds transfer memory")?,
            )?
        };
        let disabled = u32_at(args, 0x1c)?;
        if disabled & !0x1fe != 0 {
            return Err("unsupported software keyboard character flags");
        }
        if disabled & 0x80 != 0 {
            return Err("download-code character restrictions are not implemented");
        }
        let optional_symbols = [u16_at(args, 0x16)?, u16_at(args, 0x18)?].map(|c| {
            if c == 0 {
                None
            } else {
                char::from_u32(u32::from(c))
            }
        });
        if optional_symbols
            .iter()
            .flatten()
            .any(|c| !c.is_ascii() || c.is_ascii_control())
        {
            return Err("non-ASCII numeric keyboard symbols are not implemented");
        }
        let cursor = u32_at(args, 0x20)?;
        let password = u32_at(args, 0x3b4)?;
        let draw = u32_at(args, 0x3b8)?;
        if cursor > 1 || password > 1 || draw > 2 {
            return Err("invalid software keyboard cursor, password or text form mode");
        }
        let config = Self {
            kind,
            ok: utf16(&args[4..0x16])?,
            header: utf16(&args[0x24..0xa6])?,
            subtitle: utf16(&args[0xa6..0x1a8])?,
            guide: utf16(&args[0x1a8..0x3aa])?,
            initial,
            cursor_at_end: cursor == 1,
            minimum,
            maximum,
            password: password != 0,
            multiline: args[0x3bc] != 0,
            cancel_disabled: size == 0x4c8 && args[0x4b5] != 0,
            text_check: args[0x3d0] != 0,
            utf8: args[0x3bd] != 0,
            disabled,
            optional_symbols,
        };
        if !config.initial.chars().all(|c| config.allows(c)) || config.initial.len() > maximum {
            return Err("initial keyboard text is outside the supported character set or length");
        }
        Ok(config)
    }

    pub fn allows(&self, c: char) -> bool {
        if c == '\n' {
            return self.multiline && self.kind == KeyboardType::Ascii;
        }
        if !c.is_ascii() || c.is_ascii_control() {
            return false;
        }
        if self.kind == KeyboardType::Numeric
            && !c.is_ascii_digit()
            && !self.optional_symbols.contains(&Some(c))
        {
            return false;
        }
        // libnx UserName preset disables these three ASCII characters.
        // https://github.com/switchbrew/libnx/blob/master/nx/include/switch/applets/swkbd.h
        if self.disabled & 0x100 != 0 && matches!(c, '@' | '%' | '\\') {
            return false;
        }
        let flag = match c {
            ' ' => 2,
            '@' => 4,
            '%' => 8,
            '/' => 16,
            '\\' => 32,
            '0'..='9' => 64,
            _ => 0,
        };
        self.disabled & flag == 0
    }
    pub fn valid(&self, text: &str) -> bool {
        text.len() >= self.minimum
            && text.len() <= self.maximum
            && text.chars().all(|c| self.allows(c))
    }
    pub fn encode_result(&self, text: &str, cancelled: bool) -> Vec<u8> {
        let mut bytes = vec![0; 0x7d8];
        bytes[..4].copy_from_slice(&u32::from(cancelled).to_le_bytes());
        self.encode_text(if cancelled { "" } else { text }, &mut bytes[4..]);
        bytes
    }
    pub fn encode_check(&self, text: &str) -> Vec<u8> {
        let mut bytes = vec![0; 0x7dc];
        let length = if self.utf8 {
            text.len() + 1
        } else {
            (text.encode_utf16().count() + 1) * 2
        };
        bytes[..8].copy_from_slice(&(length as u64).to_le_bytes());
        self.encode_text(text, &mut bytes[8..]);
        bytes
    }
    fn encode_text(&self, text: &str, bytes: &mut [u8]) {
        if self.utf8 {
            bytes[..text.len()].copy_from_slice(text.as_bytes());
        } else {
            for (unit, pair) in text.encode_utf16().zip(bytes.chunks_exact_mut(2)) {
                pair.copy_from_slice(&unit.to_le_bytes());
            }
        }
    }
}
fn u32_at(bytes: &[u8], offset: usize) -> Result<u32, &'static str> {
    Ok(u32::from_le_bytes(
        bytes
            .get(offset..offset + 4)
            .ok_or("truncated keyboard configuration")?
            .try_into()
            .unwrap(),
    ))
}
fn u16_at(bytes: &[u8], offset: usize) -> Result<u16, &'static str> {
    Ok(u16::from_le_bytes(
        bytes
            .get(offset..offset + 2)
            .ok_or("truncated keyboard configuration")?
            .try_into()
            .unwrap(),
    ))
}
pub fn utf16(bytes: &[u8]) -> Result<String, &'static str> {
    if !bytes.len().is_multiple_of(2) {
        return Err("unaligned UTF-16 keyboard text");
    }
    let units: Vec<_> = bytes
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .take_while(|c| *c != 0)
        .collect();
    String::from_utf16(&units).map_err(|_| "invalid UTF-16 keyboard text")
}

#[derive(Clone, Debug)]
pub enum Submission {
    Accept { text: String, confirmed: bool },
    Cancel,
}
#[derive(Clone, Debug)]
pub enum Validation {
    Failure(String),
    Confirm(String),
    Retry,
}

pub struct KeyboardSession {
    pub config: KeyboardConfig,
    submit: Box<dyn Fn(Submission) + Send + Sync>,
    validation: Mutex<Option<Validation>>,
    finished: AtomicBool,
    notify: Arc<dyn Fn() + Send + Sync>,
}
impl fmt::Debug for KeyboardSession {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KeyboardSession")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}
impl KeyboardSession {
    pub fn submit(&self, value: Submission) {
        (self.submit)(value);
    }
    pub fn feedback(&self, value: Validation) {
        *self.validation.lock().unwrap() = Some(value);
        (self.notify)();
    }
    pub fn take_feedback(&self) -> Option<Validation> {
        self.validation.lock().unwrap().take()
    }
    pub fn finish(&self) {
        self.finished.store(true, Ordering::Release);
        (self.notify)();
    }
    pub fn finished(&self) -> bool {
        self.finished.load(Ordering::Acquire)
    }
}
struct HostState {
    pending: Option<Arc<KeyboardSession>>,
    controller: EmulatedControllerState,
    samples: VecDeque<EmulatedControllerState>,
    initial_sample: bool,
    current: Option<Weak<KeyboardSession>>,
}
#[derive(Clone)]
pub struct KeyboardHost {
    state: Arc<Mutex<HostState>>,
    active: Arc<AtomicBool>,
    drain_release: Arc<AtomicBool>,
    notify: Arc<dyn Fn() + Send + Sync>,
}
impl fmt::Debug for KeyboardHost {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KeyboardHost")
            .field("active", &self.active())
            .finish()
    }
}
impl KeyboardHost {
    pub fn new(notify: impl Fn() + Send + Sync + 'static) -> Self {
        Self {
            state: Arc::new(Mutex::new(HostState {
                pending: None,
                controller: EmulatedControllerState::default(),
                samples: VecDeque::new(),
                initial_sample: false,
                current: None,
            })),
            active: Arc::new(AtomicBool::new(false)),
            drain_release: Arc::new(AtomicBool::new(false)),
            notify: Arc::new(notify),
        }
    }
    pub fn open(
        &self,
        config: KeyboardConfig,
        submit: impl Fn(Submission) + Send + Sync + 'static,
    ) -> Result<Arc<KeyboardSession>, &'static str> {
        let mut state = self.state.lock().unwrap();
        if state
            .current
            .as_ref()
            .and_then(Weak::upgrade)
            .is_some_and(|current| !current.finished())
        {
            return Err("another host keyboard is already active");
        }
        self.active.store(true, Ordering::Release);
        let session = Arc::new(KeyboardSession {
            config,
            submit: Box::new(submit),
            validation: Mutex::new(None),
            finished: AtomicBool::new(false),
            notify: self.notify.clone(),
        });
        state.samples.clear();
        state.initial_sample = true;
        state.current = Some(Arc::downgrade(&session));
        state.pending = Some(session.clone());
        drop(state);
        (self.notify)();
        Ok(session)
    }
    pub fn active(&self) -> bool {
        self.active.load(Ordering::Acquire)
    }
    pub fn take_request(&self) -> Option<Arc<KeyboardSession>> {
        self.state.lock().unwrap().pending.take()
    }
    pub fn close(&self, session: &Arc<KeyboardSession>) {
        let mut state = self.state.lock().unwrap();
        if !state
            .current
            .as_ref()
            .and_then(Weak::upgrade)
            .is_some_and(|current| Arc::ptr_eq(&current, session))
        {
            return;
        }
        state.current = None;
        self.active.store(false, Ordering::Release);
        self.drain_release.store(true, Ordering::Release);
    }
    /// Called at the existing input sampling boundary. No UI polling thread.
    pub fn route_controller(&self, state: Option<&EmulatedControllerState>) -> bool {
        let state = state.copied().unwrap_or_default();
        if !self.active() {
            if self.drain_release.load(Ordering::Acquire) {
                if neutral(state) {
                    self.drain_release.store(false, Ordering::Release);
                }
                return true;
            }
            return false;
        }
        let mut host = self.state.lock().unwrap();
        if host.initial_sample
            || host.controller.buttons != state.buttons
            || host.controller.left_stick != state.left_stick
        {
            host.initial_sample = false;
            host.controller = state;
            host.samples.push_back(state);
            drop(host);
            (self.notify)();
        }
        true
    }
    pub fn take_controller_samples(&self) -> VecDeque<EmulatedControllerState> {
        std::mem::take(&mut self.state.lock().unwrap().samples)
    }
}
pub fn neutral(state: EmulatedControllerState) -> bool {
    state.buttons == Default::default()
        && state.left_stick.x.unsigned_abs() < 12000
        && state.left_stick.y.unsigned_abs() < 12000
}

#[cfg(test)]
mod tests {
    use super::*;
    fn common_args(version: u32) -> [u8; 32] {
        let mut common = [0; 32];
        common[..4].copy_from_slice(&1u32.to_le_bytes());
        common[4..8].copy_from_slice(&32u32.to_le_bytes());
        common[8..12].copy_from_slice(&version.to_le_bytes());
        common
    }
    fn config() -> KeyboardConfig {
        let mut args = vec![0; 0x4c8];
        args[0x3ac..0x3b0].copy_from_slice(&9u32.to_le_bytes());
        KeyboardConfig::decode(&common_args(0x8000d), &args, &[]).unwrap()
    }
    #[test]
    fn libnx_v5_keyboard_configuration_accepts_struct_padding_and_text_validation() {
        // Reproduces software-keyboard's actual version/size and default preset.
        let mut args = vec![0; 0x3e8];
        args[..4].copy_from_slice(&2u32.to_le_bytes());
        args[0x20..0x24].copy_from_slice(&1u32.to_le_bytes());
        args[0x3b8..0x3bc].copy_from_slice(&1u32.to_le_bytes());
        args[0x3bc] = 1;
        args[0x3d0] = 1;
        // The guest's callback pointer/padding is not executed by the applet.
        args[0x3e0..].fill(0xa5);
        let config = KeyboardConfig::decode(&common_args(5), &args, &[0; 0x1000]).unwrap();
        assert_eq!(config.kind, KeyboardType::Ascii);
        assert!(config.cursor_at_end);
        assert!(config.multiline);
        assert!(config.text_check);
        assert!(!config.utf8);
        assert!(config.initial.is_empty());
        assert_eq!((config.minimum, config.maximum), (0, 500));
        assert!(config.valid("Hello\nworld"));
        assert_eq!(&config.encode_check("Nixe")[8..18], b"N\0i\0x\0e\0\0\0");
    }
    #[test]
    fn configuration_uses_the_versioned_prefix_with_larger_storage() {
        for (version, size) in [
            (5, 0x3e0),
            (0x10006, 0x3e0),
            (0x30007, 0x400),
            (0x40008, 0x400),
            (0x50009, 0x400),
            (0x6000b, 0x4c8),
            (0x8000d, 0x4c8),
        ] {
            let mut args = vec![0; size];
            args[0x3ac..0x3b0].copy_from_slice(&12u32.to_le_bytes());
            args[0x3b0..0x3b4].copy_from_slice(&2u32.to_le_bytes());
            args[0x3c0..0x3c4].copy_from_slice(&4u32.to_le_bytes());
            args[0x3c4..0x3c8].copy_from_slice(&4u32.to_le_bytes());
            if version == 0x8000d {
                args[0x4b5] = 1;
            }
            let work = b"\0\0\0\0N\0i\0x\0e\0";
            for trailing in [0, 8, 32] {
                let mut storage = args.clone();
                storage.extend(std::iter::repeat_n(0xff, trailing));
                let config = KeyboardConfig::decode(&common_args(version), &storage, work).unwrap();
                assert_eq!(config.initial, "Nixe");
                assert_eq!((config.minimum, config.maximum), (2, 12));
                assert_eq!(config.cancel_disabled, version == 0x8000d);
                assert!(config.valid("Nixe"));
                assert!(!config.valid("x"));
                assert!(!config.valid("too many characters"));
            }
            assert!(matches!(
                KeyboardConfig::decode(&common_args(version), &args[..size - 1], work),
                Err("invalid software keyboard configuration size")
            ));
        }
        assert!(matches!(
            KeyboardConfig::decode(&common_args(0x9000f), &[0; 0x1000], &[]),
            Err("unsupported software keyboard API version")
        ));
    }
    #[test]
    fn restrictions_apply_to_all_input_sources() {
        let mut c = config();
        assert!(c.valid("Test 123"));
        assert!(!c.valid("é"));
        assert!(!c.valid("0123456789"));
        c.disabled = 2 | 64;
        assert!(!c.valid("a b"));
        assert!(!c.valid("123"));
        assert!(c.valid("abc"));
        c.disabled = 0x100;
        assert!(c.valid("a b/123"));
        assert!(!c.valid("a@b"));
        assert!(!c.valid("a%b"));
        assert!(!c.valid("a\\b"));
        c.kind = KeyboardType::Numeric;
        c.disabled = 0;
        c.optional_symbols = [Some('.'), None];
        assert!(c.valid("12.3"));
        assert!(!c.valid("test"));
    }
    #[test]
    fn results_and_validation_preserve_requested_encoding() {
        let mut c = config();
        let out = c.encode_result("abc", false);
        assert_eq!(out.len(), 0x7d8);
        assert_eq!(&out[4..12], b"a\0b\0c\0\0\0");
        c.utf8 = true;
        assert_eq!(&c.encode_result("abc", false)[4..8], b"abc\0");
        assert_eq!(
            u64::from_le_bytes(c.encode_check("abc")[..8].try_into().unwrap()),
            4
        );
        assert_eq!(
            &c.encode_result("abc", true)[..8],
            &[1, 0, 0, 0, 0, 0, 0, 0]
        );
    }
    #[test]
    fn a_completed_request_can_be_replaced_before_the_ui_closes_it() {
        let host = KeyboardHost::new(|| {});
        let first = host.open(config(), |_| {}).unwrap();
        host.take_request();
        assert!(host.open(config(), |_| {}).is_err());
        first.finish();
        let next = host.open(config(), |_| {}).unwrap();
        host.close(&first);
        assert!(host.active());
        assert!(Arc::ptr_eq(&host.take_request().unwrap(), &next));
        host.close(&next);
        assert!(!host.active());
    }
    #[test]
    fn host_captures_controller_until_accept_button_is_released() {
        let host = KeyboardHost::new(|| {});
        let session = host.open(config(), |_| {}).unwrap();
        let mut state = EmulatedControllerState::default();
        state.buttons.a = true;
        assert!(host.route_controller(Some(&state)));
        assert_eq!(host.take_controller_samples().back(), Some(&state));
        host.close(&session);
        assert!(host.route_controller(Some(&state)));
        assert!(host.route_controller(None));
        assert!(!host.route_controller(None));
    }
}
