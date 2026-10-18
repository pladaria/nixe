//! Host-rendered ASCII software keyboard. No guest GPU or firmware UI commands.
use egui::{Color32, FontId, Rect, Stroke, StrokeKind, vec2};
use nixe_applets::keyboard::{
    KeyboardConfig, KeyboardHost, KeyboardSession, KeyboardType, Submission, Validation, neutral,
};
use nixe_input::EmulatedControllerState;
use std::{
    ops::Range,
    sync::Arc,
    time::{Duration, Instant},
};
use winit::{event::WindowEvent, window::Window};

#[derive(Clone, Debug, PartialEq)]
enum Action {
    Character(char),
    Backspace,
    Shift,
    Symbols,
    Space,
    Enter,
    Accept,
    Cancel,
}
#[derive(Clone)]
struct Key {
    rect: Rect,
    label: String,
    action: Action,
}

pub(super) struct KeyboardOverlay {
    ctx: egui::Context,
    input: egui_winit::State,
    editor: Option<Editor>,
}
struct Editor {
    session: Arc<KeyboardSession>,
    text: String,
    cursor: Range<usize>,
    cursor_changed: bool,
    shifted: bool,
    symbols: bool,
    selected: usize,
    armed: bool,
    controller: EmulatedControllerState,
    repeat_at: Instant,
    direction: (i32, i32),
    waiting: bool,
    feedback: Option<Validation>,
    pointer: Option<egui::Pos2>,
    seen_controller: bool,
    pending_submission: Option<Submission>,
}
impl Editor {
    fn new(session: Arc<KeyboardSession>) -> Self {
        let text = session.config.initial.clone();
        let cursor = if session.config.cursor_at_end {
            text.len()
        } else {
            0
        };
        Self {
            session,
            text,
            cursor: cursor..cursor,
            cursor_changed: true,
            shifted: false,
            symbols: false,
            selected: 0,
            armed: false,
            controller: Default::default(),
            repeat_at: Instant::now(),
            direction: (0, 0),
            waiting: false,
            feedback: None,
            pointer: None,
            seen_controller: false,
            pending_submission: None,
        }
    }
    fn insert(&mut self, text: &str) {
        if self.waiting {
            return;
        }
        let filtered: String = text
            .chars()
            .filter(|c| self.session.config.allows(*c))
            .collect();
        if !text.is_empty() && filtered.is_empty() {
            return;
        }
        let available = self
            .session
            .config
            .maximum
            .saturating_sub(self.text.len() - self.cursor.len());
        let filtered = &filtered[..filtered.len().min(available)];
        let start = self.cursor.start;
        self.text.replace_range(self.cursor.clone(), filtered);
        let end = start + filtered.len();
        self.cursor = end..end;
        self.cursor_changed = true;
    }
    fn backspace(&mut self) {
        if self.waiting {
            return;
        }
        if self.cursor.is_empty() && self.cursor.start > 0 {
            self.cursor.start -= 1;
        }
        self.insert("");
    }
    fn submit(&mut self, value: Submission) {
        self.pending_submission = Some(value);
    }
    fn action(&mut self, action: &Action) {
        if self.waiting {
            return;
        }
        match action {
            Action::Character(c) => {
                self.insert(&c.to_string());
                if self.shifted {
                    self.shifted = false;
                }
            }
            Action::Backspace => self.backspace(),
            Action::Shift => self.shifted = !self.shifted,
            Action::Symbols => self.symbols = !self.symbols,
            Action::Space => self.insert(" "),
            Action::Enter => self.insert("\n"),
            Action::Accept => {
                if self.session.config.valid(&self.text) {
                    self.waiting = true;
                    self.submit(Submission::Accept {
                        text: self.text.clone(),
                        confirmed: false,
                    });
                }
            }
            Action::Cancel => {
                if !self.session.config.cancel_disabled {
                    self.waiting = true;
                    self.submit(Submission::Cancel);
                }
            }
        }
    }
    fn controller(&mut self, state: EmulatedControllerState, keys: &[Key], now: Instant) {
        let previous = self.controller;
        self.controller = state;
        self.seen_controller = true;
        if !self.armed {
            self.armed = neutral(state);
            return;
        }
        let b = state.buttons;
        let old = previous.buttons;
        if let Some(Validation::Confirm(_)) = &self.feedback {
            if b.a && !old.a {
                self.feedback = None;
                self.waiting = true;
                self.submit(Submission::Accept {
                    text: self.text.clone(),
                    confirmed: true,
                });
            } else if b.b && !old.b {
                self.feedback = None;
            }
            return;
        }
        if self.feedback.is_some() {
            if b.a && !old.a {
                self.feedback = None;
            }
            return;
        }
        if b.a && !old.a {
            let key = &keys[self.selected.min(keys.len() - 1)];
            if enabled(self, key) {
                self.action(&key.action);
            }
        }
        if b.b && !old.b {
            self.backspace();
        }
        if b.x && !old.x {
            self.action(&Action::Cancel);
        }
        if b.y && !old.y {
            self.insert(" ");
        }
        if b.left_stick && !old.left_stick {
            self.action(&Action::Shift);
        }
        if b.zr && !old.zr {
            self.action(&Action::Symbols);
        }
        if b.plus && !old.plus {
            self.action(&Action::Accept);
        }
        if (b.l && !old.l) || (b.r && !old.r) {
            let offset = if b.l { -1 } else { 1 };
            let cursor = self
                .cursor
                .end
                .saturating_add_signed(offset)
                .min(self.text.len());
            self.cursor = cursor..cursor;
            self.cursor_changed = true;
        }
        let x = if b.dpad_left || state.left_stick.x < -16000 {
            -1
        } else if b.dpad_right || state.left_stick.x > 16000 {
            1
        } else {
            0
        };
        let y = if b.dpad_up || state.left_stick.y > 16000 {
            -1
        } else if b.dpad_down || state.left_stick.y < -16000 {
            1
        } else {
            0
        };
        let direction = (x, y);
        if direction != (0, 0) && (direction != self.direction || now >= self.repeat_at) {
            self.selected = neighbor(keys, self.selected, direction);
            self.repeat_at =
                now + Duration::from_millis(if direction != self.direction { 350 } else { 80 });
        }
        self.direction = direction;
    }
}
fn neighbor(keys: &[Key], selected: usize, (x, y): (i32, i32)) -> usize {
    let center = keys[selected.min(keys.len() - 1)].rect.center();
    let score = |key: &Key| {
        let d = key.rect.center() - center;
        if x != 0 {
            if d.x * x as f32 <= 1.0
                || d.y.abs() > key.rect.height().min(keys[selected].rect.height()) * 0.55
            {
                return f32::INFINITY;
            }
            d.x.abs() + d.y.abs() * 3.0
        } else {
            if d.y * y as f32 <= 1.0 {
                return f32::INFINITY;
            }
            d.y.abs() + d.x.abs() * 2.0
        }
    };
    keys.iter()
        .enumerate()
        .filter_map(|(i, key)| {
            let s = score(key);
            s.is_finite().then_some((i, s))
        })
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map_or(selected, |v| v.0)
}
fn keys(
    config: &KeyboardConfig,
    shift: bool,
    symbols: bool,
    origin: egui::Pos2,
    unit: f32,
) -> Vec<Key> {
    let mut keys = Vec::new();
    let mut add = |x: f32, y: f32, w: f32, h: f32, label: String, action: Action| {
        keys.push(Key {
            rect: Rect::from_min_size(origin + vec2(x * unit, y * unit), vec2(w * unit, h * unit))
                .shrink(2.0 * unit / 90.0),
            label,
            action,
        })
    };
    if config.kind == KeyboardType::Numeric {
        for (i, c) in "123456789".chars().enumerate() {
            add(
                (i % 3) as f32 * 3.0,
                (i / 3) as f32,
                3.0,
                1.0,
                c.to_string(),
                Action::Character(c),
            );
        }
        add(3.0, 3.0, 3.0, 1.0, "0".into(), Action::Character('0'));
        for (i, c) in config.optional_symbols.into_iter().enumerate() {
            if let Some(c) = c {
                add(
                    i as f32 * 6.0,
                    3.0,
                    3.0,
                    1.0,
                    c.to_string(),
                    Action::Character(c),
                );
            }
        }
        add(9.0, 0.0, 3.0, 1.0, "⌫".into(), Action::Backspace);
        add(9.0, 1.0, 3.0, 3.0, ok_label(config), Action::Accept);
    } else {
        let rows = if symbols {
            ["!@#$%^&*()_", "[]{}<>\\|/~`", "+=:;\"'?-.,/", "0123456789!"]
        } else {
            ["1234567890-", "qwertyuiop/", "asdfghjkl:'", "zxcvbnm,.?!"]
        };
        for (y, row) in rows.into_iter().enumerate() {
            for (x, c) in row.chars().enumerate() {
                let c = if shift { c.to_ascii_uppercase() } else { c };
                add(
                    x as f32,
                    y as f32,
                    1.0,
                    1.0,
                    c.to_string(),
                    Action::Character(c),
                );
            }
        }
        add(11.0, 0.0, 1.0, 1.0, "⌫".into(), Action::Backspace);
        add(11.0, 1.0, 1.0, 2.0, "Return".into(), Action::Enter);
        add(11.0, 3.0, 1.0, 2.0, ok_label(config), Action::Accept);
        add(0.0, 4.0, 1.0, 1.0, "ABC".into(), Action::Symbols);
        add(1.0, 4.0, 1.0, 1.0, "⇧".into(), Action::Shift);
        add(2.0, 4.0, 1.0, 1.0, "#+=".into(), Action::Symbols);
        add(3.0, 4.0, 8.0, 1.0, "Space".into(), Action::Space);
    }
    keys
}
fn ok_label(config: &KeyboardConfig) -> String {
    if config.ok.is_empty() {
        "OK".into()
    } else {
        config.ok.clone()
    }
}
fn enabled(editor: &Editor, key: &Key) -> bool {
    if editor.waiting || editor.feedback.is_some() {
        return false;
    }
    match key.action {
        Action::Character(c) => editor.session.config.allows(c),
        Action::Space => editor.session.config.allows(' '),
        Action::Enter => editor.session.config.multiline,
        Action::Accept => editor.session.config.valid(&editor.text),
        Action::Cancel => !editor.session.config.cancel_disabled,
        _ => true,
    }
}
impl KeyboardOverlay {
    pub fn new(window: &Window, max_texture: usize) -> Self {
        let ctx = egui::Context::default();
        let mut fonts = egui::FontDefinitions::empty();
        fonts.font_data.insert(
            "DejaVu".into(),
            Arc::new(egui::FontData::from_static(include_bytes!(
                "../../../assets/DejaVuSans.ttf"
            ))),
        );
        fonts
            .families
            .insert(egui::FontFamily::Proportional, vec!["DejaVu".into()]);
        fonts
            .families
            .insert(egui::FontFamily::Monospace, vec!["DejaVu".into()]);
        ctx.set_fonts(fonts);
        ctx.set_visuals(egui::Visuals::light());
        let input = egui_winit::State::new(
            ctx.clone(),
            egui::ViewportId::ROOT,
            window,
            Some(window.scale_factor() as f32),
            None,
            Some(max_texture),
        );
        Self {
            ctx,
            input,
            editor: None,
        }
    }
    pub fn open(&mut self, session: Arc<KeyboardSession>) {
        self.editor = Some(Editor::new(session));
    }
    pub fn active(&self) -> bool {
        self.editor.is_some()
    }
    pub fn event(&mut self, window: &Window, event: &WindowEvent) {
        let _ = self.input.on_window_event(window, event);
    }
    pub fn dispatch_submission(&mut self) {
        if let Some(editor) = &mut self.editor
            && let Some(value) = editor.pending_submission.take()
        {
            editor.session.submit(value);
        }
    }
    pub fn prepare(
        &mut self,
        host: &KeyboardHost,
        window: &Window,
    ) -> Option<nixe_gpu_vulkan::presentation::UiFrame> {
        let editor = self.editor.as_mut()?;
        if editor.session.finished() {
            host.close(&editor.session);
            self.editor = None;
            window.request_redraw();
            return None;
        }
        if let Some(feedback) = editor.session.take_feedback() {
            editor.waiting = false;
            editor.feedback = if matches!(feedback, Validation::Retry) {
                None
            } else {
                Some(feedback)
            };
        }
        self.input.egui_input_mut().focused = window.has_focus();
        egui_winit::update_viewport_info(
            self.input
                .egui_input_mut()
                .viewports
                .entry(egui::ViewportId::ROOT)
                .or_default(),
            &self.ctx,
            window,
            false,
        );
        let mut input = self.input.take_egui_input(window);
        input.events.retain_mut(|event| match event {
            egui::Event::Text(text) | egui::Event::Paste(text) => {
                let nonempty = !text.is_empty();
                text.retain(|c| editor.session.config.allows(c));
                !nonempty || !text.is_empty()
            }
            _ => true,
        });
        input.events.retain(|event| {
            if let egui::Event::Key {
                key, pressed: true, ..
            } = event
            {
                if *key == egui::Key::Enter {
                    editor.action(&Action::Accept);
                    return false;
                }
                if *key == egui::Key::Escape {
                    editor.action(&Action::Cancel);
                    return false;
                }
            }
            true
        });
        let ctx = self.ctx.clone();
        let output = ctx.run_ui(input, |root| {
            let ctx = root.ctx();
            let screen = ctx.viewport_rect();
            let scale = (screen.width() / 1280.0).min(screen.height() / 720.0);
            let origin = screen.center() - vec2(640.0, 360.0) * scale;
            let mut key_list = keys(
                &editor.session.config,
                editor.shifted,
                editor.symbols,
                origin + vec2(40.0, 350.0) * scale,
                100.0 * scale,
            );
            // The reference layout uses wide, shallow keys.
            for key in &mut key_list {
                let min = key.rect.min - origin;
                let max = key.rect.max - origin;
                key.rect.min.y = origin.y + 350.0 * scale + (min.y - 350.0 * scale) * 0.56;
                key.rect.max.y = origin.y + 350.0 * scale + (max.y - 350.0 * scale) * 0.56;
            }
            for sample in host.take_controller_samples() {
                editor.controller(sample, &key_list, Instant::now());
            }
            if editor.seen_controller {
                let current = editor.controller;
                editor.controller(current, &key_list, Instant::now());
            }
            egui::Area::new(egui::Id::new("software-keyboard"))
                .fixed_pos(screen.min)
                .movable(false)
                .show(ctx, |ui| {
                    ui.set_min_size(screen.size());
                    let painter = ui.painter();
                    painter.rect_filled(screen, 0.0, Color32::from_rgb(28, 31, 33));
                    let header_rect = Rect::from_min_size(
                        origin + vec2(40.0, 35.0) * scale,
                        vec2(1200.0, 55.0) * scale,
                    );
                    painter.text(
                        header_rect.center(),
                        egui::Align2::CENTER_CENTER,
                        &editor.session.config.header,
                        FontId::proportional(28.0 * scale),
                        Color32::WHITE,
                    );
                    painter.text(
                        origin + vec2(640.0, 115.0) * scale,
                        egui::Align2::CENTER_CENTER,
                        &editor.session.config.subtitle,
                        FontId::proportional(20.0 * scale),
                        Color32::LIGHT_GRAY,
                    );
                    let field = Rect::from_min_size(
                        origin + vec2(380.0, 215.0) * scale,
                        vec2(520.0, 55.0) * scale,
                    );
                    let id = egui::Id::new("keyboard-text");
                    if editor.cursor_changed {
                        let mut state =
                            egui::text_edit::TextEditState::load(ctx, id).unwrap_or_default();
                        state
                            .cursor
                            .set_char_range(Some(egui::text::CCursorRange::two(
                                egui::text::CCursor::new(editor.cursor.start),
                                egui::text::CCursor::new(editor.cursor.end),
                            )));
                        state.store(ctx, id);
                        editor.cursor_changed = false;
                    }
                    let text_before = editor.text.clone();
                    let output = ui
                        .scope_builder(egui::UiBuilder::new().max_rect(field), |ui| {
                            ui.visuals_mut().override_text_color = Some(Color32::WHITE);
                            let edit = if editor.session.config.multiline {
                                egui::TextEdit::multiline(&mut editor.text)
                            } else {
                                egui::TextEdit::singleline(&mut editor.text)
                            };
                            edit.id(id)
                                .font(FontId::proportional(30.0 * scale))
                                .password(editor.session.config.password)
                                .char_limit(editor.session.config.maximum)
                                .interactive(!editor.waiting && editor.feedback.is_none())
                                .frame(egui::Frame::NONE)
                                .desired_width(field.width())
                                .show(ui)
                        })
                        .inner;
                    if !editor.text.chars().all(|c| editor.session.config.allows(c)) {
                        editor.text = text_before;
                    }
                    if let Some(range) = output.cursor_range {
                        let indices = range.as_sorted_char_range();
                        editor.cursor = usize::from(indices.start).min(editor.text.len())
                            ..usize::from(indices.end).min(editor.text.len());
                    }
                    output.response.request_focus();
                    ui.painter().line_segment(
                        [field.left_bottom(), field.right_bottom()],
                        Stroke::new(2.0 * scale, Color32::LIGHT_GRAY),
                    );
                    ui.painter().text(
                        field.right_bottom() + vec2(0.0, 20.0) * scale,
                        egui::Align2::RIGHT_CENTER,
                        format!("{}/{}", editor.text.len(), editor.session.config.maximum),
                        FontId::proportional(18.0 * scale),
                        Color32::LIGHT_GRAY,
                    );
                    ui.painter().text(
                        origin + vec2(640.0, 315.0) * scale,
                        egui::Align2::CENTER_CENTER,
                        &editor.session.config.guide,
                        FontId::proportional(18.0 * scale),
                        Color32::LIGHT_GRAY,
                    );
                    ui.painter().rect_filled(
                        Rect::from_min_max(
                            origin + vec2(0.0, 335.0) * scale,
                            origin + vec2(1280.0, 720.0) * scale,
                        ),
                        0.0,
                        Color32::from_rgb(242, 243, 244),
                    );
                    let pointer = ctx.input(|i| i.pointer.hover_pos());
                    let pointer_moved = pointer != editor.pointer;
                    editor.pointer = pointer;
                    for (index, key) in key_list.iter().enumerate() {
                        let can = enabled(editor, key);
                        let response = ui.interact(
                            key.rect,
                            egui::Id::new(("osk-key", index)),
                            egui::Sense::click(),
                        );
                        if pointer_moved && response.hovered() {
                            editor.selected = index;
                        }
                        let accepting = key.action == Action::Accept;
                        let fill = if accepting {
                            Color32::from_rgb(55, 74, 220)
                        } else {
                            Color32::from_rgb(227, 229, 231)
                        };
                        ui.painter().rect_filled(key.rect, 0.0, fill);
                        let color = if !can {
                            Color32::from_gray(158)
                        } else if accepting {
                            Color32::WHITE
                        } else {
                            Color32::from_gray(45)
                        };
                        let font = if key.label.len() == 1 { 30.0 } else { 20.0 };
                        ui.painter().text(
                            key.rect.center(),
                            egui::Align2::CENTER_CENTER,
                            &key.label,
                            FontId::proportional(font * scale),
                            color,
                        );
                        if index == editor.selected {
                            ui.painter().rect_stroke(
                                key.rect,
                                0.0,
                                Stroke::new(4.0 * scale, Color32::from_rgb(30, 218, 191)),
                                StrokeKind::Inside,
                            );
                        }
                        if can && response.clicked() {
                            editor.selected = index;
                            editor.action(&key.action);
                        }
                    }
                    let cancel = Rect::from_min_size(
                        origin + vec2(960.0, 655.0) * scale,
                        vec2(260.0, 60.0) * scale,
                    );
                    let cancel_response = ui.interact(
                        cancel,
                        egui::Id::new("keyboard-cancel"),
                        egui::Sense::click(),
                    );
                    ui.painter().text(
                        cancel.center(),
                        egui::Align2::CENTER_CENTER,
                        "X Cancel",
                        FontId::proportional(20.0 * scale),
                        if editor.session.config.cancel_disabled {
                            Color32::GRAY
                        } else {
                            Color32::from_gray(45)
                        },
                    );
                    if cancel_response.clicked() {
                        editor.action(&Action::Cancel);
                    }
                    let footer = "L/R Cursor    Stick click Shift    B Delete    A Select    + OK";
                    ui.painter().text(
                        origin + vec2(485.0, 687.0) * scale,
                        egui::Align2::CENTER_CENTER,
                        footer,
                        FontId::proportional(18.0 * scale),
                        Color32::from_gray(45),
                    );
                });
            if let Some(feedback) = editor.feedback.clone() {
                let confirming = matches!(feedback, Validation::Confirm(_));
                let message = match feedback {
                    Validation::Failure(m) | Validation::Confirm(m) => m,
                    Validation::Retry => String::new(),
                };
                egui::Window::new(if confirming { "Confirm" } else { "Check input" })
                    .collapsible(false)
                    .resizable(false)
                    .anchor(egui::Align2::CENTER_CENTER, vec2(0.0, 0.0))
                    .show(ctx, |ui| {
                        ui.label(message);
                        ui.horizontal(|ui| {
                            if confirming && ui.button("Cancel").clicked() {
                                editor.feedback = None;
                            }
                            if ui.button("OK").clicked() {
                                editor.feedback = None;
                                if confirming {
                                    editor.waiting = true;
                                    editor.submit(Submission::Accept {
                                        text: editor.text.clone(),
                                        confirmed: true,
                                    });
                                }
                            }
                        });
                    });
            }
        });
        self.input
            .handle_platform_output(window, output.platform_output);
        let pixels = output.pixels_per_point;
        let jobs = ctx.tessellate(output.shapes, pixels);
        Some(nixe_gpu_vulkan::presentation::UiFrame {
            primitives: jobs,
            textures: output.textures_delta,
            pixels_per_point: pixels,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn editor() -> (Editor, Arc<std::sync::Mutex<Vec<Submission>>>) {
        let mut common = vec![0; 32];
        common[..4].copy_from_slice(&1u32.to_le_bytes());
        common[4..8].copy_from_slice(&32u32.to_le_bytes());
        common[8..12].copy_from_slice(&0x8000du32.to_le_bytes());
        let mut args = vec![0; 0x4c8];
        args[0x3ac..0x3b0].copy_from_slice(&8u32.to_le_bytes());
        let config = KeyboardConfig::decode(&common, &args, &[]).unwrap();
        let host = KeyboardHost::new(|| {});
        let submissions = Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink = submissions.clone();
        host.open(config, move |value| sink.lock().unwrap().push(value))
            .unwrap();
        (Editor::new(host.take_request().unwrap()), submissions)
    }
    #[test]
    fn editing_replaces_selection_and_filters_ascii_before_length_limit() {
        let (mut e, _) = editor();
        e.insert("abcd");
        e.cursor = 1..3;
        e.insert("éXYZ");
        assert_eq!(e.text, "aXYZd");
        assert_eq!(e.cursor, 4..4);
        e.backspace();
        assert_eq!(e.text, "aXYd");
        e.insert("123456789");
        assert_eq!(e.text, "aXY1234d");
        e.cursor = 0..e.text.len();
        e.insert("é");
        assert_eq!(e.text, "aXY1234d");
        e.backspace();
        assert!(e.text.is_empty());
    }
    #[test]
    fn controller_open_requires_release_and_short_pulses_select_without_leaking() {
        let (mut e, submissions) = editor();
        let keys = keys(&e.session.config, false, false, egui::pos2(0.0, 0.0), 100.0);
        let now = Instant::now();
        let mut state = EmulatedControllerState::default();
        state.buttons.a = true;
        e.controller(state, &keys, now);
        assert!(e.text.is_empty());
        e.controller(Default::default(), &keys, now);
        e.controller(state, &keys, now);
        assert_eq!(e.text, "1");
        e.controller(state, &keys, now);
        assert_eq!(e.text, "1");
        state.buttons.a = false;
        state.buttons.dpad_right = true;
        e.controller(state, &keys, now);
        assert_eq!(e.selected, 1);
        state.buttons.dpad_right = false;
        state.buttons.a = true;
        e.controller(state, &keys, now);
        assert_eq!(e.text, "12");
        e.controller(Default::default(), &keys, now);
        state = Default::default();
        state.buttons.plus = true;
        e.controller(state, &keys, now);
        assert!(
            matches!(e.pending_submission.as_ref(),Some(Submission::Accept {text,..}) if text=="12")
        );
        assert!(submissions.lock().unwrap().is_empty());
        e.session.submit(e.pending_submission.take().unwrap());
        assert!(
            matches!(submissions.lock().unwrap().as_slice(),[Submission::Accept {text,..}] if text=="12")
        );
    }
    #[test]
    fn switch_stick_positive_y_moves_up_like_the_dpad() {
        let (mut editor, _) = editor();
        let keys = keys(
            &editor.session.config,
            false,
            false,
            egui::pos2(0.0, 0.0),
            100.0,
        );
        let now = Instant::now();
        editor.controller(Default::default(), &keys, now);
        let mut state = EmulatedControllerState::default();
        state.left_stick.y = -32768;
        editor.controller(state, &keys, now);
        assert_eq!(editor.selected, 11);
        editor.controller(Default::default(), &keys, now);
        state.left_stick.y = 32767;
        editor.controller(state, &keys, now);
        assert_eq!(editor.selected, 0);
    }
    #[test]
    fn ascii_pages_cover_every_printable_character_and_spatial_navigation() {
        let mut common = vec![0; 32];
        common[..4].copy_from_slice(&1u32.to_le_bytes());
        common[4..8].copy_from_slice(&32u32.to_le_bytes());
        common[8..12].copy_from_slice(&0x8000du32.to_le_bytes());
        let config = KeyboardConfig::decode(&common, &vec![0; 0x4c8], &[]).unwrap();
        let mut available = std::collections::BTreeSet::from([' ']);
        for (shift, symbols) in [(false, false), (true, false), (false, true)] {
            for key in keys(&config, shift, symbols, egui::pos2(0.0, 0.0), 100.0) {
                if let Action::Character(c) = key.action {
                    available.insert(c);
                }
            }
        }
        for c in ' '..='~' {
            assert!(available.contains(&c), "missing {c}");
        }
        let keys = keys(&config, false, false, egui::pos2(0.0, 0.0), 100.0);
        assert_eq!(neighbor(&keys, 0, (1, 0)), 1);
        assert_eq!(neighbor(&keys, 0, (0, 1)), 11);
    }
}
