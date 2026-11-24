//! Shared host UI rendering; inactive overlays do not produce egui frames.
use super::{Viewport, keyboard::KeyboardOverlay};
use egui::{Color32, FontFamily, FontId, Rect, pos2, vec2};
use nixe_applets::keyboard::{KeyboardHost, KeyboardSession};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use winit::{event::WindowEvent, window::Window};

const VOLUME_DURATION: Duration = Duration::from_secs(2);

mod icons {
    pub(super) use egui_phosphor::variants::fill::{SPEAKER_HIGH, SPEAKER_X};

    const GLYPHS: &[&str] = &[SPEAKER_HIGH, SPEAKER_X];
    const SOURCE: &[u8] = egui_phosphor::variants::bytes::fill::FONT;
    const FONT_LEN: usize = egui_phosphor::subset::subset_len(SOURCE, GLYPHS);
    static FONT: [u8; FONT_LEN] = egui_phosphor::subset::subset_into::<FONT_LEN>(SOURCE, GLYPHS);
    const FONT_NAME: &str = module_path!();

    pub(super) fn family() -> egui::FontFamily {
        egui::FontFamily::Name(FONT_NAME.into())
    }

    pub(super) fn add_as_family(fonts: &mut egui::FontDefinitions) {
        egui_phosphor::add_font_bytes_as_family(fonts, FONT_NAME, &FONT);
    }
}

pub(super) struct Overlay {
    ctx: egui::Context,
    input: egui_winit::State,
    keyboard: KeyboardOverlay,
    keyboard_host: Option<KeyboardHost>,
    volume: Option<VolumeNotice>,
    pub(super) compilation: nixe_gpu_vulkan::cache::CompilationCounts,
}

impl Overlay {
    pub fn new(window: &Window) -> Self {
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
            .insert(FontFamily::Proportional, vec!["DejaVu".into()]);
        fonts
            .families
            .insert(FontFamily::Monospace, vec!["DejaVu".into()]);
        icons::add_as_family(&mut fonts);
        ctx.set_fonts(fonts);
        ctx.set_visuals(egui::Visuals::light());
        let input = egui_winit::State::new(
            ctx.clone(),
            egui::ViewportId::ROOT,
            window,
            Some(window.scale_factor() as f32),
            None,
            Some(4096),
        );
        Self {
            ctx,
            input,
            keyboard: KeyboardOverlay::default(),
            keyboard_host: None,
            volume: None,
            compilation: Default::default(),
        }
    }

    pub fn open_keyboard(&mut self, session: Arc<KeyboardSession>, host: KeyboardHost) {
        self.keyboard.open(session);
        self.keyboard_host = Some(host);
    }

    pub fn keyboard_active(&self) -> bool {
        self.keyboard.active()
    }

    pub fn event(&mut self, window: &Window, event: &WindowEvent) {
        let _ = self.input.on_window_event(window, event);
    }

    pub fn dispatch_submission(&mut self) {
        self.keyboard.dispatch_submission();
    }

    pub fn show_volume(&mut self, value: u8, now: Instant) {
        self.volume = Some(VolumeNotice {
            value,
            expires: now + VOLUME_DURATION,
        });
    }

    pub fn volume_deadline(&self) -> Option<Instant> {
        self.volume.as_ref().map(|volume| volume.expires)
    }

    pub fn expire_volume(&mut self, now: Instant) -> bool {
        if self
            .volume_deadline()
            .is_some_and(|deadline| now >= deadline)
        {
            self.volume = None;
            return true;
        }
        false
    }

    pub fn prepare(
        &mut self,
        window: &Window,
        viewport: Viewport,
    ) -> Option<nixe_gpu_vulkan::presentation::UiFrame> {
        if !self.keyboard_active()
            && self.volume.is_none()
            && self.compilation == Default::default()
        {
            return None;
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
        if let Some(host) = &self.keyboard_host {
            self.keyboard.prepare(host, window, &mut input);
        }
        let output = self.ctx.run_ui(input, |root| {
            let ctx = root.ctx();
            if let Some(host) = &self.keyboard_host {
                self.keyboard.show(host, ctx);
            }
            if self.compilation != Default::default() {
                let px = ctx.pixels_per_point().recip();
                let bottom = pos2(viewport.x + viewport.width, viewport.y + viewport.height) * px;
                let mut lines = Vec::with_capacity(2);
                for (n, verb, noun) in [
                    (self.compilation.shaders, "Compiling", "shader"),
                    (self.compilation.pipelines, "Building", "pipeline"),
                ] {
                    if n != 0 {
                        lines.push(format!(
                            "{verb} {n} {noun}{}",
                            if n == 1 { "" } else { "s" }
                        ));
                    }
                }
                let painter = ctx.layer_painter(egui::LayerId::new(
                    egui::Order::Tooltip,
                    egui::Id::new("compilation"),
                ));
                let galley = painter.layout_no_wrap(
                    lines.join("\n"),
                    FontId::proportional(17. * px),
                    Color32::WHITE,
                );
                let panel =
                    Rect::from_min_max(bottom - galley.size() - vec2(28., 16.) * px, bottom);
                painter.rect_filled(panel, 0., Color32::from_rgba_unmultiplied(32, 32, 32, 153));
                painter.galley(panel.min + vec2(14., 8.) * px, galley, Color32::WHITE);
            }
            if let Some(volume) = &self.volume {
                volume.show(ctx, viewport);
            }
        });
        self.input
            .handle_platform_output(window, output.platform_output);
        let pixels = output.pixels_per_point;
        Some(nixe_gpu_vulkan::presentation::UiFrame {
            primitives: self.ctx.tessellate(output.shapes, pixels),
            textures: output.textures_delta,
            pixels_per_point: pixels,
        })
    }
}

struct VolumeNotice {
    value: u8,
    expires: Instant,
}

impl VolumeNotice {
    fn show(&self, ctx: &egui::Context, viewport: Viewport) {
        // Keep the indicator's physical pixel size independent of render scaling and DPI.
        let pixel = ctx.pixels_per_point().recip();
        let origin = pos2(viewport.x, viewport.y) * pixel;
        let panel = Rect::from_min_size(origin, vec2(256., 56.) * pixel);
        let painter = ctx.layer_painter(egui::LayerId::new(
            egui::Order::Tooltip,
            egui::Id::new("volume-indicator"),
        ));
        painter.rect_filled(
            panel,
            0.,
            Color32::from_rgba_unmultiplied(0x20, 0x20, 0x20, 153),
        );
        let icon = if self.value == 0 {
            icons::SPEAKER_X
        } else {
            icons::SPEAKER_HIGH
        };
        painter.text(
            origin + vec2(40., 28.) * pixel,
            egui::Align2::CENTER_CENTER,
            icon,
            FontId::new(32. * pixel, icons::family()),
            Color32::WHITE,
        );
        let bar = Rect::from_min_size(origin + vec2(68., 25.) * pixel, vec2(164., 6.) * pixel);
        painter.rect_filled(bar, 3. * pixel, Color32::from_gray(100));
        if self.value != 0 {
            let filled = Rect::from_min_size(
                bar.min,
                vec2(bar.width() * f32::from(self.value) / 100., bar.height()),
            );
            painter.rect_filled(filled, 3. * pixel, Color32::WHITE);
        }
    }
}
