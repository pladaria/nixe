//! On-demand presentation capture. No readback or worker exists until requested.

use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::thread::JoinHandle;
use std::time::Duration;

pub(super) struct Screenshots {
    title: String,
    directory: PathBuf,
    requested: Option<PathBuf>,
    worker: Option<JoinHandle<()>>,
}

impl Screenshots {
    pub(super) fn has_request(&self) -> bool {
        self.requested.is_some()
    }

    pub(super) fn native(
        &mut self,
        client: &nixe_gpu_vulkan::presentation::PresentationClient,
    ) -> Option<std::sync::mpsc::Sender<Result<nixe_gpu_vulkan::presentation::Screenshot, String>>>
    {
        let path = self.requested.take()?;
        self.join_worker();
        let (sender, receiver) =
            std::sync::mpsc::channel::<Result<nixe_gpu_vulkan::presentation::Screenshot, String>>();
        let client = client.clone();
        match std::thread::Builder::new()
            .name("nixe-screenshot".into())
            .spawn(move || {
                let capture = loop {
                    match receiver.recv_timeout(Duration::from_millis(50)) {
                        Ok(result) => break result,
                        Err(std::sync::mpsc::RecvTimeoutError::Timeout) if !client.stopped() => {
                            continue;
                        }
                        _ => {
                            log::info!("screenshot canceled during presentation shutdown");
                            return;
                        }
                    }
                };
                let result = capture.map_err(io::Error::other).and_then(|capture| {
                    save_png(&path, (capture.width, capture.height), &capture.rgba)
                });
                match result {
                    Ok(()) => log::info!("saved screenshot {}", path.display()),
                    Err(error) => log::error!("cannot save screenshot {}: {error}", path.display()),
                }
            }) {
            Ok(worker) => {
                self.worker = Some(worker);
                Some(sender)
            }
            Err(error) => {
                log::error!("cannot start native screenshot writer: {error}");
                None
            }
        }
    }
    pub(super) fn new(title: String, directory: PathBuf) -> Self {
        Self {
            title: filename_title(&title),
            directory,
            requested: None,
            worker: None,
        }
    }

    pub(super) fn request(&mut self) {
        if self.requested.is_some() || self.worker.as_ref().is_some_and(|w| !w.is_finished()) {
            log::info!("screenshot already pending");
            return;
        }
        self.join_worker();
        let timestamp = chrono::Utc::now().format("%Y%m%dT%H%M%S%.9fZ");
        self.requested = Some(
            self.directory
                .join(format!("{}-{timestamp}.png", self.title)),
        );
        log::info!("screenshot requested; waiting for the next presented frame");
    }

    fn join_worker(&mut self) {
        if let Some(worker) = self.worker.take()
            && worker.join().is_err()
        {
            log::error!("screenshot writer panicked");
        }
    }
}

impl Drop for Screenshots {
    fn drop(&mut self) {
        // Finish an accepted capture even when the user immediately closes the
        // window. GPU waiting is bounded and is never done with the queue lock.
        self.join_worker();
        if self.requested.is_some() {
            log::warn!("screenshot cancelled: no new frame was presented before shutdown");
        }
    }
}

fn save_png(path: &Path, dimensions: (u32, u32), bytes: &[u8]) -> io::Result<()> {
    let (width, height) = dimensions;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    // Never overwrite an existing capture, even after a clock adjustment.
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    let result = (|| {
        let mut output = BufWriter::new(file);
        let mut encoder = png::Encoder::new(&mut output, width, height);
        // Like present.wgsl, display RGB without the guest's internal alpha.
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        // Only one CPU row of scratch, not another full image.
        // https://docs.rs/png/0.18.1/png/struct.Writer.html
        let mut writer = encoder.write_header()?;
        {
            let mut stream = writer.stream_writer()?;
            let mut row = vec![0; width as usize * 3];
            for y in 0..height {
                let start = y as usize * width as usize * 4;
                let rgba = &bytes[start..start + width as usize * 4];
                for (pixel, source) in row.chunks_exact_mut(3).zip(rgba.chunks_exact(4)) {
                    pixel.copy_from_slice(&source[..3]);
                }
                stream.write_all(&row)?;
            }
            stream.finish()?;
        }
        writer.finish()?;
        output.flush()
    })();
    if result.is_err() {
        // Only this invocation's newly created incomplete file can be removed.
        let _ = std::fs::remove_file(path);
    }
    result
}

fn filename_title(title: &str) -> String {
    let mut name = String::new();
    for c in title.chars().flat_map(char::to_lowercase) {
        let c = if c.is_alphanumeric() || c == '_' {
            c
        } else {
            '-'
        };
        if c == '-' && (name.is_empty() || name.ends_with('-')) {
            continue;
        }
        if name.len() + c.len_utf8() > 120 {
            break;
        }
        name.push(c);
    }
    let name = name.trim_end_matches('-');
    if name.is_empty() {
        "untitled".into()
    } else {
        name.into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn title_is_a_bounded_single_filename_component() {
        assert_eq!(
            filename_title("../Deko Examples2: Raster / tests"),
            "deko-examples2-raster-tests"
        );
        assert_eq!(filename_title("deko_examples2"), "deko_examples2");
        assert_eq!(filename_title(".../\\:*?"), "untitled");
        assert_eq!(filename_title("Café 日本"), "café-日本");
        assert!(filename_title(&"界".repeat(100)).len() <= 120);
    }

    #[test]
    fn repeated_requests_are_coalesced_until_a_frame_arrives() {
        let mut captures = Screenshots::new("Demo".into(), "captures".into());
        captures.request();
        let path = captures.requested.clone();
        captures.request();
        assert_eq!(captures.requested, path);
        assert!(
            path.unwrap()
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .starts_with("demo-")
        );
    }

    #[test]
    fn busy_writer_does_not_queue_additional_readbacks() {
        let mut captures = Screenshots::new("Demo".into(), "captures".into());
        let (tx, rx) = std::sync::mpsc::channel();
        captures.worker = Some(std::thread::spawn(move || rx.recv().unwrap()));
        captures.request();
        assert!(captures.requested.is_none());
        tx.send(()).unwrap();
        captures.join_worker();
        captures.request();
        assert!(captures.requested.is_some());
    }
}
