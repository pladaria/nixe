//! On-demand presentation capture. No readback or worker exists until requested.

use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::thread::JoinHandle;
use std::time::Duration;

use nixe_video::FrameTransform;

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
                    save_png(
                        &path,
                        (capture.width, capture.height),
                        capture.width * 4,
                        &capture.rgba,
                        FrameTransform::default(),
                        false,
                    )
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

fn save_png(
    path: &Path,
    dimensions: (u32, u32),
    pitch: u32,
    bytes: &[u8],
    transform: FrameTransform,
    bgra: bool,
) -> io::Result<()> {
    let (width, height) = if transform.rotate_90_clockwise {
        (dimensions.1, dimensions.0)
    } else {
        dimensions
    };
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
                rgb_row(&mut row, y, dimensions, pitch, bytes, transform, bgra);
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

fn rgb_row(
    row: &mut [u8],
    y: u32,
    (width, height): (u32, u32),
    pitch: u32,
    bytes: &[u8],
    transform: FrameTransform,
    bgra: bool,
) {
    for (x, pixel) in row.chunks_exact_mut(3).enumerate() {
        // Exact integer version of present.wgsl's normalized-coordinate mapping:
        // rotate first, then flip each source axis, with no interpolation.
        let (mut sx, mut sy) = if transform.rotate_90_clockwise {
            (y, height - 1 - x as u32)
        } else {
            (x as u32, y)
        };
        if transform.flip_horizontal {
            sx = width - 1 - sx;
        }
        if transform.flip_vertical {
            sy = height - 1 - sy;
        }
        let offset = sy as usize * pitch as usize + sx as usize * 4;
        let source = &bytes[offset..offset + 4];
        if bgra {
            pixel.copy_from_slice(&[source[2], source[1], source[0]]);
        } else {
            pixel.copy_from_slice(&source[..3]);
        }
    }
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
    fn png_preserves_channels_rows_and_does_not_overwrite() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("nested/capture.png");
        let pixels = [10, 20, 30, 255, 40, 50, 60, 255];
        let mut padded = vec![99; 512];
        padded[..4].copy_from_slice(&pixels[..4]);
        padded[256..260].copy_from_slice(&pixels[4..]);
        save_png(
            &path,
            (1, 2),
            256,
            &padded,
            FrameTransform::default(),
            false,
        )
        .unwrap();
        let original = std::fs::read(&path).unwrap();
        let mut reader = png::Decoder::new(std::io::Cursor::new(&original))
            .read_info()
            .unwrap();
        let mut decoded = vec![0; reader.output_buffer_size().unwrap()];
        let info = reader.next_frame(&mut decoded).unwrap();
        assert_eq!((info.width, info.height), (1, 2));
        assert_eq!(decoded, [10, 20, 30, 40, 50, 60]);
        assert_eq!(
            save_png(
                &path,
                (1, 2),
                256,
                &padded,
                FrameTransform::default(),
                false
            )
            .unwrap_err()
            .kind(),
            io::ErrorKind::AlreadyExists
        );
        assert_eq!(std::fs::read(&path).unwrap(), original);
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

    #[test]
    fn output_directory_error_preserves_existing_file() {
        let directory = tempfile::tempdir().unwrap();
        let blocker = directory.path().join("not-a-directory");
        std::fs::write(&blocker, "preserved").unwrap();
        let path = blocker.join("capture.png");
        assert!(save_png(&path, (1, 2), 4, &[], FrameTransform::default(), false).is_err());
        assert!(!path.exists());
        assert_eq!(std::fs::read_to_string(blocker).unwrap(), "preserved");
    }

    const TRANSFORMED: [[u8; 6]; 8] = [
        [1, 2, 3, 4, 5, 6],
        [2, 1, 4, 3, 6, 5],
        [5, 6, 3, 4, 1, 2],
        [6, 5, 4, 3, 2, 1],
        [5, 3, 1, 6, 4, 2],
        [6, 4, 2, 5, 3, 1],
        [1, 3, 5, 2, 4, 6],
        [2, 4, 6, 1, 3, 5],
    ];

    fn transform(bits: usize) -> FrameTransform {
        FrameTransform {
            flip_horizontal: bits & 1 != 0,
            flip_vertical: bits & 2 != 0,
            rotate_90_clockwise: bits & 4 != 0,
        }
    }

    #[test]
    fn every_transform_matches_presentation_pixel_mapping() {
        let bytes: Vec<_> = (1..=6).flat_map(|n| [n, n + 10, n + 20, 0]).collect();
        for (bits, expected) in TRANSFORMED.iter().enumerate() {
            let width = if bits & 4 != 0 { 3 } else { 2 };
            let mut actual = vec![0; 18];
            for (y, row) in actual.chunks_exact_mut(width * 3).enumerate() {
                rgb_row(row, y as u32, (2, 3), 8, &bytes, transform(bits), false);
            }
            let expected: Vec<_> = expected.iter().flat_map(|n| [*n, n + 10, n + 20]).collect();
            assert_eq!(actual, expected, "transform bits={bits}");
        }
    }
}
