//! On-demand presentation capture. No readback or worker exists until requested.

use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::thread::JoinHandle;
use std::time::Duration;

use nixe_video::{FrameCrop, FrameTransform, PresentationFrame};
use wgpu::{CommandEncoder, Device};

pub(super) struct Screenshots {
    title: String,
    directory: PathBuf,
    requested: Option<PathBuf>,
    worker: Option<JoinHandle<()>>,
}

impl Screenshots {
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

    pub(super) fn encode(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        frame: &PresentationFrame,
    ) -> Option<Capture> {
        let path = self.requested.take()?;
        let texture =
            nixe_gpu_wgpu::resident_texture(frame.image()).expect("validated resident frame");
        match Capture::encode(
            device,
            encoder,
            texture,
            frame.crop(),
            frame.transform(),
            path,
        ) {
            Ok(capture) => Some(capture),
            Err(error) => {
                log::error!("cannot capture presentation image: {error}");
                None
            }
        }
    }

    pub(super) fn save(
        &mut self,
        device: Device,
        submission: wgpu::SubmissionIndex,
        capture: Capture,
    ) {
        // Bounded to one in-flight capture. Do not hold WgpuQueueAccess while
        // polling or writing the file: guest execution/presentation can continue.
        match std::thread::Builder::new()
            .name("nixe-screenshot".into())
            .spawn(move || match capture.save(&device, submission) {
                Ok(()) => log::info!("saved screenshot {}", capture.path.display()),
                Err(error) => {
                    log::error!("cannot save screenshot {}: {error}", capture.path.display())
                }
            }) {
            Ok(worker) => self.worker = Some(worker),
            Err(error) => log::error!("cannot start screenshot writer: {error}"),
        }
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

pub(super) struct Capture {
    buffer: wgpu::Buffer,
    dimensions: (u32, u32),
    row_pitch: u32,
    path: PathBuf,
    transform: FrameTransform,
    bgra: bool,
}

impl Capture {
    fn encode(
        device: &Device,
        encoder: &mut CommandEncoder,
        texture: &wgpu::Texture,
        crop: FrameCrop,
        transform: FrameTransform,
        path: PathBuf,
    ) -> io::Result<Self> {
        // No rendering, filtering or color-space conversion: copy the actual
        // resident presentation bytes. A future format must be handled explicitly.
        let bgra = match texture.format() {
            wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => false,
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => true,
            format => {
                return Err(io::Error::other(format!(
                    "unsupported screenshot format {format:?}"
                )));
            }
        };
        let (width, height) = (crop.width, crop.height);
        let row_pitch = (width * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Nixe on-demand screenshot readback"),
            size: u64::from(row_pitch) * u64::from(height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                origin: wgpu::Origin3d {
                    x: crop.left,
                    y: crop.top,
                    z: 0,
                },
                ..texture.as_image_copy()
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row_pitch),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        Ok(Self {
            buffer,
            dimensions: (width, height),
            row_pitch,
            path,
            transform,
            bgra,
        })
    }

    fn save(&self, device: &Device, submission: wgpu::SubmissionIndex) -> io::Result<()> {
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        self.buffer
            .map_async(wgpu::MapMode::Read, .., move |result| {
                let _ = tx.send(result);
            });
        // https://docs.rs/wgpu/30.0.0/wgpu/type.PollType.html
        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: Some(Duration::from_secs(30)),
            })
            .map_err(io::Error::other)?;
        // Another device-polling thread may be delivering the mapping callback.
        // Wait for that callback rather than treating its scheduling as failure.
        rx.recv_timeout(Duration::from_secs(5))
            .map_err(io::Error::other)?
            .map_err(io::Error::other)?;
        let result = {
            let bytes = self.buffer.get_mapped_range(..).map_err(io::Error::other)?;
            save_png(
                &self.path,
                self.dimensions,
                self.row_pitch,
                &bytes,
                self.transform,
                self.bgra,
            )
        };
        self.buffer.unmap();
        result
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

    #[test]
    #[ignore = "requires a Vulkan GPU; tests real cropped RGBA/BGRA readback to PNG"]
    fn gpu_framebuffer_dump_preserves_pixels_crop_and_all_transforms() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let Some(adapter) = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::VULKAN))
            .into_iter()
            .find(|adapter| {
                matches!(
                    adapter.get_info().device_type,
                    wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::IntegratedGpu
                )
            })
        else {
            eprintln!("SKIP: framebuffer readback requires a physical Vulkan GPU");
            return;
        };
        let (device, queue) =
            pollster::block_on(adapter.request_device(&Default::default())).unwrap();
        let directory = tempfile::tempdir().unwrap();
        for (bgra, format) in [
            (false, wgpu::TextureFormat::Rgba8Unorm),
            (true, wgpu::TextureFormat::Bgra8Unorm),
            (false, wgpu::TextureFormat::Rgba8UnormSrgb),
            (true, wgpu::TextureFormat::Bgra8UnormSrgb),
        ] {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("screenshot source with crop padding"),
                size: wgpu::Extent3d {
                    width: 4,
                    height: 5,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let mut bytes = vec![199; 4 * 5 * 4];
            for y in 0..3 {
                for x in 0..2 {
                    let n = (y * 2 + x + 1) as u8;
                    let offset = ((y + 1) * 4 + x + 1) * 4;
                    let pixel = if bgra {
                        [n + 20, n + 10, n, 0]
                    } else {
                        [n, n + 10, n + 20, 0]
                    };
                    bytes[offset..offset + 4].copy_from_slice(&pixel);
                }
            }
            queue.write_texture(
                texture.as_image_copy(),
                &bytes,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(16),
                    rows_per_image: Some(5),
                },
                texture.size(),
            );
            for (bits, expected) in TRANSFORMED.iter().enumerate() {
                let path = directory.path().join(format!("{format:?}-{bits}.png"));
                let mut encoder = device.create_command_encoder(&Default::default());
                let capture = Capture::encode(
                    &device,
                    &mut encoder,
                    &texture,
                    FrameCrop {
                        left: 1,
                        top: 1,
                        width: 2,
                        height: 3,
                    },
                    transform(bits),
                    path.clone(),
                )
                .unwrap();
                let submission = queue.submit([encoder.finish()]);
                capture.save(&device, submission).unwrap();
                let mut reader =
                    png::Decoder::new(std::io::BufReader::new(std::fs::File::open(path).unwrap()))
                        .read_info()
                        .unwrap();
                let mut decoded = vec![0; reader.output_buffer_size().unwrap()];
                let info = reader.next_frame(&mut decoded).unwrap();
                let dimensions = if bits & 4 != 0 { (3, 2) } else { (2, 3) };
                assert_eq!((info.width, info.height), dimensions);
                let expected: Vec<_> = expected.iter().flat_map(|n| [*n, n + 10, n + 20]).collect();
                assert_eq!(decoded, expected, "bgra={bgra} transform bits={bits}");
            }
        }
    }
}
