//! Optional per-region output from the direct Cranelift compiler.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use nixe_memory::GuestVirtualAddress;

static NEXT_SESSION: AtomicU64 = AtomicU64::new(0);

pub(crate) struct RegionDump {
    pub(crate) clif: String,
    pub(crate) native: Vec<u8>,
}

pub(super) struct Diagnostics {
    directory: PathBuf,
    next_region: AtomicU64,
}

impl Diagnostics {
    pub(super) fn new(root: &Path) -> Result<Self, Box<str>> {
        let session = NEXT_SESSION.fetch_add(1, Ordering::Relaxed);
        let directory = root.join(format!("session-{}-{session}", std::process::id()));
        fs::create_dir_all(&directory).map_err(|error| {
            format!(
                "cannot create JIT diagnostic directory {}: {error}",
                directory.display()
            )
            .into_boxed_str()
        })?;
        Ok(Self {
            directory,
            next_region: AtomicU64::new(0),
        })
    }

    pub(super) fn directory(&self) -> &Path {
        &self.directory
    }

    pub(super) fn write(
        &self,
        start: GuestVirtualAddress,
        dump: RegionDump,
    ) -> Result<(), Box<str>> {
        let sequence = self.next_region.fetch_add(1, Ordering::Relaxed);
        let stem = format!("region-{sequence:08}-{:016x}", start.get());
        write(
            &self.directory.join(format!("{stem}.clif")),
            dump.clif.as_bytes(),
        )?;
        write(&self.directory.join(format!("{stem}.bin")), &dump.native)
    }
}

fn write(path: &Path, contents: &[u8]) -> Result<(), Box<str>> {
    fs::write(path, contents).map_err(|error| {
        format!("cannot write JIT diagnostic {}: {error}", path.display()).into_boxed_str()
    })
}
