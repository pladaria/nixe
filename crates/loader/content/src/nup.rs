//! Unencrypted package trees and metadata, without NCA or console-runtime assumptions.
//! https://github.com/a55uka/nup/blob/meow/SPEC.md

mod cache;

use std::collections::BTreeSet;
use std::ops::Range;
use std::sync::{Arc, Mutex};

use nixe_loader_storage::{FormatLoader, LoadError, StorageRef, SubStorage};
use sha2::{Digest, Sha256};

use crate::binary::{read_u16, read_u32, read_u64};

const NONE: u32 = u32::MAX;
const MAX_TOC: u64 = 256 * 1024 * 1024;
const MAX_META: u64 = 16 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NupPackageKind {
    Base,
    Update,
    AddOn,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NupCodec {
    Raw,
    Zstandard,
}

/// Container metadata is independent of the optional guest NACP.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NupMetadata {
    pub npdm: Option<Arc<[u8]>>,
    pub nacp: Option<Arc<[u8]>>,
    pub icon: Option<Arc<[u8]>>,
    pub display_version: Option<String>,
    pub title: Option<String>,
    pub publisher: Option<String>,
    pub build_id: Option<[u8; 32]>,
    pub source_name: Option<String>,
    pub localized_titles: Vec<(String, String)>,
    pub localized_icons: Vec<(String, Arc<[u8]>)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NupPackage {
    pub title_id: u64,
    pub version: u32,
    pub kind: NupPackageKind,
    pub standalone_update: bool,
    pub root: u32,
    pub exefs: Option<u32>,
    pub romfs: Option<u32>,
    pub metadata: NupMetadata,
    /// Content identity independent of physical extent placement and compression.
    pub content_hash: [u8; 32],
}

#[derive(Clone, Debug)]
pub struct NupDirectory {
    pub name: String,
    children: Range<usize>,
}

#[derive(Clone, Debug)]
pub struct NupFile {
    pub name: String,
    pub size: u64,
    pub stored_offset: u64,
    pub stored_size: u64,
    pub sha256: [u8; 32],
    pub codec: NupCodec,
}

pub struct NupArchive {
    source: StorageRef,
    packages: Vec<NupPackage>,
    directories: Vec<NupDirectory>,
    files: Vec<NupFile>,
    children: Vec<u32>,
    blob_offset: u64,
    cache: Arc<Mutex<cache::Cache>>,
}

impl std::fmt::Debug for NupArchive {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NupArchive")
            .field("packages", &self.packages)
            .field("directories", &self.directories.len())
            .field("files", &self.files.len())
            .finish_non_exhaustive()
    }
}

impl NupArchive {
    pub fn packages(&self) -> &[NupPackage] {
        &self.packages
    }
    pub fn files(&self) -> &[NupFile] {
        &self.files
    }
    pub fn directories(&self) -> &[NupDirectory] {
        &self.directories
    }
    pub fn blob_offset(&self) -> u64 {
        self.blob_offset
    }

    /// Looks up a relative file path under a package directory. No host extraction occurs.
    pub fn find_file(&self, root: u32, path: &str) -> Option<usize> {
        let mut node = root as usize;
        let mut parts = path.split('/').peekable();
        while let Some(name) = parts.next() {
            if name.is_empty() || name == "." || name == ".." {
                return None;
            }
            let directory = self.directories.get(node)?;
            let children = &self.children[directory.children.clone()];
            let index = children
                .binary_search_by(|child| self.node_name(*child).cmp(name))
                .ok()?;
            node = children[index] as usize;
            if parts.peek().is_none() {
                return node.checked_sub(self.directories.len());
            }
        }
        None
    }

    pub fn open_file(&self, index: usize) -> Result<StorageRef, LoadError> {
        let file = self
            .files
            .get(index)
            .ok_or_else(|| invalid("file index is out of bounds"))?;
        let source: StorageRef = Arc::new(SubStorage::new(
            self.source.clone(),
            file.stored_offset,
            file.stored_size,
        )?);
        if file.codec == NupCodec::Raw {
            return Ok(source);
        }
        Ok(Arc::new(cache::CompressedFile {
            source,
            key: file.stored_offset,
            size: file.size,
            hash: file.sha256,
            cache: self.cache.clone(),
        }))
    }

    /// Materializes names and bounded sources, without reading or decoding file data.
    pub fn tree(&self, root: u32) -> Result<crate::ReadOnlyTree, LoadError> {
        if root as usize >= self.directories.len() {
            return Err(invalid("tree root is not a directory"));
        }
        let mut tree = crate::ReadOnlyTree::default();
        let mut pending = vec![(root, String::new())];
        while let Some((index, parent)) = pending.pop() {
            for child in &self.children[self.directories[index as usize].children.clone()] {
                let path = format!("{parent}/{}", self.node_name(*child));
                if (*child as usize) < self.directories.len() {
                    tree.insert_directory(&path)?;
                    pending.push((*child, path));
                } else {
                    tree.insert_file(
                        &path,
                        self.open_file(*child as usize - self.directories.len())?,
                    )?;
                }
            }
        }
        Ok(tree)
    }

    fn node_name(&self, node: u32) -> &str {
        let node = node as usize;
        if node < self.directories.len() {
            &self.directories[node].name
        } else {
            &self.files[node - self.directories.len()].name
        }
    }

    fn contains_directory(&self, root: u32, wanted: u32) -> bool {
        let mut pending = vec![root];
        let mut visited = BTreeSet::new();
        while let Some(node) = pending.pop() {
            if node == wanted {
                return true;
            }
            if visited.insert(node) {
                pending.extend(
                    self.children[self.directories[node as usize].children.clone()]
                        .iter()
                        .copied()
                        .filter(|child| (*child as usize) < self.directories.len()),
                );
            }
        }
        false
    }
}

pub struct NupLoader;

impl FormatLoader for NupLoader {
    type Output = NupArchive;
    const FORMAT_NAME: &'static str = "NUP";
    fn load(source: StorageRef) -> Result<NupArchive, LoadError> {
        parse(source)
    }
}

fn invalid(reason: impl Into<String>) -> LoadError {
    LoadError::invalid("NUP", reason)
}
fn zeros(bytes: &[u8]) -> Result<(), LoadError> {
    if bytes.iter().any(|b| *b != 0) {
        return Err(invalid("nonzero reserved field"));
    }
    Ok(())
}
fn region(offset: u64, size: u64, len: u64) -> Result<Range<u64>, LoadError> {
    let end = offset
        .checked_add(size)
        .filter(|end| *end <= len)
        .ok_or_else(|| invalid("region exceeds container bounds"))?;
    Ok(offset..end)
}
fn text(bytes: &[u8]) -> Result<String, LoadError> {
    let value = std::str::from_utf8(bytes).map_err(|_| invalid("metadata or name is not UTF-8"))?;
    if value.contains('\0') {
        return Err(invalid("embedded NUL in string"));
    }
    Ok(value.to_owned())
}

fn parse(source: StorageRef) -> Result<NupArchive, LoadError> {
    let size = source.len()?;
    if size < 0x80 || size > i64::MAX as u64 {
        return Err(invalid("invalid container size"));
    }
    let mut header = [0; 0x80];
    source.read_at(0, &mut header)?;
    let header_size = u64::from(read_u16(&header, 4));
    let toc_offset = read_u64(&header, 8);
    let toc_size = read_u64(&header, 16);
    let blob_offset = read_u64(&header, 56);
    let count = read_u32(&header, 72) as usize;
    if &header[..4] != b"NUP\x01"
        || header_size < 0x80
        || read_u16(&header, 6) == 0
        || toc_offset != header_size
        || !(48..=MAX_TOC).contains(&toc_size)
        || read_u64(&header, 64) != size
        || read_u32(&header, 76) != 0
        || count == 0
        || count > 1024
        || !blob_offset.is_multiple_of(16)
        || blob_offset > size
    {
        return Err(invalid("unsupported or invalid superblock"));
    }
    zeros(&header[96..])?;
    let toc_range = region(toc_offset, toc_size, size)?;
    if toc_range.end > blob_offset {
        return Err(invalid("TOC overlaps blob region"));
    }
    let mut toc = vec![0; toc_size as usize];
    source.read_at(toc_offset, &mut toc)?;
    if Sha256::digest(&toc).as_slice() != &header[24..56] {
        return Err(invalid("TOC SHA-256 mismatch"));
    }
    if &toc[..4] != b"TOC\x01" || read_u32(&toc, 4) as usize != count {
        return Err(invalid("invalid TOC header"));
    }
    zeros(&toc[40..48])?;
    let dirs = read_u64(&toc, 8);
    let files = read_u64(&toc, 16);
    if dirs == 0 || dirs > MAX_TOC / 24 || files > MAX_TOC / 80 {
        return Err(invalid("node count exceeds TOC limits"));
    }
    let pool = region(read_u64(&toc, 24), read_u64(&toc, 32), toc_size)?;
    let dir_offset = 48 + count * 64;
    let file_offset = dir_offset + dirs as usize * 24;
    let child_offset = file_offset + files as usize * 80;
    if child_offset as u64 > pool.start || !(pool.start - child_offset as u64).is_multiple_of(4) {
        return Err(invalid("TOC tables overlap string pool"));
    }
    let strings = &toc[pool.start as usize..pool.end as usize];
    let mut name_bytes = 0_u64;
    let mut name = |record: &[u8]| -> Result<String, LoadError> {
        let r = region(
            u64::from(read_u32(record, 0)),
            u64::from(read_u32(record, 4)),
            strings.len() as u64,
        )?;
        name_bytes += r.end - r.start;
        if name_bytes > MAX_TOC || r.end - r.start > 4096 {
            return Err(invalid(
                "expanded names exceed the loader's metadata limits",
            ));
        }
        let value = text(&strings[r.start as usize..r.end as usize])?;
        if value.contains(['/', '\\']) || value == "." || value == ".." {
            return Err(invalid("invalid file or directory name"));
        }
        Ok(value)
    };
    let children = toc[child_offset..pool.start as usize]
        .chunks_exact(4)
        .map(|b| read_u32(b, 0))
        .collect::<Vec<_>>();
    if children
        .iter()
        .any(|child| u64::from(*child) >= dirs + files)
    {
        return Err(invalid("child index is out of bounds"));
    }
    let mut directories = Vec::with_capacity(dirs as usize);
    for r in toc[dir_offset..file_offset].chunks_exact(24) {
        zeros(&r[16..])?;
        let range = region(
            u64::from(read_u32(r, 8)),
            u64::from(read_u32(r, 12)),
            children.len() as u64,
        )?;
        directories.push(NupDirectory {
            name: name(r)?,
            children: range.start as usize..range.end as usize,
        });
    }
    let mut entries = Vec::with_capacity(files as usize);
    // Exact shared extents are valid only for identical content, codec and sizes.
    let mut extents = Vec::with_capacity(files as usize);
    for r in toc[file_offset..child_offset].chunks_exact(80) {
        zeros(&r[65..])?;
        let codec = match r[64] {
            0 => NupCodec::Raw,
            1 => NupCodec::Zstandard,
            value => return Err(invalid(format!("unsupported file codec {value}"))),
        };
        let file = NupFile {
            name: name(r)?,
            size: read_u64(r, 8),
            stored_offset: read_u64(r, 16),
            stored_size: read_u64(r, 24),
            sha256: r[32..64].try_into().expect("fixed hash"),
            codec,
        };
        if file.name.is_empty()
            || file.size > i64::MAX as u64
            || (codec == NupCodec::Raw && file.stored_size != file.size)
        {
            return Err(invalid("invalid file name or size"));
        }
        if file.size == 0 {
            if file.stored_offset != 0 || file.stored_size != 0 || codec != NupCodec::Raw {
                return Err(invalid("invalid empty file"));
            }
        } else {
            let r = region(file.stored_offset, file.stored_size, size)?;
            if r.start < blob_offset || !r.start.is_multiple_of(16) || r.is_empty() {
                return Err(invalid("invalid file extent"));
            }
            extents.push((r.start, r.end, entries.len()));
        }
        entries.push(file);
    }
    extents.sort_unstable();
    for pair in extents.windows(2) {
        let (a, b) = (&entries[pair[0].2], &entries[pair[1].2]);
        if pair[0].1 > pair[1].0
            && !(a.stored_offset == b.stored_offset
                && a.stored_size == b.stored_size
                && a.size == b.size
                && a.sha256 == b.sha256
                && a.codec == b.codec)
        {
            return Err(invalid("overlapping or incompatible shared file extents"));
        }
    }
    let mut archive = NupArchive {
        source,
        packages: Vec::with_capacity(count),
        directories,
        files: entries,
        children,
        blob_offset,
        cache: Arc::new(Mutex::new(cache::Cache::default())),
    };
    let hashes = validate_tree(&archive)?;
    let mut metadata_ranges = Vec::new();
    let mut metadata_bytes = 0_u64;
    for r in toc[48..dir_offset].chunks_exact(64) {
        zeros(&r[26..32])?;
        zeros(&r[44..])?;
        let kind = match r[24] {
            0 => NupPackageKind::Base,
            1 => NupPackageKind::Update,
            2 => NupPackageKind::AddOn,
            v => return Err(invalid(format!("unsupported package kind {v}"))),
        };
        if r[25] & !1 != 0 || (r[25] != 0 && kind != NupPackageKind::Update) {
            return Err(invalid("invalid package flags"));
        }
        let root = read_u32(r, 8);
        let exefs = (read_u32(r, 12) != NONE).then(|| read_u32(r, 12));
        let romfs = (read_u32(r, 16) != NONE).then(|| read_u32(r, 16));
        if root as usize >= archive.directories.len()
            || exefs.into_iter().chain(romfs).any(|d| {
                d as usize >= archive.directories.len() || !archive.contains_directory(root, d)
            })
        {
            return Err(invalid("package directory is outside its tree"));
        }
        let offset = read_u64(r, 32);
        let len = u64::from(read_u32(r, 40));
        metadata_bytes += len;
        if len > MAX_META || metadata_bytes > 64 * 1024 * 1024 || (len == 0) != (offset == 0) {
            return Err(invalid("invalid metadata size or offset"));
        }
        let mut metadata = NupMetadata::default();
        if len != 0 {
            let range = region(offset, len, size)?;
            if offset < toc_range.end {
                return Err(invalid("metadata overlaps header or TOC"));
            }
            // Search the last extent beginning before the metadata ends.
            let index = extents.partition_point(|extent| extent.0 < range.end);
            if index != 0 && extents[index - 1].1 > range.start {
                return Err(invalid("metadata overlaps file extent"));
            }
            metadata_ranges.push(range);
            let mut bytes = vec![0; len as usize];
            archive.source.read_at(offset, &mut bytes)?;
            metadata = parse_metadata(&bytes)?;
        }
        let mut identity = Sha256::new();
        identity.update(hashes[root as usize]);
        for directory in [exefs, romfs] {
            match directory {
                Some(index) => {
                    identity.update([1]);
                    identity.update(hashes[index as usize]);
                }
                None => identity.update([0]),
            }
        }
        archive.packages.push(NupPackage {
            title_id: read_u64(r, 0),
            version: read_u32(r, 20),
            kind,
            standalone_update: r[25] != 0,
            root,
            exefs,
            romfs,
            metadata,
            content_hash: identity.finalize().into(),
        });
    }
    metadata_ranges.sort_by_key(|r| r.start);
    if metadata_ranges
        .windows(2)
        .any(|p| p[0].end > p[1].start && p[0] != p[1])
    {
        return Err(invalid("overlapping metadata blocks"));
    }
    let base_count = archive
        .packages
        .iter()
        .filter(|p| p.kind == NupPackageKind::Base)
        .count();
    if base_count > 1
        || archive
            .packages
            .iter()
            .any(|p| p.kind == NupPackageKind::Update && p.standalone_update != (base_count == 0))
    {
        return Err(invalid("inconsistent base/standalone update relationship"));
    }
    Ok(archive)
}

fn validate_tree(archive: &NupArchive) -> Result<Vec<[u8; 32]>, LoadError> {
    for dir in &archive.directories {
        let mut previous = None;
        for child in &archive.children[dir.children.clone()] {
            let name = archive.node_name(*child);
            if name.is_empty() || previous.is_some_and(|p| p >= name) {
                return Err(invalid("children are not uniquely sorted by name"));
            }
            previous = Some(name);
        }
    }
    let mut state = vec![0_u8; archive.directories.len()];
    let mut height = vec![0_usize; state.len()];
    let mut hashes = vec![[0; 32]; state.len()];
    for start in 0..state.len() {
        if state[start] != 0 {
            continue;
        }
        let mut stack = vec![(start, 0)];
        state[start] = 1;
        while let Some((node, next)) = stack.last_mut() {
            let children = &archive.children[archive.directories[*node].children.clone()];
            if *next == children.len() {
                let mut hash = Sha256::new();
                for child in children {
                    let index = *child as usize;
                    let name = archive.node_name(*child);
                    hash.update((name.len() as u64).to_le_bytes());
                    hash.update(name);
                    if index < state.len() {
                        height[*node] = height[*node].max(height[index] + 1);
                        hash.update([0]);
                        hash.update(hashes[index]);
                    } else {
                        let file = &archive.files[index - state.len()];
                        hash.update([1]);
                        hash.update(file.size.to_le_bytes());
                        hash.update(file.sha256);
                    }
                }
                if height[*node] > 256 {
                    return Err(invalid("directory tree exceeds depth limit"));
                }
                hashes[*node] = hash.finalize().into();
                state[*node] = 2;
                stack.pop();
                continue;
            }
            let child = children[*next] as usize;
            *next += 1;
            if child < state.len() {
                if state[child] == 1 {
                    return Err(invalid("directory cycle"));
                }
                if state[child] == 0 {
                    if stack.len() >= 256 {
                        return Err(invalid("directory tree exceeds depth limit"));
                    }
                    state[child] = 1;
                    stack.push((child, 0));
                }
            }
        }
    }
    Ok(hashes)
}

fn parse_metadata(bytes: &[u8]) -> Result<NupMetadata, LoadError> {
    let mut result = NupMetadata::default();
    let mut position = 0;
    let mut seen = BTreeSet::new();
    while position < bytes.len() {
        let header = bytes
            .get(position..position + 8)
            .ok_or_else(|| invalid("truncated metadata header"))?;
        let kind = read_u16(header, 0);
        zeros(&header[2..4])?;
        let range = region(
            (position + 8) as u64,
            u64::from(read_u32(header, 4)),
            bytes.len() as u64,
        )?;
        let value = &bytes[range.start as usize..range.end as usize];
        position = range.end as usize;
        if (1..=10).contains(&kind) && !seen.insert(kind) {
            return Err(invalid("duplicate metadata type"));
        }
        match kind {
            1 => result.npdm = Some(Arc::from(value)),
            2 => result.nacp = Some(Arc::from(value)),
            3 => {
                image(value)?;
                result.icon = Some(Arc::from(value));
            }
            4 => result.display_version = Some(text(value)?),
            5 => result.title = Some(text(value)?),
            6 => result.publisher = Some(text(value)?),
            7 => {
                result.build_id = Some(
                    value
                        .try_into()
                        .map_err(|_| invalid("invalid build ID size"))?,
                )
            }
            8 => result.source_name = Some(text(value)?),
            9 => {
                result.localized_titles = localized(value)?
                    .into_iter()
                    .map(|(lang, value)| Ok((lang, text(value)?)))
                    .collect::<Result<_, LoadError>>()?
            }
            10 => {
                result.localized_icons = localized(value)?
                    .into_iter()
                    .map(|(lang, value)| {
                        image(value)?;
                        Ok((lang, Arc::from(value)))
                    })
                    .collect::<Result<_, LoadError>>()?
            }
            _ => {}
        }
    }
    Ok(result)
}

fn localized(mut bytes: &[u8]) -> Result<Vec<(String, &[u8])>, LoadError> {
    let mut result = Vec::new();
    let mut seen = BTreeSet::new();
    while let Some((&len, rest)) = bytes.split_first() {
        let len = usize::from(len);
        if len == 0 || rest.len() < len + 4 {
            return Err(invalid("invalid localized metadata header"));
        }
        let lang = text(&rest[..len])?.to_ascii_lowercase();
        if !lang.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-')
            || !seen.insert(lang.clone())
        {
            return Err(invalid("invalid or duplicate language tag"));
        }
        let size = read_u32(rest, len) as usize;
        let payload = &rest[len + 4..];
        let data = payload
            .get(..size)
            .ok_or_else(|| invalid("truncated localized metadata"))?;
        result.push((lang, data));
        bytes = &payload[size..];
    }
    Ok(result)
}

fn image(bytes: &[u8]) -> Result<(), LoadError> {
    if bytes.starts_with(&[0xff, 0xd8, 0xff]) || bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        Ok(())
    } else {
        Err(invalid("icon is neither JPEG nor PNG"))
    }
}
