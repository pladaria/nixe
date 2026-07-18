use sha2::{Digest, Sha256};
use swiitx_loader_storage::{LoadError, StorageRef};

const HASH_SIZE: u64 = 0x20;
const HASH_BUFFER_SIZE: usize = 1024 * 1024;

/// Identifies an integrity check performed for an NCA section.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntegrityCheckKind {
    FsHeader,
    HierarchicalSha256Master,
    HierarchicalSha256Data,
    IvfcLevel(u8),
    BktrVirtualImage,
}

/// Result of an individual integrity check.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntegrityStatus {
    Valid,
    Invalid,
    Unchecked,
}

/// One named result in an NCA section integrity report.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IntegrityCheck {
    pub kind: IntegrityCheckKind,
    pub status: IntegrityStatus,
}

/// Integrity results for one NCA section.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IntegrityReport {
    section_index: u8,
    checks: Vec<IntegrityCheck>,
}

impl IntegrityReport {
    pub(crate) fn new(section_index: u8, checks: Vec<IntegrityCheck>) -> Self {
        Self {
            section_index,
            checks,
        }
    }

    pub const fn section_index(&self) -> u8 {
        self.section_index
    }

    pub fn checks(&self) -> &[IntegrityCheck] {
        &self.checks
    }

    /// Returns true only when every advertised check was performed and passed.
    pub fn is_valid(&self) -> bool {
        !self.checks.is_empty()
            && self
                .checks
                .iter()
                .all(|check| check.status == IntegrityStatus::Valid)
    }
}

#[derive(Clone, Debug)]
pub(crate) enum IntegrityLayout {
    None,
    HierarchicalSha256(Sha256Layout),
    Ivfc(IvfcLayout),
    Bktr(IvfcLayout),
}

#[derive(Clone, Debug)]
pub(crate) struct Sha256Layout {
    pub master_hash: [u8; 32],
    pub hash_table_offset: u64,
    pub hash_table_size: u64,
    pub data_offset: u64,
    pub data_size: u64,
    pub block_size: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct IvfcLayout {
    pub master_hashes: Vec<[u8; 32]>,
    pub levels: Vec<IvfcLevel>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct IvfcLevel {
    pub offset: u64,
    pub size: u64,
    pub block_size: u64,
}

pub(crate) fn validate(
    section_index: u8,
    header_hash_valid: bool,
    storage: &StorageRef,
    layout: &IntegrityLayout,
) -> Result<IntegrityReport, LoadError> {
    let mut checks = vec![IntegrityCheck {
        kind: IntegrityCheckKind::FsHeader,
        status: status(header_hash_valid),
    }];

    match layout {
        IntegrityLayout::None => {}
        IntegrityLayout::Bktr(_) => checks.push(IntegrityCheck {
            kind: IntegrityCheckKind::BktrVirtualImage,
            status: IntegrityStatus::Unchecked,
        }),
        IntegrityLayout::HierarchicalSha256(layout) => {
            let actual_master =
                hash_range(storage, layout.hash_table_offset, layout.hash_table_size)?;
            checks.push(IntegrityCheck {
                kind: IntegrityCheckKind::HierarchicalSha256Master,
                status: status(actual_master == layout.master_hash),
            });

            let valid = validate_hash_layer(
                storage,
                layout.data_offset,
                layout.data_size,
                layout.block_size,
                false,
                ExpectedHashes::Storage {
                    offset: layout.hash_table_offset,
                },
            )?;
            checks.push(IntegrityCheck {
                kind: IntegrityCheckKind::HierarchicalSha256Data,
                status: status(valid),
            });
        }
        IntegrityLayout::Ivfc(layout) => {
            for (index, level) in layout.levels.iter().enumerate() {
                let expected = if index == 0 {
                    ExpectedHashes::Inline(&layout.master_hashes)
                } else {
                    ExpectedHashes::Storage {
                        offset: layout.levels[index - 1].offset,
                    }
                };
                let valid = validate_hash_layer(
                    storage,
                    level.offset,
                    level.size,
                    level.block_size,
                    true,
                    expected,
                )?;
                checks.push(IntegrityCheck {
                    kind: IntegrityCheckKind::IvfcLevel(
                        u8::try_from(index).expect("IVFC has at most six levels"),
                    ),
                    status: status(valid),
                });
            }
        }
    }

    Ok(IntegrityReport::new(section_index, checks))
}

fn status(valid: bool) -> IntegrityStatus {
    if valid {
        IntegrityStatus::Valid
    } else {
        IntegrityStatus::Invalid
    }
}

enum ExpectedHashes<'a> {
    Inline(&'a [[u8; 32]]),
    Storage { offset: u64 },
}

fn validate_hash_layer(
    storage: &StorageRef,
    data_offset: u64,
    data_size: u64,
    block_size: u64,
    pad_last_block: bool,
    expected_hashes: ExpectedHashes<'_>,
) -> Result<bool, LoadError> {
    let block_count = data_size.div_ceil(block_size);
    for block_index in 0..block_count {
        let relative_offset = block_index
            .checked_mul(block_size)
            .ok_or_else(|| LoadError::invalid("NCA", "integrity block offset overflows"))?;
        let offset = data_offset
            .checked_add(relative_offset)
            .ok_or_else(|| LoadError::invalid("NCA", "integrity data offset overflows"))?;
        let remaining = data_size - relative_offset;
        let read_size = remaining.min(block_size);
        let actual = hash_padded_range(
            storage,
            offset,
            read_size,
            pad_last_block.then_some(block_size),
        )?;

        let expected = match expected_hashes {
            ExpectedHashes::Inline(hashes) => {
                let index = usize::try_from(block_index)
                    .map_err(|_| LoadError::invalid("NCA", "integrity hash index is too large"))?;
                *hashes
                    .get(index)
                    .ok_or_else(|| LoadError::invalid("NCA", "master hash table is truncated"))?
            }
            ExpectedHashes::Storage { offset } => {
                let hash_offset = offset
                    .checked_add(
                        block_index
                            .checked_mul(HASH_SIZE)
                            .ok_or_else(|| LoadError::invalid("NCA", "hash offset overflows"))?,
                    )
                    .ok_or_else(|| LoadError::invalid("NCA", "hash offset overflows"))?;
                let mut expected = [0_u8; HASH_SIZE as usize];
                storage.read_at(hash_offset, &mut expected)?;
                expected
            }
        };

        if actual != expected {
            return Ok(false);
        }
    }
    Ok(true)
}

fn hash_range(storage: &StorageRef, offset: u64, size: u64) -> Result<[u8; 32], LoadError> {
    hash_padded_range(storage, offset, size, None)
}

fn hash_padded_range(
    storage: &StorageRef,
    offset: u64,
    size: u64,
    padded_size: Option<u64>,
) -> Result<[u8; 32], LoadError> {
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; HASH_BUFFER_SIZE];
    let mut consumed = 0_u64;

    while consumed < size {
        let chunk_size = (size - consumed).min(HASH_BUFFER_SIZE as u64);
        let chunk_len = usize::try_from(chunk_size)
            .map_err(|_| LoadError::invalid("NCA", "hash chunk is too large"))?;
        storage.read_at(
            offset
                .checked_add(consumed)
                .ok_or_else(|| LoadError::invalid("NCA", "hash read offset overflows"))?,
            &mut buffer[..chunk_len],
        )?;
        hasher.update(&buffer[..chunk_len]);
        consumed += chunk_size;
    }

    if let Some(padded_size) = padded_size {
        if padded_size < size {
            return Err(LoadError::invalid(
                "NCA",
                "integrity padded size is smaller than its data",
            ));
        }
        let zeros = [0_u8; 8192];
        let mut padding = padded_size - size;
        while padding != 0 {
            let chunk = padding.min(zeros.len() as u64);
            hasher.update(&zeros[..usize::try_from(chunk).expect("zero chunk fits")]);
            padding -= chunk;
        }
    }

    Ok(hasher.finalize().into())
}
