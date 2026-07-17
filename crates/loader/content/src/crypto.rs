use std::sync::Arc;

use aes::Aes128;
use aes::cipher::{BlockDecrypt, BlockEncrypt, KeyInit};
use swiitx_loader_storage::{Storage, StorageError, StorageRef};

const AES_BLOCK_SIZE: u64 = 0x10;

pub(crate) fn decrypt_ecb_blocks(key: &[u8; 16], data: &mut [u8]) {
    debug_assert_eq!(data.len() % AES_BLOCK_SIZE as usize, 0);
    let cipher = Aes128::new(key.into());
    for block in data.chunks_exact_mut(AES_BLOCK_SIZE as usize) {
        cipher.decrypt_block(block.into());
    }
}

pub(crate) fn decrypt_xts(key: &[u8; 32], data: &mut [u8], first_sector: u64, sector_size: usize) {
    debug_assert!(sector_size >= AES_BLOCK_SIZE as usize);
    debug_assert_eq!(sector_size % AES_BLOCK_SIZE as usize, 0);
    debug_assert_eq!(data.len() % sector_size, 0);

    let data_cipher = Aes128::new((&key[..16]).into());
    let tweak_cipher = Aes128::new((&key[16..]).into());

    for (sector_offset, sector) in data.chunks_exact_mut(sector_size).enumerate() {
        let sector_number = first_sector + u64::try_from(sector_offset).expect("sector index fits");
        let mut tweak = [0_u8; 16];
        tweak[8..].copy_from_slice(&sector_number.to_be_bytes());
        tweak_cipher.encrypt_block((&mut tweak).into());

        for block in sector.chunks_exact_mut(AES_BLOCK_SIZE as usize) {
            xor_block(block, &tweak);
            data_cipher.decrypt_block(block.into());
            xor_block(block, &tweak);
            multiply_tweak(&mut tweak);
        }
    }
}

fn xor_block(block: &mut [u8], mask: &[u8; 16]) {
    for (byte, mask_byte) in block.iter_mut().zip(mask) {
        *byte ^= mask_byte;
    }
}

fn multiply_tweak(tweak: &mut [u8; 16]) {
    let mut carry = 0_u8;
    for byte in tweak.iter_mut() {
        let next_carry = *byte >> 7;
        *byte = (*byte << 1) | carry;
        carry = next_carry;
    }
    if carry != 0 {
        tweak[0] ^= 0x87;
    }
}

#[derive(Clone)]
pub(crate) struct AesCtrStorage {
    parent: StorageRef,
    key: [u8; 16],
    counter_prefix: [u8; 8],
    absolute_offset: u64,
}

impl AesCtrStorage {
    pub(crate) fn new(
        parent: StorageRef,
        key: [u8; 16],
        counter_prefix: [u8; 8],
        absolute_offset: u64,
    ) -> Self {
        Self {
            parent,
            key,
            counter_prefix,
            absolute_offset,
        }
    }
}

impl std::fmt::Debug for AesCtrStorage {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AesCtrStorage")
            .field("absolute_offset", &self.absolute_offset)
            .finish_non_exhaustive()
    }
}

impl Storage for AesCtrStorage {
    fn len(&self) -> Result<u64, StorageError> {
        self.parent.len()
    }

    fn read_at(&self, offset: u64, buffer: &mut [u8]) -> Result<(), StorageError> {
        validate_range(self.parent.as_ref(), offset, buffer.len())?;
        if buffer.is_empty() {
            return Ok(());
        }

        let request_end = offset
            .checked_add(u64::try_from(buffer.len()).map_err(|_| StorageError::OutOfBounds)?)
            .ok_or(StorageError::OutOfBounds)?;
        let aligned_start = offset & !(AES_BLOCK_SIZE - 1);
        let aligned_end = align_up(request_end, AES_BLOCK_SIZE)?;
        let aligned_len =
            usize::try_from(aligned_end - aligned_start).map_err(|_| StorageError::OutOfBounds)?;
        let mut encrypted = vec![0_u8; aligned_len];
        self.parent.read_at(aligned_start, &mut encrypted)?;

        let absolute_start = self
            .absolute_offset
            .checked_add(aligned_start)
            .ok_or(StorageError::OutOfBounds)?;
        apply_ctr(
            &self.key,
            self.counter_prefix,
            absolute_start / AES_BLOCK_SIZE,
            &mut encrypted,
        );

        let within =
            usize::try_from(offset - aligned_start).map_err(|_| StorageError::OutOfBounds)?;
        buffer.copy_from_slice(&encrypted[within..within + buffer.len()]);
        Ok(())
    }
}

fn apply_ctr(key: &[u8; 16], prefix: [u8; 8], first_block: u64, data: &mut [u8]) {
    let cipher = Aes128::new(key.into());
    for (index, block) in data.chunks_mut(AES_BLOCK_SIZE as usize).enumerate() {
        let mut counter = [0_u8; 16];
        counter[..8].copy_from_slice(&prefix);
        let block_number =
            first_block + u64::try_from(index).expect("in-memory block index always fits");
        counter[8..].copy_from_slice(&block_number.to_be_bytes());
        cipher.encrypt_block((&mut counter).into());
        for (byte, key_byte) in block.iter_mut().zip(counter) {
            *byte ^= key_byte;
        }
    }
}

#[derive(Clone)]
pub(crate) struct AesXtsStorage {
    parent: StorageRef,
    key: [u8; 32],
    sector_size: u64,
}

impl AesXtsStorage {
    pub(crate) fn new(parent: StorageRef, key: [u8; 32], sector_size: u64) -> Self {
        Self {
            parent,
            key,
            sector_size,
        }
    }
}

impl std::fmt::Debug for AesXtsStorage {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AesXtsStorage")
            .field("sector_size", &self.sector_size)
            .finish_non_exhaustive()
    }
}

impl Storage for AesXtsStorage {
    fn len(&self) -> Result<u64, StorageError> {
        self.parent.len()
    }

    fn read_at(&self, offset: u64, buffer: &mut [u8]) -> Result<(), StorageError> {
        validate_range(self.parent.as_ref(), offset, buffer.len())?;
        if buffer.is_empty() {
            return Ok(());
        }

        let request_end = offset
            .checked_add(u64::try_from(buffer.len()).map_err(|_| StorageError::OutOfBounds)?)
            .ok_or(StorageError::OutOfBounds)?;
        let aligned_start = offset / self.sector_size * self.sector_size;
        let aligned_end = align_up(request_end, self.sector_size)?;
        let aligned_len =
            usize::try_from(aligned_end - aligned_start).map_err(|_| StorageError::OutOfBounds)?;
        let mut encrypted = vec![0_u8; aligned_len];
        self.parent.read_at(aligned_start, &mut encrypted)?;

        decrypt_xts(
            &self.key,
            &mut encrypted,
            aligned_start / self.sector_size,
            usize::try_from(self.sector_size).map_err(|_| StorageError::OutOfBounds)?,
        );

        let within =
            usize::try_from(offset - aligned_start).map_err(|_| StorageError::OutOfBounds)?;
        buffer.copy_from_slice(&encrypted[within..within + buffer.len()]);
        Ok(())
    }
}

fn validate_range(storage: &dyn Storage, offset: u64, len: usize) -> Result<(), StorageError> {
    let len = u64::try_from(len).map_err(|_| StorageError::OutOfBounds)?;
    let end = offset.checked_add(len).ok_or(StorageError::OutOfBounds)?;
    if end > storage.len()? {
        return Err(StorageError::OutOfBounds);
    }
    Ok(())
}

fn align_up(value: u64, alignment: u64) -> Result<u64, StorageError> {
    value
        .checked_add(alignment - 1)
        .map(|value| value / alignment * alignment)
        .ok_or(StorageError::OutOfBounds)
}

pub(crate) fn ctr_storage(
    parent: StorageRef,
    key: [u8; 16],
    section_counter: [u8; 8],
    absolute_offset: u64,
) -> StorageRef {
    let mut counter_prefix = section_counter;
    counter_prefix.reverse();
    Arc::new(AesCtrStorage::new(
        parent,
        key,
        counter_prefix,
        absolute_offset,
    ))
}

pub(crate) fn xts_storage(parent: StorageRef, key: [u8; 32]) -> StorageRef {
    Arc::new(AesXtsStorage::new(parent, key, 0x200))
}
#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct VecStorage(Vec<u8>);

    impl Storage for VecStorage {
        fn len(&self) -> Result<u64, StorageError> {
            Ok(self.0.len() as u64)
        }

        fn read_at(&self, offset: u64, buffer: &mut [u8]) -> Result<(), StorageError> {
            let start = usize::try_from(offset).map_err(|_| StorageError::OutOfBounds)?;
            let end = start
                .checked_add(buffer.len())
                .ok_or(StorageError::OutOfBounds)?;
            buffer.copy_from_slice(self.0.get(start..end).ok_or(StorageError::OutOfBounds)?);
            Ok(())
        }
    }

    #[test]
    fn decrypts_aes_ecb_known_vector() {
        let key = [0_u8; 16];
        let mut ciphertext = [
            0x66, 0xe9, 0x4b, 0xd4, 0xef, 0x8a, 0x2c, 0x3b, 0x88, 0x4c, 0xfa, 0x59, 0xca, 0x34,
            0x2b, 0x2e,
        ];
        decrypt_ecb_blocks(&key, &mut ciphertext);
        assert_eq!(ciphertext, [0_u8; 16]);
    }

    #[test]
    fn ctr_view_supports_unaligned_random_reads() {
        let ciphertext = vec![
            0x66, 0xe9, 0x4b, 0xd4, 0xef, 0x8a, 0x2c, 0x3b, 0x88, 0x4c, 0xfa, 0x59, 0xca, 0x34,
            0x2b, 0x2e,
        ];
        let parent: StorageRef = Arc::new(VecStorage(ciphertext));
        let view = AesCtrStorage::new(parent, [0_u8; 16], [0_u8; 8], 0);

        let mut whole = [0xFF_u8; 16];
        view.read_at(0, &mut whole).unwrap();
        assert_eq!(whole, [0_u8; 16]);

        let mut partial = [0xFF_u8; 7];
        view.read_at(3, &mut partial).unwrap();
        assert_eq!(partial, [0_u8; 7]);
    }
}
