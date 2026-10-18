use std::hash::{Hash, Hasher};
use twox_hash::xxhash3_128::{DEFAULT_SECRET_LENGTH, RawHasher, SecretBuffer};

/// Computes the non-cryptographic 128-bit identity used by in-memory GPU
/// caches. It is intentionally not a persistent serialization format.
///
/// Optimized builds treat the result as the identity. Development builds must
/// additionally compare the exact semantic key on a hit so collisions and
/// incomplete key definitions fail immediately.
#[must_use]
#[inline]
pub(super) fn cache_fingerprint<T: Hash + ?Sized>(value: &T) -> u128 {
    // The streaming convenience hasher allocates its default secret per call.
    // Borrow the same immutable secret with inline streaming state instead;
    // keep the algorithm/seed and therefore every existing cache identity.
    // https://docs.rs/twox-hash/2.1.3/twox_hash/xxhash3_128/struct.SecretBuffer.html#method.default
    let mut hasher = CacheFingerprintHasher(RawHasher::new(SecretBuffer::default()));
    value.hash(&mut hasher);
    hasher.0.finish_128()
}

struct CacheFingerprintHasher(RawHasher<&'static [u8; DEFAULT_SECRET_LENGTH]>);

impl Hasher for CacheFingerprintHasher {
    fn finish(&self) -> u64 {
        self.0.finish_128() as u64
    }

    fn write(&mut self, bytes: &[u8]) {
        self.0.write(bytes);
    }
}
