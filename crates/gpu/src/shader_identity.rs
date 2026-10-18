//! Content identities shared by shader capture, artifacts and host caches.
use sha2::{Digest, Sha256};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ContentId(pub [u8; 32]);
impl ContentId {
    /// Length framing makes distinct tuples unambiguous; all callers supply
    /// explicit encodings, never raw Rust struct memory or pointer addresses.
    pub fn of(domain: &[u8], parts: &[&[u8]]) -> Self {
        let mut h = Sha256::new();
        for part in std::iter::once(domain).chain(parts.iter().copied()) {
            h.update((part.len() as u64).to_le_bytes());
            h.update(part);
        }
        Self(h.finalize().into())
    }
    pub fn hex(self) -> String {
        self.0.iter().map(|b| format!("{b:02x}")).collect()
    }
}

/// Selected once by the concrete guest frontend, independently of host features.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ShaderNamespace {
    pub guest: [u8; 16],
    pub isa: u32,
}
