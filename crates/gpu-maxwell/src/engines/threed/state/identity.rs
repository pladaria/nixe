//! Incremental identity of consumed register values. Intermediate writes
//! which restore the same register values restore the same key too. Exact COW
//! pages reject fingerprint collisions and preserve unset versus programmed zero.
use std::sync::Arc;
const WORDS: usize = 64;
const PAGES: usize = 0x1000 / WORDS;
type Page = [Option<u32>; WORDS];
#[derive(Clone, Debug)]
pub(super) struct Registers {
    pub fingerprint: u128,
    pages: Arc<[Option<Arc<Page>>; PAGES]>,
}
impl Default for Registers {
    fn default() -> Self {
        Self {
            fingerprint: 0,
            pages: Arc::new(std::array::from_fn(|_| None)),
        }
    }
}
impl Registers {
    pub fn insert(&mut self, method: u32, value: u32) {
        let word = method as usize / 4;
        let old = self.pages[word / WORDS]
            .as_ref()
            .and_then(|p| p[word % WORDS]);
        if old == Some(value) {
            return;
        }
        if let Some(old) = old {
            self.fingerprint ^= nixe_gpu::cache_fingerprint(&(method, old));
        }
        self.fingerprint ^= nixe_gpu::cache_fingerprint(&(method, value));
        let page = Arc::make_mut(&mut self.pages)[word / WORDS]
            .get_or_insert_with(|| Arc::new([None; WORDS]));
        Arc::make_mut(page)[word % WORDS] = Some(value);
    }
    pub fn matches(&self, other: &Self) -> bool {
        self.fingerprint == other.fingerprint
            && (Arc::ptr_eq(&self.pages, &other.pages)
                || self
                    .pages
                    .iter()
                    .zip(other.pages.iter())
                    .all(|(a, b)| match (a, b) {
                        (Some(a), Some(b)) => Arc::ptr_eq(a, b) || a.as_ref() == b.as_ref(),
                        (None, None) => true,
                        _ => false,
                    }))
    }
}
// Placement is cheap; equality always checks the exact retained register pages.
impl PartialEq for Registers {
    fn eq(&self, other: &Self) -> bool {
        self.matches(other)
    }
}
impl Eq for Registers {}
impl std::hash::Hash for Registers {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::hash::Hash::hash(&self.fingerprint, state);
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn restored_values_reuse_identity_without_mutating_retained_pages() {
        let mut state = Registers::default();
        state.insert(0x1164, 0x3880_0001);
        let old = state.clone();
        state.insert(0x1164, 0x3880_0600);
        assert!(!old.matches(&state));
        state.insert(0x1164, 0x3880_0001);
        assert!(old.matches(&state));
        assert!(!Arc::ptr_eq(&old.pages, &state.pages));
        let empty = Registers::default();
        state.insert(0x1168, 0);
        assert!(!state.matches(&empty));
    }
    #[test]
    fn equal_fingerprints_never_substitute_for_exact_configuration() {
        let mut a = Registers::default();
        let mut b = Registers::default();
        a.insert(0x1c00, 0x100c);
        b.insert(0x1c00, 0x1018);
        b.fingerprint = a.fingerprint;
        assert!(!a.matches(&b));
    }
}
