//! Direct lookup of validated byte-addressed class registers. Allocate storage
//! by 64-register pages, retaining unset values and exact write provenance.
use super::MaxwellThreeDRegister;

const PAGE_WORDS: usize = 64;
// The pushbuffer method address is twelve dword bits (pushbuffer/packet.rs).
const PAGES: usize = 0x1000 / PAGE_WORDS;
type Page = [Option<MaxwellThreeDRegister<u32>>; PAGE_WORDS];

#[derive(Clone, Debug, Eq, PartialEq)]
pub(in crate::engines::threed) struct Registers {
    pages: Box<[Option<Box<Page>>; PAGES]>,
}

impl Default for Registers {
    fn default() -> Self {
        Self {
            pages: Box::new(std::array::from_fn(|_| None)),
        }
    }
}

impl Registers {
    pub fn get(&self, method: &u32) -> Option<&MaxwellThreeDRegister<u32>> {
        if !method.is_multiple_of(4) {
            return None;
        }
        let word = *method as usize / 4;
        self.pages.get(word / PAGE_WORDS)?.as_ref()?[word % PAGE_WORDS].as_ref()
    }

    pub fn insert(&mut self, method: u32, value: MaxwellThreeDRegister<u32>) {
        assert!(method.is_multiple_of(4), "validated aligned class method");
        let word = method as usize / 4;
        let page = self.pages[word / PAGE_WORDS]
            .get_or_insert_with(|| Box::new(std::array::from_fn(|_| None)));
        page[word % PAGE_WORDS] = Some(value);
    }
}
