//! Indexed admission and dirty domains, derived from the class definitions once.
use super::{Class, describe, fixed_graphics_method};

const WORDS: usize = 0xe00;
const VALID: u64 = 1 << 63;
const ORDINARY: u64 = 1 << 62;

/// Dirty domains carried with an admitted block, independently of its writes.
#[derive(Clone, Copy, Debug, Default)]
pub struct DirtyFlags(pub(crate) u64);
impl DirtyFlags {
    pub(crate) const BITS: u32 = 46;
    pub(crate) const INDEX: u64 = 1 << 32;
    pub(crate) const TARGETS: u32 = 33;
    pub(crate) const PREPARATION: u64 = 1 << 42;
    pub(crate) const DYNAMIC: u64 = 1 << 43;
    pub(crate) const VERTEX_INPUT: u64 = 1 << 44;
    pub(crate) const SHADER: u64 = 1 << 45;
}

#[derive(Clone, Copy)]
pub(crate) struct Method(u64);
impl Method {
    pub fn ordinary(self) -> bool {
        self.0 & ORDINARY != 0
    }
    pub fn dirty(self) -> DirtyFlags {
        DirtyFlags(self.0 & !(VALID | ORDINARY))
    }
}

pub(super) struct Table {
    methods: [Method; WORDS],
    #[cfg(target_arch = "x86_64")]
    pub valid: [u32; WORDS / 32],
}
impl Table {
    const fn new(class: Class) -> Self {
        let mut table = Self {
            methods: [Method(0); WORDS],
            #[cfg(target_arch = "x86_64")]
            valid: [0; WORDS / 32],
        };
        let mut index = 0;
        while index < WORDS {
            let method = index as u32 * 4;
            if let Some((_, ordinary)) = describe(class, method) {
                let dirty = if matches!(class, Class::ThreeD) {
                    graphics_dirty(method)
                } else {
                    0
                };
                assert!(dirty >> DirtyFlags::BITS == 0);
                table.methods[index] = Method(VALID | if ordinary { ORDINARY } else { 0 } | dirty);
                #[cfg(target_arch = "x86_64")]
                {
                    table.valid[index / 32] |= 1 << (index % 32);
                }
            }
            index += 1;
        }
        table
    }
    #[inline]
    pub fn method(&self, method: u32) -> Option<Method> {
        if method & 3 != 0 {
            return None;
        }
        let &info = self.methods.get((method / 4) as usize)?;
        (info.0 & VALID != 0).then_some(info)
    }
}

pub(super) fn table(class: Class) -> &'static Table {
    static GRAPHICS: Table = Table::new(Class::ThreeD);
    static COMPUTE: Table = Table::new(Class::Compute);
    static INLINE: Table = Table::new(Class::Inline);
    static TWO_D: Table = Table::new(Class::TwoD);
    static COPY: Table = Table::new(Class::Copy);
    match class {
        Class::ThreeD => &GRAPHICS,
        Class::Compute => &COMPUTE,
        Class::Inline => &INLINE,
        Class::TwoD => &TWO_D,
        Class::Copy => &COPY,
    }
}

const fn graphics_dirty(m: u32) -> u64 {
    let mut flags = 0;
    if matches!(m, 0x2000..=0x217c | 0x1240..=0x125c | 0x16a4 | 0x192c) {
        flags |= DirtyFlags::SHADER | DirtyFlags::PREPARATION;
    }
    flags |= match m {
        0xa00..=0xa1c
        | 0xe00..=0xe08
        | 0x131c..=0x1328
        | 0x1394..=0x139c
        | 0xf54..=0xf5c
        | 0x13b0..=0x13b4
        | 0x20c
        | 0x15bc
        | 0x187c
        | 0x156c => DirtyFlags::DYNAMIC,
        0x1160..=0x11dc | 0x1880..=0x18fc => DirtyFlags::VERTEX_INPUT,
        0x600..=0x6fc => 1 << ((m - 0x600) / 8),
        0x1c00..=0x1dfc => {
            (1 << ((m - 0x1c00) / 16))
                | if (m - 0x1c00).is_multiple_of(16) || (m - 0x1c00) % 16 == 12 {
                    DirtyFlags::VERTEX_INPUT
                } else {
                    0
                }
        }
        0x17c8..=0x17d8 => DirtyFlags::INDEX,
        // Resource contents/draw arguments have separate consumers. BEGIN's
        // topology depends on the value and is handled by the effect consumer.
        0x17dc..=0x17f8 | 0xd74..=0xd78 | 0x2380..=0x2490 | 0x1618 => 0,
        _ if fixed_graphics_method(m) => DirtyFlags::PREPARATION,
        _ => 0,
    };
    flags |= (match m {
        0x800..=0x9e4 => 1 << ((m - 0x800) / 0x40),
        0xfe0..=0xff0 | 0x1228..=0x1230 | 0x1538 => 0x100,
        0xfb8 | 0xff4 | 0xff8 | 0x121c => 0x1ff,
        _ => 0,
    }) << DirtyFlags::TARGETS;
    flags
}
