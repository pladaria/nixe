/// Reference type flags: unions are used by resource-width metadata and signatures.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Type(pub u32);

#[allow(non_upper_case_globals)]
impl Type {
    pub const Void: Self = Self(0);
    pub const Opaque: Self = Self(1 << 0);
    pub const Reg: Self = Self(1 << 1);
    pub const Pred: Self = Self(1 << 2);
    pub const Attribute: Self = Self(1 << 3);
    pub const Patch: Self = Self(1 << 4);
    pub const U1: Self = Self(1 << 5);
    pub const U8: Self = Self(1 << 6);
    pub const U16: Self = Self(1 << 7);
    pub const U32: Self = Self(1 << 8);
    pub const U64: Self = Self(1 << 9);
    pub const F16: Self = Self(1 << 10);
    pub const F32: Self = Self(1 << 11);
    pub const F64: Self = Self(1 << 12);
    pub const U32x2: Self = Self(1 << 13);
    pub const U32x3: Self = Self(1 << 14);
    pub const U32x4: Self = Self(1 << 15);
    pub const F16x2: Self = Self(1 << 16);
    pub const F16x3: Self = Self(1 << 17);
    pub const F16x4: Self = Self(1 << 18);
    pub const F32x2: Self = Self(1 << 19);
    pub const F32x3: Self = Self(1 << 20);
    pub const F32x4: Self = Self(1 << 21);
    pub const F64x2: Self = Self(1 << 22);
    pub const F64x3: Self = Self(1 << 23);
    pub const F64x4: Self = Self(1 << 24);
}

impl std::ops::BitOr for Type {
    type Output = Self;
    fn bitor(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}
impl std::ops::BitOrAssign for Type {
    fn bitor_assign(&mut self, other: Self) {
        self.0 |= other.0;
    }
}
impl std::ops::BitAnd for Type {
    type Output = Self;
    fn bitand(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }
}
