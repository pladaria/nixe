/// Bit fields are checked at monomorphization time, including full-width fields.
pub const fn extract<const START: u32, const WIDTH: u32>(word: u64) -> u64 {
    const {
        assert!(WIDTH > 0 && WIDTH <= 64 && START <= 64 - WIDTH);
    }
    (word >> START) & (u64::MAX >> (64 - WIDTH))
}

pub const fn extract_signed<const START: u32, const WIDTH: u32>(word: u64) -> i64 {
    let bits = extract::<START, WIDTH>(word);
    ((bits << (64 - WIDTH)) as i64) >> (64 - WIDTH)
}

pub const fn insert<const START: u32, const WIDTH: u32>(word: u64, value: u64) -> u64 {
    const {
        assert!(WIDTH > 0 && WIDTH <= 64 && START <= 64 - WIDTH);
    }
    let mask = (u64::MAX >> (64 - WIDTH)) << START;
    (word & !mask) | ((value << START) & mask)
}
