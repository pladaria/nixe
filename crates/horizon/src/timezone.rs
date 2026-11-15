//! Horizon's serialized time-zone rules and calendar conversion.
//! https://switchbrew.org/wiki/Glue_services#TimeZoneRule

use chrono::{Datelike, Timelike};

use crate::HorizonIpcResult;

pub(crate) const RULE_SIZE: usize = 0x4000;
const ATS: usize = 0x10;
const TYPES: usize = 0x1f50;
const TTIS: usize = 0x2338;
const CHARS: usize = 0x2b38;
const DEFAULT_TYPE: usize = 0x2d38;
const REPEAT_SECONDS: i128 = 146_097 * 86_400;

unsafe extern "C" {
    fn nixe_timezone_parse(data: *const u8, size: usize, output: *mut u8) -> i32;
}

#[derive(Debug)]
pub(crate) struct TimeZoneRule {
    bytes: Box<[u8; RULE_SIZE]>,
}

pub(crate) fn location_names() -> impl Iterator<Item = &'static str> {
    jiff_tzdb::available().filter(|name| name.len() < 0x24)
}

impl TimeZoneRule {
    pub(crate) fn load(name: &str) -> Result<Self, HorizonIpcResult> {
        let (_, tzif) = jiff_tzdb::get(name).ok_or(HorizonIpcResult::TIME_ZONE_NOT_FOUND)?;
        // Firmware uses the 64-bit TZif body. Skip the compatibility 32-bit
        // block in the embedded IANA files, without copying the payload.
        let binary = tzif_64(tzif).ok_or(HorizonIpcResult::TIME_ZONE_PARSE_FAILED)?;
        let mut bytes = Box::new([0; RULE_SIZE]);
        // SAFETY: embedded TZif bytes are immutable and the output spans the
        // complete rule. The native parser retains neither pointer.
        if unsafe { nixe_timezone_parse(binary.as_ptr(), binary.len(), bytes.as_mut_ptr()) } != 0 {
            return Err(HorizonIpcResult::TIME_ZONE_PARSE_FAILED);
        }
        Self::decode(bytes)
    }

    pub(crate) fn into_bytes(self) -> Vec<u8> {
        let bytes: Box<[u8]> = self.bytes;
        bytes.into_vec()
    }

    pub(crate) fn decode(bytes: Box<[u8; RULE_SIZE]>) -> Result<Self, HorizonIpcResult> {
        let rule = Self { bytes };
        let time_count = rule.i32(0);
        let type_count = rule.i32(4);
        let char_count = rule.i32(8);
        if !(0..=1000).contains(&time_count)
            || !(1..=128).contains(&type_count)
            || !(0..=512).contains(&char_count)
            || !(0..type_count).contains(&rule.i32(DEFAULT_TYPE))
            || rule.bytes[12] > 1
            || rule.bytes[13] > 1
            || (time_count == 0 && (rule.bytes[12] != 0 || rule.bytes[13] != 0))
        {
            return Err(HorizonIpcResult::TIME_ZONE_OUT_OF_RANGE);
        }
        for i in 0..time_count as usize {
            if i32::from(rule.bytes[TYPES + i]) >= type_count
                || (i != 0 && rule.transition(i - 1) >= rule.transition(i))
            {
                return Err(HorizonIpcResult::TIME_ZONE_OUT_OF_RANGE);
            }
        }
        for i in 0..type_count as usize {
            let base = TTIS + i * 16;
            let name = rule.i32(base + 8);
            if !(0..512).contains(&name)
                || !rule.bytes[CHARS + name as usize..CHARS + 512].contains(&0)
                || [4, 12, 13]
                    .iter()
                    .any(|&offset| rule.bytes[base + offset] > 1)
            {
                return Err(HorizonIpcResult::TIME_ZONE_OUT_OF_RANGE);
            }
        }
        Ok(rule)
    }

    pub(crate) fn calendar(&self, timestamp: i64) -> Result<[u8; 0x20], HorizonIpcResult> {
        let index = self.type_at(timestamp)?;
        let info = TTIS + index * 16;
        let offset = self.i32(info);
        let local = timestamp
            .checked_add(i64::from(offset))
            .and_then(|time| chrono::DateTime::from_timestamp(time, 0))
            .ok_or(HorizonIpcResult::TIME_OVERFLOWED)?;
        let year = i16::try_from(local.year()).map_err(|_| HorizonIpcResult::TIME_OVERFLOWED)?;
        let mut data = [0; 0x20];
        data[..2].copy_from_slice(&year.to_le_bytes());
        data[2..7].copy_from_slice(&[
            local.month() as u8,
            local.day() as u8,
            local.hour() as u8,
            local.minute() as u8,
            local.second() as u8,
        ]);
        data[8..12].copy_from_slice(&local.weekday().num_days_from_sunday().to_le_bytes());
        data[12..16].copy_from_slice(&local.ordinal0().to_le_bytes());
        let name = &self.bytes[CHARS + self.i32(info + 8) as usize..CHARS + 512];
        let length = name.iter().position(|&byte| byte == 0).unwrap().min(7);
        data[16..16 + length].copy_from_slice(&name[..length]);
        data[24..28].copy_from_slice(&u32::from(self.bytes[info + 4]).to_le_bytes());
        data[28..32].copy_from_slice(&offset.to_le_bytes());
        Ok(data)
    }

    fn type_at(&self, timestamp: i64) -> Result<usize, HorizonIpcResult> {
        let count = self.i32(0) as usize;
        let mut time = i128::from(timestamp);
        if count != 0 {
            let first = i128::from(self.transition(0));
            let last = i128::from(self.transition(count - 1));
            // tzcode extrapolates recurring rules in 400-year Gregorian cycles.
            // Wide arithmetic also handles the full guest timestamp range.
            if self.bytes[12] != 0 && time < first {
                time += ((first - time - 1) / REPEAT_SECONDS + 1) * REPEAT_SECONDS;
                if time > last {
                    return Err(HorizonIpcResult::TIME_OVERFLOWED);
                }
            } else if self.bytes[13] != 0 && time > last {
                time -= ((time - last - 1) / REPEAT_SECONDS + 1) * REPEAT_SECONDS;
                if time < first {
                    return Err(HorizonIpcResult::TIME_OVERFLOWED);
                }
            }
        }
        let mut low = 0;
        let mut high = count;
        while low < high {
            let mid = (low + high) / 2;
            if i128::from(self.transition(mid)) <= time {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        Ok(if low == 0 {
            self.i32(DEFAULT_TYPE) as usize
        } else {
            usize::from(self.bytes[TYPES + low - 1])
        })
    }

    fn i32(&self, offset: usize) -> i32 {
        i32::from_le_bytes(self.bytes[offset..offset + 4].try_into().unwrap())
    }

    fn transition(&self, index: usize) -> i64 {
        let offset = ATS + index * 8;
        i64::from_le_bytes(self.bytes[offset..offset + 8].try_into().unwrap())
    }
}

fn tzif_64(tzif: &[u8]) -> Option<&[u8]> {
    if tzif.get(..4)? != b"TZif" || !matches!(*tzif.get(4)?, b'2' | b'3' | b'4') {
        return None;
    }
    let mut offset = 44_usize;
    for (position, stride) in [(20, 1), (24, 1), (28, 8), (32, 5), (36, 6), (40, 1)] {
        let count = u32::from_be_bytes(tzif.get(position..position + 4)?.try_into().ok()?);
        offset = offset.checked_add((count as usize).checked_mul(stride)?)?;
    }
    let binary = tzif.get(offset..)?;
    (binary.get(..4)? == b"TZif").then_some(binary)
}
