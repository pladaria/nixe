//! Bounded Android Parcel codec used by the Switch VI Binder relay.

use std::fmt::{Display, Formatter};

const HEADER_SIZE: usize = 16;
const MAX_PARCEL_BYTES: usize = 0x4000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ParcelError(pub(crate) &'static str);

impl Display for ParcelError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.0)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ParcelReader<'a> {
    payload: &'a [u8],
    position: usize,
}

impl<'a> ParcelReader<'a> {
    pub(crate) fn decode(encoded: &'a [u8]) -> Result<Self, ParcelError> {
        if encoded.len() < HEADER_SIZE || encoded.len() > MAX_PARCEL_BYTES {
            return Err(ParcelError("Parcel has an invalid total size"));
        }
        let payload_size = usize::try_from(read_u32(encoded, 0)?)
            .map_err(|_| ParcelError("Parcel payload size overflows"))?;
        let payload_offset = usize::try_from(read_u32(encoded, 4)?)
            .map_err(|_| ParcelError("Parcel payload offset overflows"))?;
        let objects_size = usize::try_from(read_u32(encoded, 8)?)
            .map_err(|_| ParcelError("Parcel objects size overflows"))?;
        let objects_offset = usize::try_from(read_u32(encoded, 12)?)
            .map_err(|_| ParcelError("Parcel objects offset overflows"))?;
        let payload_end = payload_offset
            .checked_add(payload_size)
            .ok_or(ParcelError("Parcel payload range overflows"))?;
        let objects_end = objects_offset
            .checked_add(objects_size)
            .ok_or(ParcelError("Parcel object range overflows"))?;
        if payload_offset < HEADER_SIZE
            || payload_end > encoded.len()
            || objects_end > encoded.len()
        {
            return Err(ParcelError("Parcel range exceeds its buffer"));
        }
        if objects_size != 0 {
            return Err(ParcelError("Parcel Binder objects are not supported"));
        }
        Ok(Self {
            payload: &encoded[payload_offset..payload_end],
            position: 0,
        })
    }

    pub(crate) fn read_i32(&mut self) -> Result<i32, ParcelError> {
        Ok(i32::from_le_bytes(self.read_exact(4)?.try_into().unwrap()))
    }

    pub(crate) fn read_u32(&mut self) -> Result<u32, ParcelError> {
        Ok(u32::from_le_bytes(self.read_exact(4)?.try_into().unwrap()))
    }

    pub(crate) fn read_flattened(&mut self) -> Result<&'a [u8], ParcelError> {
        let size = self.read_i32()?;
        let fd_count = self.read_i32()?;
        if size < 0 || fd_count != 0 {
            return Err(ParcelError("flattened Parcel object header is invalid"));
        }
        self.read_exact(usize::try_from(size).map_err(|_| ParcelError("flattened size overflows"))?)
    }

    /// Validates and consumes libnx's strict UTF-16 Binder interface token.
    pub(crate) fn read_interface_token(&mut self, expected: &str) -> Result<(), ParcelError> {
        if self.read_u32()? != 0x100 {
            return Err(ParcelError("Binder interface token policy is invalid"));
        }
        let length = self.read_i32()?;
        if length < 0 {
            return Err(ParcelError("Binder interface token length is negative"));
        }
        let length =
            usize::try_from(length).map_err(|_| ParcelError("interface length overflows"))?;
        let encoded_bytes = length
            .checked_add(1)
            .and_then(|length| length.checked_mul(2))
            .ok_or(ParcelError("interface token size overflows"))?;
        let encoded = self.read_exact(encoded_bytes)?;
        if encoded
            .chunks_exact(2)
            .take(length)
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
            .ne(expected.encode_utf16())
            || encoded.get(length * 2..length * 2 + 2) != Some(&[0, 0])
        {
            return Err(ParcelError("Binder interface token does not match"));
        }
        Ok(())
    }

    fn read_exact(&mut self, size: usize) -> Result<&'a [u8], ParcelError> {
        let aligned = size
            .checked_add(3)
            .map(|size| size & !3)
            .ok_or(ParcelError("Parcel read size overflows"))?;
        let end = self
            .position
            .checked_add(aligned)
            .ok_or(ParcelError("Parcel cursor overflows"))?;
        let data_end = self
            .position
            .checked_add(size)
            .ok_or(ParcelError("Parcel cursor overflows"))?;
        let data = self
            .payload
            .get(self.position..data_end)
            .ok_or(ParcelError("Parcel payload is truncated"))?;
        if end > self.payload.len() {
            return Err(ParcelError("Parcel alignment exceeds payload"));
        }
        self.position = end;
        Ok(data)
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct ParcelWriter {
    payload: Vec<u8>,
}

impl ParcelWriter {
    pub(crate) fn write_i32(&mut self, value: i32) {
        self.payload.extend_from_slice(&value.to_le_bytes());
    }

    pub(crate) fn write_u32(&mut self, value: u32) {
        self.payload.extend_from_slice(&value.to_le_bytes());
    }

    pub(crate) fn write_flattened(&mut self, value: &[u8]) -> Result<(), ParcelError> {
        let size = i32::try_from(value.len())
            .map_err(|_| ParcelError("flattened Parcel object is too large"))?;
        self.write_i32(size);
        self.write_i32(0);
        self.payload.extend_from_slice(value);
        self.payload.resize((self.payload.len() + 3) & !3, 0);
        Ok(())
    }

    pub(crate) fn finish(self) -> Result<Vec<u8>, ParcelError> {
        let payload_size = u32::try_from(self.payload.len())
            .map_err(|_| ParcelError("Parcel response size overflows"))?;
        let total = HEADER_SIZE
            .checked_add(self.payload.len())
            .ok_or(ParcelError("Parcel response size overflows"))?;
        let total_u32 =
            u32::try_from(total).map_err(|_| ParcelError("Parcel response size overflows"))?;
        let mut encoded = Vec::with_capacity(total);
        encoded.extend_from_slice(&payload_size.to_le_bytes());
        encoded.extend_from_slice(&(HEADER_SIZE as u32).to_le_bytes());
        encoded.extend_from_slice(&0_u32.to_le_bytes());
        encoded.extend_from_slice(&total_u32.to_le_bytes());
        encoded.extend_from_slice(&self.payload);
        Ok(encoded)
    }
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, ParcelError> {
    Ok(u32::from_le_bytes(
        bytes
            .get(offset..offset + 4)
            .ok_or(ParcelError("Parcel header is truncated"))?
            .try_into()
            .unwrap(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_truncated_parcel_header_is_rejected() {
        let valid = ParcelWriter::default().finish().unwrap();
        for end in 0..HEADER_SIZE {
            assert!(ParcelReader::decode(&valid[..end]).is_err());
        }
    }

    #[test]
    fn scalar_reply_round_trips() {
        let mut writer = ParcelWriter::default();
        writer.write_i32(-7);
        writer.write_u32(9);
        let encoded = writer.finish().unwrap();
        let mut reader = ParcelReader::decode(&encoded).unwrap();
        assert_eq!(reader.read_i32().unwrap(), -7);
        assert_eq!(reader.read_u32().unwrap(), 9);
    }

    #[test]
    fn flattened_reply_round_trips_with_alignment() {
        let mut writer = ParcelWriter::default();
        writer.write_flattened(&[1, 2, 3]).unwrap();
        writer.write_u32(7);
        let encoded = writer.finish().unwrap();
        let mut reader = ParcelReader::decode(&encoded).unwrap();
        assert_eq!(reader.read_flattened().unwrap(), &[1, 2, 3]);
        assert_eq!(reader.read_u32().unwrap(), 7);
    }
}
