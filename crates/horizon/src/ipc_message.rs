//! Checked, bounded codecs for the HIPC transport and CMIF framing layers.

use std::fmt::{Display, Formatter};

/// Size of the per-thread HIPC command buffer used by Horizon clients.
pub(crate) const COMMAND_BUFFER_SIZE: usize = 0x100;

const CMIF_IN_MAGIC: u32 = 0x4943_4653;
const CMIF_OUT_MAGIC: u32 = 0x4f43_4653;
const CMIF_COMMAND_LEGACY_REQUEST: u16 = 1;
const CMIF_COMMAND_CLOSE: u16 = 2;
const CMIF_COMMAND_LEGACY_CONTROL: u16 = 3;
const CMIF_COMMAND_REQUEST: u16 = 4;
const CMIF_COMMAND_CONTROL: u16 = 5;
const CMIF_COMMAND_REQUEST_WITH_CONTEXT: u16 = 6;
const CMIF_COMMAND_CONTROL_WITH_CONTEXT: u16 = 7;
const CMIF_DOMAIN_SEND_MESSAGE: u8 = 1;
const CMIF_DOMAIN_CLOSE: u8 = 2;
const MAX_DESCRIPTOR_COUNT: usize = 15;
const MAX_HANDLE_COUNT: usize = 15;

/// A deterministic rejection reason for an invalid HIPC or CMIF message.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct MessageError(pub(crate) &'static str);

impl Display for MessageError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.0)
    }
}

/// Kernel mapping mode carried by a HIPC buffer descriptor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BufferMode {
    Normal,
    NonSecure,
    Invalid,
    NonDevice,
}

impl BufferMode {
    fn from_raw(raw: u32) -> Self {
        match raw {
            0 => Self::Normal,
            1 => Self::NonSecure,
            2 => Self::Invalid,
            3 => Self::NonDevice,
            _ => unreachable!("HIPC buffer mode occupies two bits"),
        }
    }
}

/// One HIPC send-static (pointer) descriptor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SendStaticDescriptor {
    pub(crate) address: u64,
    pub(crate) size: u16,
    pub(crate) index: u8,
}

/// One HIPC map-alias buffer descriptor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct BufferDescriptor {
    pub(crate) address: u64,
    pub(crate) size: u64,
    pub(crate) mode: BufferMode,
}

/// One HIPC receive-static (pointer) descriptor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ReceiveStaticDescriptor {
    pub(crate) address: u64,
    pub(crate) size: u16,
}

/// Receive-static mode encoded in the HIPC header.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ReceiveStatics {
    None,
    Auto,
    Entries(Vec<ReceiveStaticDescriptor>),
}

/// Fully decoded HIPC request metadata and descriptor tables.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct HipcRequest<'a> {
    pub(crate) command_type: u16,
    pub(crate) pid: Option<u64>,
    pub(crate) copy_handles: Vec<u32>,
    pub(crate) move_handles: Vec<u32>,
    pub(crate) send_statics: Vec<SendStaticDescriptor>,
    pub(crate) send_buffers: Vec<BufferDescriptor>,
    pub(crate) receive_buffers: Vec<BufferDescriptor>,
    pub(crate) exchange_buffers: Vec<BufferDescriptor>,
    pub(crate) receive_statics: ReceiveStatics,
    raw: &'a [u8],
    handle_offset: usize,
    data_offset: usize,
    data_end: usize,
}

impl<'a> HipcRequest<'a> {
    /// Decodes one complete TLS command buffer without reading outside it.
    pub(crate) fn decode(raw: &'a [u8]) -> Result<Self, MessageError> {
        if raw.len() > COMMAND_BUFFER_SIZE {
            return Err(MessageError("HIPC command buffer exceeds 0x100 bytes"));
        }
        let word0 = get_u32(raw, 0)?;
        let word1 = get_u32(raw, 4)?;
        let command_type = word0 as u16;
        let send_static_count = field(word0, 16, 4);
        let send_buffer_count = field(word0, 20, 4);
        let receive_buffer_count = field(word0, 24, 4);
        let exchange_buffer_count = field(word0, 28, 4);
        let data_word_count = field(word1, 0, 10);
        let receive_static_mode = field(word1, 10, 4);
        if field(word1, 14, 6) != 0 {
            return Err(MessageError("HIPC header padding is nonzero"));
        }
        let has_special_header = word1 >> 31 != 0;

        let mut offset = 8;
        let (pid, copy_handle_count, move_handle_count) = if has_special_header {
            let special = get_u32(raw, offset)?;
            offset = checked_add(offset, 4)?;
            if special >> 9 != 0 {
                return Err(MessageError("HIPC special-header padding is nonzero"));
            }
            let sends_pid = special & 1 != 0;
            let copy_count = field(special, 1, 4);
            let move_count = field(special, 5, 4);
            let pid = if sends_pid {
                let value = get_u64(raw, offset)?;
                offset = checked_add(offset, 8)?;
                Some(value)
            } else {
                None
            };
            (pid, copy_count, move_count)
        } else {
            (None, 0, 0)
        };

        let handle_offset = offset;
        let (copy_handles, next) = decode_words(raw, offset, copy_handle_count)?;
        offset = next;
        let (move_handles, next) = decode_words(raw, offset, move_handle_count)?;
        offset = next;
        let (send_statics, next) = decode_send_statics(raw, offset, send_static_count)?;
        offset = next;
        let (send_buffers, next) = decode_buffers(raw, offset, send_buffer_count)?;
        offset = next;
        let (receive_buffers, next) = decode_buffers(raw, offset, receive_buffer_count)?;
        offset = next;
        let (exchange_buffers, next) = decode_buffers(raw, offset, exchange_buffer_count)?;
        offset = next;

        let data_size = checked_mul(data_word_count, 4)?;
        let data_end = checked_add(offset, data_size)?;
        if data_end > raw.len() {
            return Err(MessageError("HIPC data words exceed the command buffer"));
        }
        offset = data_end;
        let receive_statics = match receive_static_mode {
            0 => ReceiveStatics::None,
            1 => return Err(MessageError("HIPC receive-static mode 1 is reserved")),
            2 => ReceiveStatics::Auto,
            mode => {
                let count = mode - 2;
                let (entries, next) = decode_receive_statics(raw, offset, count)?;
                offset = next;
                ReceiveStatics::Entries(entries)
            }
        };
        if offset > raw.len() {
            return Err(MessageError(
                "HIPC receive-static list exceeds the command buffer",
            ));
        }

        Ok(Self {
            command_type,
            pid,
            copy_handles,
            move_handles,
            send_statics,
            send_buffers,
            receive_buffers,
            exchange_buffers,
            receive_statics,
            raw,
            handle_offset,
            data_offset: data_end - data_size,
            data_end,
        })
    }

    pub(crate) const fn handle_offset(&self) -> usize {
        self.handle_offset
    }

    fn aligned_data(&self) -> Result<&'a [u8], MessageError> {
        let start = align_up(self.data_offset, 16)?;
        self.raw
            .get(start..self.data_end)
            .ok_or(MessageError("aligned CMIF data exceeds HIPC data words"))
    }

    fn data_word_bytes(&self) -> usize {
        self.data_end - self.data_offset
    }
}

/// Domain-specific CMIF request information.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum DomainRequest {
    SendMessage {
        object_id: u32,
        input_objects: Vec<u32>,
    },
    Close {
        object_id: u32,
    },
}

/// Decoded request/control/close message at the CMIF layer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CmifRequest<'a> {
    pub(crate) command_type: u16,
    pub(crate) command_id: u32,
    pub(crate) token: u32,
    pub(crate) context: Option<u32>,
    pub(crate) data: &'a [u8],
    pub(crate) domain: Option<DomainRequest>,
}

impl<'a> CmifRequest<'a> {
    /// Decodes the CMIF layer carried by a checked HIPC request.
    pub(crate) fn decode(hipc: &HipcRequest<'a>, is_domain: bool) -> Result<Self, MessageError> {
        let command_type = hipc.command_type;
        if command_type == CMIF_COMMAND_CLOSE {
            if is_domain {
                return Err(MessageError("session close cannot target a domain object"));
            }
            if hipc.data_word_bytes() != 0 {
                return Err(MessageError("CMIF session close contains data words"));
            }
            return Ok(Self {
                command_type,
                command_id: 0,
                token: 0,
                context: None,
                data: &[],
                domain: None,
            });
        }
        if !matches!(
            command_type,
            CMIF_COMMAND_LEGACY_REQUEST
                | CMIF_COMMAND_LEGACY_CONTROL
                | CMIF_COMMAND_REQUEST
                | CMIF_COMMAND_CONTROL
                | CMIF_COMMAND_REQUEST_WITH_CONTEXT
                | CMIF_COMMAND_CONTROL_WITH_CONTEXT
        ) {
            return Err(MessageError("unsupported HIPC command type for CMIF"));
        }

        let data_words = hipc.aligned_data()?;
        if is_domain
            && matches!(
                command_type,
                CMIF_COMMAND_REQUEST | CMIF_COMMAND_REQUEST_WITH_CONTEXT
            )
        {
            return Self::decode_domain(command_type, data_words);
        }
        Self::decode_plain(command_type, data_words)
    }

    fn decode_plain(command_type: u16, bytes: &'a [u8]) -> Result<Self, MessageError> {
        let header = decode_in_header(bytes, 0)?;
        let has_context = matches!(
            command_type,
            CMIF_COMMAND_REQUEST_WITH_CONTEXT | CMIF_COMMAND_CONTROL_WITH_CONTEXT
        );
        validate_version(header.version, has_context)?;
        Ok(Self {
            command_type,
            command_id: header.command_id,
            token: header.token,
            context: has_context.then_some(header.token),
            data: bytes
                .get(16..)
                .ok_or(MessageError("truncated CMIF input header"))?,
            domain: None,
        })
    }

    fn decode_domain(command_type: u16, bytes: &'a [u8]) -> Result<Self, MessageError> {
        let domain_type = *bytes
            .first()
            .ok_or(MessageError("truncated CMIF domain header"))?;
        let input_object_count = usize::from(
            *bytes
                .get(1)
                .ok_or(MessageError("truncated CMIF domain input-object count"))?,
        );
        let data_size = usize::from(get_u16(bytes, 2)?);
        let object_id = get_u32(bytes, 4)?;
        if get_u32(bytes, 8)? != 0 {
            return Err(MessageError("CMIF domain-header padding is nonzero"));
        }
        let domain_token = get_u32(bytes, 12)?;
        match domain_type {
            CMIF_DOMAIN_CLOSE => {
                if input_object_count != 0 || data_size != 0 {
                    return Err(MessageError(
                        "CMIF domain close contains payload or objects",
                    ));
                }
                Ok(Self {
                    command_type,
                    command_id: 0,
                    token: 0,
                    context: None,
                    data: &[],
                    domain: Some(DomainRequest::Close { object_id }),
                })
            }
            CMIF_DOMAIN_SEND_MESSAGE => {
                if data_size < 16 {
                    return Err(MessageError("CMIF domain payload omits its input header"));
                }
                let payload_end = checked_add(16, data_size)?;
                if payload_end > bytes.len() {
                    return Err(MessageError("CMIF domain payload exceeds HIPC data words"));
                }
                let header = decode_in_header(bytes, 16)?;
                let has_context = command_type == CMIF_COMMAND_REQUEST_WITH_CONTEXT;
                validate_version(header.version, has_context)?;
                if header.token != 0 {
                    return Err(MessageError("domain CMIF input-header token is nonzero"));
                }
                let (input_objects, objects_end) =
                    decode_words(bytes, payload_end, input_object_count)?;
                if objects_end > bytes.len() {
                    return Err(MessageError("CMIF domain object list exceeds data words"));
                }
                Ok(Self {
                    command_type,
                    command_id: header.command_id,
                    token: domain_token,
                    context: has_context.then_some(domain_token),
                    data: &bytes[32..payload_end],
                    domain: Some(DomainRequest::SendMessage {
                        object_id,
                        input_objects,
                    }),
                })
            }
            _ => Err(MessageError("invalid CMIF domain request type")),
        }
    }
}

#[derive(Clone, Copy)]
struct CmifInHeader {
    version: u32,
    command_id: u32,
    token: u32,
}

fn decode_in_header(bytes: &[u8], offset: usize) -> Result<CmifInHeader, MessageError> {
    if get_u32(bytes, offset)? != CMIF_IN_MAGIC {
        return Err(MessageError("invalid CMIF input-header magic"));
    }
    Ok(CmifInHeader {
        version: get_u32(bytes, checked_add(offset, 4)?)?,
        command_id: get_u32(bytes, checked_add(offset, 8)?)?,
        token: get_u32(bytes, checked_add(offset, 12)?)?,
    })
}

fn validate_version(version: u32, has_context: bool) -> Result<(), MessageError> {
    let expected = u32::from(has_context);
    if version == expected {
        Ok(())
    } else {
        Err(MessageError(
            "CMIF input-header version does not match command type",
        ))
    }
}

/// Inputs used to encode a checked HIPC/CMIF server response.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct CmifResponse<'a> {
    pub(crate) token: u32,
    pub(crate) result: u32,
    pub(crate) data: &'a [u8],
    pub(crate) pid: Option<u64>,
    pub(crate) copy_handles: &'a [u32],
    pub(crate) move_handles: &'a [u32],
    pub(crate) send_statics: &'a [SendStaticDescriptor],
    pub(crate) is_domain: bool,
    pub(crate) domain_objects: &'a [u32],
}

impl CmifResponse<'_> {
    /// Encodes one response consumable by libnx's HIPC and CMIF parsers.
    pub(crate) fn encode(&self) -> Result<Vec<u8>, MessageError> {
        if !self.is_domain && !self.domain_objects.is_empty() {
            return Err(MessageError(
                "non-domain CMIF response contains domain objects",
            ));
        }
        bounded_count(
            self.copy_handles.len(),
            MAX_HANDLE_COUNT,
            "too many copied handles",
        )?;
        bounded_count(
            self.move_handles.len(),
            MAX_HANDLE_COUNT,
            "too many moved handles",
        )?;
        bounded_count(
            self.send_statics.len(),
            MAX_DESCRIPTOR_COUNT,
            "too many send statics",
        )?;

        let has_special =
            self.pid.is_some() || !self.copy_handles.is_empty() || !self.move_handles.is_empty();
        let special_size = if has_special {
            4 + usize::from(self.pid.is_some()) * 8
                + checked_mul(self.copy_handles.len() + self.move_handles.len(), 4)?
        } else {
            0
        };
        let descriptor_size = checked_mul(self.send_statics.len(), 8)?;
        let data_offset = checked_add(8, checked_add(special_size, descriptor_size)?)?;
        let data_start = align_up(data_offset, 16)?;
        let domain_header_size = usize::from(self.is_domain) * 16;
        let body_size = checked_add(
            checked_add(domain_header_size, 16)?,
            checked_add(self.data.len(), checked_mul(self.domain_objects.len(), 4)?)?,
        )?;
        // CMIF reserves 16 bytes so its aligned header fits regardless of the
        // descriptor layout preceding the data-word region.
        let data_word_bytes = align_up(checked_add(16, body_size)?, 4)?;
        let data_words = data_word_bytes / 4;
        if data_words > 0x3ff {
            return Err(MessageError("CMIF response exceeds HIPC data-word limit"));
        }
        let total = checked_add(data_offset, data_word_bytes)?;
        if total > COMMAND_BUFFER_SIZE {
            return Err(MessageError("CMIF response exceeds the TLS command buffer"));
        }

        let mut output = vec![0; total];
        let word0 = u32::try_from(self.send_statics.len()).expect("count is bounded") << 16;
        let word1 = u32::try_from(data_words).expect("data-word count is bounded")
            | (u32::from(has_special) << 31);
        put_u32(&mut output, 0, word0)?;
        put_u32(&mut output, 4, word1)?;
        let mut offset = 8;
        if has_special {
            let special = u32::from(self.pid.is_some())
                | (u32::try_from(self.copy_handles.len()).expect("count is bounded") << 1)
                | (u32::try_from(self.move_handles.len()).expect("count is bounded") << 5);
            put_u32(&mut output, offset, special)?;
            offset += 4;
            if let Some(pid) = self.pid {
                put_u64(&mut output, offset, pid)?;
                offset += 8;
            }
            for handle in self.copy_handles.iter().chain(self.move_handles) {
                put_u32(&mut output, offset, *handle)?;
                offset += 4;
            }
        }
        for descriptor in self.send_statics {
            encode_send_static(&mut output, offset, *descriptor)?;
            offset += 8;
        }
        debug_assert_eq!(offset, data_offset);

        let mut cmif_offset = data_start;
        if self.is_domain {
            put_u32(
                &mut output,
                cmif_offset,
                u32::try_from(self.domain_objects.len())
                    .map_err(|_| MessageError("domain object count overflows"))?,
            )?;
            cmif_offset += 16;
        }
        put_u32(&mut output, cmif_offset, CMIF_OUT_MAGIC)?;
        put_u32(&mut output, cmif_offset + 4, 0)?;
        put_u32(&mut output, cmif_offset + 8, self.result)?;
        put_u32(&mut output, cmif_offset + 12, self.token)?;
        let payload_start = cmif_offset + 16;
        let payload_end = checked_add(payload_start, self.data.len())?;
        output[payload_start..payload_end].copy_from_slice(self.data);
        let mut object_offset = payload_end;
        for object in self.domain_objects {
            put_u32(&mut output, object_offset, *object)?;
            object_offset += 4;
        }
        Ok(output)
    }
}

fn decode_words(
    bytes: &[u8],
    mut offset: usize,
    count: usize,
) -> Result<(Vec<u32>, usize), MessageError> {
    bounded_count(count, MAX_HANDLE_COUNT.max(0xff), "too many HIPC words")?;
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        values.push(get_u32(bytes, offset)?);
        offset = checked_add(offset, 4)?;
    }
    Ok((values, offset))
}

fn decode_send_statics(
    bytes: &[u8],
    mut offset: usize,
    count: usize,
) -> Result<(Vec<SendStaticDescriptor>, usize), MessageError> {
    bounded_count(count, MAX_DESCRIPTOR_COUNT, "too many send statics")?;
    let mut descriptors = Vec::with_capacity(count);
    for _ in 0..count {
        let first = get_u32(bytes, offset)?;
        let low = get_u32(bytes, checked_add(offset, 4)?)?;
        descriptors.push(SendStaticDescriptor {
            address: u64::from(low)
                | ((field(first, 12, 4) as u64) << 32)
                | ((field(first, 6, 6) as u64) << 36),
            size: (first >> 16) as u16,
            index: (first & 0x3f) as u8,
        });
        offset = checked_add(offset, 8)?;
    }
    Ok((descriptors, offset))
}

fn decode_buffers(
    bytes: &[u8],
    mut offset: usize,
    count: usize,
) -> Result<(Vec<BufferDescriptor>, usize), MessageError> {
    bounded_count(count, MAX_DESCRIPTOR_COUNT, "too many HIPC buffers")?;
    let mut descriptors = Vec::with_capacity(count);
    for _ in 0..count {
        let size_low = get_u32(bytes, offset)?;
        let address_low = get_u32(bytes, checked_add(offset, 4)?)?;
        let high = get_u32(bytes, checked_add(offset, 8)?)?;
        descriptors.push(BufferDescriptor {
            address: u64::from(address_low)
                | ((field(high, 28, 4) as u64) << 32)
                | ((field(high, 2, 22) as u64) << 36),
            size: u64::from(size_low) | ((field(high, 24, 4) as u64) << 32),
            mode: BufferMode::from_raw(field(high, 0, 2) as u32),
        });
        offset = checked_add(offset, 12)?;
    }
    Ok((descriptors, offset))
}

fn decode_receive_statics(
    bytes: &[u8],
    mut offset: usize,
    count: usize,
) -> Result<(Vec<ReceiveStaticDescriptor>, usize), MessageError> {
    bounded_count(count, 13, "too many receive statics")?;
    let mut descriptors = Vec::with_capacity(count);
    for _ in 0..count {
        let low = get_u32(bytes, offset)?;
        let high = get_u32(bytes, checked_add(offset, 4)?)?;
        descriptors.push(ReceiveStaticDescriptor {
            address: u64::from(low) | (u64::from(high & 0xffff) << 32),
            size: (high >> 16) as u16,
        });
        offset = checked_add(offset, 8)?;
    }
    Ok((descriptors, offset))
}

fn encode_send_static(
    bytes: &mut [u8],
    offset: usize,
    descriptor: SendStaticDescriptor,
) -> Result<(), MessageError> {
    if descriptor.address >> 42 != 0 || descriptor.index > 0x3f {
        return Err(MessageError("send-static descriptor field is out of range"));
    }
    let first = u32::from(descriptor.index)
        | (((descriptor.address >> 36) as u32 & 0x3f) << 6)
        | (((descriptor.address >> 32) as u32 & 0xf) << 12)
        | (u32::from(descriptor.size) << 16);
    put_u32(bytes, offset, first)?;
    put_u32(bytes, offset + 4, descriptor.address as u32)
}

fn bounded_count(count: usize, maximum: usize, reason: &'static str) -> Result<(), MessageError> {
    if count <= maximum {
        Ok(())
    } else {
        Err(MessageError(reason))
    }
}

fn field(value: u32, shift: u32, width: u32) -> usize {
    ((value >> shift) & ((1_u32 << width) - 1)) as usize
}

fn get_u16(bytes: &[u8], offset: usize) -> Result<u16, MessageError> {
    let end = checked_add(offset, 2)?;
    let value = bytes
        .get(offset..end)
        .ok_or(MessageError("truncated HIPC halfword"))?;
    Ok(u16::from_le_bytes(
        value.try_into().expect("length checked"),
    ))
}

fn get_u32(bytes: &[u8], offset: usize) -> Result<u32, MessageError> {
    let end = checked_add(offset, 4)?;
    let value = bytes
        .get(offset..end)
        .ok_or(MessageError("truncated HIPC word"))?;
    Ok(u32::from_le_bytes(
        value.try_into().expect("length checked"),
    ))
}

fn get_u64(bytes: &[u8], offset: usize) -> Result<u64, MessageError> {
    let end = checked_add(offset, 8)?;
    let value = bytes
        .get(offset..end)
        .ok_or(MessageError("truncated HIPC doubleword"))?;
    Ok(u64::from_le_bytes(
        value.try_into().expect("length checked"),
    ))
}

fn put_u32(bytes: &mut [u8], offset: usize, value: u32) -> Result<(), MessageError> {
    let end = checked_add(offset, 4)?;
    let target = bytes
        .get_mut(offset..end)
        .ok_or(MessageError("HIPC output word is out of bounds"))?;
    target.copy_from_slice(&value.to_le_bytes());
    Ok(())
}

fn put_u64(bytes: &mut [u8], offset: usize, value: u64) -> Result<(), MessageError> {
    let end = checked_add(offset, 8)?;
    let target = bytes
        .get_mut(offset..end)
        .ok_or(MessageError("HIPC output doubleword is out of bounds"))?;
    target.copy_from_slice(&value.to_le_bytes());
    Ok(())
}

fn checked_add(left: usize, right: usize) -> Result<usize, MessageError> {
    left.checked_add(right)
        .ok_or(MessageError("HIPC size calculation overflows"))
}

fn checked_mul(left: usize, right: usize) -> Result<usize, MessageError> {
    left.checked_mul(right)
        .ok_or(MessageError("HIPC size calculation overflows"))
}

fn align_up(value: usize, alignment: usize) -> Result<usize, MessageError> {
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or(MessageError("HIPC alignment overflows"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn put_word(bytes: &mut [u8], offset: usize, value: u32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    #[test]
    fn decodes_every_hipc_descriptor_class_and_special_metadata() {
        let mut bytes = [0_u8; COMMAND_BUFFER_SIZE];
        put_word(&mut bytes, 0, 4 | 1 << 16 | 1 << 20 | 1 << 24 | 1 << 28);
        put_word(&mut bytes, 4, 8 | 3 << 10 | 1 << 31);
        put_word(&mut bytes, 8, 1 | 1 << 1 | 1 << 5);
        bytes[12..20].copy_from_slice(&0x1122_3344_5566_7788_u64.to_le_bytes());
        put_word(&mut bytes, 20, 0x31);
        put_word(&mut bytes, 24, 0x32);
        // Send static: address 0x123456789a, size 0x44, index 3.
        put_word(&mut bytes, 28, 3 | 0x12 << 6 | 3 << 12 | 0x44 << 16);
        put_word(&mut bytes, 32, 0x4567_89ab);
        // Three buffer classes use the same verified 12-byte encoding.
        for offset in [36, 48, 60] {
            put_word(&mut bytes, offset, 0x89ab_cdef);
            put_word(&mut bytes, offset + 4, 0x7654_3210);
            put_word(&mut bytes, offset + 8, 3 | 0x12345 << 2 | 7 << 24 | 4 << 28);
        }
        // Eight data words occupy 72..104; the CMIF header starts at 80.
        put_word(&mut bytes, 80, CMIF_IN_MAGIC);
        put_word(&mut bytes, 84, 0);
        put_word(&mut bytes, 88, 9);
        put_word(&mut bytes, 92, 7);
        put_word(&mut bytes, 104, 0x7654_3210);
        put_word(&mut bytes, 108, 0x20 | 0x1234 << 16);

        let hipc = HipcRequest::decode(&bytes[..112]).unwrap();
        assert_eq!(hipc.pid, Some(0x1122_3344_5566_7788));
        assert_eq!(hipc.copy_handles, [0x31]);
        assert_eq!(hipc.move_handles, [0x32]);
        assert_eq!(hipc.send_statics[0].index, 3);
        assert_eq!(hipc.send_buffers[0].mode, BufferMode::NonDevice);
        assert_eq!(hipc.receive_buffers.len(), 1);
        assert_eq!(hipc.exchange_buffers.len(), 1);
        assert_eq!(
            hipc.receive_statics,
            ReceiveStatics::Entries(vec![ReceiveStaticDescriptor {
                address: 0x0020_7654_3210,
                size: 0x1234,
            }])
        );
        let cmif = CmifRequest::decode(&hipc, false).unwrap();
        assert_eq!(cmif.command_id, 9);
        assert_eq!(cmif.token, 7);
    }

    #[test]
    fn decodes_context_domain_request_and_input_objects() {
        let mut bytes = [0_u8; 80];
        put_word(&mut bytes, 0, u32::from(CMIF_COMMAND_REQUEST_WITH_CONTEXT));
        put_word(&mut bytes, 4, 16);
        bytes[16] = CMIF_DOMAIN_SEND_MESSAGE;
        bytes[17] = 2;
        bytes[18..20].copy_from_slice(&20_u16.to_le_bytes());
        put_word(&mut bytes, 20, 0x44);
        put_word(&mut bytes, 28, 0x55);
        put_word(&mut bytes, 32, CMIF_IN_MAGIC);
        put_word(&mut bytes, 36, 1);
        put_word(&mut bytes, 40, 8);
        put_word(&mut bytes, 44, 0);
        put_word(&mut bytes, 48, 0x99);
        put_word(&mut bytes, 52, 0x66);
        put_word(&mut bytes, 56, 0x77);

        let hipc = HipcRequest::decode(&bytes).unwrap();
        let cmif = CmifRequest::decode(&hipc, true).unwrap();
        assert_eq!(cmif.command_id, 8);
        assert_eq!(cmif.context, Some(0x55));
        assert_eq!(cmif.data, 0x99_u32.to_le_bytes());
        assert_eq!(
            cmif.domain,
            Some(DomainRequest::SendMessage {
                object_id: 0x44,
                input_objects: vec![0x66, 0x77],
            })
        );
    }

    #[test]
    fn response_round_trips_handles_statics_pid_and_domain_objects() {
        let response = CmifResponse {
            token: 7,
            result: 0,
            data: &0x100_u16.to_le_bytes(),
            pid: Some(0x1234),
            copy_handles: &[0x41],
            move_handles: &[0x42],
            send_statics: &[SendStaticDescriptor {
                address: 0x1234_5678,
                size: 0x20,
                index: 1,
            }],
            is_domain: true,
            domain_objects: &[0x51, 0x52],
        }
        .encode()
        .unwrap();

        let hipc = HipcRequest::decode(&response).unwrap();
        assert_eq!(hipc.pid, Some(0x1234));
        assert_eq!(hipc.copy_handles, [0x41]);
        assert_eq!(hipc.move_handles, [0x42]);
        assert_eq!(hipc.send_statics.len(), 1);
        let cmif_start = align_up(hipc.data_offset, 16).unwrap();
        assert_eq!(get_u32(&response, cmif_start).unwrap(), 2);
        assert_eq!(get_u32(&response, cmif_start + 16).unwrap(), CMIF_OUT_MAGIC);
        assert_eq!(get_u32(&response, cmif_start + 28).unwrap(), 7);
    }

    #[test]
    fn rejects_reserved_modes_bad_counts_and_truncated_tables() {
        let mut reserved = [0_u8; 8];
        put_word(&mut reserved, 4, 1 << 10);
        assert_eq!(
            HipcRequest::decode(&reserved).unwrap_err(),
            MessageError("HIPC receive-static mode 1 is reserved")
        );

        let mut truncated = [0_u8; 12];
        put_word(&mut truncated, 0, 1 << 20);
        assert_eq!(
            HipcRequest::decode(&truncated).unwrap_err(),
            MessageError("truncated HIPC word")
        );

        let too_many_handles = [0_u32; 16];
        assert_eq!(
            CmifResponse {
                copy_handles: &too_many_handles,
                ..CmifResponse::default()
            }
            .encode()
            .unwrap_err(),
            MessageError("too many copied handles")
        );
        assert_eq!(
            CmifResponse {
                domain_objects: &[1],
                ..CmifResponse::default()
            }
            .encode()
            .unwrap_err(),
            MessageError("non-domain CMIF response contains domain objects")
        );
    }

    #[test]
    fn accepts_empty_session_close_and_rejects_data_bearing_close() {
        let mut close = [0_u8; 12];
        put_word(&mut close, 0, u32::from(CMIF_COMMAND_CLOSE));
        let hipc = HipcRequest::decode(&close).unwrap();
        let cmif = CmifRequest::decode(&hipc, false).unwrap();
        assert_eq!(cmif.command_type, CMIF_COMMAND_CLOSE);

        put_word(&mut close, 4, 1);
        assert_eq!(
            CmifRequest::decode(&HipcRequest::decode(&close).unwrap(), false).unwrap_err(),
            MessageError("CMIF session close contains data words")
        );
    }

    #[test]
    fn every_truncated_prefix_of_a_complex_message_is_rejected() {
        let mut bytes = [0_u8; 112];
        put_word(&mut bytes, 0, 4 | 1 << 16 | 1 << 20 | 1 << 24 | 1 << 28);
        put_word(&mut bytes, 4, 8 | 3 << 10 | 1 << 31);
        put_word(&mut bytes, 8, 1 | 1 << 1 | 1 << 5);
        put_word(&mut bytes, 80, CMIF_IN_MAGIC);
        for prefix_length in 0..bytes.len() {
            assert!(HipcRequest::decode(&bytes[..prefix_length]).is_err());
        }
        assert!(HipcRequest::decode(&bytes).is_ok());
    }
}
