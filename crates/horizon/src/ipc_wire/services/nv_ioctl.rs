//! Common NV ioctl HIPC buffer ABI, independent of the device generation.
use super::prelude::*;
use crate::ipc_wire::message::ReceiveStaticDescriptor;
const NV_IOCTL_WRITE: u32 = 1 << 30;
const NV_IOCTL_READ: u32 = 1 << 31;
const NV_IOCTL_SIZE_MASK: u32 = 0x3fff;

// The ioctl direction selects the buffers consumed by the driver and their
// common size. libnx also clears the address on the unused side; official
// NvOsDrvIoctl clients retain the argument address with a zero size:
// https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/nv.c#L137-L170

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct NvIoctlBuffers {
    pub input: Option<BufferDescriptor>,
    pub additional_input: Option<BufferDescriptor>,
    pub output: Option<BufferDescriptor>,
    pub additional_output: Option<BufferDescriptor>,
}

pub(super) fn nv_ioctl_buffers(
    hipc: &HipcRequest<'_>,
    request: u32,
) -> Result<NvIoctlBuffers, IpcWireError> {
    let ReceiveStatics::Entries(pointers) = &hipc.receive_statics else {
        return Err(IpcWireError::Malformed(
            "nvdrv ioctl requires one output auto-select pointer",
        ));
    };
    nv_ioctl_buffer_descriptors(
        &hipc.send_statics,
        &hipc.send_buffers,
        &hipc.receive_buffers,
        pointers,
        request,
    )
}

fn nv_ioctl_buffer_descriptors(
    send_statics: &[SendStaticDescriptor],
    send_buffers: &[BufferDescriptor],
    receive_buffers: &[BufferDescriptor],
    receive_statics: &[ReceiveStaticDescriptor],
    request: u32,
) -> Result<NvIoctlBuffers, IpcWireError> {
    let [input] = send_buffers else {
        return Err(IpcWireError::Malformed(
            "nvdrv ioctl requires exactly one input auto-select buffer",
        ));
    };
    let [output] = receive_buffers else {
        return Err(IpcWireError::Malformed(
            "nvdrv ioctl requires exactly one output auto-select buffer",
        ));
    };
    let [input_pointer] = send_statics else {
        return Err(IpcWireError::Malformed(
            "nvdrv ioctl requires exactly one input auto-select pointer",
        ));
    };
    let [output_pointer] = receive_statics else {
        return Err(IpcWireError::Malformed(
            "nvdrv ioctl requires exactly one output auto-select pointer",
        ));
    };

    // Nixe reports a zero CMIF pointer-buffer size, so non-empty AutoSelect
    // transfers use map-alias buffers. A zero-length pointer transfers nothing
    // and may retain an address; libnx happens to clear that address as well.
    // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/sf/cmif.h#L157-L177
    if input_pointer.size != 0 || output_pointer.size != 0 {
        return Err(IpcWireError::Malformed(
            "nvdrv ioctl pointer buffers exceed the advertised zero pointer-buffer size",
        ));
    }

    let encoded_size = u64::from((request >> 16) & NV_IOCTL_SIZE_MASK);
    let select = |descriptor: BufferDescriptor,
                  present: bool,
                  present_error: &'static str,
                  absent_error: &'static str|
     -> Result<Option<BufferDescriptor>, IpcWireError> {
        if present {
            if descriptor.size != encoded_size || (encoded_size != 0 && descriptor.address == 0) {
                return Err(IpcWireError::Malformed(present_error));
            }
            Ok(Some(descriptor))
        } else if descriptor.size == 0 {
            // A zero-length buffer has no memory to access, even when its
            // descriptor retains the caller's argument address.
            Ok(None)
        } else {
            Err(IpcWireError::Malformed(absent_error))
        }
    };

    Ok(NvIoctlBuffers {
        additional_output: None,
        input: select(
            *input,
            request & NV_IOCTL_WRITE != 0,
            "nvdrv ioctl input buffer does not match its encoded direction and size",
            "nvdrv ioctl without input carries a non-empty input buffer",
        )?,
        additional_input: None,
        output: select(
            *output,
            request & NV_IOCTL_READ != 0,
            "nvdrv ioctl output buffer does not match its encoded direction and size",
            "nvdrv ioctl without output carries a non-empty output buffer",
        )?,
    })
}

pub(super) fn nv_ioctl2_buffers(
    hipc: &HipcRequest<'_>,
    request: u32,
) -> Result<NvIoctlBuffers, IpcWireError> {
    let ReceiveStatics::Entries(pointers) = &hipc.receive_statics else {
        return Err(IpcWireError::Malformed(
            "nvdrv Ioctl2 requires one output auto-select pointer",
        ));
    };
    nv_ioctl2_buffer_descriptors(
        &hipc.send_statics,
        &hipc.send_buffers,
        &hipc.receive_buffers,
        pointers,
        request,
    )
}

fn nv_ioctl2_buffer_descriptors(
    send_statics: &[SendStaticDescriptor],
    send_buffers: &[BufferDescriptor],
    receive_buffers: &[BufferDescriptor],
    receive_statics: &[ReceiveStaticDescriptor],
    request: u32,
) -> Result<NvIoctlBuffers, IpcWireError> {
    let [input, additional_input] = send_buffers else {
        return Err(IpcWireError::Malformed(
            "nvdrv Ioctl2 requires exactly two input auto-select buffers",
        ));
    };
    let [output] = receive_buffers else {
        return Err(IpcWireError::Malformed(
            "nvdrv Ioctl2 requires exactly one output auto-select buffer",
        ));
    };
    let [input_pointer, additional_input_pointer] = send_statics else {
        return Err(IpcWireError::Malformed(
            "nvdrv Ioctl2 requires exactly two input auto-select pointers",
        ));
    };
    let [output_pointer] = receive_statics else {
        return Err(IpcWireError::Malformed(
            "nvdrv Ioctl2 requires exactly one output auto-select pointer",
        ));
    };
    if [input_pointer, additional_input_pointer]
        .into_iter()
        .any(|pointer| pointer.size != 0)
        || output_pointer.size != 0
    {
        return Err(IpcWireError::Malformed(
            "nvdrv Ioctl2 pointer buffers exceed the advertised zero pointer-buffer size",
        ));
    }
    if additional_input.size != 0 && additional_input.address == 0 {
        return Err(IpcWireError::Malformed(
            "nvdrv Ioctl2 additional input buffer is invalid",
        ));
    }

    let ordinary = nv_ioctl_buffer_descriptors(
        &[*input_pointer],
        &[*input],
        &[*output],
        std::slice::from_ref(output_pointer),
        request,
    )?;
    Ok(NvIoctlBuffers {
        additional_output: None,
        input: ordinary.input,
        additional_input: Some(*additional_input),
        output: ordinary.output,
    })
}

// Ioctl3 adds a second output AutoSelect buffer to the ordinary ioctl ABI.
// https://github.com/switchbrew/libnx/blob/master/nx/source/services/nv.c
pub(super) fn nv_ioctl3_buffers(
    hipc: &HipcRequest<'_>,
    request: u32,
) -> Result<NvIoctlBuffers, IpcWireError> {
    let [output, inline] = hipc.receive_buffers.as_slice() else {
        return Err(IpcWireError::Malformed(
            "nvdrv Ioctl3 requires two output buffers",
        ));
    };
    let ReceiveStatics::Entries(pointers) = &hipc.receive_statics else {
        return Err(IpcWireError::Malformed(
            "nvdrv Ioctl3 requires two output pointers",
        ));
    };
    let [output_pointer, inline_pointer] = pointers.as_slice() else {
        return Err(IpcWireError::Malformed(
            "nvdrv Ioctl3 requires two output pointers",
        ));
    };
    if inline_pointer.size != 0 || (inline.size != 0 && inline.address == 0) {
        return Err(IpcWireError::Malformed(
            "nvdrv Ioctl3 inline output buffer is invalid",
        ));
    }
    let mut buffers = nv_ioctl_buffer_descriptors(
        &hipc.send_statics,
        &hipc.send_buffers,
        &[*output],
        std::slice::from_ref(output_pointer),
        request,
    )?;
    buffers.additional_output = Some(*inline);
    Ok(buffers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipc_wire::message::BufferMode;

    fn buffer(address: u64, size: u64) -> BufferDescriptor {
        BufferDescriptor {
            address,
            size,
            mode: BufferMode::Normal,
        }
    }

    fn ioctl_descriptors(
        input: BufferDescriptor,
        output: BufferDescriptor,
        request: u32,
    ) -> Result<NvIoctlBuffers, IpcWireError> {
        nv_ioctl_buffer_descriptors(
            &[SendStaticDescriptor {
                address: 0,
                size: 0,
                index: 0,
            }],
            &[input],
            &[output],
            &[ReceiveStaticDescriptor {
                address: 0,
                size: 0,
            }],
            request,
        )
    }

    #[test]
    fn write_only_ioctl_accepts_the_libnx_null_output_placeholder() {
        assert_eq!(
            (ioctl_descriptors(buffer(0x1000, 40), buffer(0, 0), 0x4028_4109)).unwrap(),
            NvIoctlBuffers {
                additional_output: None,
                input: Some(buffer(0x1000, 40)),
                additional_input: None,
                output: None,
            }
        );
    }

    #[test]
    fn ioctl_accepts_empty_pointer_descriptors_with_retained_addresses() {
        let input = buffer(0x1000, 40);
        let output = buffer(0x2000, 40);
        for address in [0, 0x10_73ef_f87c, (1 << 42) - 1] {
            let mut input_pointer = SendStaticDescriptor {
                address,
                size: 0,
                index: 0,
            };
            let mut output_pointer = ReceiveStaticDescriptor { address, size: 0 };
            let decode = |input_pointer, output_pointer| {
                nv_ioctl_buffer_descriptors(
                    &[input_pointer],
                    &[input],
                    &[output],
                    &[output_pointer],
                    0xc028_4702,
                )
            };
            assert_eq!(
                (decode(input_pointer, output_pointer)).unwrap(),
                NvIoctlBuffers {
                    additional_output: None,
                    input: Some(input),
                    additional_input: None,
                    output: Some(output),
                }
            );
            // Non-empty pointers remain invalid even when the map-alias side
            // has the expected size: no pointer transfer was negotiated.
            input_pointer.size = 1;
            assert!(decode(input_pointer, output_pointer).is_err());
            input_pointer.size = 0;
            output_pointer.size = 1;
            assert!(decode(input_pointer, output_pointer).is_err());
        }
    }

    #[test]
    fn ioctl_direction_accepts_zero_length_buffers_with_retained_addresses() {
        for mode in [
            BufferMode::Normal,
            BufferMode::NonSecure,
            BufferMode::NonDevice,
        ] {
            let unused = BufferDescriptor {
                mode,
                ..buffer(0x2000, 0)
            };
            assert_eq!(
                (ioctl_descriptors(unused, buffer(0x1000, 4), 0x8004_4701)).unwrap(),
                NvIoctlBuffers {
                    additional_output: None,
                    input: None,
                    additional_input: None,
                    output: Some(buffer(0x1000, 4)),
                }
            );
            assert_eq!(
                (ioctl_descriptors(buffer(0x1000, 40), unused, 0x4028_4109)).unwrap(),
                NvIoctlBuffers {
                    additional_output: None,
                    input: Some(buffer(0x1000, 40)),
                    additional_input: None,
                    output: None,
                }
            );
        }
    }

    #[test]
    fn ioctl_direction_rejects_a_non_empty_inactive_side() {
        assert!(matches!(
            ioctl_descriptors(buffer(0x1000, 40), buffer(0x2000, 40), 0x4028_4109),
            Err(IpcWireError::Malformed(
                "nvdrv ioctl without output carries a non-empty output buffer"
            ))
        ));
        assert!(matches!(
            ioctl_descriptors(buffer(0x1000, 4), buffer(0x2000, 4), 0x8004_4701),
            Err(IpcWireError::Malformed(
                "nvdrv ioctl without input carries a non-empty input buffer"
            ))
        ));
    }

    #[test]
    fn ioctl_still_validates_active_buffer_sizes_and_addresses() {
        for input in [
            buffer(0, 40),
            buffer(0x1000, 0),
            buffer(0x1000, 39),
            buffer(0x1000, 41),
        ] {
            assert!(ioctl_descriptors(input, buffer(0x2000, 0), 0x4028_4109).is_err());
        }
        for output in [
            buffer(0, 4),
            buffer(0x1000, 0),
            buffer(0x1000, 3),
            buffer(0x1000, 5),
        ] {
            assert!(ioctl_descriptors(buffer(0x2000, 0), output, 0x8004_4701).is_err());
        }
    }

    #[test]
    fn ioctl2_preserves_additional_input_with_a_zero_length_unused_output() {
        let input = buffer(0x1000, 40);
        let additional = buffer(0x2000, 16);
        let pointer = SendStaticDescriptor {
            address: 0,
            size: 0,
            index: 0,
        };
        assert_eq!(
            (nv_ioctl2_buffer_descriptors(
                &[
                    pointer,
                    SendStaticDescriptor {
                        index: 1,
                        ..pointer
                    }
                ],
                &[input, additional],
                &[buffer(0x1000, 0)],
                &[ReceiveStaticDescriptor {
                    address: 0,
                    size: 0
                }],
                0x4028_4109,
            ))
            .unwrap(),
            NvIoctlBuffers {
                additional_output: None,
                input: Some(input),
                additional_input: Some(additional),
                output: None,
            }
        );
    }

    #[test]
    fn ioctl2_accepts_only_empty_pointers_regardless_of_address() {
        let mut pointers = [
            SendStaticDescriptor {
                address: 0x10_73ef_f87c,
                size: 0,
                index: 0,
            },
            SendStaticDescriptor {
                address: 0x10_73ef_f97c,
                size: 0,
                index: 1,
            },
        ];
        let output_pointer = ReceiveStaticDescriptor {
            address: 0x10_73ef_fa7c,
            size: 0,
        };
        let input = buffer(0x1000, 24);
        let additional = buffer(0x2000, 16);
        let output = buffer(0x1000, 24);
        let decode = |pointers: &[SendStaticDescriptor], output_pointer| {
            nv_ioctl2_buffer_descriptors(
                pointers,
                &[input, additional],
                &[output],
                &[output_pointer],
                0xc018_481b,
            )
        };
        assert_eq!(
            (decode(&pointers, output_pointer)).unwrap(),
            NvIoctlBuffers {
                additional_output: None,
                input: Some(input),
                additional_input: Some(additional),
                output: Some(output),
            }
        );
        for index in 0..pointers.len() {
            pointers[index].size = 1;
            assert!(decode(&pointers, output_pointer).is_err());
            pointers[index].size = 0;
        }
        assert!(
            decode(
                &pointers,
                ReceiveStaticDescriptor {
                    size: 1,
                    ..output_pointer
                }
            )
            .is_err()
        );
    }
}
