//! Shared NvDrv/NvDrv2 zero-bandwidth-clear table ABI and registrations.
use super::{
    NV_BAD_PARAMETER, NV_OVERFLOW, NvDrvCallError, input_u32, require_input_size, write_u32,
};

// Native clears carry their values directly and do not emit guest compression
// tags. The driver still retains registrations for subsequent table queries.
// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/service/nvdrv/devices/nvhost_ctrl_gpu.cpp
#[derive(Debug, Default)]
pub(crate) struct NvHostZbcTables {
    colors: Vec<ZbcColor>,
    depths: Vec<ZbcDepth>,
}

#[derive(Debug)]
struct ZbcColor {
    values: [u32; 8],
    format: u32,
    references: u32,
}

#[derive(Debug)]
struct ZbcDepth {
    value: u32,
    format: u32,
    references: u32,
}

impl NvHostZbcTables {
    // The ioctl carries two representations of the same clear color. Keep
    // both bit patterns; converting through a host float loses guest data.
    // https://github.com/switchbrew/libnx/blob/master/nx/source/nvidia/ioctl/nvhost-ctrl-gpu.c#L41-L78
    pub(crate) fn set(&mut self, input: &[u8]) -> Result<(), NvDrvCallError> {
        require_input_size(input, 44)?;
        let format = input_u32(input, 36)?;
        match input_u32(input, 40)? {
            1 => {
                let mut values = [0; 8];
                for (index, value) in values.iter_mut().enumerate() {
                    *value = input_u32(input, index * 4)?;
                }
                if let Some(entry) = self
                    .colors
                    .iter_mut()
                    .find(|entry| entry.format == format && entry.values == values)
                {
                    entry.references = entry
                        .references
                        .checked_add(1)
                        .ok_or(NvDrvCallError::GuestResult(NV_OVERFLOW))?;
                } else {
                    self.colors.push(ZbcColor {
                        values,
                        format,
                        references: 1,
                    });
                }
            }
            2 => {
                let value = input_u32(input, 32)?;
                if let Some(entry) = self
                    .depths
                    .iter_mut()
                    .find(|entry| entry.format == format && entry.value == value)
                {
                    entry.references = entry
                        .references
                        .checked_add(1)
                        .ok_or(NvDrvCallError::GuestResult(NV_OVERFLOW))?;
                } else {
                    self.depths.push(ZbcDepth {
                        value,
                        format,
                        references: 1,
                    });
                }
            }
            _ => return Err(NvDrvCallError::GuestResult(NV_BAD_PARAMETER)),
        }
        Ok(())
    }

    pub(crate) fn query(&self, input: &[u8]) -> Result<Vec<u8>, NvDrvCallError> {
        require_input_size(input, 52)?;
        let kind = input_u32(input, 44)?;
        let index = input_u32(input, 48)? as usize;
        let mut output = vec![0; 52];
        let (references, format, count) = match kind {
            1 => {
                let entry = self
                    .colors
                    .get(index)
                    .ok_or(NvDrvCallError::GuestResult(NV_BAD_PARAMETER))?;
                for (index, value) in entry.values.iter().enumerate() {
                    write_u32(&mut output, index * 4, *value)?;
                }
                (entry.references, entry.format, self.colors.len())
            }
            2 => {
                let entry = self
                    .depths
                    .get(index)
                    .ok_or(NvDrvCallError::GuestResult(NV_BAD_PARAMETER))?;
                write_u32(&mut output, 32, entry.value)?;
                (entry.references, entry.format, self.depths.len())
            }
            _ => return Err(NvDrvCallError::GuestResult(NV_BAD_PARAMETER)),
        };
        write_u32(&mut output, 36, references)?;
        write_u32(&mut output, 40, format)?;
        write_u32(&mut output, 44, kind)?;
        write_u32(&mut output, 48, count as u32)?;
        Ok(output)
    }
}
