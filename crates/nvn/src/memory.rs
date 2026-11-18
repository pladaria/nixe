use super::*;
use bytemuck::Pod;
use nixe_memory::{Access, AccessResult, GuestVirtualAddress};

impl Call<'_> {
    pub(crate) fn read<T: Pod>(&self, address: u64) -> Result<T> {
        let mut value = T::zeroed();
        self.read_bytes(address, bytemuck::bytes_of_mut(&mut value))?;
        Ok(value)
    }
    pub(crate) fn read_bytes(&self, address: u64, dst: &mut [u8]) -> Result<()> {
        match self
            .memory
            .read_into(GuestVirtualAddress::new(address), dst)
            .map_err(|e| e.to_string())?
        {
            Access::Ready(()) => Ok(()),
            Access::Pending(p) => match p.wait().map_err(|e| e.to_string())? {
                AccessResult::Read(bytes) if bytes.len() == dst.len() => {
                    dst.copy_from_slice(&bytes);
                    Ok(())
                }
                _ => Err("unexpected device memory read result".into()),
            },
        }
    }
    pub(crate) fn write<T: Pod>(&self, address: u64, value: &T) -> Result<()> {
        self.write_bytes(address, bytemuck::bytes_of(value))
    }
    pub(crate) fn write_bytes(&self, address: u64, bytes: &[u8]) -> Result<()> {
        if let Access::Pending(p) = self
            .memory
            .write(GuestVirtualAddress::new(address), bytes)
            .map_err(|e| e.to_string())?
        {
            p.wait().map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}
