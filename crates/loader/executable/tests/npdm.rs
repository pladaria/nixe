use std::sync::Arc;

use nixe_loader_executable::{AddressSpaceType, NpdmLoader};
use nixe_loader_storage::{FormatLoader, LoadError, Storage, StorageError, StorageRef};

#[derive(Debug)]
struct Bytes(Vec<u8>);

impl Storage for Bytes {
    fn len(&self) -> Result<u64, StorageError> {
        u64::try_from(self.0.len()).map_err(|_| StorageError::OutOfBounds)
    }

    fn read_at(&self, offset: u64, output: &mut [u8]) -> Result<(), StorageError> {
        let start = usize::try_from(offset).map_err(|_| StorageError::OutOfBounds)?;
        let end = start
            .checked_add(output.len())
            .ok_or(StorageError::OutOfBounds)?;
        output.copy_from_slice(self.0.get(start..end).ok_or(StorageError::OutOfBounds)?);
        Ok(())
    }
}

fn put32(data: &mut [u8], offset: usize, value: usize) {
    data[offset..offset + 4].copy_from_slice(&(value as u32).to_le_bytes());
}

fn put64(data: &mut [u8], offset: usize, value: u64) {
    data[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn load(data: Vec<u8>) -> Result<nixe_loader_executable::Npdm, LoadError> {
    let storage: StorageRef = Arc::new(Bytes(data));
    NpdmLoader::load(storage)
}

fn build_npdm() -> Vec<u8> {
    const META: usize = 0x80;
    const ACID_HEADER: usize = 0x240;
    const FAC_SIZE: usize = 0x2c;
    const FAH_SIZE: usize = 0x1c;

    let acid_sac = [1, b'f', b'*'];
    let aci_sac = [1, b'f', b's'];
    let kac = (0x15_u32 << 5 | 0xf).to_le_bytes();

    let fac = ACID_HEADER;
    let acid_sac_offset = fac + FAC_SIZE;
    let acid_kac = acid_sac_offset + acid_sac.len();
    let acid_size = acid_kac + kac.len();
    let acid = META;

    let aci = acid + acid_size;
    let fah = 0x40;
    let aci_sac_offset = fah + FAH_SIZE;
    let aci_kac = aci_sac_offset + aci_sac.len();
    let aci_size = aci_kac + kac.len();
    let mut data = vec![0; aci + aci_size];

    data[..4].copy_from_slice(b"META");
    data[0xc] = 1 | (3 << 1);
    data[0xe] = 0x20;
    data[0xf] = 2;
    put32(&mut data, 0x1c, 0x4000);
    data[0x20..0x2b].copy_from_slice(b"Application");
    data[0x30..0x34].copy_from_slice(b"TEST");
    put32(&mut data, 0x70, aci);
    put32(&mut data, 0x74, aci_size);
    put32(&mut data, 0x78, acid);
    put32(&mut data, 0x7c, acid_size);

    data[acid + 0x200..acid + 0x204].copy_from_slice(b"ACID");
    put32(&mut data, acid + 0x204, acid_size - 0x100);
    data[acid + 0x208] = 1;
    put32(&mut data, acid + 0x20c, 1);
    put64(&mut data, acid + 0x210, 0x0100_0000_0000_1000);
    put64(&mut data, acid + 0x218, 0x0100_0000_0000_1fff);
    put32(&mut data, acid + 0x220, fac);
    put32(&mut data, acid + 0x224, FAC_SIZE);
    put32(&mut data, acid + 0x228, acid_sac_offset);
    put32(&mut data, acid + 0x22c, acid_sac.len());
    put32(&mut data, acid + 0x230, acid_kac);
    put32(&mut data, acid + 0x234, kac.len());
    data[acid + fac] = 1;
    put64(&mut data, acid + fac + 4, 3);
    put64(&mut data, acid + fac + 0x14, u64::MAX);
    put64(&mut data, acid + fac + 0x24, u64::MAX);
    data[acid + acid_sac_offset..acid + acid_kac].copy_from_slice(&acid_sac);
    data[acid + acid_kac..acid + acid_size].copy_from_slice(&kac);

    data[aci..aci + 4].copy_from_slice(b"ACI0");
    put64(&mut data, aci + 0x10, 0x0100_0000_0000_1234);
    put32(&mut data, aci + 0x20, fah);
    put32(&mut data, aci + 0x24, FAH_SIZE);
    put32(&mut data, aci + 0x28, aci_sac_offset);
    put32(&mut data, aci + 0x2c, aci_sac.len());
    put32(&mut data, aci + 0x30, aci_kac);
    put32(&mut data, aci + 0x34, kac.len());
    data[aci + fah] = 1;
    put64(&mut data, aci + fah + 4, 1);
    data[aci + aci_sac_offset..aci + aci_kac].copy_from_slice(&aci_sac);
    data[aci + aci_kac..aci + aci_size].copy_from_slice(&kac);
    data
}

fn invalid(data: Vec<u8>, expected: &str) {
    match load(data) {
        Err(LoadError::InvalidFormat { format, reason }) => {
            assert_eq!(format, "NPDM");
            assert!(reason.contains(expected), "{expected:?} not in {reason:?}");
        }
        result => panic!("expected invalid NPDM, got {result:?}"),
    }
}

#[test]
fn loads_metadata_and_effective_policy() {
    let npdm = load(build_npdm()).unwrap();
    assert_eq!(npdm.name_str(), Some("Application"));
    assert_eq!(npdm.product_code_str(), Some("TEST"));
    assert_eq!(npdm.program_id(), 0x0100_0000_0000_1234);
    assert_eq!(
        npdm.flags().address_space(),
        AddressSpaceType::AddressSpace64Bit
    );
    assert_eq!(npdm.main_thread_stack_size(), 0x4000);
    assert!(npdm.effective_policy().allows_client(b"fs"));
    assert!(!npdm.effective_policy().allows_host(b"fs"));
    assert_eq!(npdm.requested_filesystem().permissions().raw(), 1);
    assert_eq!(npdm.authorized_filesystem().permissions().raw(), 3);
}

#[test]
fn rejects_escalated_permissions_and_services() {
    let mut filesystem = build_npdm();
    let aci = u32::from_le_bytes(filesystem[0x70..0x74].try_into().unwrap()) as usize;
    put64(&mut filesystem, aci + 0x44, 4);
    invalid(filesystem, "filesystem permissions exceed");

    let mut service = build_npdm();
    let aci = u32::from_le_bytes(service[0x70..0x74].try_into().unwrap()) as usize;
    let sac = u32::from_le_bytes(service[aci + 0x28..aci + 0x2c].try_into().unwrap()) as usize;
    service[aci + sac + 1..aci + sac + 3].copy_from_slice(b"sm");
    invalid(service, "service access exceeds");

    let mut kernel = build_npdm();
    let aci = u32::from_le_bytes(kernel[0x70..0x74].try_into().unwrap()) as usize;
    let kac = u32::from_le_bytes(kernel[aci + 0x30..aci + 0x34].try_into().unwrap()) as usize;
    let descriptor = u32::from_le_bytes(kernel[aci + kac..aci + kac + 4].try_into().unwrap());
    kernel[aci + kac..aci + kac + 4].copy_from_slice(&(descriptor | (1 << 20)).to_le_bytes());
    invalid(kernel, "kernel capabilities exceed");

    let mut identity = build_npdm();
    let aci = u32::from_le_bytes(identity[0x70..0x74].try_into().unwrap()) as usize;
    put64(&mut identity, aci + 0x10, 0x0200_0000_0000_0000);
    invalid(identity, "program ID is outside");
}

#[test]
fn rejects_corruption_and_all_truncated_prefixes() {
    let mut overlap = build_npdm();
    let acid = u32::from_le_bytes(overlap[0x78..0x7c].try_into().unwrap()) as usize;
    put32(&mut overlap, 0x70, acid);
    invalid(overlap, "overlap");

    let complete = build_npdm();
    for length in 0..complete.len() {
        assert!(
            load(complete[..length].to_vec()).is_err(),
            "prefix {length:#x} parsed"
        );
    }
}

#[test]
fn rejects_or_safely_exposes_malformed_edge_cases() {
    let mut magic = build_npdm();
    magic[0] = b'X';
    invalid(magic, "META magic");

    let mut flags = build_npdm();
    flags[0x0c] = 4 << 1;
    invalid(flags, "process flag bits");

    let mut offset = build_npdm();
    put32(&mut offset, 0x70, u32::MAX as usize);
    invalid(offset, "outside its parent");

    let mut service = build_npdm();
    let aci = u32::from_le_bytes(service[0x70..0x74].try_into().unwrap()) as usize;
    let sac = u32::from_le_bytes(service[aci + 0x28..aci + 0x2c].try_into().unwrap()) as usize;
    service[aci + sac] = 7;
    invalid(service, "service entry is truncated");

    let mut text = build_npdm();
    text[0x20] = 0xff;
    let npdm = load(text).unwrap();
    assert_eq!(npdm.name()[0], 0xff);
    assert_eq!(npdm.name_str(), None);
}
