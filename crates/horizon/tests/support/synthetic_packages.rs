//! Redistributable package builders for end-to-end Horizon launch tests.
//!
//! Every byte produced here is synthetic. The builders deliberately emit only
//! the small, decrypted subset of each format needed by the public loaders.

use sha2::{Digest, Sha256};

pub const APPLICATION_ID: u64 = 0x0100_5a11_0000_0000;
pub const PATCH_ID: u64 = APPLICATION_ID + 0x800;
pub const FIRST_DLC_ID: u64 = APPLICATION_ID + 0x1001;
pub const SECOND_DLC_ID: u64 = APPLICATION_ID + 0x1002;

const NCA_HEADER_SIZE: usize = 0xc00;
const MEDIA_UNIT_SIZE: usize = 0x200;
const BKTR_NODE_SIZE: usize = 0x4000;

#[derive(Clone, Copy)]
pub enum MetaKind {
    Application,
    Patch,
    AddOnContent { required_application_version: u32 },
}

pub struct Content {
    pub id: [u8; 16],
    pub content_type: u8,
    pub id_offset: u8,
    pub nca: Vec<u8>,
}

pub struct Package {
    pub title_id: u64,
    pub version: u32,
    pub kind: MetaKind,
    pub contents: Vec<Content>,
}

pub fn content_id(seed: u8) -> [u8; 16] {
    std::array::from_fn(|index| seed.wrapping_add(index as u8))
}

pub fn program_content(id: [u8; 16], modules: &[(&str, u8)]) -> Content {
    program_content_with_npdm_services(id, modules, true)
}

pub fn program_content_without_services(id: [u8; 16], modules: &[(&str, u8)]) -> Content {
    program_content_with_npdm_services(id, modules, false)
}

fn program_content_with_npdm_services(
    id: [u8; 16],
    modules: &[(&str, u8)],
    include_services: bool,
) -> Content {
    let npdm = build_npdm(APPLICATION_ID, include_services);
    let module_bytes = modules
        .iter()
        .map(|(name, seed)| ((*name).to_owned(), build_nso(*seed)))
        .collect::<Vec<_>>();
    let mut files = module_bytes
        .iter()
        .map(|(name, bytes)| (name.as_str(), bytes.as_slice()))
        .collect::<Vec<_>>();
    files.push(("main.npdm", npdm.as_slice()));
    let exefs = build_pfs0(&files);
    Content {
        id,
        content_type: 1,
        id_offset: 0,
        nca: build_nca(APPLICATION_ID, 0, vec![pfs0_section(exefs)]),
    }
}

pub fn data_content(id: [u8; 16], title_id: u64, id_offset: u8, romfs: Vec<u8>) -> Content {
    Content {
        id,
        content_type: 2,
        id_offset,
        nca: build_nca(title_id, 4, vec![romfs_section(romfs)]),
    }
}

pub fn bktr_data_content(
    id: [u8; 16],
    id_offset: u8,
    base_romfs: &[u8],
    effective_romfs: &[u8],
) -> Content {
    assert_eq!(base_romfs.len(), effective_romfs.len());
    Content {
        id,
        content_type: 2,
        id_offset,
        nca: build_nca(
            APPLICATION_ID,
            4,
            vec![bktr_section(base_romfs, effective_romfs)],
        ),
    }
}

pub fn build_nsp(package: &Package) -> Vec<u8> {
    let meta = build_meta_nca(package);
    let meta_name = format!("{:032x}.cnmt.nca", package.title_id);
    let mut owned = vec![(meta_name, meta)];
    owned.extend(
        package
            .contents
            .iter()
            .map(|content| (format!("{}.nca", hex(&content.id)), content.nca.clone())),
    );
    let files = owned
        .iter()
        .map(|(name, bytes)| (name.as_str(), bytes.as_slice()))
        .collect::<Vec<_>>();
    build_pfs0(&files)
}

pub fn build_xci(packages: &[Package]) -> Vec<u8> {
    let mut owned = Vec::new();
    for package in packages {
        owned.push((
            format!("{:032x}.cnmt.nca", package.title_id),
            build_meta_nca(package),
        ));
        owned.extend(
            package
                .contents
                .iter()
                .map(|content| (format!("{}.nca", hex(&content.id)), content.nca.clone())),
        );
    }
    let secure_files = owned
        .iter()
        .map(|(name, bytes)| (name.as_str(), bytes.as_slice()))
        .collect::<Vec<_>>();
    let secure = build_hfs0(&secure_files);
    let root = build_hfs0(&[("secure", secure.as_slice())]);
    let root_header_size = 0x10 + 0x40 + "secure".len() + 1;
    let root_offset = 0x200;
    let image_size = root_offset + root.len();
    let page_count = image_size.div_ceil(MEDIA_UNIT_SIZE);
    let mut image = vec![0_u8; page_count * MEDIA_UNIT_SIZE];
    image[0x100..0x104].copy_from_slice(b"HEAD");
    put_u32(&mut image, 0x118, (page_count - 1) as u32);
    put_u64(&mut image, 0x130, root_offset as u64);
    put_u64(&mut image, 0x138, root_header_size as u64);
    image[0x140..0x160].copy_from_slice(&Sha256::digest(&root[..root_header_size]));
    image[root_offset..root_offset + root.len()].copy_from_slice(&root);
    image
}

pub fn build_romfs(files: &[(&str, &[u8])]) -> Vec<u8> {
    const EMPTY: u32 = u32::MAX;
    let mut file_meta = Vec::new();
    let mut data_offset = 0_u64;
    for (index, (name, data)) in files.iter().enumerate() {
        let next = if index + 1 == files.len() {
            EMPTY
        } else {
            (file_meta.len() + 0x20 + name.len().next_multiple_of(4)) as u32
        };
        file_meta.extend_from_slice(&0_u32.to_le_bytes());
        file_meta.extend_from_slice(&next.to_le_bytes());
        file_meta.extend_from_slice(&data_offset.to_le_bytes());
        file_meta.extend_from_slice(&(data.len() as u64).to_le_bytes());
        file_meta.extend_from_slice(&EMPTY.to_le_bytes());
        file_meta.extend_from_slice(&(name.len() as u32).to_le_bytes());
        file_meta.extend_from_slice(name.as_bytes());
        while file_meta.len() % 4 != 0 {
            file_meta.push(0);
        }
        data_offset += data.len() as u64;
    }
    let file_data_offset = (0x70 + file_meta.len()).next_multiple_of(0x10);
    let mut bytes = vec![0_u8; file_data_offset];
    for (offset, value) in [
        (0, 0x50),
        (0x08, 0x50),
        (0x10, 4),
        (0x18, 0x54),
        (0x20, 0x18),
        (0x28, 0x6c),
        (0x30, 4),
        (0x38, 0x70),
        (0x40, file_meta.len() as u64),
        (0x48, file_data_offset as u64),
    ] {
        put_u64(&mut bytes, offset, value);
    }
    put_u32(&mut bytes, 0x54, 0);
    put_u32(&mut bytes, 0x58, EMPTY);
    put_u32(&mut bytes, 0x5c, EMPTY);
    put_u32(&mut bytes, 0x60, if files.is_empty() { EMPTY } else { 0 });
    put_u32(&mut bytes, 0x64, EMPTY);
    put_u32(&mut bytes, 0x68, 0);
    bytes[0x70..0x70 + file_meta.len()].copy_from_slice(&file_meta);
    for (_, data) in files {
        bytes.extend_from_slice(data);
    }
    bytes
}

fn build_meta_nca(package: &Package) -> Vec<u8> {
    let cnmt = build_cnmt(package);
    let pfs0 = build_pfs0(&[("ContentMeta.cnmt", cnmt.as_slice())]);
    build_nca(package.title_id, 1, vec![pfs0_section(pfs0)])
}

fn build_cnmt(package: &Package) -> Vec<u8> {
    let (raw_type, extended) = match package.kind {
        MetaKind::Application => {
            let mut bytes = vec![0_u8; 0x10];
            put_u64(&mut bytes, 0, PATCH_ID);
            (0x80, bytes)
        }
        MetaKind::Patch => {
            let mut bytes = vec![0_u8; 0x18];
            put_u64(&mut bytes, 0, APPLICATION_ID);
            (0x81, bytes)
        }
        MetaKind::AddOnContent {
            required_application_version,
        } => {
            let mut bytes = vec![0_u8; 0x18];
            put_u64(&mut bytes, 0, APPLICATION_ID);
            put_u32(&mut bytes, 8, required_application_version);
            (0x82, bytes)
        }
    };
    let mut bytes = vec![0_u8; 0x20];
    put_u64(&mut bytes, 0, package.title_id);
    put_u32(&mut bytes, 8, package.version);
    bytes[0x0c] = raw_type;
    put_u16(&mut bytes, 0x0e, extended.len() as u16);
    put_u16(&mut bytes, 0x10, package.contents.len() as u16);
    bytes[0x17] = 1;
    bytes.extend_from_slice(&extended);
    for content in &package.contents {
        let mut record = [0_u8; 0x38];
        record[..0x20].copy_from_slice(&Sha256::digest(&content.nca));
        record[0x20..0x30].copy_from_slice(&content.id);
        record[0x30..0x35].copy_from_slice(&(content.nca.len() as u64).to_le_bytes()[..5]);
        record[0x36] = content.content_type;
        record[0x37] = content.id_offset;
        bytes.extend_from_slice(&record);
    }
    bytes.extend_from_slice(&[0xa5; 0x20]);
    bytes
}

struct NcaSectionFixture {
    bytes: Vec<u8>,
    fs_header: [u8; 0x200],
}

fn pfs0_section(payload: Vec<u8>) -> NcaSectionFixture {
    let block_size = 0x1000_usize;
    let mut hashes = Vec::new();
    for block in payload.chunks(block_size) {
        hashes.extend_from_slice(&Sha256::digest(block));
    }
    let data_offset = hashes.len().next_multiple_of(MEDIA_UNIT_SIZE);
    let mut section = vec![0_u8; data_offset + payload.len()];
    section[..hashes.len()].copy_from_slice(&hashes);
    section[data_offset..].copy_from_slice(&payload);
    let mut fs = [0_u8; 0x200];
    fs[2] = 1;
    fs[3] = 2;
    fs[4] = 1;
    fs[0x08..0x28].copy_from_slice(&Sha256::digest(&hashes));
    put_u32(&mut fs, 0x28, block_size as u32);
    put_u64(&mut fs, 0x30, 0);
    put_u64(&mut fs, 0x38, hashes.len() as u64);
    put_u64(&mut fs, 0x40, data_offset as u64);
    put_u64(&mut fs, 0x48, payload.len() as u64);
    NcaSectionFixture {
        bytes: section,
        fs_header: fs,
    }
}

fn romfs_section(payload: Vec<u8>) -> NcaSectionFixture {
    let mut fs = [0_u8; 0x200];
    fs[3] = 3;
    fs[4] = 1;
    ivfc_header(&mut fs, payload.len() as u64);
    NcaSectionFixture {
        bytes: payload,
        fs_header: fs,
    }
}

fn bktr_section(base: &[u8], effective: &[u8]) -> NcaSectionFixture {
    let mut relocations = Vec::new();
    let mut start = 0;
    let mut from_base = base[0] == effective[0];
    for index in 1..effective.len() {
        let next_from_base = base[index] == effective[index];
        if next_from_base != from_base {
            relocations.push(relocation_entry(start as u64, u32::from(!from_base)));
            start = index;
            from_base = next_from_base;
        }
    }
    relocations.push(relocation_entry(start as u64, u32::from(!from_base)));
    let indirect_offset = effective.len().next_multiple_of(MEDIA_UNIT_SIZE);
    let indirect = build_bucket_table(&relocations, 0x14, effective.len() as u64);
    let subsection_offset = indirect_offset + indirect.len();
    let subsection = build_bucket_table(&[subsection_entry()], 0x10, subsection_offset as u64);
    let mut bytes = vec![0_u8; subsection_offset + subsection.len()];
    bytes[..effective.len()].copy_from_slice(effective);
    bytes[indirect_offset..indirect_offset + indirect.len()].copy_from_slice(&indirect);
    bytes[subsection_offset..].copy_from_slice(&subsection);
    let mut fs = [0_u8; 0x200];
    fs[3] = 3;
    fs[4] = 4;
    ivfc_header(&mut fs, effective.len() as u64);
    put_u64(&mut fs, 0x100, indirect_offset as u64);
    put_u64(&mut fs, 0x108, indirect.len() as u64);
    write_bucket_header(&mut fs[0x110..0x120], relocations.len() as u32);
    put_u64(&mut fs, 0x120, subsection_offset as u64);
    put_u64(&mut fs, 0x128, subsection.len() as u64);
    write_bucket_header(&mut fs[0x130..0x140], 1);
    NcaSectionFixture {
        bytes,
        fs_header: fs,
    }
}

fn build_nca(title_id: u64, content_type: u8, sections: Vec<NcaSectionFixture>) -> Vec<u8> {
    assert!(sections.len() <= 4);
    let mut offsets = Vec::with_capacity(sections.len());
    let mut cursor = NCA_HEADER_SIZE;
    for section in &sections {
        let size = section.bytes.len().next_multiple_of(MEDIA_UNIT_SIZE);
        offsets.push((cursor, size));
        cursor += size;
    }
    let mut nca = vec![0_u8; cursor];
    nca[0x200..0x204].copy_from_slice(b"NCA3");
    nca[0x205] = content_type;
    nca[0x206] = 1;
    put_u64(&mut nca, 0x208, cursor as u64);
    put_u64(&mut nca, 0x210, title_id);
    for (index, (section, (offset, size))) in sections.iter().zip(offsets).enumerate() {
        put_u32(
            &mut nca,
            0x240 + index * 0x10,
            (offset / MEDIA_UNIT_SIZE) as u32,
        );
        put_u32(
            &mut nca,
            0x244 + index * 0x10,
            ((offset + size) / MEDIA_UNIT_SIZE) as u32,
        );
        let fs_offset = 0x400 + index * 0x200;
        nca[fs_offset..fs_offset + 0x200].copy_from_slice(&section.fs_header);
        nca[0x280 + index * 0x20..0x2a0 + index * 0x20]
            .copy_from_slice(&Sha256::digest(section.fs_header));
        nca[offset..offset + section.bytes.len()].copy_from_slice(&section.bytes);
    }
    nca
}

fn build_npdm(program_id: u64, include_services: bool) -> Vec<u8> {
    const META: usize = 0x80;
    const ACID_HEADER: usize = 0x240;
    const FAC_SIZE: usize = 0x2c;
    const FAH_SIZE: usize = 0x1c;
    const SERVICE_ACCESS: &[u8; 14] = b"\x06fsp-srv\x04aoc:u";
    let service_access = if include_services {
        SERVICE_ACCESS.as_slice()
    } else {
        &[]
    };
    let sac_size = service_access.len();
    let kac = (0x15_u32 << 5 | 0xf).to_le_bytes();
    let fac = ACID_HEADER;
    let acid_sac = fac + FAC_SIZE;
    let acid_kac = acid_sac + sac_size;
    let acid_size = acid_kac + kac.len();
    let acid = META;
    let aci = acid + acid_size;
    let fah = 0x40;
    let aci_sac = fah + FAH_SIZE;
    let aci_kac = aci_sac + sac_size;
    let aci_size = aci_kac + kac.len();
    let mut data = vec![0_u8; aci + aci_size];
    data[..4].copy_from_slice(b"META");
    data[0x0c] = 1 | (3 << 1);
    data[0x0e] = 0x20;
    data[0x0f] = 2;
    put_u32(&mut data, 0x1c, 0x4000);
    data[0x20..0x2f].copy_from_slice(b"SyntheticTitle\0");
    data[0x30..0x34].copy_from_slice(b"SWTX");
    put_u32(&mut data, 0x70, aci as u32);
    put_u32(&mut data, 0x74, aci_size as u32);
    put_u32(&mut data, 0x78, acid as u32);
    put_u32(&mut data, 0x7c, acid_size as u32);
    data[acid + 0x200..acid + 0x204].copy_from_slice(b"ACID");
    put_u32(&mut data, acid + 0x204, (acid_size - 0x100) as u32);
    data[acid + 0x208] = 1;
    put_u64(&mut data, acid + 0x210, program_id);
    put_u64(&mut data, acid + 0x218, program_id);
    put_u32(&mut data, acid + 0x220, fac as u32);
    put_u32(&mut data, acid + 0x224, FAC_SIZE as u32);
    put_u32(&mut data, acid + 0x228, acid_sac as u32);
    put_u32(&mut data, acid + 0x22c, sac_size as u32);
    put_u32(&mut data, acid + 0x230, acid_kac as u32);
    put_u32(&mut data, acid + 0x234, kac.len() as u32);
    data[acid + fac] = 1;
    put_u64(&mut data, acid + fac + 4, 1_u64 << 63);
    put_u64(&mut data, acid + fac + 0x14, u64::MAX);
    put_u64(&mut data, acid + fac + 0x24, u64::MAX);
    data[acid + acid_sac..acid + acid_sac + sac_size].copy_from_slice(service_access);
    data[acid + acid_kac..acid + acid_size].copy_from_slice(&kac);
    data[aci..aci + 4].copy_from_slice(b"ACI0");
    put_u64(&mut data, aci + 0x10, program_id);
    put_u32(&mut data, aci + 0x20, fah as u32);
    put_u32(&mut data, aci + 0x24, FAH_SIZE as u32);
    put_u32(&mut data, aci + 0x28, aci_sac as u32);
    put_u32(&mut data, aci + 0x2c, sac_size as u32);
    put_u32(&mut data, aci + 0x30, aci_kac as u32);
    put_u32(&mut data, aci + 0x34, kac.len() as u32);
    data[aci + fah] = 1;
    put_u64(&mut data, aci + fah + 4, 1_u64 << 63);
    data[aci + aci_sac..aci + aci_sac + sac_size].copy_from_slice(service_access);
    data[aci + aci_kac..aci + aci_size].copy_from_slice(&kac);
    data
}

fn build_nso(seed: u8) -> Vec<u8> {
    let mut payloads = [
        vec![seed; 0x100],
        vec![seed + 1; 0x80],
        vec![seed + 2; 0x40],
    ];
    payloads[0][..4].copy_from_slice(&0xd503_201f_u32.to_le_bytes());
    payloads[0][4..8].fill(0);
    let mut bytes = vec![0_u8; 0x106];
    bytes[..4].copy_from_slice(b"NSO0");
    put_u32(&mut bytes, 4, 3);
    put_u32(&mut bytes, 0x0c, 0x38);
    put_u32(&mut bytes, 0x1c, 0x100);
    put_u32(&mut bytes, 0x2c, 6);
    bytes[0x100..0x106].copy_from_slice(b"main\0\0");
    for index in 0..32 {
        bytes[0x40 + index] = seed.wrapping_add(index as u8);
    }
    for (index, payload) in payloads.iter().enumerate() {
        let descriptor = [0x10, 0x20, 0x30][index];
        let file_offset = bytes.len();
        put_u32(&mut bytes, descriptor, file_offset as u32);
        put_u32(&mut bytes, descriptor + 4, [0, 0x1000, 0x2000][index]);
        put_u32(&mut bytes, descriptor + 8, payload.len() as u32);
        put_u32(&mut bytes, [0x60, 0x64, 0x68][index], payload.len() as u32);
        bytes[0xa0 + index * 0x20..0xc0 + index * 0x20].copy_from_slice(&Sha256::digest(payload));
        bytes.extend_from_slice(payload);
    }
    put_u32(&mut bytes, 0x3c, 0x41);
    bytes
}

fn build_pfs0(files: &[(&str, &[u8])]) -> Vec<u8> {
    let mut strings = Vec::new();
    let mut name_offsets = Vec::new();
    for (name, _) in files {
        name_offsets.push(strings.len() as u32);
        strings.extend_from_slice(name.as_bytes());
        strings.push(0);
    }
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"PFS0");
    bytes.extend_from_slice(&(files.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&(strings.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&0_u32.to_le_bytes());
    let mut relative_offset = 0_u64;
    for ((_, data), name_offset) in files.iter().zip(name_offsets) {
        bytes.extend_from_slice(&relative_offset.to_le_bytes());
        bytes.extend_from_slice(&(data.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&name_offset.to_le_bytes());
        bytes.extend_from_slice(&0_u32.to_le_bytes());
        relative_offset += data.len() as u64;
    }
    bytes.extend_from_slice(&strings);
    for (_, data) in files {
        bytes.extend_from_slice(data);
    }
    bytes
}

fn build_hfs0(files: &[(&str, &[u8])]) -> Vec<u8> {
    let mut strings = Vec::new();
    let mut name_offsets = Vec::new();
    for (name, _) in files {
        name_offsets.push(strings.len() as u32);
        strings.extend_from_slice(name.as_bytes());
        strings.push(0);
    }
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"HFS0");
    bytes.extend_from_slice(&(files.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&(strings.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&0_u32.to_le_bytes());
    let mut relative_offset = 0_u64;
    for ((_, data), name_offset) in files.iter().zip(name_offsets) {
        bytes.extend_from_slice(&relative_offset.to_le_bytes());
        bytes.extend_from_slice(&(data.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&name_offset.to_le_bytes());
        bytes.extend_from_slice(&(data.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&[0; 8]);
        bytes.extend_from_slice(&Sha256::digest(data));
        relative_offset += data.len() as u64;
    }
    bytes.extend_from_slice(&strings);
    for (_, data) in files {
        bytes.extend_from_slice(data);
    }
    bytes
}

fn build_bucket_table(entries: &[Vec<u8>], entry_size: usize, end_offset: u64) -> Vec<u8> {
    let mut bytes = vec![0_u8; BKTR_NODE_SIZE * 2];
    put_i32(&mut bytes, 0, 0);
    put_i32(&mut bytes, 4, 1);
    put_u64(&mut bytes, 8, end_offset);
    put_u64(&mut bytes, 0x10, read_u64(&entries[0], 0));
    put_i32(&mut bytes, BKTR_NODE_SIZE, 0);
    put_i32(&mut bytes, BKTR_NODE_SIZE + 4, entries.len() as i32);
    put_u64(&mut bytes, BKTR_NODE_SIZE + 8, end_offset);
    for (index, entry) in entries.iter().enumerate() {
        let offset = BKTR_NODE_SIZE + 0x10 + index * entry_size;
        bytes[offset..offset + entry_size].copy_from_slice(entry);
    }
    bytes
}

fn relocation_entry(offset: u64, selector: u32) -> Vec<u8> {
    let mut entry = vec![0_u8; 0x14];
    put_u64(&mut entry, 0, offset);
    put_u64(&mut entry, 8, offset);
    put_u32(&mut entry, 0x10, selector);
    entry
}

fn subsection_entry() -> Vec<u8> {
    let mut entry = vec![0_u8; 0x10];
    entry[8] = 1;
    entry
}

fn ivfc_header(fs: &mut [u8], data_size: u64) {
    fs[0x08..0x0c].copy_from_slice(b"IVFC");
    put_u32(fs, 0x10, 0x20);
    put_u32(fs, 0x14, 2);
    put_u64(fs, 0x18, 0);
    put_u64(fs, 0x20, data_size);
    put_u32(fs, 0x28, 20);
}

fn write_bucket_header(bytes: &mut [u8], count: u32) {
    bytes[..4].copy_from_slice(b"BKTR");
    put_u32(bytes, 4, 1);
    put_u32(bytes, 8, count);
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

fn put_i32(bytes: &mut [u8], offset: usize, value: i32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u16(bytes: &mut [u8], offset: usize, value: u16) {
    bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}
