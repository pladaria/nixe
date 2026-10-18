use crate::{CompileError, Result, diagnostics::FailureCategory};

#[derive(Clone, Copy, Debug)]
pub struct Encoding {
    pub opcode: Opcode,
    pub mask: u64,
    pub value: u64,
    pub name: &'static str,
    /// True only for forms whose visitor rejects the entire opcode. Other forms
    /// may still reject unsupported modifiers when translated.
    pub unsupported: bool,
}

macro_rules! encodings {
    ($($name:ident: $mask:literal, $value:literal, $unsupported:literal;)*) => {
        #[allow(non_camel_case_types)]
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        #[repr(u16)]
        pub enum Opcode { $($name),* }
        impl Opcode {
            pub const fn encoding(self) -> &'static Encoding { &ENCODINGS[self as usize] }
        }
        pub const ENCODINGS: &[Encoding] = &[$(Encoding {
            opcode: Opcode::$name, mask: $mask, value: $value,
            name: stringify!($name), unsupported: $unsupported,
        }),*];
    };
}

encodings! {
    CCTLT: 0xfff9000000000000, 0xebf0000000000000, true;
    AL2P: 0xfff8000000000000, 0xefa0000000000000, false;
    ALD: 0xfff8000000000000, 0xefd8000000000000, false;
    AST: 0xfff8000000000000, 0xeff0000000000000, false;
    B2R: 0xfff8000000000000, 0xf0b8000000000000, true;
    BAR: 0xfff8000000000000, 0xf0a8000000000000, false;
    BFE_cbuf: 0xfff8000000000000, 0x4c00000000000000, false;
    BFE_reg: 0xfff8000000000000, 0x5c00000000000000, false;
    BFI_cr: 0xfff8000000000000, 0x4bf0000000000000, false;
    BFI_rc: 0xfff8000000000000, 0x53f0000000000000, false;
    BFI_reg: 0xfff8000000000000, 0x5bf0000000000000, false;
    CS2R: 0xfff8000000000000, 0x50c8000000000000, true;
    CSET: 0xfff8000000000000, 0x5098000000000000, false;
    CSETP: 0xfff8000000000000, 0x50a0000000000000, false;
    DADD_cbuf: 0xfff8000000000000, 0x4c70000000000000, false;
    DADD_reg: 0xfff8000000000000, 0x5c70000000000000, false;
    DEPBAR: 0xfff8000000000000, 0xf0f0000000000000, false;
    DMNMX_cbuf: 0xfff8000000000000, 0x4c50000000000000, false;
    DMNMX_reg: 0xfff8000000000000, 0x5c50000000000000, false;
    DMUL_cbuf: 0xfff8000000000000, 0x4c80000000000000, false;
    DMUL_reg: 0xfff8000000000000, 0x5c80000000000000, false;
    F2F_cbuf: 0xfff8000000000000, 0x4ca8000000000000, false;
    F2F_reg: 0xfff8000000000000, 0x5ca8000000000000, false;
    F2I_cbuf: 0xfff8000000000000, 0x4cb0000000000000, false;
    F2I_reg: 0xfff8000000000000, 0x5cb0000000000000, false;
    FADD_cbuf: 0xfff8000000000000, 0x4c58000000000000, false;
    FADD_reg: 0xfff8000000000000, 0x5c58000000000000, false;
    FCHK_cbuf: 0xfff8000000000000, 0x4c88000000000000, true;
    FCHK_reg: 0xfff8000000000000, 0x5c88000000000000, true;
    FLO_cbuf: 0xfff8000000000000, 0x4c30000000000000, false;
    FLO_reg: 0xfff8000000000000, 0x5c30000000000000, false;
    FMNMX_cbuf: 0xfff8000000000000, 0x4c60000000000000, false;
    FMNMX_reg: 0xfff8000000000000, 0x5c60000000000000, false;
    FMUL_cbuf: 0xfff8000000000000, 0x4c68000000000000, false;
    FMUL_reg: 0xfff8000000000000, 0x5c68000000000000, false;
    FSWZADD: 0xfff8000000000000, 0x50f8000000000000, false;
    HADD2_reg: 0xfff8000000000000, 0x5d10000000000000, false;
    HFMA2_reg: 0xfff8000000000000, 0x5d00000000000000, false;
    HMUL2_reg: 0xfff8000000000000, 0x5d08000000000000, false;
    HSET2_reg: 0xfff8000000000000, 0x5d18000000000000, false;
    HSETP2_reg: 0xfff8000000000000, 0x5d20000000000000, false;
    I2F_cbuf: 0xfff8000000000000, 0x4cb8000000000000, false;
    I2F_reg: 0xfff8000000000000, 0x5cb8000000000000, false;
    I2I_cbuf: 0xfff8000000000000, 0x4ce0000000000000, false;
    I2I_reg: 0xfff8000000000000, 0x5ce0000000000000, false;
    IADD_cbuf: 0xfff8000000000000, 0x4c10000000000000, false;
    IADD_reg: 0xfff8000000000000, 0x5c10000000000000, false;
    IDP_imm: 0xfff8000000000000, 0x53d8000000000000, true;
    IDP_reg: 0xfff8000000000000, 0x53f8000000000000, true;
    IMNMX_cbuf: 0xfff8000000000000, 0x4c20000000000000, false;
    IMNMX_reg: 0xfff8000000000000, 0x5c20000000000000, false;
    IMUL_cbuf: 0xfff8000000000000, 0x4c38000000000000, true;
    IMUL_reg: 0xfff8000000000000, 0x5c38000000000000, true;
    ISBERD: 0xfff8000000000000, 0xefd0000000000000, false;
    ISCADD_cbuf: 0xfff8000000000000, 0x4c18000000000000, false;
    ISCADD_reg: 0xfff8000000000000, 0x5c18000000000000, false;
    LDC: 0xfff8000000000000, 0xef90000000000000, false;
    LDG: 0xfff8000000000000, 0xeed0000000000000, false;
    LDL: 0xfff8000000000000, 0xef40000000000000, false;
    LDS: 0xfff8000000000000, 0xef48000000000000, false;
    LEA_hi_reg: 0xfff8000000000000, 0x5bd8000000000000, false;
    LEA_lo_reg: 0xfff8000000000000, 0x5bd0000000000000, false;
    LEPC: 0xfff8000000000000, 0x50d0000000000000, true;
    LOP3_reg: 0xfff8000000000000, 0x5be0000000000000, false;
    LOP_cbuf: 0xfff8000000000000, 0x4c40000000000000, false;
    LOP_reg: 0xfff8000000000000, 0x5c40000000000000, false;
    MEMBAR: 0xfff8000000000000, 0xef98000000000000, false;
    MOV_cbuf: 0xfff8000000000000, 0x4c98000000000000, false;
    MOV_reg: 0xfff8000000000000, 0x5c98000000000000, false;
    MUFU: 0xfff8000000000000, 0x5080000000000000, false;
    NOP: 0xfff8000000000000, 0x50b0000000000000, false;
    OUT_cbuf: 0xfff8000000000000, 0xebe0000000000000, false;
    OUT_reg: 0xfff8000000000000, 0xfbe0000000000000, false;
    P2R_cbuf: 0xfff8000000000000, 0x4ce8000000000000, true;
    P2R_imm: 0xfff8000000000000, 0x38e8000000000000, false;
    P2R_reg: 0xfff8000000000000, 0x5ce8000000000000, true;
    PIXLD: 0xfff8000000000000, 0xefe8000000000000, false;
    POPC_cbuf: 0xfff8000000000000, 0x4c08000000000000, false;
    POPC_reg: 0xfff8000000000000, 0x5c08000000000000, false;
    PSET: 0xfff8000000000000, 0x5088000000000000, false;
    PSETP: 0xfff8000000000000, 0x5090000000000000, false;
    R2B: 0xfff8000000000000, 0xf0c0000000000000, true;
    R2P_cbuf: 0xfff8000000000000, 0x4cf0000000000000, false;
    R2P_reg: 0xfff8000000000000, 0x5cf0000000000000, false;
    RED: 0xfff8000000000000, 0xebf8000000000000, false;
    RRO_cbuf: 0xfff8000000000000, 0x4c90000000000000, false;
    RRO_reg: 0xfff8000000000000, 0x5c90000000000000, false;
    S2R: 0xfff8000000000000, 0xf0c8000000000000, false;
    SEL_cbuf: 0xfff8000000000000, 0x4ca0000000000000, false;
    SEL_reg: 0xfff8000000000000, 0x5ca0000000000000, false;
    SHFL: 0xfff8000000000000, 0xef10000000000000, false;
    SHF_l_reg: 0xfff8000000000000, 0x5bf8000000000000, false;
    SHF_r_reg: 0xfff8000000000000, 0x5cf8000000000000, false;
    SHL_cbuf: 0xfff8000000000000, 0x4c48000000000000, false;
    SHL_reg: 0xfff8000000000000, 0x5c48000000000000, false;
    SHR_cbuf: 0xfff8000000000000, 0x4c28000000000000, false;
    SHR_reg: 0xfff8000000000000, 0x5c28000000000000, false;
    STG: 0xfff8000000000000, 0xeed8000000000000, false;
    STL: 0xfff8000000000000, 0xef50000000000000, false;
    STP: 0xfff8000000000000, 0xeea0000000000000, true;
    STS: 0xfff8000000000000, 0xef58000000000000, false;
    SYNC: 0xfff8000000000000, 0xf0f8000000000000, false;
    TMML: 0xfff8000000000000, 0xdf58000000000000, false;
    TMML_b: 0xfff8000000000000, 0xdf60000000000000, false;
    TXA: 0xfff8000000000000, 0xdf40000000000000, true;
    TXQ: 0xfff8000000000000, 0xdf48000000000000, false;
    TXQ_b: 0xfff8000000000000, 0xdf50000000000000, false;
    VOTE: 0xfff8000000000000, 0x50d8000000000000, false;
    VOTE_vtg: 0xfff8000000000000, 0x50e0000000000000, false;
    VSETP: 0xfff8000000000000, 0x50f0000000000000, false;
    ATOM_cas: 0xfff0000000000000, 0xeef0000000000000, true;
    BFE_imm: 0xfef8000000000000, 0x3800000000000000, false;
    BFI_imm: 0xfef8000000000000, 0x36f0000000000000, false;
    BPT: 0xfff0000000000000, 0xe3a0000000000000, true;
    BRA: 0xfff0000000000000, 0xe240000000000000, false;
    BRK: 0xfff0000000000000, 0xe340000000000000, false;
    BRX: 0xfff0000000000000, 0xe250000000000000, false;
    CAL: 0xfff0000000000000, 0xe260000000000000, false;
    CONT: 0xfff0000000000000, 0xe350000000000000, false;
    DADD_imm: 0xfef8000000000000, 0x3870000000000000, false;
    DFMA_cr: 0xfff0000000000000, 0x4b70000000000000, false;
    DFMA_rc: 0xfff0000000000000, 0x5370000000000000, false;
    DFMA_reg: 0xfff0000000000000, 0x5b70000000000000, false;
    DMNMX_imm: 0xfef8000000000000, 0x3850000000000000, false;
    DMUL_imm: 0xfef8000000000000, 0x3880000000000000, false;
    DSETP_cbuf: 0xfff0000000000000, 0x4b80000000000000, false;
    DSETP_reg: 0xfff0000000000000, 0x5b80000000000000, false;
    EXIT: 0xfff0000000000000, 0xe300000000000000, false;
    F2F_imm: 0xfef8000000000000, 0x38a8000000000000, false;
    F2I_imm: 0xfef8000000000000, 0x38b0000000000000, true;
    FADD_imm: 0xfef8000000000000, 0x3858000000000000, false;
    FCHK_imm: 0xfef8000000000000, 0x3888000000000000, true;
    FCMP_cr: 0xfff0000000000000, 0x4ba0000000000000, false;
    FCMP_rc: 0xfff0000000000000, 0x53a0000000000000, false;
    FCMP_reg: 0xfff0000000000000, 0x5ba0000000000000, false;
    FLO_imm: 0xfef8000000000000, 0x3830000000000000, false;
    FMNMX_imm: 0xfef8000000000000, 0x3860000000000000, false;
    FMUL_imm: 0xfef8000000000000, 0x3868000000000000, false;
    FSETP_cbuf: 0xfff0000000000000, 0x4bb0000000000000, false;
    FSETP_reg: 0xfff0000000000000, 0x5bb0000000000000, false;
    GETCRSPTR: 0xfff0000000000000, 0xe2c0000000000000, true;
    GETLMEMBASE: 0xfff0000000000000, 0xe2d0000000000000, true;
    I2F_imm: 0xfef8000000000000, 0x38b8000000000000, false;
    I2I_imm: 0xfef8000000000000, 0x38e0000000000000, false;
    IADD3_cbuf: 0xfff0000000000000, 0x4cc0000000000000, false;
    IADD3_reg: 0xfff0000000000000, 0x5cc0000000000000, false;
    IADD_imm: 0xfef8000000000000, 0x3810000000000000, false;
    ICMP_cr: 0xfff0000000000000, 0x4b40000000000000, false;
    ICMP_rc: 0xfff0000000000000, 0x5340000000000000, false;
    ICMP_reg: 0xfff0000000000000, 0x5b40000000000000, false;
    IDE: 0xfff0000000000000, 0xe390000000000000, true;
    IMNMX_imm: 0xfef8000000000000, 0x3820000000000000, false;
    IMUL_imm: 0xfef8000000000000, 0x3838000000000000, true;
    ISCADD_imm: 0xfef8000000000000, 0x3818000000000000, false;
    ISETP_cbuf: 0xfff0000000000000, 0x4b60000000000000, false;
    ISETP_reg: 0xfff0000000000000, 0x5b60000000000000, false;
    ISET_cbuf: 0xfff0000000000000, 0x4b50000000000000, false;
    ISET_reg: 0xfff0000000000000, 0x5b50000000000000, false;
    JCAL: 0xfff0000000000000, 0xe220000000000000, false;
    JMP: 0xfff0000000000000, 0xe210000000000000, false;
    JMX: 0xfff0000000000000, 0xe200000000000000, false;
    KIL: 0xfff0000000000000, 0xe330000000000000, false;
    LEA_lo_cbuf: 0xfff0000000000000, 0x4bd0000000000000, false;
    LEA_lo_imm: 0xfef8000000000000, 0x36d0000000000000, false;
    LONGJMP: 0xfff0000000000000, 0xe310000000000000, false;
    LOP_imm: 0xfef8000000000000, 0x3840000000000000, false;
    MOV32I: 0xfff0000000000000, 0x0100000000000000, false;
    MOV_imm: 0xfef8000000000000, 0x3898000000000000, false;
    OUT_imm: 0xfef8000000000000, 0xf6e0000000000000, false;
    PBK: 0xfff0000000000000, 0xe2a0000000000000, false;
    PCNT: 0xfff0000000000000, 0xe2b0000000000000, false;
    PEXIT: 0xfff0000000000000, 0xe230000000000000, false;
    PLONGJMP: 0xfff0000000000000, 0xe280000000000000, false;
    POPC_imm: 0xfef8000000000000, 0x3808000000000000, false;
    PRET: 0xfff0000000000000, 0xe270000000000000, false;
    PRMT_cr: 0xfff0000000000000, 0x4bc0000000000000, true;
    PRMT_rc: 0xfff0000000000000, 0x53c0000000000000, true;
    PRMT_reg: 0xfff0000000000000, 0x5bc0000000000000, true;
    R2P_imm: 0xfef8000000000000, 0x38f0000000000000, false;
    RAM: 0xfff0000000000000, 0xe380000000000000, false;
    RET: 0xfff0000000000000, 0xe320000000000000, false;
    RRO_imm: 0xfef8000000000000, 0x3890000000000000, true;
    RTT: 0xfff0000000000000, 0xe360000000000000, true;
    SAM: 0xfff0000000000000, 0xe370000000000000, false;
    SEL_imm: 0xfef8000000000000, 0x38a0000000000000, false;
    SETCRSPTR: 0xfff0000000000000, 0xe2e0000000000000, true;
    SETLMEMBASE: 0xfff0000000000000, 0xe2f0000000000000, true;
    SHF_l_imm: 0xfef8000000000000, 0x36f8000000000000, false;
    SHF_r_imm: 0xfef8000000000000, 0x38f8000000000000, false;
    SHL_imm: 0xfef8000000000000, 0x3848000000000000, false;
    SHR_imm: 0xfef8000000000000, 0x3828000000000000, false;
    SSY: 0xfff0000000000000, 0xe290000000000000, false;
    CCTL: 0xffe0000000000000, 0xef60000000000000, true;
    CCTLL: 0xffe0000000000000, 0xef80000000000000, true;
    DFMA_imm: 0xfef0000000000000, 0x3670000000000000, false;
    DSETP_imm: 0xfef0000000000000, 0x3680000000000000, false;
    FCMP_imm: 0xfef0000000000000, 0x36a0000000000000, false;
    FSETP_imm: 0xfef0000000000000, 0x36b0000000000000, false;
    IADD3_imm: 0xfef0000000000000, 0x38c0000000000000, false;
    ICMP_imm: 0xfef0000000000000, 0x3640000000000000, false;
    ISETP_imm: 0xfef0000000000000, 0x3660000000000000, false;
    ISET_imm: 0xfef0000000000000, 0x3650000000000000, false;
    PRMT_imm: 0xfef0000000000000, 0x36c0000000000000, true;
    SULD: 0xffe0000000000000, 0xeb00000000000000, false;
    SURED: 0xffe0000000000000, 0xeb40000000000000, false;
    SUST: 0xffe0000000000000, 0xeb20000000000000, false;
    TEX_b: 0xffc0000000000000, 0xde80000000000000, false;
    TLD4_b: 0xffc0000000000000, 0xdec0000000000000, false;
    TXD: 0xffc0000000000000, 0xde00000000000000, false;
    TXD_b: 0xffc0000000000000, 0xde40000000000000, false;
    XMAD_reg: 0xffc0000000000000, 0x5b00000000000000, false;
    DSET_cbuf: 0xff80000000000000, 0x4900000000000000, false;
    DSET_reg: 0xff80000000000000, 0x5900000000000000, false;
    FFMA_cr: 0xff80000000000000, 0x4980000000000000, false;
    FFMA_rc: 0xff80000000000000, 0x5180000000000000, false;
    FFMA_reg: 0xff80000000000000, 0x5980000000000000, false;
    IMADSP_cr: 0xff80000000000000, 0x4a80000000000000, true;
    IMADSP_rc: 0xff80000000000000, 0x5280000000000000, true;
    IMADSP_reg: 0xff80000000000000, 0x5a80000000000000, true;
    IMAD_cr: 0xff80000000000000, 0x4a00000000000000, true;
    IMAD_rc: 0xff80000000000000, 0x5200000000000000, true;
    IMAD_reg: 0xff80000000000000, 0x5a00000000000000, true;
    SUATOM: 0xff80000000000000, 0xea00000000000000, false;
    SUATOM_cas: 0xff80000000000000, 0xea80000000000000, true;
    TLD4S: 0xff40000000000000, 0xdf00000000000000, false;
    VABSDIFF4: 0xff80000000000000, 0x5000000000000000, true;
    XMAD_imm: 0xfec0000000000000, 0x3600000000000000, false;
    XMAD_rc: 0xff80000000000000, 0x5100000000000000, false;
    ATOM: 0xff00000000000000, 0xed00000000000000, false;
    ATOMS: 0xff00000000000000, 0xec00000000000000, false;
    ATOMS_cas: 0xff00000000000000, 0xee00000000000000, true;
    DSET_imm: 0xfe80000000000000, 0x3200000000000000, false;
    FFMA_imm: 0xfe80000000000000, 0x3280000000000000, false;
    FMUL32I: 0xff00000000000000, 0x1e00000000000000, false;
    FSET_cbuf: 0xff00000000000000, 0x4800000000000000, false;
    FSET_reg: 0xff00000000000000, 0x5800000000000000, false;
    HADD2_cbuf: 0xfe80000000000000, 0x7a80000000000000, false;
    HADD2_imm: 0xfe80000000000000, 0x7a00000000000000, false;
    HMUL2_cbuf: 0xfe80000000000000, 0x7880000000000000, false;
    HMUL2_imm: 0xfe80000000000000, 0x7800000000000000, false;
    HSET2_cbuf: 0xfe80000000000000, 0x7c80000000000000, false;
    HSET2_imm: 0xfe80000000000000, 0x7c00000000000000, false;
    HSETP2_cbuf: 0xfe80000000000000, 0x7e80000000000000, false;
    HSETP2_imm: 0xfe80000000000000, 0x7e00000000000000, false;
    IMADSP_imm: 0xfe80000000000000, 0x3480000000000000, true;
    IMAD_imm: 0xfe80000000000000, 0x3400000000000000, true;
    IMUL32I: 0xff00000000000000, 0x1f00000000000000, true;
    IPA: 0xff00000000000000, 0xe000000000000000, false;
    TLD: 0xff00000000000000, 0xdc00000000000000, false;
    TLD_b: 0xff00000000000000, 0xdd00000000000000, false;
    VABSDIFF: 0xff00000000000000, 0x5400000000000000, true;
    VMAD: 0xff00000000000000, 0x5f00000000000000, false;
    VSHL: 0xff00000000000000, 0x5700000000000000, true;
    VSHR: 0xff00000000000000, 0x5600000000000000, true;
    FSET_imm: 0xfe00000000000000, 0x3000000000000000, false;
    HADD2_32I: 0xfe00000000000000, 0x2c00000000000000, false;
    HFMA2_32I: 0xfe00000000000000, 0x2800000000000000, false;
    HMUL2_32I: 0xfe00000000000000, 0x2a00000000000000, false;
    IADD32I: 0xfe00000000000000, 0x1c00000000000000, false;
    LOP3_cbuf: 0xfe00000000000000, 0x0200000000000000, false;
    VMNMX: 0xfe00000000000000, 0x3a00000000000000, false;
    VSET: 0xfe00000000000000, 0x4000000000000000, true;
    XMAD_cr: 0xfe00000000000000, 0x4e00000000000000, false;
    FADD32I: 0xfc00000000000000, 0x0800000000000000, false;
    FFMA32I: 0xfc00000000000000, 0x0c00000000000000, false;
    HFMA2_cr: 0xf880000000000000, 0x7080000000000000, false;
    HFMA2_imm: 0xf880000000000000, 0x7000000000000000, false;
    HFMA2_rc: 0xf880000000000000, 0x6080000000000000, false;
    IMAD32I: 0xfc00000000000000, 0x8000000000000000, true;
    ISCADD32I: 0xfc00000000000000, 0x1400000000000000, false;
    LEA_hi_cbuf: 0xfc00000000000000, 0x1800000000000000, false;
    LOP32I: 0xfc00000000000000, 0x0400000000000000, false;
    LOP3_imm: 0xfc00000000000000, 0x3c00000000000000, false;
    TEXS: 0xf600000000000000, 0xd000000000000000, false;
    TLD4: 0xfc00000000000000, 0xc800000000000000, false;
    TLDS: 0xf600000000000000, 0xd200000000000000, false;
    VADD: 0xfc00000000000000, 0x2000000000000000, true;
    TEX: 0xf800000000000000, 0xc000000000000000, false;
    LD: 0xe000000000000000, 0x8000000000000000, true;
    ST: 0xe000000000000000, 0xa000000000000000, true;
}

/// Bucket candidates retain decoder precedence, including overlapping patterns.
pub fn decode(word: u64) -> Result<Opcode> {
    for &index in BUCKETS[(word >> 56) as usize] {
        let encoding = &ENCODINGS[index as usize];
        if word & encoding.mask == encoding.value {
            return Ok(encoding.opcode);
        }
    }
    let mut error = CompileError::new(
        FailureCategory::UnknownInstruction,
        format!("Unknown Maxwell instruction {word:#018x}"),
    );
    error.instruction = Some(word);
    Err(error)
}

#[derive(Clone, Copy, Debug)]
pub struct DecodedInstruction {
    pub location: super::location::Location,
    pub instruction: super::instruction::Instruction,
    pub opcode: Opcode,
}

pub fn fetch<E: crate::Environment>(
    environment: &mut E,
    location: super::location::Location,
) -> Result<DecodedInstruction> {
    let stage = environment.metadata().stage;
    let address = location.address();
    let word = environment
        .read_instruction(address)
        .map_err(|error| error.at(stage, address, None))?;
    let opcode = decode(word).map_err(|error| error.at(stage, address, Some(word)))?;
    Ok(DecodedInstruction {
        location,
        instruction: super::instruction::Instruction(word),
        opcode,
    })
}

const BUCKETS: &[&[u16]; 256] = &[
    &[],
    &[166],
    &[259],
    &[259],
    &[271],
    &[271],
    &[271],
    &[271],
    &[263],
    &[263],
    &[263],
    &[263],
    &[264],
    &[264],
    &[264],
    &[264],
    &[],
    &[],
    &[],
    &[],
    &[269],
    &[269],
    &[269],
    &[269],
    &[270],
    &[270],
    &[270],
    &[270],
    &[258],
    &[258],
    &[233],
    &[246],
    &[276],
    &[276],
    &[276],
    &[276],
    &[],
    &[],
    &[],
    &[],
    &[256],
    &[256],
    &[257],
    &[257],
    &[255],
    &[255],
    &[],
    &[],
    &[254],
    &[254],
    &[231, 232],
    &[231, 232],
    &[244, 245],
    &[244, 245],
    &[112, 163, 187, 194, 195, 196, 197, 199, 200, 201, 202, 226],
    &[112, 163, 187, 194, 195, 196, 197, 199, 200, 201, 202, 226],
    &[
        74, 111, 119, 123, 124, 128, 129, 130, 131, 135, 136, 137, 142, 143, 146, 151, 152, 153,
        165, 167, 173, 178, 181, 184, 188, 189, 190, 198,
    ],
    &[
        111, 119, 123, 124, 128, 129, 130, 131, 135, 136, 137, 142, 143, 146, 151, 152, 153, 165,
        167, 173, 178, 181, 184, 188, 189, 190, 198,
    ],
    &[260],
    &[260],
    &[272],
    &[272],
    &[272],
    &[272],
    &[261],
    &[261],
    &[],
    &[],
    &[],
    &[],
    &[],
    &[],
    &[234],
    &[211, 213],
    &[216, 219],
    &[8, 120, 125, 132, 138, 147, 154, 156, 162, 175],
    &[
        6, 14, 17, 19, 21, 23, 25, 27, 29, 31, 33, 41, 43, 45, 49, 51, 54, 64, 67, 73, 77, 82, 85,
        88, 93, 95, 144,
    ],
    &[],
    &[262],
    &[262],
    &[11, 12, 13, 35, 62, 69, 70, 79, 80, 107, 108, 109, 225],
    &[214, 227],
    &[217, 220],
    &[9, 47, 48, 121, 133, 148, 176],
    &[250],
    &[],
    &[253],
    &[252],
    &[235],
    &[212, 215],
    &[218, 221],
    &[
        10, 60, 61, 63, 91, 122, 126, 134, 139, 149, 155, 157, 177, 210,
    ],
    &[
        7, 15, 18, 20, 22, 24, 26, 28, 30, 32, 34, 42, 44, 46, 50, 52, 55, 65, 68, 75, 78, 83, 86,
        89, 92, 94, 96, 145,
    ],
    &[36, 37, 38, 39, 40],
    &[],
    &[251],
    &[267],
    &[267],
    &[267],
    &[267],
    &[267],
    &[267],
    &[267],
    &[267],
    &[],
    &[],
    &[],
    &[],
    &[],
    &[],
    &[],
    &[],
    &[265, 266],
    &[265, 266],
    &[265, 266],
    &[265, 266],
    &[265, 266],
    &[265, 266],
    &[265, 266],
    &[265, 266],
    &[238, 239],
    &[238, 239],
    &[236, 237],
    &[236, 237],
    &[240, 241],
    &[240, 241],
    &[242, 243],
    &[242, 243],
    &[268, 278],
    &[268, 278],
    &[268, 278],
    &[268, 278],
    &[278],
    &[278],
    &[278],
    &[278],
    &[278],
    &[278],
    &[278],
    &[278],
    &[278],
    &[278],
    &[278],
    &[278],
    &[278],
    &[278],
    &[278],
    &[278],
    &[278],
    &[278],
    &[278],
    &[278],
    &[278],
    &[278],
    &[278],
    &[278],
    &[278],
    &[278],
    &[278],
    &[278],
    &[279],
    &[279],
    &[279],
    &[279],
    &[279],
    &[279],
    &[279],
    &[279],
    &[279],
    &[279],
    &[279],
    &[279],
    &[279],
    &[279],
    &[279],
    &[279],
    &[279],
    &[279],
    &[279],
    &[279],
    &[279],
    &[279],
    &[279],
    &[279],
    &[279],
    &[279],
    &[279],
    &[279],
    &[279],
    &[279],
    &[279],
    &[279],
    &[277],
    &[277],
    &[277],
    &[277],
    &[277],
    &[277],
    &[277],
    &[277],
    &[274],
    &[274],
    &[274],
    &[274],
    &[],
    &[],
    &[],
    &[],
    &[273],
    &[273],
    &[275],
    &[275],
    &[],
    &[],
    &[],
    &[],
    &[273],
    &[273],
    &[275],
    &[275],
    &[248],
    &[249],
    &[206, 207, 208, 209],
    &[102, 103, 104, 105, 106, 224],
    &[247],
    &[],
    &[
        114, 116, 117, 140, 141, 158, 159, 160, 169, 170, 171, 172, 174, 185, 186, 191,
    ],
    &[113, 115, 118, 127, 150, 161, 164, 179, 180, 182, 183],
    &[],
    &[],
    &[],
    &[],
    &[],
    &[],
    &[222, 223],
    &[0, 71, 84, 203, 204, 205],
    &[229],
    &[228],
    &[57, 97, 99, 110, 230],
    &[1, 2, 3, 53, 56, 58, 59, 66, 76, 90, 98, 100, 192, 193],
    &[4, 5, 16, 81, 87, 101],
    &[],
    &[],
    &[],
    &[],
    &[],
    &[168],
    &[168],
    &[],
    &[],
    &[],
    &[72],
    &[],
    &[],
    &[],
    &[],
];
