use super::*;
use crate::{CompileError, Result, diagnostics::FailureCategory};

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum FlowTest {
    #[default]
    F = 0,
    LT = 1,
    EQ = 2,
    LE = 3,
    GT = 4,
    NE = 5,
    GE = 6,
    NUM = 7,
    NaN = 8,
    LTU = 9,
    EQU = 10,
    LEU = 11,
    GTU = 12,
    NEU = 13,
    GEU = 14,
    T = 15,
    OFF = 16,
    LO = 17,
    SFF = 18,
    LS = 19,
    HI = 20,
    SFT = 21,
    HS = 22,
    OFT = 23,
    CSM_TA = 24,
    CSM_TR = 25,
    CSM_MX = 26,
    FCSM_TA = 27,
    FCSM_TR = 28,
    FCSM_MX = 29,
    RLE = 30,
    RGT = 31,
}

impl TryFrom<u8> for FlowTest {
    type Error = CompileError;
    fn try_from(value: u8) -> Result<Self> {
        match value {
            0 => Ok(Self::F),
            1 => Ok(Self::LT),
            2 => Ok(Self::EQ),
            3 => Ok(Self::LE),
            4 => Ok(Self::GT),
            5 => Ok(Self::NE),
            6 => Ok(Self::GE),
            7 => Ok(Self::NUM),
            8 => Ok(Self::NaN),
            9 => Ok(Self::LTU),
            10 => Ok(Self::EQU),
            11 => Ok(Self::LEU),
            12 => Ok(Self::GTU),
            13 => Ok(Self::NEU),
            14 => Ok(Self::GEU),
            15 => Ok(Self::T),
            16 => Ok(Self::OFF),
            17 => Ok(Self::LO),
            18 => Ok(Self::SFF),
            19 => Ok(Self::LS),
            20 => Ok(Self::HI),
            21 => Ok(Self::SFT),
            22 => Ok(Self::HS),
            23 => Ok(Self::OFT),
            24 => Ok(Self::CSM_TA),
            25 => Ok(Self::CSM_TR),
            26 => Ok(Self::CSM_MX),
            27 => Ok(Self::FCSM_TA),
            28 => Ok(Self::FCSM_TR),
            29 => Ok(Self::FCSM_MX),
            30 => Ok(Self::RLE),
            31 => Ok(Self::RGT),
            _ => Err(CompileError::new(
                FailureCategory::InvalidArgument,
                "Invalid flow test",
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Condition {
    pub flow_test: FlowTest,
    pub predicate: Pred,
    pub negated: bool,
}

impl Condition {
    pub const fn always(value: bool) -> Self {
        Self {
            flow_test: FlowTest::T,
            predicate: Pred::PT,
            negated: !value,
        }
    }
}

impl Emitter<'_> {
    pub fn condition(&mut self, condition: Condition) -> Result<Value> {
        let pred = self.get_pred(condition.predicate, condition.negated)?;
        if condition.flow_test == FlowTest::T {
            return Ok(pred);
        }
        let test = self.get_flow_test_result(condition.flow_test)?;
        self.logical_and(pred, test)
    }

    pub fn get_flow_test_result(&mut self, test: FlowTest) -> Result<Value> {
        match test {
            FlowTest::F => Ok(Value::U1(false)),
            FlowTest::LT => {
                let v0 = self.get_s_flag()?;
                let v1 = self.get_z_flag()?;
                let v2 = self.logical_not(v1)?;
                let v3 = self.logical_and(v0, v2)?;
                let v4 = self.get_o_flag()?;
                let v5 = self.logical_xor(v3, v4)?;
                Ok(v5)
            }
            FlowTest::EQ => {
                let v0 = self.get_s_flag()?;
                let v1 = self.logical_not(v0)?;
                let v2 = self.get_z_flag()?;
                let v3 = self.logical_and(v1, v2)?;
                Ok(v3)
            }
            FlowTest::LE => {
                let v0 = self.get_s_flag()?;
                let v1 = self.get_z_flag()?;
                let v2 = self.get_o_flag()?;
                let v3 = self.logical_or(v1, v2)?;
                let v4 = self.logical_xor(v0, v3)?;
                Ok(v4)
            }
            FlowTest::GT => {
                let v0 = self.get_s_flag()?;
                let v1 = self.logical_not(v0)?;
                let v2 = self.get_o_flag()?;
                let v3 = self.logical_xor(v1, v2)?;
                let v4 = self.get_z_flag()?;
                let v5 = self.logical_not(v4)?;
                let v6 = self.logical_and(v3, v5)?;
                Ok(v6)
            }
            FlowTest::NE => {
                let v0 = self.get_z_flag()?;
                let v1 = self.logical_not(v0)?;
                Ok(v1)
            }
            FlowTest::GE => {
                let v0 = self.get_s_flag()?;
                let v1 = self.get_o_flag()?;
                let v2 = self.logical_xor(v0, v1)?;
                let v3 = self.logical_not(v2)?;
                Ok(v3)
            }
            FlowTest::NUM => {
                let v0 = self.get_s_flag()?;
                let v1 = self.logical_not(v0)?;
                let v2 = self.get_z_flag()?;
                let v3 = self.logical_not(v2)?;
                let v4 = self.logical_or(v1, v3)?;
                Ok(v4)
            }
            FlowTest::NaN => {
                let v0 = self.get_s_flag()?;
                let v1 = self.get_z_flag()?;
                let v2 = self.logical_and(v0, v1)?;
                Ok(v2)
            }
            FlowTest::LTU => {
                let v0 = self.get_s_flag()?;
                let v1 = self.get_o_flag()?;
                let v2 = self.logical_xor(v0, v1)?;
                Ok(v2)
            }
            FlowTest::EQU => {
                let v0 = self.get_z_flag()?;
                Ok(v0)
            }
            FlowTest::LEU => {
                let v0 = self.get_s_flag()?;
                let v1 = self.get_o_flag()?;
                let v2 = self.logical_xor(v0, v1)?;
                let v3 = self.get_z_flag()?;
                let v4 = self.logical_or(v2, v3)?;
                Ok(v4)
            }
            FlowTest::GTU => {
                let v0 = self.get_s_flag()?;
                let v1 = self.logical_not(v0)?;
                let v2 = self.get_z_flag()?;
                let v3 = self.get_o_flag()?;
                let v4 = self.logical_or(v2, v3)?;
                let v5 = self.logical_xor(v1, v4)?;
                Ok(v5)
            }
            FlowTest::NEU => {
                let v0 = self.get_s_flag()?;
                let v1 = self.get_z_flag()?;
                let v2 = self.logical_not(v1)?;
                let v3 = self.logical_or(v0, v2)?;
                Ok(v3)
            }
            FlowTest::GEU => {
                let v0 = self.get_s_flag()?;
                let v1 = self.logical_not(v0)?;
                let v2 = self.get_z_flag()?;
                let v3 = self.logical_or(v1, v2)?;
                let v4 = self.get_o_flag()?;
                let v5 = self.logical_xor(v3, v4)?;
                Ok(v5)
            }
            FlowTest::T => Ok(Value::U1(true)),
            FlowTest::OFF => {
                let v0 = self.get_o_flag()?;
                let v1 = self.logical_not(v0)?;
                Ok(v1)
            }
            FlowTest::LO => {
                let v0 = self.get_c_flag()?;
                let v1 = self.logical_not(v0)?;
                Ok(v1)
            }
            FlowTest::SFF => {
                let v0 = self.get_s_flag()?;
                let v1 = self.logical_not(v0)?;
                Ok(v1)
            }
            FlowTest::LS => {
                let v0 = self.get_z_flag()?;
                let v1 = self.get_c_flag()?;
                let v2 = self.logical_not(v1)?;
                let v3 = self.logical_or(v0, v2)?;
                Ok(v3)
            }
            FlowTest::HI => {
                let v0 = self.get_c_flag()?;
                let v1 = self.get_z_flag()?;
                let v2 = self.logical_not(v1)?;
                let v3 = self.logical_and(v0, v2)?;
                Ok(v3)
            }
            FlowTest::SFT => {
                let v0 = self.get_s_flag()?;
                Ok(v0)
            }
            FlowTest::HS => {
                let v0 = self.get_c_flag()?;
                Ok(v0)
            }
            FlowTest::OFT => {
                let v0 = self.get_o_flag()?;
                Ok(v0)
            }
            FlowTest::RLE => {
                let v0 = self.get_s_flag()?;
                let v1 = self.get_z_flag()?;
                let v2 = self.logical_or(v0, v1)?;
                Ok(v2)
            }
            FlowTest::RGT => {
                let v0 = self.get_s_flag()?;
                let v1 = self.logical_not(v0)?;
                let v2 = self.get_z_flag()?;
                let v3 = self.logical_not(v2)?;
                let v4 = self.logical_and(v1, v3)?;
                Ok(v4)
            }
            FlowTest::FCSM_TR => Ok(Value::U1(false)),
            _ => Err(CompileError::new(
                FailureCategory::Unsupported,
                format!("Flow test {test:?} is not implemented"),
            )),
        }
    }
}
