use super::{Result, Value, Visitor};
use crate::{ir::Type, maxwell::special_register::SpecialRegister as S};

impl Visitor<'_> {
    pub(super) fn s2r(&mut self) -> Result<()> {
        let value = match S::try_from(self.field::<20, 8>() as u8) {
            Ok(S::SR_INVOCATION_ID) => self.emit.invocation_id()?,
            Ok(S::SR_THREAD_KILL) => {
                let helper = self.emit.is_helper_invocation()?;
                self.emit
                    .select(helper, Value::U32(u32::MAX), Value::U32(0))?
            }
            Ok(S::SR_INVOCATION_INFO) => self.emit.invocation_info()?,
            Ok(S::SR_TID) => {
                let tid = self.emit.local_invocation_id()?;
                let x = self.emit.composite_extract(tid, 0)?;
                let y = self.emit.composite_extract(tid, 1)?;
                let z = self.emit.composite_extract(tid, 2)?;
                let xy = self
                    .emit
                    .bit_field_insert(x, y, Value::U32(16), Value::U32(8))?;
                self.emit
                    .bit_field_insert(xy, z, Value::U32(26), Value::U32(6))?
            }
            Ok(S::SR_TID_X) => self.emit.local_invocation_id_x()?,
            Ok(S::SR_TID_Y) => self.emit.local_invocation_id_y()?,
            Ok(S::SR_TID_Z) => self.emit.local_invocation_id_z()?,
            Ok(S::SR_CTAID_X) => self.emit.workgroup_id_x()?,
            Ok(S::SR_CTAID_Y) => self.emit.workgroup_id_y()?,
            Ok(S::SR_CTAID_Z) => self.emit.workgroup_id_z()?,
            Ok(S::SR_WSCALEFACTOR_XY) => self.emit.sr_w_scale_factor_xy()?,
            Ok(S::SR_WSCALEFACTOR_Z) => self.emit.sr_w_scale_factor_z()?,
            Ok(S::SR_LANEID) => self.emit.lane_id()?,
            Ok(S::SR_EQMASK) => self.emit.subgroup_eq_mask()?,
            Ok(S::SR_LTMASK) => self.emit.subgroup_lt_mask()?,
            Ok(S::SR_LEMASK) => self.emit.subgroup_le_mask()?,
            Ok(S::SR_GTMASK) => self.emit.subgroup_gt_mask()?,
            Ok(S::SR_GEMASK) => self.emit.subgroup_ge_mask()?,
            Ok(S::SR_Y_DIRECTION) => {
                let direction = self.emit.y_direction()?;
                self.emit.bit_cast(Type::U32, direction)?
            }
            // Affinity and unhandled selectors have an explicit zero-valued stub.
            _ => Value::U32(0),
        };
        self.write(value)
    }
}
