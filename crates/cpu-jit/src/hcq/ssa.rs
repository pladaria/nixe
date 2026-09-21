//! Real public ingress labels and shared internal SSA parameter contracts.
//! This constructs no body copies, native bridges or architectural home traffic.

use super::{Graph, flow::Analysis};
use crate::abi::{GuestValue, LazyFlags};
use crate::jit_error::Error;
use crate::lowering::{
    IntegerLowering,
    values::{Values, guest_type, register_operands},
};
use cranelift_codegen::ir::{self, AbiParam, InstBuilder, types};
use cranelift_codegen::isa::CallConv;
use cranelift_frontend::FunctionBuilder;

// Backend map IDs are unique across entries AND exits. Ordinary exits start
// at 1, faults at 1<<60 and FP continuations near i64::MAX. These are unit-local
// metadata identities, not public dispatch ordinals or a runtime selector.
const ENTRY_ID_BASE: u64 = 1 << 59;

pub(super) struct Block {
    pub label: ir::Block,
    pub flag_mask: u8,
    pub operands: Vec<GuestValue>,
    pub flags: Option<LazyFlags<ir::Value>>,
}

pub(super) struct Entry {
    pub label: ir::Block,
    pub target: usize,
    pub id: u64,
}

pub(super) struct Ssa {
    pub blocks: Vec<Block>,
    pub entries: Vec<Entry>,
}

impl Ssa {
    pub(super) fn new(
        builder: &mut FunctionBuilder<'_>,
        graph: &Graph,
        analysis: &Analysis,
        entries: &[usize],
    ) -> Self {
        // Declare all parameters before emission, including loop headers and
        // public joins. Every predecessor must supply this exact ordering.
        let blocks: Vec<_> = (0..graph.blocks.len())
            .map(|index| {
                let label = builder.create_block();
                let operands = register_operands(analysis.native.blocks[index].live_in);
                for &guest in &operands {
                    builder.append_block_param(label, guest_type(guest));
                }
                let flags = analysis.flags.blocks[index].input.as_ref().map(|shape| {
                    shape
                        .try_map_with_bits(&mut |_, bits| {
                            Ok::<_, std::convert::Infallible>(builder.append_block_param(
                                label,
                                match bits {
                                    8 => types::I8,
                                    32 => types::I32,
                                    64 => types::I64,
                                    _ => unreachable!("invalid analyzed NZCV operand width"),
                                },
                            ))
                        })
                        .unwrap()
                });
                Block {
                    label,
                    flag_mask: analysis.native.blocks[index].live_in.nzcv,
                    operands,
                    flags,
                }
            })
            .collect();
        let entries = entries
            .iter()
            .enumerate()
            .map(|(id, &target)| {
                let id = ENTRY_ID_BASE + id as u64;
                let block = &blocks[target];
                let label = builder.create_block();
                builder.switch_to_block(label);
                let mut signature = ir::Signature::new(CallConv::SystemV);
                signature.returns = block
                    .operands
                    .iter()
                    .map(|&guest| AbiParam::new(guest_type(guest)))
                    .chain(block.flags.as_ref().map(|flags| {
                        assert_eq!(flags.shape(), LazyFlags::Packed(()));
                        AbiParam::new(types::I32)
                    }))
                    .collect();
                let signature = builder.import_signature(signature);
                let inst = builder.ins().nixe_entry(signature, id as i64);
                let args: Vec<_> = builder
                    .func
                    .dfg
                    .inst_results(inst)
                    .iter()
                    .copied()
                    .map(ir::BlockArg::from)
                    .collect();
                builder.ins().jump(block.label, &args);
                Entry { label, target, id }
            })
            .collect();
        Self { blocks, entries }
    }
}

impl Block {
    /// Rebind on block entry, not by inheriting the last emitted predecessor's
    /// register array. Keep only the parameter list, not a full value file per
    /// block, while the region is waiting to be emitted.
    pub(super) fn values(&self, builder: &FunctionBuilder<'_>) -> Values {
        let mut values = Values::default();
        for (&guest, &value) in self.operands.iter().zip(builder.block_params(self.label)) {
            values.bind(guest, value);
        }
        values
    }

    /// Reconcile only when this destination needs packed bits. Arithmetic
    /// recipes retain their captured operands; no memory/bridge is involved.
    /// The native flow's dirty mask, not flags.dirty(), governs HCQ observations.
    pub(super) fn reconcile<'a>(
        &self,
        lowerer: &mut impl IntegerLowering<'a>,
        flags: Option<&LazyFlags<ir::Value>>,
    ) -> Result<Option<LazyFlags<ir::Value>>, Error> {
        let Some(required) = &self.flags else {
            return Ok(None);
        };
        let flags = flags.ok_or_else(|| Error::internal("missing HCQ edge NZCV operands"))?;
        if required.shape() == flags.shape() {
            return Ok(Some(flags.clone()));
        }
        if matches!(required, LazyFlags::Packed(_)) {
            return Ok(Some(LazyFlags::Packed(
                lowerer.packed_flag_subset(flags, self.flag_mask),
            )));
        }
        Err(Error::internal("incompatible HCQ lazy-NZCV edge shape"))
    }

    /// Call after reconcile. Missing SSA values are a compile error, never a
    /// reason to load possibly stale canonical storage on an internal edge.
    pub(super) fn arguments(
        &self,
        values: &Values,
        flags: Option<&LazyFlags<ir::Value>>,
    ) -> Result<Vec<ir::BlockArg>, Error> {
        let mut args = self
            .operands
            .iter()
            .map(|&guest| values.get(guest).map(ir::BlockArg::from))
            .collect::<Result<Vec<_>, _>>()?;
        if let Some(required) = &self.flags {
            let flags = flags.ok_or_else(|| Error::internal("missing HCQ edge NZCV operands"))?;
            if required.shape() != flags.shape() {
                return Err(Error::internal("unreconciled HCQ lazy-NZCV edge"));
            }
            flags
                .try_map(&mut |value| {
                    args.push((*value).into());
                    Ok::<_, std::convert::Infallible>(())
                })
                .unwrap();
        }
        Ok(args)
    }
}

#[cfg(test)]
pub(super) mod tests;
