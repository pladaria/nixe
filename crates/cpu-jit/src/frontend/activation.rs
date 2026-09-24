//! A native FP activation/continuation pair using the fork's existing boundary
//! contracts. No call ABI, canonical checkpoint, or guessed allocator state.

use super::*;
use cranelift_codegen::nixe::Location;

const ID_BASE: i64 = i64::MAX;

pub(crate) struct Pending {
    pub entry: ir::Block,
    pub pc: GuestVirtualAddress,
    pub(crate) id: u64,
    dirty: StateSet,
    operands: Vec<(GuestValue, usize)>,
    flags: Option<LazyFlags<(usize, ir::Type)>>,
}

impl Translator<'_> {
    /// Cold guards exit before this point. Only this successful native path
    /// acquires the proof; HCQ resets it at each block from definite CFG flow.
    pub(crate) fn ensure_fp(&mut self, pc: GuestVirtualAddress, flags: &mut LazyFlags<ir::Value>) {
        if self.fp_active {
            return;
        }
        // Disjoint pairs, separate from ordinary exits, public entries and
        // faults. The bounded instruction count keeps this range near i64::MAX.
        let id = ID_BASE - 2 * self.fp_activations.len() as i64;
        let mut values = Vec::new();
        let mut push = |value| {
            let index = values.len();
            values.push(value);
            index
        };
        let registers = self.values.registers.map(|value| value.map(&mut push));
        let vectors = self.values.vectors.map(|value| value.map(&mut push));
        let system = self.values.system.map(|value| value.map(&mut push));
        let recipe = flags
            .try_map(&mut |value| Ok::<_, std::convert::Infallible>(push(*value)))
            .unwrap();
        let mut dirty = self.dirty;
        // Activation may be a no-op for a segment inherited from another unit.
        dirty.fpsr = true;
        let source_flags = (dirty.nzcv != 0).then(|| {
            recipe
                .try_map(&mut |index| {
                    Ok::<_, std::convert::Infallible>((
                        *index,
                        self.builder.func.dfg.value_type(values[*index]),
                    ))
                })
                .unwrap()
        });
        let operands = register_operands(dirty)
            .into_iter()
            .map(|guest| {
                let index = match guest {
                    GuestValue::Vector(index) => vectors[usize::from(index)],
                    GuestValue::Fpcr | GuestValue::TpidrEl0 | GuestValue::TpidrroEl0 => {
                        system[system_index(guest)]
                    }
                    _ => registers[register_index(guest)],
                }
                .expect("dirty guest value has SSA storage");
                (guest, index)
            })
            .collect();
        let original_types: Vec<_> = values
            .iter()
            .map(|&v| self.builder.func.dfg.value_type(v))
            .collect();
        // Transfers use 32/64-bit scalars and vectors. Extend small recipe
        // operands explicitly; never read beyond an allocated byte-sized spill.
        for (value, ty) in values.iter_mut().zip(&original_types) {
            if ty.bytes() < 4 {
                *value = self.builder.ins().uextend(types::I32, *value);
            }
        }
        let mut signature = ir::Signature::new(CallConv::SystemV);
        signature.returns = values
            .iter()
            .map(|&value| AbiParam::new(self.builder.func.dfg.value_type(value)))
            .collect();
        self.builder.ins().nixe_exit(id, &values);
        let continuation = self.builder.create_block();
        self.builder.switch_to_block(continuation);
        let signature = self.builder.import_signature(signature);
        let inst = self.builder.ins().nixe_entry(signature, id - 1);
        let mut incoming = self.builder.func.dfg.inst_results(inst).to_vec();
        for (value, ty) in incoming.iter_mut().zip(original_types) {
            if ty.bytes() < 4 {
                *value = self.builder.ins().ireduce(ty, *value);
            }
        }
        self.values.registers = registers.map(|index| index.map(|i| incoming[i]));
        self.values.vectors = vectors.map(|index| index.map(|i| incoming[i]));
        self.values.system = system.map(|index| index.map(|i| incoming[i]));
        *flags = recipe
            .try_map(&mut |index| Ok::<_, std::convert::Infallible>(incoming[*index]))
            .unwrap();
        self.fp_active = true;
        self.fp_activations.push(Pending {
            entry: continuation,
            pc,
            id: id as u64,
            dirty,
            operands,
            flags: source_flags,
        });
    }
}

impl Pending {
    pub(crate) fn adapter(
        &self,
        abi: HostAbi,
        code: &cranelift_codegen::CompiledCode,
        site: ExitSiteKey,
    ) -> Result<(StateMap, Vec<u8>, u32, StateRecord), Error> {
        let find = |entry| {
            code.buffer
                .nixe_states
                .iter()
                .find(|map| map.entry == entry && map.id == self.id - u64::from(entry))
                .ok_or_else(|| {
                    Error::internal(format!(
                        "missing allocated FP activation boundary at guest PC {:#x}",
                        self.pc.get()
                    ))
                })
        };
        let source = AllocatedBoundary::new(abi, code, find(false)?).map_err(fail)?;
        let target = AllocatedBoundary::new(abi, code, find(true)?).map_err(fail)?;
        let mut copies = Vec::new();
        for (index, value) in target.map.values.iter().enumerate() {
            if value.location == Location::Unused {
                continue;
            }
            copies.push((
                source.location(index, value.ty).map_err(fail)?,
                target.location(index, value.ty).map_err(fail)?,
                value.ty.bytes() as u8,
            ));
        }
        let mut bytes = crate::native::emit_fp_activation(abi);
        bytes.extend(crate::native::emit_operand_transfer(abi, &copies).map_err(fail)?);
        let nzcv = match &self.flags {
            Some(flags) => NzcvLocation::Deferred(
                flags
                    .try_map(&mut |(index, original_type)| {
                        let location = source.location(*index, source.map.values[*index].ty)?;
                        // Transport widens small recipes to I32; their semantic
                        // locations retain the original byte/halfword extent.
                        Ok::<_, crate::native::TransferError>(match location {
                            crate::abi::ValueLocation::Spill { offset, .. } => {
                                crate::abi::ValueLocation::Spill {
                                    offset,
                                    bytes: original_type.bytes() as u8,
                                }
                            }
                            location => location,
                        })
                    })
                    .map_err(fail)?,
            ),
            None => NzcvLocation::Canonical,
        };
        let state = StateRecord {
            native_offset: source.map.offset,
            state: ExitStateMap {
                abi,
                site,
                live: self.dirty,
                dirty_live: self.dirty,
                bindings: source.bindings(&self.operands).map_err(fail)?,
                nzcv,
                host_fpsr_pending: true,
            },
            exit: None,
            transfer: None,
        };
        state.state.validate().map_err(fail)?;
        Ok((source.map.clone(), bytes, target.map.offset, state))
    }
}
