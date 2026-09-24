//! Canonical/fast entry contracts from final backend allocation, shared by tiers.

use super::*;
use crate::abi::EntryContract;
use cranelift_codegen::nixe::Location;

pub(crate) fn contract(
    abi: HostAbi,
    allocated: &AllocatedBoundary<'_>,
    inputs: &[GuestValue],
    flags: Option<u8>,
) -> Result<EntryContract, Error> {
    let map = allocated.map;
    if !map.entry || map.values.len() != inputs.len() + usize::from(flags.is_some()) {
        return Err(Error::internal("native entry operand contract mismatch"));
    }
    let operands: Vec<_> = inputs
        .iter()
        .enumerate()
        .filter(|(index, _)| map.values[*index].location != Location::Unused)
        .map(|(index, &guest)| (guest, index))
        .collect();
    let mut live = operands
        .iter()
        .fold(StateSet::default(), |state, (guest, _)| {
            state.union(guest.state().unwrap())
        });
    let nzcv = match flags {
        Some(mask) => {
            let index = inputs.len();
            let value = &map.values[index];
            if value.location == Location::Unused {
                NzcvLocation::Canonical
            } else {
                live.nzcv = mask;
                NzcvLocation::Packed(allocated.location(index, types::I32).map_err(fail)?)
            }
        }
        None => NzcvLocation::Canonical,
    };
    let entry = EntryContract {
        abi,
        live_in: live,
        bindings: allocated.bindings(&operands).map_err(fail)?,
        nzcv,
    };
    entry.validate().map_err(fail)?;
    Ok(entry)
}
