//! Choose a predecessor's register convention at compilation, using existing
//! target-keyed adjacency. No global rescans, execution counters or new roots.

use super::*;
use crate::abi::InstructionKey;
use crate::frontend::entry::Plan;
use crate::lifetime::State;

impl Lifetime {
    pub(crate) fn entry_plan(&self, key: BlockKey) -> Plan {
        self.lock().entry_plan(key, |_| false)
    }
}

impl State {
    pub(in crate::lifetime) fn entry_plan(
        &self,
        key: BlockKey,
        contains: impl Fn(InstructionKey) -> bool,
    ) -> Plan {
        let mut selected = Plan::default();
        let mut rank = (false, false, 0);
        let mut consider = |unit: UnitHandle, map: u32, observed: bool| {
            let code = &self.units.records.get(unit.0).unwrap().code;
            let record = &code.states[map as usize];
            let source = record
                .exit
                .unwrap()
                .source(|i| code.instructions.get(i))
                .unwrap()
                .1
                .key;
            if contains(source) {
                return;
            }
            let plan = Plan::from_exit(&record.state);
            // Executed indirect edges are evidence of demand; within that set
            // prefer optimized predecessors and then the largest avoided save.
            let candidate = (observed, code.tier == Tier::Hcq, plan.len());
            if plan.len() != 0 && candidate > rank {
                selected = plan;
                rank = candidate;
            }
        };
        for site in self.units.static_sources(key) {
            consider(site.source, site.state_map, false);
        }
        if let Some(slot) = self.keys.get(&key).and_then(|h| self.dispatch.get(*h)) {
            for &owner in slot.owners.iter().flatten() {
                for (source, map) in self.dynamic_sources(owner, key) {
                    consider(source, map, true);
                }
            }
        }
        selected
    }
}
