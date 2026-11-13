//! Channel ownership is independent of descriptor and CMIF-session lifetimes.
use std::{
    collections::BTreeSet,
    sync::{Arc, Mutex},
};

#[derive(Debug, Default)]
pub(crate) struct Channel {
    pub closed: bool,
    pub runlist: Option<u32>,
    pub space: Option<u32>,
    pub group: Option<(Arc<Mutex<super::Tsg>>, u32)>,
    pub entries: Option<u32>,
    pub inflight_jobs: u32,
    pub deterministic: bool,
    pub timeline: Option<Arc<Mutex<crate::gsv::syncpt::Counter>>>,
    pub objects: BTreeSet<u32>,
    pub client_data: u64,
    pub timeout_ms: u32,
    pub timeslice_us: u32,
    pub priority: u32,
    pub error_notifier: bool,
    pub preemption: [u32; 2],
    pub zcull: Option<(u64, u32)>,
}
