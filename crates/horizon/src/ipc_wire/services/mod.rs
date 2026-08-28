//! Built-in Horizon service dispatchers, grouped by service family.

use crate::IpcService;

pub(in crate::ipc_wire) const fn semantic_service_name(service: IpcService) -> &'static str {
    match service {
        IpcService::FileSystem => "fsp-srv",
        IpcService::AddOnContent => "aoc:u",
    }
}

mod account;
mod aoc;
mod apm;
mod applet;
pub(super) mod content;
pub(super) mod fsp;
mod hid;
mod lm;
mod nvdrv;
mod pctl;
mod prelude;
pub(super) mod response;
mod settings;
mod sm;
mod time;
mod vi;

pub(super) use account::dispatch_account;
pub(super) use apm::{dispatch_performance_manager, dispatch_performance_session};
pub(super) use applet::{applet_object_name, applet_requests_self_exit, dispatch_applet};
pub(super) use hid::{dispatch_hid, dispatch_hid_applet_resource};
pub(super) use lm::{dispatch_log_manager, dispatch_logger};
pub(super) use nvdrv::dispatch_nvdrv;
pub(super) use pctl::{dispatch_parental_control, dispatch_parental_control_service};
pub(super) use settings::{dispatch_system_settings, dispatch_user_settings};
pub(super) use sm::dispatch_service_manager;
pub(super) use time::{
    dispatch_steady_clock, dispatch_system_clock, dispatch_time, dispatch_timezone,
};
pub(super) use vi::{dispatch_vi, vi_object_name};
