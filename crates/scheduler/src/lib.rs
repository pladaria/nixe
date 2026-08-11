//! Console- and engine-independent scheduler ownership boundary.
//!
//! Phase A intentionally places no scheduling behavior here. Later phases add
//! typed topology and policy without importing CPU engines, Horizon, or graphics.
