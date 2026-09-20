//! Compatibility re-export of the runtime-owned branch-state contract.
//!
//! New generic runtime code should import [`opendecision_runtime::branch`]
//! directly. This module remains so downstream backend users do not need a
//! flag-day import change.

pub use opendecision_runtime::branch::*;
