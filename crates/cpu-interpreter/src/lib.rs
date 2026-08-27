//! Direct reference-interpreter backend.
//!
//! This crate owns instruction interpretation, the bounded execution loop,
//! interpreter-local dispatch resources, and its semantic/differential tests.
//! `nixe-cpu` supplies only reusable architectural frontend primitives.

mod interpreter;
mod process;

pub use interpreter::{
    InstructionStep, InterpreterContext, InterpreterError, execute_one, execute_one_with_context,
};
pub use process::{InterpreterProcess, InterpreterRunRequest, InterpreterThread};
