//! Complete reference-interpreter engine for the neutral execution protocol.
//!
//! This crate owns instruction interpretation, the bounded execution loop,
//! interpreter-local dispatch resources, and its semantic/differential tests.
//! `nixe-cpu` supplies only reusable architectural frontend primitives.

mod engine;
mod interpreter;

pub use engine::{
    INTERPRETER_ENGINE_ID, InterpreterDomain, InterpreterExecutor, InterpreterProvider,
};
pub use interpreter::{
    InterpreterContext, InterpreterError, InterpreterOutcome, InterpreterPolicy, execute_fallback,
    execute_fallback_with_context, execute_one, execute_one_with_context,
};
