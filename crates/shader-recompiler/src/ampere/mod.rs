//! Native 128-bit shader decoding into the shared semantic IR.
//! Encoding reference: https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/shader_recompiler/frontend/ampere/lower_to_maxwell.h
pub mod flow;
pub mod instruction;
pub mod translate;
