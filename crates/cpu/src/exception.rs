//! Engine-independent architectural exception descriptions.

/// Classification of one precise architectural exception.
///
/// This type belongs to the architectural CPU contract rather than one engine:
/// the reference interpreter and native providers report the same values to the
/// runtime.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ExceptionKind {
    SupervisorCall,
    Breakpoint,
    UndefinedInstruction,
    InstructionAbort,
    DataAbort,
    AlignmentFault,
    FloatingPoint,
    SystemRegisterTrap,
}
