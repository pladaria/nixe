//! Engine-independent architectural exception descriptions.

/// Classification of one precise architectural exception.
///
/// Native execution reports these architectural values to the runtime.
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
