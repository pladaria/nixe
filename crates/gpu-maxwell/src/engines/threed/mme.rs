//! Source-preserving Macro Method Expander program storage for `MAXWELL_B`.

use std::{
    collections::BTreeMap,
    sync::{Arc, OnceLock},
};
mod direct;
use direct::DirectProgram;

use nixe_gpu::GpuMethodId;

use crate::MaxwellMethodSource;

use super::{MaxwellThreeDRegister, state::MaxwellThreeDRawRegisterFile as ShadowRegisters};

/// Number of indexed MME shadow scratch registers exposed by `MAXWELL_B`.
///
/// NVIDIA defines the family as `0x3400 + i * 4`; the aperture ends where
/// `CALL_MME_MACRO` begins at `0x3800`:
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L4156-L4159>
pub const MAXWELL_THREE_D_MME_SHADOW_SCRATCH_COUNT: usize = 256;

/// How host method writes interact with the MME register shadow.
///
/// The field and all four encodings are published by NVIDIA:
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L67-L72>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u8)]
pub enum MaxwellThreeDMmeShadowRamControl {
    MethodTrack = 0,
    MethodTrackWithFilter = 1,
    MethodPassthrough = 2,
    MethodReplay = 3,
}

/// Whether Maxwell must process mutable methods through its heavyweight path.
///
/// NVIDIA publishes this as a single boolean scheduling-control field. The
/// neutral frontend already preserves strict method order, so the value is
/// retained with provenance but does not introduce a host pipeline dependency:
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L1812-L1815>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MaxwellThreeDMutableMethodControl {
    Lightweight,
    Heavyweight,
}

impl MaxwellThreeDMutableMethodControl {
    #[must_use]
    pub const fn parse(raw: u32) -> Option<Self> {
        match raw {
            0 => Some(Self::Lightweight),
            1 => Some(Self::Heavyweight),
            _ => None,
        }
    }

    #[must_use]
    pub const fn treats_mutable_as_heavyweight(self) -> bool {
        matches!(self, Self::Heavyweight)
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self.treats_mutable_as_heavyweight() as u32
    }
}

impl MaxwellThreeDMmeShadowRamControl {
    #[must_use]
    pub const fn parse(raw: u32) -> Option<Self> {
        match raw {
            0 => Some(Self::MethodTrack),
            1 => Some(Self::MethodTrackWithFilter),
            2 => Some(Self::MethodPassthrough),
            3 => Some(Self::MethodReplay),
            _ => None,
        }
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self as u32
    }

    const fn tracks(self) -> bool {
        matches!(self, Self::MethodTrack | Self::MethodTrackWithFilter)
    }
}

/// Maximum distinct instruction words retained by the current host model.
///
/// NVIDIA publishes 32-bit pointer/data fields but no physical Maxwell RAM
/// capacity. This is therefore an explicit emulator coverage bound, not a
/// guest-visible hardware limit. Exceeding it stops with a typed host error.
pub const MAXWELL_THREE_D_MME_CAPTURED_INSTRUCTION_WORDS: usize = 4096;

/// Maximum distinct macro start-address entries retained by the host model.
pub const MAXWELL_THREE_D_MME_CAPTURED_START_ADDRESSES: usize = 256;

/// Maximum instructions retired by one macro invocation.
///
/// This is an emulator watchdog rather than a hardware limit. Deko3D's
/// bounded draw macros legitimately execute roughly eight instructions per
/// emitted vertex batch and exceed the separate 4096-method output bound in
/// retired instructions well before exhausting emitted methods.
pub const MAXWELL_THREE_D_MME_EXECUTION_INSTRUCTION_LIMIT: u32 = 65_536;

/// Maximum class methods emitted by one macro invocation.
///
/// This independently bounds frontend state/operation growth. Captured
/// deko3d geometry macros legitimately emit multiple methods for each of 3600
/// iterations, so the previous 4096-entry host bound rejected finite work.
pub const MAXWELL_THREE_D_MME_EMITTED_METHOD_LIMIT: u32 = 16_384;

const MME_REGISTER_COUNT: usize = 8;
const MME_METHOD_DWORD_MASK: u32 = 0x0fff;

/// One address in MME instruction or start-address RAM.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct MaxwellThreeDMmeRamAddress(u32);

impl MaxwellThreeDMmeRamAddress {
    #[must_use]
    pub const fn new(raw: u32) -> Self {
        Self(raw)
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }
}

/// One opaque Maxwell MME instruction word.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct MaxwellThreeDMmeInstruction(u32);

impl MaxwellThreeDMmeInstruction {
    #[must_use]
    pub const fn new(raw: u32) -> Self {
        Self(raw)
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }

    const fn operation(self) -> u8 {
        (self.0 & 7) as u8
    }
    const fn result(self) -> u8 {
        ((self.0 >> 4) & 7) as u8
    }
    const fn destination(self) -> u8 {
        ((self.0 >> 8) & 7) as u8
    }
    const fn src_a(self) -> u8 {
        ((self.0 >> 11) & 7) as u8
    }
    const fn src_b(self) -> u8 {
        ((self.0 >> 14) & 7) as u8
    }
    const fn alu(self) -> u8 {
        ((self.0 >> 17) & 31) as u8
    }
    const fn source_bit(self) -> u8 {
        ((self.0 >> 17) & 31) as u8
    }
    const fn destination_bit(self) -> u8 {
        ((self.0 >> 27) & 31) as u8
    }
    const fn immediate(self) -> i32 {
        signed_immediate(self.0)
    }
    const fn mask(self) -> u32 {
        bitfield_mask(self.0)
    }
    const fn exit(self) -> bool {
        self.0 & (1 << 7) != 0
    }
    const fn branch_nonzero(self) -> bool {
        self.0 & (1 << 4) != 0
    }
    const fn branch_annul(self) -> bool {
        self.0 & (1 << 5) != 0
    }
}

/// Which independently addressed MME RAM a load targets.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MaxwellThreeDMmeRam {
    Instruction,
    StartAddress,
}

/// Why one syntactically valid MME load exceeds current host coverage.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MaxwellThreeDMmeLoadError {
    PointerUnset,
    PointerOverflow,
    StorageLimitExceeded { limit: usize },
}

/// Why one captured MME program cannot be executed faithfully.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MaxwellThreeDMmeExecutionError {
    DataWithoutCall,
    MissingStartAddress {
        macro_index: u8,
    },
    MissingInstruction {
        address: MaxwellThreeDMmeRamAddress,
    },
    InvalidOperation {
        address: MaxwellThreeDMmeRamAddress,
        operation: u8,
    },
    InvalidAluOperation {
        address: MaxwellThreeDMmeRamAddress,
        operation: u8,
    },
    BranchInDelaySlot {
        address: MaxwellThreeDMmeRamAddress,
    },
    ProgramCounterOverflow {
        address: MaxwellThreeDMmeRamAddress,
    },
    ParameterUnavailable {
        index: usize,
    },
    UnconsumedParameters {
        consumed: usize,
        supplied: usize,
    },
    RegisterReadUnavailable {
        method_dword: u16,
    },
    RecursiveMacroCall {
        method_dword: u16,
    },
    InstructionLimitExceeded {
        limit: u32,
    },
    EmittedMethodLimitExceeded {
        limit: u32,
    },
}

/// Host services used by direct programs and ISA execution.
pub(super) trait MaxwellThreeDMmeHost {
    type Error;

    fn read_register(&self, method_dword: u16) -> Result<u32, Self::Error>;
    /// Apply a send synchronously when it has no observable effect. A false
    /// result requires finish_method to flush a draw boundary or consume the
    /// effect admitted by this send before the interpreter continues.
    fn try_emit_method(&mut self, method_dword: u16, argument: u32) -> Result<bool, Self::Error>;
    async fn finish_method(&mut self, method_dword: u16, argument: u32) -> Result<(), Self::Error>;
}

pub(super) enum MaxwellThreeDMmeRunError<E> {
    Execution(MaxwellThreeDMmeExecutionError),
    Host(E),
}

// Guest instruction addresses remain sparse 32-bit values. Page storage
// bounds allocation by captured words rather than by the highest address and
// lets the interpreter reuse a direct window across sequential instructions
// and branches within the page. The distinct-word coverage limit is unchanged.
const MME_INSTRUCTION_PAGE_WORDS: usize = 64;
// Keep one canonical hardware word per uploaded instruction, like Eden's
// Macro::Opcode. Read only the fields used by the consuming operation.
// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/macro.h
#[derive(Clone, Debug, Eq, PartialEq)]
struct MmeInstructionPage {
    words: Box<[MaxwellThreeDMmeInstruction; MME_INSTRUCTION_PAGE_WORDS]>,
    // Zero is an executable opcode, so absence cannot use a word sentinel.
    present: u64,
}
impl Default for MmeInstructionPage {
    fn default() -> Self {
        Self {
            words: Box::new([MaxwellThreeDMmeInstruction::new(0); MME_INSTRUCTION_PAGE_WORDS]),
            present: 0,
        }
    }
}
impl MmeInstructionPage {
    fn instruction(&self, address: u32) -> Option<MaxwellThreeDMmeInstruction> {
        let index = address as usize % MME_INSTRUCTION_PAGE_WORDS;
        (self.present & (1_u64 << index) != 0).then(|| self.words[index])
    }
}

#[derive(Debug)]
struct MmeInstructionRam {
    pages: BTreeMap<u32, MmeInstructionPage>,
    count: usize,
    programs: Box<[OnceLock<Option<DirectProgram>>]>,
    program_pages: BTreeMap<u32, [u64; 4]>,
}
impl Default for MmeInstructionRam {
    fn default() -> Self {
        Self {
            pages: BTreeMap::new(),
            count: 0,
            programs: (0..256).map(|_| OnceLock::new()).collect(),
            program_pages: BTreeMap::new(),
        }
    }
}
impl Clone for MmeInstructionRam {
    fn clone(&self) -> Self {
        Self {
            pages: self.pages.clone(),
            count: self.count,
            program_pages: self.program_pages.clone(),
            ..Self::default()
        }
    }
}
impl PartialEq for MmeInstructionRam {
    fn eq(&self, other: &Self) -> bool {
        self.pages == other.pages && self.count == other.count
    }
}
impl Eq for MmeInstructionRam {}

impl MmeInstructionRam {
    fn index_program_pages(&mut self, index: usize, start: u32, present: bool) {
        // A direct program can examine at most 256 consecutive words. Include
        // failed compilations so a later upload can make them compilable.
        let bit = 1_u64 << (index % 64);
        for page in start / MME_INSTRUCTION_PAGE_WORDS as u32
            ..=start.saturating_add(255) / MME_INSTRUCTION_PAGE_WORDS as u32
        {
            if present {
                self.program_pages.entry(page).or_default()[index / 64] |= bit;
            } else if let Some(users) = self.program_pages.get_mut(&page) {
                users[index / 64] &= !bit;
                if users.iter().all(|&mask| mask == 0) {
                    self.program_pages.remove(&page);
                }
            }
        }
    }

    fn bind_program(&mut self, index: usize, old_start: Option<u32>, start: u32) {
        if index >= self.programs.len() {
            return;
        }
        if let Some(old_start) = old_start {
            self.index_program_pages(index, old_start, false);
        }
        self.index_program_pages(index, start, true);
        self.programs[index].take();
    }

    fn page(&self, address: u32) -> Option<&MmeInstructionPage> {
        self.pages
            .get(&(address / MME_INSTRUCTION_PAGE_WORDS as u32))
    }

    fn get(&self, address: u32) -> Option<MaxwellThreeDMmeInstruction> {
        self.page(address)?.instruction(address)
    }

    fn insert(&mut self, address: u32, instruction: MaxwellThreeDMmeInstruction) {
        if let Some(users) = self
            .program_pages
            .get(&(address / MME_INSTRUCTION_PAGE_WORDS as u32))
            .copied()
        {
            for (word, mut mask) in users.into_iter().enumerate() {
                while mask != 0 {
                    let index = word * 64 + mask.trailing_zeros() as usize;
                    self.programs[index].take();
                    mask &= mask - 1;
                }
            }
        }
        let page = self
            .pages
            .entry(address / MME_INSTRUCTION_PAGE_WORDS as u32)
            .or_default();
        let index = address as usize % MME_INSTRUCTION_PAGE_WORDS;
        let bit = 1_u64 << index;
        if page.present & bit == 0 {
            self.count += 1;
        }
        page.words[index] = instruction;
        page.present |= bit;
    }

    const fn len(&self) -> usize {
        self.count
    }
}

#[cfg(test)]
mod instruction_cache_tests {
    use super::*;

    #[test]
    fn uploads_invalidate_only_programs_whose_direct_window_can_read_the_page() {
        let mut ram = MmeInstructionRam::default();
        ram.bind_program(1, None, 0);
        ram.bind_program(2, None, 640);
        ram.programs[1].set(None).unwrap();
        ram.programs[2].set(None).unwrap();

        ram.insert(3, MaxwellThreeDMmeInstruction::new(0));
        assert!(ram.programs[1].get().is_none());
        assert!(ram.programs[2].get().is_some());

        ram.programs[1].set(None).unwrap();
        ram.bind_program(1, Some(0), 1280);
        ram.programs[1].set(None).unwrap();
        ram.insert(3, MaxwellThreeDMmeInstruction::new(0));
        assert!(ram.programs[1].get().is_some());
        ram.insert(1280, MaxwellThreeDMmeInstruction::new(0));
        assert!(ram.programs[1].get().is_none());
        assert!(ram.programs[2].get().is_some());

        // The start-address RAM can hold indices beyond the 256 callable
        // method pairs; they do not own a direct-program slot.
        ram.bind_program(256, None, 1920);
    }
}

struct MaxwellThreeDMmeInterpreter<'a, H> {
    program: &'a MaxwellThreeDMmeProgram,
    instruction_page: Option<&'a MmeInstructionPage>,
    // No 32-bit word address can have this initial page index.
    instruction_page_index: u32,
    host: &'a mut H,
    parameters: &'a [u32],
    registers: [u32; MME_REGISTER_COUNT],
    pc: u32,
    delayed_pc: Option<u32>,
    delay_slot: bool,
    exit_after_delay: bool,
    finished: bool,
    next_parameter: usize,
    method_address: u16,
    method_increment: u8,
    carry: bool,
    instructions: u32,
    emitted_methods: u32,
}

#[derive(Clone)]
pub(super) struct MaxwellThreeDMmeProgram {
    instructions: Arc<MmeInstructionRam>,
    start_addresses: Arc<BTreeMap<u32, MaxwellThreeDRegister<MaxwellThreeDMmeRamAddress>>>,
}

impl MaxwellThreeDMmeProgram {
    /// Executes one Maxwell MME program against ordered live host state.
    ///
    /// The ISA layout and behavior are pinned to yuzu's low-level Maxwell MME
    /// interpreter and independently agree with Ryujinx's interpreter:
    /// <https://source.hodakov.me/hdkv/yuzu/src/commit/8a674958a730a36dbcc43910412521420a804c69/src/video_core/macro/macro.h>
    /// <https://source.hodakov.me/hdkv/yuzu/src/commit/8a674958a730a36dbcc43910412521420a804c69/src/video_core/macro/macro_interpreter.cpp>
    /// <https://git.axenov.dev/Museum/ryujinx/src/commit/ec3e848d7998038ce22c41acdbf81032bf47991f/Ryujinx.Graphics.Gpu/Engine/MME/MacroInterpreter.cs>
    pub(super) async fn execute<H: MaxwellThreeDMmeHost>(
        &self,
        macro_index: u8,
        parameters: &[u32],
        host: &mut H,
    ) -> Result<(), MaxwellThreeDMmeRunError<H::Error>> {
        let start = self
            .start_addresses
            .get(&u32::from(macro_index))
            .and_then(MaxwellThreeDRegister::value)
            .copied()
            .ok_or(MaxwellThreeDMmeRunError::Execution(
                MaxwellThreeDMmeExecutionError::MissingStartAddress { macro_index },
            ))?;
        if parameters.is_empty() {
            return Err(MaxwellThreeDMmeRunError::Execution(
                MaxwellThreeDMmeExecutionError::ParameterUnavailable { index: 0 },
            ));
        }
        if let Some(program) = self.instructions.programs[macro_index as usize].get_or_init(|| {
            let program = DirectProgram::compile(&self.instructions, start.raw());
            nixe_gpu::trace::event(
                "mme_program",
                u64::from(macro_index),
                u64::from(start.raw()),
                u64::from(program.is_some()),
            );
            program
        }) {
            nixe_gpu::trace::detail_event(
                "mme_direct_macro",
                u64::from(macro_index),
                u64::from(start.raw()),
                parameters.len() as u64,
            );
            return program.execute(parameters, host).await;
        }
        let mut interpreter = MaxwellThreeDMmeInterpreter {
            program: self,
            instruction_page: None,
            instruction_page_index: u32::MAX,
            host,
            parameters,
            registers: [0; MME_REGISTER_COUNT],
            pc: start.raw(),
            delayed_pc: None,
            delay_slot: false,
            exit_after_delay: false,
            finished: false,
            next_parameter: 1,
            method_address: 0,
            method_increment: 0,
            carry: false,
            instructions: 0,
            emitted_methods: 0,
        };
        interpreter.registers[1] = parameters[0];
        while let Some((method, value)) = interpreter.run_until_effect()? {
            interpreter
                .host
                .finish_method(method, value)
                .await
                .map_err(MaxwellThreeDMmeRunError::Host)?;
        }
        if interpreter.next_parameter != parameters.len() {
            return Err(MaxwellThreeDMmeRunError::Execution(
                MaxwellThreeDMmeExecutionError::UnconsumedParameters {
                    consumed: interpreter.next_parameter,
                    supplied: parameters.len(),
                },
            ));
        }
        Ok(())
    }
}

impl<H: MaxwellThreeDMmeHost> MaxwellThreeDMmeInterpreter<'_, H> {
    fn execution_error<E>(error: MaxwellThreeDMmeExecutionError) -> MaxwellThreeDMmeRunError<E> {
        MaxwellThreeDMmeRunError::Execution(error)
    }

    // Arithmetic, branches and register-only sends stay in this synchronous loop.
    // Only an actual engine effect returns to the suspendable caller.
    fn run_until_effect(
        &mut self,
    ) -> Result<Option<(u16, u32)>, MaxwellThreeDMmeRunError<H::Error>> {
        while !self.finished {
            if self.instructions == MAXWELL_THREE_D_MME_EXECUTION_INSTRUCTION_LIMIT {
                return Err(Self::execution_error(
                    MaxwellThreeDMmeExecutionError::InstructionLimitExceeded {
                        limit: MAXWELL_THREE_D_MME_EXECUTION_INSTRUCTION_LIMIT,
                    },
                ));
            }
            let address = MaxwellThreeDMmeRamAddress::new(self.pc);
            let page_index = self.pc / MME_INSTRUCTION_PAGE_WORDS as u32;
            if self.instruction_page_index != page_index {
                let program = self.program;
                self.instruction_page = program.instructions.page(self.pc);
                self.instruction_page_index = page_index;
            }
            let instruction = self
                .instruction_page
                .and_then(|page| page.instruction(self.pc))
                .ok_or_else(|| {
                    Self::execution_error(MaxwellThreeDMmeExecutionError::MissingInstruction {
                        address,
                    })
                })?;
            self.instructions += 1;
            let base = self.pc;
            self.pc = self.pc.checked_add(1).ok_or_else(|| {
                Self::execution_error(MaxwellThreeDMmeExecutionError::ProgramCounterOverflow {
                    address,
                })
            })?;
            if let Some(delayed_pc) = self.delayed_pc.take() {
                self.pc = delayed_pc;
            }

            let is_delay_slot = std::mem::take(&mut self.delay_slot);
            let mut sent = None;
            let operation = instruction.operation();
            if operation == 7 {
                if is_delay_slot {
                    return Err(Self::execution_error(
                        MaxwellThreeDMmeExecutionError::BranchInDelaySlot { address },
                    ));
                }
                let value = self.register(instruction.src_a());
                let taken = if !instruction.branch_nonzero() {
                    value == 0
                } else {
                    value != 0
                };
                if taken {
                    let target = add_signed_18(base, instruction.immediate()).ok_or_else(|| {
                        Self::execution_error(
                            MaxwellThreeDMmeExecutionError::ProgramCounterOverflow { address },
                        )
                    })?;
                    if instruction.branch_annul() {
                        self.pc = target;
                        continue;
                    }
                    self.delayed_pc = Some(target);
                    self.delay_slot = true;
                    continue;
                }
            } else {
                let src_a = self.register(instruction.src_a());
                let src_b = self.register(instruction.src_b());
                let result = match operation {
                    0 => self.alu(address, instruction.alu(), src_a, src_b)?,
                    1 => src_a.wrapping_add_signed(instruction.immediate()),
                    2 => {
                        let mask = instruction.mask();
                        let source = (src_b >> instruction.source_bit()) & mask;
                        (src_a & !(mask << instruction.destination_bit()))
                            | (source << instruction.destination_bit())
                    }
                    3 => {
                        ((src_b >> (src_a & 0x1f)) & instruction.mask())
                            << instruction.destination_bit()
                    }
                    4 => {
                        ((src_b >> instruction.source_bit()) & instruction.mask()) << (src_a & 0x1f)
                    }
                    5 => {
                        let method = src_a.wrapping_add_signed(instruction.immediate());
                        if method > MME_METHOD_DWORD_MASK {
                            return Err(Self::execution_error(
                                MaxwellThreeDMmeExecutionError::RegisterReadUnavailable {
                                    method_dword: method as u16,
                                },
                            ));
                        }
                        self.host
                            .read_register(method as u16)
                            .map_err(MaxwellThreeDMmeRunError::Host)?
                    }
                    _ => {
                        return Err(Self::execution_error(
                            MaxwellThreeDMmeExecutionError::InvalidOperation { address, operation },
                        ));
                    }
                };
                sent = self.process_result(
                    instruction.result(),
                    usize::from(instruction.destination()),
                    result,
                )?;
            }

            if instruction.exit() && !is_delay_slot {
                self.delay_slot = true;
                self.exit_after_delay = true;
            } else if is_delay_slot && self.exit_after_delay {
                self.finished = true;
            }
            if let Some(value) = sent {
                if self.emitted_methods == MAXWELL_THREE_D_MME_EMITTED_METHOD_LIMIT {
                    return Err(Self::execution_error(
                        MaxwellThreeDMmeExecutionError::EmittedMethodLimitExceeded {
                            limit: MAXWELL_THREE_D_MME_EMITTED_METHOD_LIMIT,
                        },
                    ));
                }
                let method = self.method_address;
                self.emitted_methods += 1;
                self.method_address = (u32::from(method)
                    .wrapping_add(u32::from(self.method_increment))
                    & MME_METHOD_DWORD_MASK) as u16;
                if !self
                    .host
                    .try_emit_method(method, value)
                    .map_err(MaxwellThreeDMmeRunError::Host)?
                {
                    return Ok(Some((method, value)));
                }
            }
        }
        Ok(None)
    }

    fn alu(
        &mut self,
        address: MaxwellThreeDMmeRamAddress,
        operation: u8,
        a: u32,
        b: u32,
    ) -> Result<u32, MaxwellThreeDMmeRunError<H::Error>> {
        let result = match operation {
            0 => {
                let (result, carry) = a.overflowing_add(b);
                self.carry = carry;
                result
            }
            1 => {
                let (partial, first) = a.overflowing_add(b);
                let (result, second) = partial.overflowing_add(u32::from(self.carry));
                self.carry = first || second;
                result
            }
            2 => {
                let (result, borrow) = a.overflowing_sub(b);
                self.carry = !borrow;
                result
            }
            3 => {
                let borrow_in = u32::from(!self.carry);
                let (partial, first) = a.overflowing_sub(b);
                let (result, second) = partial.overflowing_sub(borrow_in);
                self.carry = !(first || second);
                result
            }
            8 => a ^ b,
            9 => a | b,
            10 => a & b,
            11 => a & !b,
            12 => !(a & b),
            _ => {
                return Err(Self::execution_error(
                    MaxwellThreeDMmeExecutionError::InvalidAluOperation { address, operation },
                ));
            }
        };
        Ok(result)
    }

    fn process_result(
        &mut self,
        operation: u8,
        destination: usize,
        result: u32,
    ) -> Result<Option<u32>, MaxwellThreeDMmeRunError<H::Error>> {
        match operation {
            0 => {
                let parameter = self.fetch_parameter()?;
                self.set_register(destination, parameter);
            }
            1 => self.set_register(destination, result),
            2 => {
                self.set_register(destination, result);
                self.set_method_address(result);
            }
            3 => {
                let parameter = self.fetch_parameter()?;
                self.set_register(destination, parameter);
                return Ok(Some(result));
            }
            4 => {
                self.set_register(destination, result);
                return Ok(Some(result));
            }
            5 => {
                let parameter = self.fetch_parameter()?;
                self.set_register(destination, parameter);
                self.set_method_address(result);
            }
            6 => {
                self.set_register(destination, result);
                self.set_method_address(result);
                let parameter = self.fetch_parameter()?;
                return Ok(Some(parameter));
            }
            7 => {
                self.set_register(destination, result);
                self.set_method_address(result);
                return Ok(Some((result >> 12) & 0x3f));
            }
            _ => unreachable!(),
        }
        Ok(None)
    }

    fn register(&self, register: u8) -> u32 {
        self.registers[register as usize]
    }

    fn set_register(&mut self, register: usize, value: u32) {
        if register != 0 {
            self.registers[register] = value;
        }
    }

    fn fetch_parameter(&mut self) -> Result<u32, MaxwellThreeDMmeRunError<H::Error>> {
        let index = self.next_parameter;
        let value = self.parameters.get(index).copied().ok_or_else(|| {
            Self::execution_error(MaxwellThreeDMmeExecutionError::ParameterUnavailable { index })
        })?;
        self.next_parameter += 1;
        Ok(value)
    }

    fn set_method_address(&mut self, raw: u32) {
        self.method_address = (raw & MME_METHOD_DWORD_MASK) as u16;
        self.method_increment = ((raw >> 12) & 0x3f) as u8;
    }
}

const fn signed_immediate(raw: u32) -> i32 {
    (raw as i32) >> 14
}

const fn add_signed_18(base: u32, immediate: i32) -> Option<u32> {
    if immediate >= 0 {
        base.checked_add(immediate as u32)
    } else {
        base.checked_sub(immediate.unsigned_abs())
    }
}

const fn bitfield_mask(raw: u32) -> u32 {
    (1_u32 << ((raw >> 22) & 0x1f)).wrapping_sub(1)
}

/// Complete captured MME program state for one `MAXWELL_B` channel.
///
/// The four load methods and their 32-bit fields are pinned to NVIDIA's public
/// class header:
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L55-L65>
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaxwellThreeDMmeState {
    instruction_pointer: MaxwellThreeDRegister<MaxwellThreeDMmeRamAddress>,
    next_instruction_address: Option<MaxwellThreeDMmeRamAddress>,
    instructions: Arc<MmeInstructionRam>,
    start_address_pointer: MaxwellThreeDRegister<MaxwellThreeDMmeRamAddress>,
    next_start_address_index: Option<MaxwellThreeDMmeRamAddress>,
    start_addresses: Arc<BTreeMap<u32, MaxwellThreeDRegister<MaxwellThreeDMmeRamAddress>>>,
    mutable_method_control: MaxwellThreeDRegister<MaxwellThreeDMutableMethodControl>,
    shadow_registers: ShadowRegisters,
}

impl Default for MaxwellThreeDMmeState {
    fn default() -> Self {
        // Maxwell's class register file is zero-initialized and its shadow
        // file is copied from that state. NVIDIA assigns zero to METHOD_TRACK,
        // so ordinary class writes must be shadowed before the guest's first
        // explicit SET_MME_SHADOW_RAM_CONTROL. This is observable when NVN
        // later switches through PASSTHROUGH and replays its initial state.
        //
        // NVIDIA encoding:
        // https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L67-L72
        // Register/shadow initialization:
        // https://github.com/yuzu-emu-mirror/yuzu-mainline/blob/310c1f50beb77fc5c6f9075029973161d4e51a4a/src/video_core/engines/maxwell_3d.cpp#L34-L104
        Self {
            instruction_pointer: MaxwellThreeDRegister::default(),
            next_instruction_address: None,
            instructions: Arc::new(MmeInstructionRam::default()),
            start_address_pointer: MaxwellThreeDRegister::default(),
            next_start_address_index: None,
            start_addresses: Arc::new(BTreeMap::new()),
            mutable_method_control: MaxwellThreeDRegister::default(),
            shadow_registers: ShadowRegisters::default(),
        }
    }
}

impl MaxwellThreeDMmeState {
    pub(super) fn program(&self) -> MaxwellThreeDMmeProgram {
        MaxwellThreeDMmeProgram {
            instructions: Arc::clone(&self.instructions),
            start_addresses: Arc::clone(&self.start_addresses),
        }
    }

    #[must_use]
    pub const fn instruction_pointer(&self) -> &MaxwellThreeDRegister<MaxwellThreeDMmeRamAddress> {
        &self.instruction_pointer
    }

    #[must_use]
    pub const fn next_instruction_address(&self) -> Option<MaxwellThreeDMmeRamAddress> {
        self.next_instruction_address
    }

    /// Returns the canonical word only when this address has been uploaded.
    #[must_use]
    pub fn instruction(
        &self,
        address: MaxwellThreeDMmeRamAddress,
    ) -> Option<MaxwellThreeDMmeInstruction> {
        self.instructions.get(address.raw())
    }

    #[must_use]
    pub fn instruction_count(&self) -> usize {
        self.instructions.len()
    }

    #[must_use]
    pub const fn start_address_pointer(
        &self,
    ) -> &MaxwellThreeDRegister<MaxwellThreeDMmeRamAddress> {
        &self.start_address_pointer
    }

    #[must_use]
    pub const fn next_start_address_index(&self) -> Option<MaxwellThreeDMmeRamAddress> {
        self.next_start_address_index
    }

    #[must_use]
    pub fn start_address(
        &self,
        index: MaxwellThreeDMmeRamAddress,
    ) -> Option<&MaxwellThreeDRegister<MaxwellThreeDMmeRamAddress>> {
        self.start_addresses.get(&index.raw())
    }

    #[must_use]
    pub fn start_address_count(&self) -> usize {
        self.start_addresses.len()
    }

    #[must_use]
    pub fn shadow_ram_control(&self) -> MaxwellThreeDRegister<MaxwellThreeDMmeShadowRamControl> {
        // The mode is the shadow bank's control register, as in Eden's Regs
        // union. Keep one authority even when the live control word is replayed.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/engines/maxwell_3d.cpp
        self.shadow_registers
            .read(0x0124, MaxwellThreeDMmeShadowRamControl::parse)
    }

    #[must_use]
    pub const fn mutable_method_control(
        &self,
    ) -> &MaxwellThreeDRegister<MaxwellThreeDMutableMethodControl> {
        &self.mutable_method_control
    }

    #[cfg(test)]
    #[must_use]
    pub(in crate::engines) fn shadow_value(&self, method: GpuMethodId) -> Option<u32> {
        self.shadow_registers.raw(method)
    }

    #[inline]
    pub(super) fn shadow_word(&self, method: GpuMethodId) -> u32 {
        self.shadow_registers.word(method)
    }

    #[inline]
    pub(super) fn resolve_shadow_argument(
        &self,
        method: GpuMethodId,
        submitted_argument: u32,
    ) -> u32 {
        if self.shadow_registers.word(GpuMethodId(0x0124))
            == MaxwellThreeDMmeShadowRamControl::MethodReplay.raw()
        {
            self.shadow_registers.word(method)
        } else {
            submitted_argument
        }
    }

    pub(super) fn track_shadow_register(
        &mut self,
        control: Option<MaxwellThreeDMmeShadowRamControl>,
        method: GpuMethodId,
        argument: u32,
    ) {
        // Public Maxwell implementations agree that TrackWithFilter follows
        // Track for ordinary class-register writes; the undocumented filter
        // distinction is intentionally not guessed here. MME call/data writes
        // bypass this path entirely.
        if control.is_some_and(|mode| mode.tracks()) {
            // PFIFO/MME methods index the same fixed 12-bit register file as
            // the live state. Replay consumes only the tracked value.
            self.write_shadow_register(method, argument);
        }
    }

    pub(super) fn write_shadow_register(&mut self, method: GpuMethodId, argument: u32) {
        self.shadow_registers.write(method, argument);
    }

    pub(super) fn apply(&mut self, write: MaxwellThreeDMmeStateWrite) {
        match write {
            MaxwellThreeDMmeStateWrite::InstructionPointer { value, source: _ } => {
                self.instruction_pointer = MaxwellThreeDRegister::decoded(value.raw(), value);
                self.next_instruction_address = Some(value);
            }
            MaxwellThreeDMmeStateWrite::Instruction {
                address,
                value,
                source: _,
            } => {
                nixe_gpu::trace::event(
                    "mme_upload",
                    u64::from(address.raw()),
                    u64::from(value.raw()),
                    0,
                );
                Arc::make_mut(&mut self.instructions).insert(address.raw(), value);
                self.next_instruction_address = Some(MaxwellThreeDMmeRamAddress::new(
                    address
                        .raw()
                        .checked_add(1)
                        .expect("MME address was preflighted"),
                ));
            }
            MaxwellThreeDMmeStateWrite::StartAddressPointer { value, source: _ } => {
                self.start_address_pointer = MaxwellThreeDRegister::decoded(value.raw(), value);
                self.next_start_address_index = Some(value);
            }
            MaxwellThreeDMmeStateWrite::StartAddress {
                index,
                address,
                source: _,
            } => {
                nixe_gpu::trace::event(
                    "mme_start",
                    u64::from(index.raw()),
                    u64::from(address.raw()),
                    0,
                );
                if index.raw() < 256 {
                    let old_start = self
                        .start_addresses
                        .get(&index.raw())
                        .and_then(MaxwellThreeDRegister::value)
                        .map(|start| start.raw());
                    Arc::make_mut(&mut self.instructions).bind_program(
                        index.raw() as usize,
                        old_start,
                        address.raw(),
                    );
                }
                Arc::make_mut(&mut self.start_addresses).insert(
                    index.raw(),
                    MaxwellThreeDRegister::decoded(address.raw(), address),
                );
                self.next_start_address_index = Some(MaxwellThreeDMmeRamAddress::new(
                    index
                        .raw()
                        .checked_add(1)
                        .expect("MME index was preflighted"),
                ));
            }
            MaxwellThreeDMmeStateWrite::ShadowRamControl { value, source: _ } => {
                self.shadow_registers
                    .write(GpuMethodId(0x0124), value.raw());
            }
            MaxwellThreeDMmeStateWrite::MutableMethodControl { value, source: _ } => {
                self.mutable_method_control = MaxwellThreeDRegister::decoded(value.raw(), value);
            }
        }
    }
}

/// One checked MME RAM transition ready for direct application.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellThreeDMmeStateWrite {
    InstructionPointer {
        value: MaxwellThreeDMmeRamAddress,
        source: MaxwellMethodSource,
    },
    Instruction {
        address: MaxwellThreeDMmeRamAddress,
        value: MaxwellThreeDMmeInstruction,
        source: MaxwellMethodSource,
    },
    StartAddressPointer {
        value: MaxwellThreeDMmeRamAddress,
        source: MaxwellMethodSource,
    },
    StartAddress {
        index: MaxwellThreeDMmeRamAddress,
        address: MaxwellThreeDMmeRamAddress,
        source: MaxwellMethodSource,
    },
    ShadowRamControl {
        value: MaxwellThreeDMmeShadowRamControl,
        source: MaxwellMethodSource,
    },
    MutableMethodControl {
        value: MaxwellThreeDMutableMethodControl,
        source: MaxwellMethodSource,
    },
}

#[cfg(test)]
mod execution_benchmark {
    use super::*;

    #[derive(Default)]
    struct Host {
        value: u32,
        sends: u64,
        sum: u64,
    }
    impl MaxwellThreeDMmeHost for Host {
        type Error = &'static str;
        fn read_register(&self, method: u16) -> Result<u32, Self::Error> {
            if method != 0x3400 / 4 {
                return Err("unexpected benchmark register");
            }
            Ok(self.value)
        }
        fn try_emit_method(&mut self, method: u16, value: u32) -> Result<bool, Self::Error> {
            if method != 0x3400 / 4 {
                return Err("unexpected benchmark send");
            }
            self.sends += 1;
            self.sum = self.sum.wrapping_add(u64::from(value));
            Ok(true)
        }
        async fn finish_method(&mut self, method: u16, value: u32) -> Result<(), Self::Error> {
            self.try_emit_method(method, value).map(|_| ())
        }
    }
    fn run(program: &MaxwellThreeDMmeProgram, parameters: &[u32], host: &mut Host) {
        match crate::test_memory::ready(program.execute(0, parameters, host)) {
            Ok(()) => {}
            Err(MaxwellThreeDMmeRunError::Execution(error)) => {
                panic!("benchmark MME ISA error: {error:?}")
            }
            Err(MaxwellThreeDMmeRunError::Host(error)) => {
                panic!("benchmark MME host error: {error}")
            }
        }
    }
    fn program(start: u32, words: &[u32]) -> MaxwellThreeDMmeProgram {
        let mut instructions = MmeInstructionRam::default();
        for (offset, &raw) in words.iter().enumerate() {
            instructions.insert(
                start.checked_add(offset as u32).unwrap(),
                MaxwellThreeDMmeInstruction::new(raw),
            );
        }
        instructions.bind_program(0, None, start);
        MaxwellThreeDMmeProgram {
            instructions: Arc::new(instructions),
            start_addresses: Arc::new(BTreeMap::from([(
                0,
                MaxwellThreeDRegister::decoded(start, MaxwellThreeDMmeRamAddress::new(start)),
            )])),
        }
    }

    #[test]
    fn a_zero_opcode_is_executable_and_unwritten_words_remain_missing() {
        let mut host = Host::default();
        let zero = program(7, &[0]);
        let result = crate::test_memory::ready(zero.execute(0, &[0, 1], &mut host));
        assert!(matches!(result,
            Err(MaxwellThreeDMmeRunError::Execution(
                MaxwellThreeDMmeExecutionError::MissingInstruction { address }
            )) if address.raw() == 8));
        let mut hole = program(0, &[]);
        Arc::get_mut(&mut hole.instructions)
            .unwrap()
            .insert(32, MaxwellThreeDMmeInstruction::new(0x11));
        let result = crate::test_memory::ready(hole.execute(0, &[0], &mut host));
        assert!(matches!(result,
            Err(MaxwellThreeDMmeRunError::Execution(
                MaxwellThreeDMmeExecutionError::MissingInstruction { address }
            )) if address.raw() == 0));
    }

    #[test]
    #[cfg(target_os = "linux")]
    #[ignore = "controlled MME interpreter CPU/allocation measurement"]
    fn interpreter_throughput() {
        use crate::engines::tests::register_benchmark::ALLOCATIONS;
        use std::hint::black_box;
        use std::time::Instant;
        let iterations: usize = std::env::var("NIXE_MME_BENCH_ITERS")
            .ok()
            .and_then(|value| value.parse().ok())
            .filter(|&n| n > 0)
            .unwrap_or(5000);
        let cpu_ns = || {
            std::fs::read_to_string("/proc/thread-self/schedstat")
                .unwrap()
                .split_whitespace()
                .next()
                .unwrap()
                .parse::<u64>()
                .unwrap()
        };
        // The bounded counter and branch loops also occur in the existing
        // architectural tests. A register read forces ISA execution in the
        // send case; none of these programs can use the direct-program path.
        let decrement = 1 | (1 << 4) | (1 << 8) | (1 << 11) | ((-1_i32 as u32) << 14);
        let branch = |offset: i32| 7 | (1 << 4) | (1 << 5) | (1 << 11) | ((offset as u32) << 14);
        let exit = 1 | (1 << 4) | (1 << 7);
        let delay = 1 | (1 << 4);
        let method = 0x3400 / 4;
        let set_method = 1 | (2 << 4) | (method << 14);
        let read_send = 5 | (4 << 4) | (method << 14);
        for (name, start, count, words, per_call, sends) in [
            (
                "alu_branch",
                0,
                20_000,
                vec![decrement, branch(-1), exit, delay],
                40_002,
                0,
            ),
            (
                "page_branch",
                63,
                20_000,
                vec![decrement, branch(-1), exit, delay],
                40_002,
                0,
            ),
            (
                "read_send",
                0,
                5_000,
                vec![set_method, read_send, decrement, branch(-2), exit, delay],
                15_003,
                5_000,
            ),
        ] {
            let program = program(start, &words);
            let parameters = [count];
            let mut host = Host {
                value: black_box(0x1234_abcd),
                ..Host::default()
            };
            for _ in 0..4 {
                run(&program, &parameters, &mut host);
            }
            assert!(program.instructions.programs[0].get().unwrap().is_none());
            host.sends = 0;
            host.sum = 0;
            let before = cpu_ns();
            let started = Instant::now();
            ALLOCATIONS.with(|n| n.set(Some(0)));
            for _ in 0..iterations {
                run(&program, black_box(&parameters), &mut host);
            }
            let allocations = ALLOCATIONS.with(|n| n.replace(None).unwrap());
            let wall_ns = started.elapsed().as_nanos();
            let elapsed_cpu_ns = cpu_ns() - before;
            let emitted = sends * iterations as u64;
            assert_eq!(host.sends, emitted);
            assert_eq!(host.sum, u64::from(host.value) * emitted);
            assert_eq!(allocations, 0);
            let instructions = per_call * iterations as u64;
            let word_bytes = std::mem::size_of::<MaxwellThreeDMmeInstruction>();
            println!(
                "MME_WORD_BENCH {{\"scenario\":\"{name}\",\"iterations\":{iterations},\"instructions\":{instructions},\"cpu_ns\":{elapsed_cpu_ns},\"wall_ns\":{wall_ns},\"allocations\":{allocations},\"instruction_bytes\":{word_bytes}}}"
            );
        }
    }
}
