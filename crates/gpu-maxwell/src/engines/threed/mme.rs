//! Uploaded macro bodies and the selected contiguous MME interpreter program.

use rustc_hash::FxHashMap as HashMap;
use std::sync::Arc;

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

// Maxwell3D::macro_positions aliases the callable method pairs onto this bank.
// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/engines/maxwell_3d.h#L3259
pub(super) const MME_MACRO_POSITIONS: usize = 0x80;

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

/// Live engine services consumed by the MME interpreter.
pub(super) trait MaxwellThreeDMmeHost {
    type Error;

    fn read_register(&self, method_dword: u16) -> Result<u32, Self::Error>;
    /// Execute the method and any pending primitive boundary before continuing.
    fn emit_method(&mut self, method_dword: u16, argument: u32) -> Result<(), Self::Error>;
}

pub(super) enum MaxwellThreeDMmeRunError<E> {
    Execution(MaxwellThreeDMmeExecutionError),
    Host(E),
}

struct MaxwellThreeDMmeInterpreter<'a, H> {
    program: &'a MaxwellThreeDMmeProgram,
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

/// A selected macro owns only its contiguous uploaded body. An invocation may
/// suspend while an emitted engine effect completes, so it retains that body.
/// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/macro.cpp#L1340-L1434
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct MaxwellThreeDMmeProgram {
    instructions: Arc<[MaxwellThreeDMmeInstruction]>,
    start: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CachedMmeProgram {
    upload_address: u32,
    program: MaxwellThreeDMmeProgram,
}

impl MaxwellThreeDMmeProgram {
    /// Executes one Maxwell MME program against ordered live host state.
    ///
    /// The ISA layout and behavior are pinned to yuzu's low-level Maxwell MME
    /// interpreter and independently agree with Ryujinx's interpreter:
    /// <https://source.hodakov.me/hdkv/yuzu/src/commit/8a674958a730a36dbcc43910412521420a804c69/src/video_core/macro/macro.h>
    /// <https://source.hodakov.me/hdkv/yuzu/src/commit/8a674958a730a36dbcc43910412521420a804c69/src/video_core/macro/macro_interpreter.cpp>
    /// <https://git.axenov.dev/Museum/ryujinx/src/commit/ec3e848d7998038ce22c41acdbf81032bf47991f/Ryujinx.Graphics.Gpu/Engine/MME/MacroInterpreter.cs>
    pub(super) fn execute<H: MaxwellThreeDMmeHost>(
        &self,
        parameters: &[u32],
        host: &mut H,
    ) -> Result<(), MaxwellThreeDMmeRunError<H::Error>> {
        if parameters.is_empty() {
            return Err(MaxwellThreeDMmeRunError::Execution(
                MaxwellThreeDMmeExecutionError::ParameterUnavailable { index: 0 },
            ));
        }
        let mut interpreter = MaxwellThreeDMmeInterpreter {
            program: self,
            host,
            parameters,
            registers: [0; MME_REGISTER_COUNT],
            pc: 0,
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
        interpreter.run()?;
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
    fn run(&mut self) -> Result<(), MaxwellThreeDMmeRunError<H::Error>> {
        while !self.finished {
            if self.instructions == MAXWELL_THREE_D_MME_EXECUTION_INSTRUCTION_LIMIT {
                return Err(Self::execution_error(
                    MaxwellThreeDMmeExecutionError::InstructionLimitExceeded {
                        limit: MAXWELL_THREE_D_MME_EXECUTION_INSTRUCTION_LIMIT,
                    },
                ));
            }
            let address = MaxwellThreeDMmeRamAddress::new(self.program.start.wrapping_add(self.pc));
            let instruction = self
                .program
                .instructions
                .get(self.pc as usize)
                .copied()
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
                self.host
                    .emit_method(method, value)
                    .map_err(MaxwellThreeDMmeRunError::Host)?;
            }
        }
        Ok(())
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
    uploaded_code: HashMap<u32, Vec<MaxwellThreeDMmeInstruction>>,
    programs: HashMap<u32, CachedMmeProgram>,
    start_address_pointer: MaxwellThreeDRegister<MaxwellThreeDMmeRamAddress>,
    next_start_address_index: Option<MaxwellThreeDMmeRamAddress>,
    start_addresses: Box<[MaxwellThreeDRegister<MaxwellThreeDMmeRamAddress>; MME_MACRO_POSITIONS]>,
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
            uploaded_code: HashMap::default(),
            programs: HashMap::default(),
            start_address_pointer: MaxwellThreeDRegister::default(),
            next_start_address_index: None,
            start_addresses: Box::new(std::array::from_fn(|_| MaxwellThreeDRegister::default())),
            mutable_method_control: MaxwellThreeDRegister::default(),
            shadow_registers: ShadowRegisters::default(),
        }
    }
}

impl MaxwellThreeDMmeState {
    pub(super) fn program(
        &mut self,
        macro_index: u8,
    ) -> Result<MaxwellThreeDMmeProgram, MaxwellThreeDMmeExecutionError> {
        let index = usize::from(macro_index) % MME_MACRO_POSITIONS;
        let start = self.start_addresses[index]
            .value()
            .ok_or(MaxwellThreeDMmeExecutionError::MissingStartAddress { macro_index })?
            .raw();
        if let Some(cached) = self.programs.get(&start) {
            return Ok(cached.program.clone());
        }
        // MacroEngine selects a body by its upload address. An entry in the
        // middle of an uploaded body is rebased once, before ISA execution.
        // Uploads and executable bodies have distinct lifetimes, as in Eden.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/macro.cpp#L1387-L1420
        let (upload_address, code) = self
            .uploaded_code
            .get_key_value(&start)
            .or_else(|| {
                self.uploaded_code.iter().find(|&(base, code)| {
                    start >= *base && u64::from(start - *base) < code.len() as u64
                })
            })
            .ok_or(MaxwellThreeDMmeExecutionError::MissingInstruction {
                address: MaxwellThreeDMmeRamAddress::new(start),
            })?;
        let offset = (start - upload_address) as usize;
        let program = MaxwellThreeDMmeProgram {
            start,
            instructions: Arc::from(&code[offset..]),
        };
        self.programs.insert(
            start,
            CachedMmeProgram {
                upload_address: *upload_address,
                program: program.clone(),
            },
        );
        Ok(program)
    }

    #[must_use]
    pub const fn instruction_pointer(&self) -> &MaxwellThreeDRegister<MaxwellThreeDMmeRamAddress> {
        &self.instruction_pointer
    }

    #[must_use]
    pub const fn next_instruction_address(&self) -> Option<MaxwellThreeDMmeRamAddress> {
        self.next_instruction_address
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
        self.start_addresses.get(index.raw() as usize)
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
                self.uploaded_code.remove(&value.raw());
                self.programs
                    .retain(|_, cached| cached.upload_address != value.raw());
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
                let base = self
                    .instruction_pointer
                    .value()
                    .expect("admitted MME upload")
                    .raw();
                self.programs
                    .retain(|_, cached| cached.upload_address != base);
                self.uploaded_code.entry(base).or_default().push(value);
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
                self.start_addresses[index.raw() as usize] =
                    MaxwellThreeDRegister::decoded(address.raw(), address);
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
