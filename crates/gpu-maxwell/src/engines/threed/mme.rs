//! Source-preserving Macro Method Expander program storage for `MAXWELL_B`.

use std::collections::BTreeMap;

use nixe_gpu::GpuMethodId;

use crate::MaxwellMethodSource;

use super::{MaxwellThreeDRegister, state::verified_raw_register_reset};

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

/// Why one shadow-RAM transition cannot be represented faithfully.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MaxwellThreeDMmeShadowRamError {
    ReplayRegisterUnavailable { method_dword: u16 },
}

/// Index of one `SET_MME_SHADOW_SCRATCH(i)` register.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct MaxwellThreeDMmeShadowScratchIndex(u8);

impl MaxwellThreeDMmeShadowScratchIndex {
    #[must_use]
    pub const fn new(raw: u8) -> Self {
        Self(raw)
    }

    #[must_use]
    pub const fn raw(self) -> u8 {
        self.0
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
pub const MAXWELL_THREE_D_MME_EXECUTION_INSTRUCTION_LIMIT: u32 = 4096;

/// Maximum class methods emitted by one macro invocation.
pub const MAXWELL_THREE_D_MME_EMITTED_METHOD_LIMIT: u32 = 4096;

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

/// Successful, bounded execution statistics for one MME invocation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellThreeDMmeExecutionReport {
    pub instructions: u32,
    pub emitted_methods: u32,
}

/// Host services used by the ISA interpreter.
pub(super) trait MaxwellThreeDMmeHost {
    type Error;

    fn read_register(&self, method_dword: u16) -> Result<u32, Self::Error>;
    fn emit_method(&mut self, method_dword: u16, argument: u32) -> Result<(), Self::Error>;
}

pub(super) enum MaxwellThreeDMmeRunError<E> {
    Execution(MaxwellThreeDMmeExecutionError),
    Host(E),
}

struct MaxwellThreeDMmeInterpreter<'a, H> {
    program: &'a MaxwellThreeDMmeState,
    host: &'a mut H,
    parameters: &'a [u32],
    registers: [u32; MME_REGISTER_COUNT],
    pc: u32,
    delayed_pc: Option<u32>,
    next_parameter: usize,
    method_address: u16,
    method_increment: u8,
    carry: bool,
    instructions: u32,
    emitted_methods: u32,
}

impl MaxwellThreeDMmeState {
    /// Executes one Maxwell MME program against a transactional host state.
    ///
    /// The ISA layout and behavior are pinned to yuzu's low-level Maxwell MME
    /// interpreter and independently agree with Ryujinx's interpreter:
    /// <https://source.hodakov.me/hdkv/yuzu/src/commit/8a674958a730a36dbcc43910412521420a804c69/src/video_core/macro/macro.h>
    /// <https://source.hodakov.me/hdkv/yuzu/src/commit/8a674958a730a36dbcc43910412521420a804c69/src/video_core/macro/macro_interpreter.cpp>
    /// <https://git.axenov.dev/Museum/ryujinx/src/commit/ec3e848d7998038ce22c41acdbf81032bf47991f/Ryujinx.Graphics.Gpu/Engine/MME/MacroInterpreter.cs>
    pub(super) fn execute<H: MaxwellThreeDMmeHost>(
        &self,
        macro_index: u8,
        parameters: &[u32],
        host: &mut H,
    ) -> Result<MaxwellThreeDMmeExecutionReport, MaxwellThreeDMmeRunError<H::Error>> {
        let start = self
            .start_address(MaxwellThreeDMmeRamAddress::new(u32::from(macro_index)))
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
        let mut interpreter = MaxwellThreeDMmeInterpreter {
            program: self,
            host,
            parameters,
            registers: [0; MME_REGISTER_COUNT],
            pc: start.raw(),
            delayed_pc: None,
            next_parameter: 1,
            method_address: 0,
            method_increment: 0,
            carry: false,
            instructions: 0,
            emitted_methods: 0,
        };
        interpreter.registers[1] = parameters[0];
        while interpreter.step(false)? {}
        if interpreter.next_parameter != parameters.len() {
            return Err(MaxwellThreeDMmeRunError::Execution(
                MaxwellThreeDMmeExecutionError::UnconsumedParameters {
                    consumed: interpreter.next_parameter,
                    supplied: parameters.len(),
                },
            ));
        }
        Ok(MaxwellThreeDMmeExecutionReport {
            instructions: interpreter.instructions,
            emitted_methods: interpreter.emitted_methods,
        })
    }
}

impl<H: MaxwellThreeDMmeHost> MaxwellThreeDMmeInterpreter<'_, H> {
    fn execution_error<E>(error: MaxwellThreeDMmeExecutionError) -> MaxwellThreeDMmeRunError<E> {
        MaxwellThreeDMmeRunError::Execution(error)
    }

    fn step(&mut self, is_delay_slot: bool) -> Result<bool, MaxwellThreeDMmeRunError<H::Error>> {
        if self.instructions == MAXWELL_THREE_D_MME_EXECUTION_INSTRUCTION_LIMIT {
            return Err(Self::execution_error(
                MaxwellThreeDMmeExecutionError::InstructionLimitExceeded {
                    limit: MAXWELL_THREE_D_MME_EXECUTION_INSTRUCTION_LIMIT,
                },
            ));
        }
        let address = MaxwellThreeDMmeRamAddress::new(self.pc);
        let raw = self
            .program
            .instruction(address)
            .and_then(MaxwellThreeDRegister::value)
            .map(|instruction| instruction.raw())
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

        let operation = (raw & 7) as u8;
        if operation == 7 {
            if is_delay_slot {
                return Err(Self::execution_error(
                    MaxwellThreeDMmeExecutionError::BranchInDelaySlot { address },
                ));
            }
            let value = self.register((raw >> 11) & 7);
            let taken = if raw & (1 << 4) == 0 {
                value == 0
            } else {
                value != 0
            };
            if taken {
                let target = add_signed_18(base, signed_immediate(raw)).ok_or_else(|| {
                    Self::execution_error(MaxwellThreeDMmeExecutionError::ProgramCounterOverflow {
                        address,
                    })
                })?;
                if raw & (1 << 5) != 0 {
                    self.pc = target;
                    return Ok(true);
                }
                self.delayed_pc = Some(target);
                return self.step(true);
            }
        } else {
            let src_a = self.register((raw >> 11) & 7);
            let src_b = self.register((raw >> 14) & 7);
            let result = match operation {
                0 => self.alu(address, ((raw >> 17) & 0x1f) as u8, src_a, src_b)?,
                1 => src_a.wrapping_add_signed(signed_immediate(raw)),
                2 => {
                    let mask = bitfield_mask(raw);
                    let source = (src_b >> ((raw >> 17) & 0x1f)) & mask;
                    (src_a & !(mask << ((raw >> 27) & 0x1f))) | (source << ((raw >> 27) & 0x1f))
                }
                3 => ((src_b >> (src_a & 0x1f)) & bitfield_mask(raw)) << ((raw >> 27) & 0x1f),
                4 => ((src_b >> ((raw >> 17) & 0x1f)) & bitfield_mask(raw)) << (src_a & 0x1f),
                5 => {
                    let method = src_a.wrapping_add_signed(signed_immediate(raw));
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
            self.process_result(((raw >> 4) & 7) as u8, ((raw >> 8) & 7) as usize, result)?;
        }

        if raw & (1 << 7) != 0 && !is_delay_slot {
            self.step(true)?;
            return Ok(false);
        }
        Ok(true)
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
    ) -> Result<(), MaxwellThreeDMmeRunError<H::Error>> {
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
                self.send(result)?;
            }
            4 => {
                self.set_register(destination, result);
                self.send(result)?;
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
                self.send(parameter)?;
            }
            7 => {
                self.set_register(destination, result);
                self.set_method_address(result);
                self.send((result >> 12) & 0x3f)?;
            }
            _ => unreachable!(),
        }
        Ok(())
    }

    fn register(&self, register: u32) -> u32 {
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

    fn send(&mut self, value: u32) -> Result<(), MaxwellThreeDMmeRunError<H::Error>> {
        if self.emitted_methods == MAXWELL_THREE_D_MME_EMITTED_METHOD_LIMIT {
            return Err(Self::execution_error(
                MaxwellThreeDMmeExecutionError::EmittedMethodLimitExceeded {
                    limit: MAXWELL_THREE_D_MME_EMITTED_METHOD_LIMIT,
                },
            ));
        }
        self.host
            .emit_method(self.method_address, value)
            .map_err(MaxwellThreeDMmeRunError::Host)?;
        self.emitted_methods += 1;
        self.method_address = (u32::from(self.method_address)
            .wrapping_add(u32::from(self.method_increment))
            & MME_METHOD_DWORD_MASK) as u16;
        Ok(())
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
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellThreeDMmeState {
    instruction_pointer: MaxwellThreeDRegister<MaxwellThreeDMmeRamAddress>,
    next_instruction_address: Option<MaxwellThreeDMmeRamAddress>,
    instructions: BTreeMap<u32, MaxwellThreeDRegister<MaxwellThreeDMmeInstruction>>,
    start_address_pointer: MaxwellThreeDRegister<MaxwellThreeDMmeRamAddress>,
    next_start_address_index: Option<MaxwellThreeDMmeRamAddress>,
    start_addresses: BTreeMap<u32, MaxwellThreeDRegister<MaxwellThreeDMmeRamAddress>>,
    shadow_ram_control: MaxwellThreeDRegister<MaxwellThreeDMmeShadowRamControl>,
    mutable_method_control: MaxwellThreeDRegister<MaxwellThreeDMutableMethodControl>,
    shadow_registers: BTreeMap<u32, MaxwellThreeDRegister<u32>>,
    shadow_scratch: BTreeMap<u8, MaxwellThreeDRegister<u32>>,
}

impl MaxwellThreeDMmeState {
    #[must_use]
    pub const fn instruction_pointer(&self) -> &MaxwellThreeDRegister<MaxwellThreeDMmeRamAddress> {
        &self.instruction_pointer
    }

    #[must_use]
    pub const fn next_instruction_address(&self) -> Option<MaxwellThreeDMmeRamAddress> {
        self.next_instruction_address
    }

    #[must_use]
    pub fn instruction(
        &self,
        address: MaxwellThreeDMmeRamAddress,
    ) -> Option<&MaxwellThreeDRegister<MaxwellThreeDMmeInstruction>> {
        self.instructions.get(&address.raw())
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
    pub const fn shadow_ram_control(
        &self,
    ) -> &MaxwellThreeDRegister<MaxwellThreeDMmeShadowRamControl> {
        &self.shadow_ram_control
    }

    #[must_use]
    pub const fn mutable_method_control(
        &self,
    ) -> &MaxwellThreeDRegister<MaxwellThreeDMutableMethodControl> {
        &self.mutable_method_control
    }

    #[must_use]
    pub fn shadow_register(&self, method: GpuMethodId) -> Option<&MaxwellThreeDRegister<u32>> {
        self.shadow_registers.get(&method.0)
    }

    pub(super) fn resolve_shadow_argument(
        &self,
        method: GpuMethodId,
        submitted_argument: u32,
    ) -> Result<u32, MaxwellThreeDMmeShadowRamError> {
        // SET_MME_SHADOW_RAM_CONTROL consumes its non-shadowed argument so the
        // command stream can always leave replay mode. This agrees with the
        // source-preserving split used by yuzu's Maxwell frontend:
        // https://ni.4a.si/anonymous/yuzu/tree/src/video_core/engines/maxwell_3d.cpp?id=9705094a576e6594e359cc0256b63385ac05de3f#n319
        if method.0 == 0x0124
            || self.shadow_ram_control.value()
                != Some(&MaxwellThreeDMmeShadowRamControl::MethodReplay)
        {
            return Ok(submitted_argument);
        }
        self.shadow_register(method)
            .and_then(MaxwellThreeDRegister::raw)
            .or_else(|| verified_raw_register_reset(method))
            .ok_or(MaxwellThreeDMmeShadowRamError::ReplayRegisterUnavailable {
                method_dword: (method.0 / 4) as u16,
            })
    }

    pub(super) fn track_shadow_register(
        &mut self,
        control: Option<MaxwellThreeDMmeShadowRamControl>,
        source: MaxwellMethodSource,
    ) {
        // Public Maxwell implementations agree that TrackWithFilter follows
        // Track for ordinary class-register writes; the undocumented filter
        // distinction is intentionally not guessed here. MME call/data writes
        // bypass this path entirely.
        if control.is_some_and(|mode| mode.tracks()) {
            self.shadow_registers.insert(
                source.method().0,
                MaxwellThreeDRegister::programmed(source.argument(), source.argument(), source),
            );
        }
    }

    #[must_use]
    pub fn shadow_scratch(
        &self,
        index: MaxwellThreeDMmeShadowScratchIndex,
    ) -> Option<&MaxwellThreeDRegister<u32>> {
        self.shadow_scratch.get(&index.raw())
    }

    #[must_use]
    pub fn shadow_scratch_count(&self) -> usize {
        self.shadow_scratch.len()
    }

    pub(super) fn apply(&mut self, write: MaxwellThreeDMmeStateWrite) {
        match write {
            MaxwellThreeDMmeStateWrite::InstructionPointer { value, source } => {
                self.instruction_pointer =
                    MaxwellThreeDRegister::programmed(value.raw(), value, source);
                self.next_instruction_address = Some(value);
            }
            MaxwellThreeDMmeStateWrite::Instruction {
                address,
                value,
                source,
            } => {
                self.instructions.insert(
                    address.raw(),
                    MaxwellThreeDRegister::programmed(value.raw(), value, source),
                );
                self.next_instruction_address = Some(MaxwellThreeDMmeRamAddress::new(
                    address
                        .raw()
                        .checked_add(1)
                        .expect("MME address was preflighted"),
                ));
            }
            MaxwellThreeDMmeStateWrite::StartAddressPointer { value, source } => {
                self.start_address_pointer =
                    MaxwellThreeDRegister::programmed(value.raw(), value, source);
                self.next_start_address_index = Some(value);
            }
            MaxwellThreeDMmeStateWrite::StartAddress {
                index,
                address,
                source,
            } => {
                self.start_addresses.insert(
                    index.raw(),
                    MaxwellThreeDRegister::programmed(address.raw(), address, source),
                );
                self.next_start_address_index = Some(MaxwellThreeDMmeRamAddress::new(
                    index
                        .raw()
                        .checked_add(1)
                        .expect("MME index was preflighted"),
                ));
            }
            MaxwellThreeDMmeStateWrite::ShadowRamControl { value, source } => {
                self.shadow_ram_control =
                    MaxwellThreeDRegister::programmed(value.raw(), value, source);
            }
            MaxwellThreeDMmeStateWrite::MutableMethodControl { value, source } => {
                self.mutable_method_control =
                    MaxwellThreeDRegister::programmed(value.raw(), value, source);
            }
            MaxwellThreeDMmeStateWrite::ShadowScratch {
                index,
                value,
                source,
            } => {
                self.shadow_scratch.insert(
                    index.raw(),
                    MaxwellThreeDRegister::programmed(value, value, source),
                );
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
    ShadowScratch {
        index: MaxwellThreeDMmeShadowScratchIndex,
        value: u32,
        source: MaxwellMethodSource,
    },
}
