//! Source-preserving Macro Method Expander program storage for `MAXWELL_B`.

use std::{collections::BTreeMap, sync::Arc};

mod instructions;
use instructions::{InstructionPage, InstructionRam, PAGE_WORDS};

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
    program: &'a MaxwellThreeDMmeProgram,
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
    page: Option<(u32, &'a InstructionPage)>,
}

#[derive(Clone)]
pub(super) struct MaxwellThreeDMmeProgram {
    instructions: Arc<InstructionRam>,
    start_addresses: Arc<BTreeMap<u32, MaxwellThreeDRegister<MaxwellThreeDMmeRamAddress>>>,
}

impl MaxwellThreeDMmeProgram {
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
            page: None,
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
        Ok(())
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
        let page_index = self.pc / PAGE_WORDS as u32;
        if self.page.is_none_or(|(index, _)| index != page_index) {
            self.page = self
                .program
                .instructions
                .page(page_index)
                .map(|page| (page_index, page));
        }
        let instruction = self
            .page
            .and_then(|(_, page)| page[self.pc as usize % PAGE_WORDS].as_ref())
            .map(|instruction| instruction.decoded)
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

        let operation = instruction.operation;
        if operation == 7 {
            if is_delay_slot {
                return Err(Self::execution_error(
                    MaxwellThreeDMmeExecutionError::BranchInDelaySlot { address },
                ));
            }
            let value = self.register(u32::from(instruction.src_a));
            let taken = if !instruction.branch_not_zero {
                value == 0
            } else {
                value != 0
            };
            if taken {
                let target = add_signed_18(base, instruction.immediate).ok_or_else(|| {
                    Self::execution_error(MaxwellThreeDMmeExecutionError::ProgramCounterOverflow {
                        address,
                    })
                })?;
                if instruction.branch_annul {
                    self.pc = target;
                    return Ok(true);
                }
                self.delayed_pc = Some(target);
                return self.step(true);
            }
        } else {
            let src_a = self.register(u32::from(instruction.src_a));
            let src_b = self.register(u32::from(instruction.src_b));
            let result = match operation {
                0 => self.alu(address, instruction.alu, src_a, src_b)?,
                1 => src_a.wrapping_add_signed(instruction.immediate),
                2 => {
                    let mask = instruction.mask;
                    let source = (src_b >> u32::from(instruction.shift)) & mask;
                    (src_a & !(mask << u32::from(instruction.destination_shift)))
                        | (source << u32::from(instruction.destination_shift))
                }
                3 => {
                    ((src_b >> (src_a & 0x1f)) & instruction.mask)
                        << u32::from(instruction.destination_shift)
                }
                4 => ((src_b >> u32::from(instruction.shift)) & instruction.mask) << (src_a & 0x1f),
                5 => {
                    let method = src_a.wrapping_add_signed(instruction.immediate);
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
            self.process_result(
                instruction.result_operation,
                usize::from(instruction.destination),
                result,
            )?;
        }

        if instruction.exit && !is_delay_slot {
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

const fn add_signed_18(base: u32, immediate: i32) -> Option<u32> {
    if immediate >= 0 {
        base.checked_add(immediate as u32)
    } else {
        base.checked_sub(immediate.unsigned_abs())
    }
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
    instructions: Arc<InstructionRam>,
    start_address_pointer: MaxwellThreeDRegister<MaxwellThreeDMmeRamAddress>,
    next_start_address_index: Option<MaxwellThreeDMmeRamAddress>,
    start_addresses: Arc<BTreeMap<u32, MaxwellThreeDRegister<MaxwellThreeDMmeRamAddress>>>,
    shadow_ram_control: MaxwellThreeDRegister<MaxwellThreeDMmeShadowRamControl>,
    mutable_method_control: MaxwellThreeDRegister<MaxwellThreeDMutableMethodControl>,
    shadow_registers: super::state::registers::Registers,
    shadow_scratch: BTreeMap<u8, MaxwellThreeDRegister<u32>>,
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
            instructions: Arc::new(InstructionRam::default()),
            start_address_pointer: MaxwellThreeDRegister::default(),
            next_start_address_index: None,
            start_addresses: Arc::new(BTreeMap::new()),
            shadow_ram_control: MaxwellThreeDRegister::verified_reset(
                MaxwellThreeDMmeShadowRamControl::MethodTrack.raw(),
                Some(MaxwellThreeDMmeShadowRamControl::MethodTrack),
            ),
            mutable_method_control: MaxwellThreeDRegister::default(),
            shadow_registers: super::state::registers::Registers::default(),
            shadow_scratch: BTreeMap::new(),
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
                Arc::make_mut(&mut self.instructions).insert(
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
                Arc::make_mut(&mut self.start_addresses).insert(
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

#[cfg(test)]
mod decoded_tests {
    use super::*;
    use std::cell::RefCell;

    #[derive(Default)]
    struct Host(RefCell<Vec<u16>>);
    impl MaxwellThreeDMmeHost for Host {
        type Error = ();
        fn read_register(&self, method: u16) -> Result<u32, ()> {
            self.0.borrow_mut().push(method);
            Ok(42)
        }
        fn emit_method(&mut self, _: u16, _: u32) -> Result<(), ()> {
            Ok(())
        }
    }
    fn word(raw: u32) -> MaxwellThreeDRegister<MaxwellThreeDMmeInstruction> {
        MaxwellThreeDRegister::verified_reset(raw, Some(MaxwellThreeDMmeInstruction::new(raw)))
    }

    #[test]
    fn decoded_programs_cross_pages_and_keep_their_uploaded_version() {
        for start in [15, 0x1234_ffff, u32::MAX - 2] {
            let mut instructions = InstructionRam::default();
            let read = |method: u32| (method << 14) | 5 | (1 << 4) | (1 << 7);
            instructions.insert(start, word(read(0xd00)));
            instructions.insert(start + 1, word(0x11)); // Exit delay slot.
            let mut program = MaxwellThreeDMmeProgram {
                instructions: Arc::new(instructions),
                start_addresses: Arc::new(BTreeMap::from([(
                    0,
                    MaxwellThreeDRegister::verified_reset(
                        start,
                        Some(MaxwellThreeDMmeRamAddress::new(start)),
                    ),
                )])),
            };
            let old = program.clone();
            Arc::make_mut(&mut program.instructions).insert(start, word(read(0xd01)));
            assert_eq!(program.instructions.len(), 2);
            for (snapshot, expected) in [(&old, 0xd00), (&program, 0xd01)] {
                let mut host = Host::default();
                assert!(snapshot.execute(0, &[0], &mut host).is_ok());
                assert_eq!(*host.0.borrow(), [expected]);
            }
        }
    }

    #[test]
    fn missing_words_and_invalid_opcodes_fail_when_executed() {
        let mut ram = InstructionRam::default();
        ram.insert(15, word(0x91)); // Exit with an absent delay slot on the next page.
        ram.insert(100, word(6)); // Invalid opcode must not fail at upload.
        let mut program = MaxwellThreeDMmeProgram {
            instructions: Arc::new(ram),
            start_addresses: Arc::new(BTreeMap::from([(
                0,
                MaxwellThreeDRegister::verified_reset(
                    15,
                    Some(MaxwellThreeDMmeRamAddress::new(15)),
                ),
            )])),
        };
        assert!(matches!(program.execute(0, &[0], &mut Host::default()),
            Err(MaxwellThreeDMmeRunError::Execution(MaxwellThreeDMmeExecutionError::MissingInstruction { address })) if address.raw() == 16));
        Arc::make_mut(&mut program.start_addresses).insert(
            0,
            MaxwellThreeDRegister::verified_reset(100, Some(MaxwellThreeDMmeRamAddress::new(100))),
        );
        assert!(matches!(program.execute(0, &[0], &mut Host::default()),
            Err(MaxwellThreeDMmeRunError::Execution(MaxwellThreeDMmeExecutionError::InvalidOperation { address, operation: 6 })) if address.raw() == 100));
    }
}
