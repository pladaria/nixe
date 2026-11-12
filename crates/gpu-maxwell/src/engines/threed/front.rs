//! Macro-visible registers and shadow RAM belong exclusively to the command thread.
//! Expanded writes cross the ordered boundary; resource effects run on its consumer.
//! https://github.com/zortos293/Lichii/blob/b4b830c8faf1fb5d7b2ad1b892117f4cfd708546/src/video_core/engines/maxwell_3d.cpp#L1380-L1466
use super::mme::{MaxwellThreeDMmeHost, MaxwellThreeDMmeRunError};
use super::*;
use crate::ordered::Writer;
use crate::{MaxwellMethodSink, MaxwellMethodSource};
#[cfg(feature = "performance-counters")]
use std::time::Instant;

pub(crate) struct Registers {
    words: state::MaxwellThreeDRawRegisterFile,
    mme: MaxwellThreeDMmeState,
    notification: Option<crate::MaxwellPushbufferSubchannel>,
    draw_pending: bool,
    instance_loop: bool,
}
impl Registers {
    pub fn new(jit: bool) -> Self {
        Self {
            words: Default::default(),
            mme: MaxwellThreeDMmeState::new(jit),
            notification: None,
            draw_pending: false,
            instance_loop: false,
        }
    }
    pub fn check_puller(
        &self,
        source: MaxwellMethodSource,
    ) -> Result<(), MaxwellEngineStreamError> {
        if let Some(armed) = self.notification
            && (source.method().0 == 0 || source.subchannel() != armed)
        {
            return Err(super::super::pending_notification_switch(source).into());
        }
        Ok(())
    }
    pub fn write<H: MaxwellMethodSink>(
        &mut self,
        source: MaxwellMethodSource,
        writer: &mut Writer<'_, H>,
    ) -> Result<(), MaxwellEngineStreamError> {
        let method = source.method();
        if method.0 >= 0x3800 || method.0 & 3 != 0 {
            return Err(MaxwellEngineDispatchError::UnknownMethod {
                source,
                class_name: CLASS_NAME,
            }
            .into());
        }
        let control = self.mme.shadow_ram_control().value().copied();
        let value = self.mme.resolve_shadow_argument(method, source.argument());
        self.mme.track_shadow_register(control, method, value);
        let same = self.words.word(method) == value;
        if !registers::executable(method) {
            if self.notification.is_some() {
                self.words.write(method, value);
                // An armed notify still requires the full effect diagnostic.
                writer.method(source.emitted_by_mme(method, value))?;
            } else if !same {
                self.words.write(method, value);
                writer.raw(source.subchannel(), method, value)?;
            }
            return Ok(());
        }
        self.words.write(method, value);
        match method.0 {
            0x1618 => match (value >> 26) & 3 {
                0 => self.instance_loop = false,
                1 => self.instance_loop = true,
                _ => {}
            },
            0x1300 | 0x15e8 | 0x15ec => self.instance_loop = false,
            _ => {}
        }
        // Equal vertex counters are state only; END does no work in an
        // instance loop. Index counters still arm indexed drawing on every send.
        // Earlier state writes are already ordered, including their draw flush.
        if same
            && self.notification.is_none()
            && (matches!(method.0, 0x0d74 | 0x0d78)
                || (method.0 == 0x1614 && self.instance_loop && value <= 1))
        {
            return Ok(());
        }
        let resolved = source.emitted_by_mme(method, value);
        if self.notification.is_some() && matches!(method.0, 0x114..=0x124 | 0x1134) {
            let name = METHODS_BY_DWORD[(method.0 / 4) as usize]
                .map_or("MME", |declaration| declaration.metadata.method_name());
            return Err(invalid_encoding(
                resolved,
                name,
                "pending notification requires completion of a method other than NO_OPERATION",
            )
            .into());
        }
        let mut raw = false;
        if matches!(method.0, 0x114..=0x120) {
            if let Some(write) = preflight_mme_state(resolved, &self.mme)? {
                self.mme.apply(write);
            }
            raw = true;
        } else if method.0 == 0x124 {
            let value =
                MaxwellThreeDMmeShadowRamControl::parse(source.argument()).ok_or_else(|| {
                    invalid_encoding(
                        source,
                        "SET_MME_SHADOW_RAM_CONTROL",
                        "invalid shadow RAM mode",
                    )
                })?;
            self.mme
                .apply(MaxwellThreeDMmeStateWrite::ShadowRamControl { value, source });
            raw = true;
        } else if method.0 == 0x1134 {
            let value = MaxwellThreeDMutableMethodControl::parse(value).ok_or_else(|| {
                invalid_encoding(
                    resolved,
                    "SET_MUTABLE_METHOD_CONTROL",
                    "invalid mutable method mode",
                )
            })?;
            self.mme
                .apply(MaxwellThreeDMmeStateWrite::MutableMethodControl { value, source });
            raw = true;
        } else if (0x2390..=0x23fc).contains(&method.0) {
            self.words.write(
                GpuMethodId(0x238c),
                self.words.word(GpuMethodId(0x238c)).wrapping_add(4),
            );
        } else if method.0 == 0x2310 {
            self.words.write(GpuMethodId(0x3400), 1);
        }
        // Side effects and pending primitive boundaries survive equal arguments.
        if raw {
            writer.raw(source.subchannel(), method, value)?;
        } else {
            writer.method(resolved)?;
        }
        self.note_primitive(method);
        if method.0 == 0x120 {
            let pointer = self.mme.next_start_address_index().unwrap().raw();
            let method = GpuMethodId(0x11c);
            // Loading a macro entry advances the visible pointer without writing
            // its shadow. The JIT reads this same register bank directly.
            self.words.write(method, pointer);
            writer.raw(source.subchannel(), method, pointer)?;
        }
        if method.0 == 0x10c {
            self.notification = Some(source.subchannel());
        } else if registers::completes_notification(method) {
            self.notification = None;
        }
        Ok(())
    }
    pub fn multi<H: MaxwellMethodSink>(
        &mut self,
        source: MaxwellMethodSource,
        data: crate::MaxwellDmaArguments<'_>,
        writer: &mut Writer<'_, H>,
    ) -> Result<bool, MaxwellEngineStreamError> {
        if self.notification.is_some() {
            return Ok(false);
        }
        let method = source.method();
        if (0x2390..=0x23fc).contains(&method.0) {
            // A multi-data upload consumes literal words even in replay mode;
            // unlike scalar sends, it does not alter the port or its shadow.
            self.words.write(
                GpuMethodId(0x238c),
                self.words
                    .word(GpuMethodId(0x238c))
                    .wrapping_add(data.len() as u32 * 4),
            );
        } else if matches!(method.0, 0x1b4 | 0x1300 | 0x15e8 | 0x15ec) {
            let control = self.mme.shadow_ram_control().value().copied();
            if control == Some(MaxwellThreeDMmeShadowRamControl::MethodReplay) {
                return Ok(false);
            }
            let last = data.argument_at(data.len() - 1);
            self.words.write(method, last);
            self.mme.track_shadow_register(control, method, last);
        } else {
            return Ok(false);
        }
        if matches!(method.0, 0x1300 | 0x15e8 | 0x15ec) {
            self.instance_loop = false;
        }
        writer.multi(source, data)?;
        self.note_primitive(method);
        Ok(true)
    }
    fn note_primitive(&mut self, method: GpuMethodId) {
        // These methods can leave a deferred instance run for the macro boundary.
        if matches!(
            method.0,
            0x1618 | 0x1614 | 0x1118 | 0x1434 | 0x1438 | 0x19d0 | 0x1300 | 0x15e8 | 0x15ec
        ) {
            self.draw_pending = true;
        }
    }
    pub fn run<H: MaxwellMethodSink>(
        &mut self,
        source: MaxwellMethodSource,
        parameters: &[u32],
        writer: &mut Writer<'_, H>,
    ) -> Result<(), MaxwellEngineStreamError> {
        #[cfg(feature = "performance-counters")]
        let start = Instant::now();
        #[cfg(feature = "performance-counters")]
        let blocked = writer.host.blocked_time();
        let program = self
            .mme
            .program(((source.method().0 - 0x3800) / 8) as u8)
            .map_err(|error| MaxwellEngineDispatchError::MmeExecution { source, error })?;
        let result = program.execute(
            parameters,
            &mut Host {
                registers: self,
                writer,
                source,
            },
        );
        #[cfg(feature = "performance-counters")]
        writer.host.mme_time(
            start
                .elapsed()
                .saturating_sub(writer.host.blocked_time().saturating_sub(blocked)),
        );
        match result {
            Ok(()) => {
                if std::mem::take(&mut self.draw_pending) {
                    writer.end_macro(source)?;
                }
                Ok(())
            }
            Err(MaxwellThreeDMmeRunError::Execution(error)) => {
                Err(MaxwellEngineDispatchError::MmeExecution { source, error }.into())
            }
            Err(MaxwellThreeDMmeRunError::Host(error)) => Err(error),
        }
    }
}
struct Host<'a, 'b, H: MaxwellMethodSink> {
    registers: &'a mut Registers,
    writer: &'a mut Writer<'b, H>,
    source: MaxwellMethodSource,
}
impl<H: MaxwellMethodSink> MaxwellThreeDMmeHost for Host<'_, '_, H> {
    type Error = MaxwellEngineStreamError;
    #[cfg(target_arch = "x86_64")]
    fn register_words_ptr(&mut self) -> *mut u32 {
        self.registers.words.values.as_mut_ptr()
    }
    fn read_register(&self, method_dword: u16) -> Result<u32, Self::Error> {
        self.registers
            .words
            .raw(GpuMethodId(u32::from(method_dword) * 4))
            .ok_or_else(|| {
                MaxwellEngineDispatchError::MmeExecution {
                    source: self.source,
                    error: MaxwellThreeDMmeExecutionError::RegisterReadUnavailable { method_dword },
                }
                .into()
            })
    }
    fn emit_method(&mut self, method_dword: u16, argument: u32) -> Result<(), Self::Error> {
        let method = GpuMethodId(u32::from(method_dword) * 4);
        if is_mme_aperture(method) {
            return Err(MaxwellEngineDispatchError::MmeExecution {
                source: self.source,
                error: MaxwellThreeDMmeExecutionError::RecursiveMacroCall { method_dword },
            }
            .into());
        }
        self.registers
            .write(self.source.emitted_by_mme(method, argument), self.writer)
    }
}

fn preflight_mme_state(
    source: crate::MaxwellMethodSource,
    mme: &MaxwellThreeDMmeState,
) -> Result<Option<MaxwellThreeDMmeStateWrite>, MaxwellEngineDispatchError> {
    let write = match source.method().0 {
        0x0114 => MaxwellThreeDMmeStateWrite::InstructionPointer {
            value: MaxwellThreeDMmeRamAddress::new(source.argument()),
            source,
        },
        0x0118 => {
            let address =
                mme.next_instruction_address()
                    .ok_or(MaxwellEngineDispatchError::MmeRamLoad {
                        source,
                        ram: MaxwellThreeDMmeRam::Instruction,
                        error: MaxwellThreeDMmeLoadError::PointerUnset,
                    })?;
            if address.raw() == u32::MAX {
                return Err(MaxwellEngineDispatchError::MmeRamLoad {
                    source,
                    ram: MaxwellThreeDMmeRam::Instruction,
                    error: MaxwellThreeDMmeLoadError::PointerOverflow,
                });
            }
            MaxwellThreeDMmeStateWrite::Instruction {
                address,
                value: MaxwellThreeDMmeInstruction::new(source.argument()),
                source,
            }
        }
        0x011c => MaxwellThreeDMmeStateWrite::StartAddressPointer {
            value: MaxwellThreeDMmeRamAddress::new(source.argument()),
            source,
        },
        0x0120 => {
            let index =
                mme.next_start_address_index()
                    .ok_or(MaxwellEngineDispatchError::MmeRamLoad {
                        source,
                        ram: MaxwellThreeDMmeRam::StartAddress,
                        error: MaxwellThreeDMmeLoadError::PointerUnset,
                    })?;
            if index.raw() == u32::MAX {
                return Err(MaxwellEngineDispatchError::MmeRamLoad {
                    source,
                    ram: MaxwellThreeDMmeRam::StartAddress,
                    error: MaxwellThreeDMmeLoadError::PointerOverflow,
                });
            }
            if index.raw() as usize >= mme::MME_MACRO_POSITIONS {
                return Err(MaxwellEngineDispatchError::MmeRamLoad {
                    source,
                    ram: MaxwellThreeDMmeRam::StartAddress,
                    error: MaxwellThreeDMmeLoadError::StorageLimitExceeded {
                        limit: mme::MME_MACRO_POSITIONS,
                    },
                });
            }
            MaxwellThreeDMmeStateWrite::StartAddress {
                index,
                address: MaxwellThreeDMmeRamAddress::new(source.argument()),
                source,
            }
        }
        _ => return Ok(None),
    };
    Ok(Some(write))
}
