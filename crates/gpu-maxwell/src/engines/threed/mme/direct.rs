//! Direct parameter/constant method programs. Unknown dataflow retains ISA execution.
use super::*;

#[derive(Clone, Copy, Debug)]
enum Value {
    Constant(u32),
    Parameter(usize),
}
#[derive(Debug)]
enum Operation {
    Parameter(usize),
    Send(u16, Value),
}
#[derive(Debug)]
pub(super) struct DirectProgram {
    operations: Box<[Operation]>,
    parameters: usize,
}
impl DirectProgram {
    pub(super) fn compile(ram: &MmeInstructionRam, mut pc: u32) -> Option<Self> {
        let mut registers = [Value::Constant(0); MME_REGISTER_COUNT];
        registers[1] = Value::Parameter(0);
        let mut parameters = 1;
        let mut method = 0u16;
        let mut increment = 0u16;
        let mut operations = Vec::new();
        let mut delay = false;
        // This fast path represents complete finite programs, including the exit
        // delay instruction. It never bypasses either interpreter watchdog.
        // Eden also caches direct macros by uploaded program contents:
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/macro.cpp
        for _ in 0..256 {
            let instruction = ram.page(pc)?.decoded[pc as usize % MME_INSTRUCTION_PAGE_WORDS]?;
            pc = pc.checked_add(1)?;
            let a = registers[instruction.src_a as usize];
            let b = registers[instruction.src_b as usize];
            let value = match instruction.operation {
                1 => match a {
                    Value::Constant(value) => {
                        Value::Constant(value.wrapping_add_signed(instruction.immediate))
                    }
                    _ if instruction.immediate == 0 => a,
                    _ => return None,
                },
                0 => match (instruction.alu, a, b) {
                    (0 | 8 | 9, Value::Constant(0), value) => value,
                    (0 | 2 | 8 | 9, value, Value::Constant(0)) => value,
                    (10, value, Value::Constant(u32::MAX)) => value,
                    (10, Value::Constant(u32::MAX), value) => value,
                    (op, Value::Constant(a), Value::Constant(b)) => Value::Constant(match op {
                        0 => a.wrapping_add(b),
                        2 => a.wrapping_sub(b),
                        8 => a ^ b,
                        9 => a | b,
                        10 => a & b,
                        11 => a & !b,
                        12 => !(a & b),
                        _ => return None,
                    }),
                    _ => return None,
                },
                _ => return None,
            };
            let destination = instruction.destination as usize;
            let result = instruction.result;
            let fetched = if matches!(result, 0 | 3 | 5 | 6) {
                operations.push(Operation::Parameter(parameters));
                let value = Value::Parameter(parameters);
                parameters += 1;
                Some(value)
            } else {
                None
            };
            if destination != 0 {
                registers[destination] = fetched.filter(|_| result != 6).unwrap_or(value);
            }
            if matches!(result, 2 | 5 | 6 | 7) {
                let Value::Constant(raw) = value else {
                    return None;
                };
                method = (raw & MME_METHOD_DWORD_MASK) as u16;
                increment = ((raw >> 12) & 0x3f) as u16;
            }
            let sent = match result {
                3 | 4 => Some(value),
                6 => fetched,
                7 => {
                    let Value::Constant(raw) = value else {
                        return None;
                    };
                    Some(Value::Constant((raw >> 12) & 0x3f))
                }
                _ => None,
            };
            if let Some(value) = sent {
                operations.push(Operation::Send(method, value));
                method = method.wrapping_add(increment) & MME_METHOD_DWORD_MASK as u16;
            }
            if delay {
                return Some(Self {
                    operations: operations.into(),
                    parameters,
                });
            }
            delay = instruction.exit;
        }
        None
    }

    pub(super) async fn execute<H: MaxwellThreeDMmeHost>(
        &self,
        parameters: &[u32],
        host: &mut H,
    ) -> Result<(), MaxwellThreeDMmeRunError<H::Error>> {
        for operation in &self.operations {
            match *operation {
                Operation::Parameter(index) => {
                    if index >= parameters.len() {
                        return Err(MaxwellThreeDMmeRunError::Execution(
                            MaxwellThreeDMmeExecutionError::ParameterUnavailable { index },
                        ));
                    }
                }
                Operation::Send(method, value) => {
                    let value = match value {
                        Value::Constant(value) => value,
                        Value::Parameter(index) => parameters[index],
                    };
                    if !host
                        .emit_register(method, value)
                        .map_err(MaxwellThreeDMmeRunError::Host)?
                    {
                        host.emit_method(method, value)
                            .await
                            .map_err(MaxwellThreeDMmeRunError::Host)?;
                    }
                }
            }
        }
        if parameters.len() != self.parameters {
            return Err(MaxwellThreeDMmeRunError::Execution(
                MaxwellThreeDMmeExecutionError::UnconsumedParameters {
                    consumed: self.parameters,
                    supplied: parameters.len(),
                },
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct Host {
        sent: Vec<(u16, u32)>,
        fail_at: Option<usize>,
    }
    impl MaxwellThreeDMmeHost for Host {
        type Error = usize;
        fn read_register(&self, _: u16) -> Result<u32, usize> {
            panic!("direct programs contain no register reads")
        }
        fn emit_register(&mut self, _: u16, _: u32) -> Result<bool, usize> {
            Ok(false)
        }
        async fn emit_method(&mut self, method: u16, value: u32) -> Result<(), usize> {
            if self.fail_at == Some(self.sent.len()) {
                return Err(23);
            }
            self.sent.push((method, value));
            Ok(())
        }
    }
    fn ram(words: &[u32]) -> MmeInstructionRam {
        let mut ram = MmeInstructionRam::default();
        for (pc, &word) in words.iter().enumerate() {
            let page = ram
                .pages
                .entry(pc as u32 / MME_INSTRUCTION_PAGE_WORDS as u32)
                .or_insert_with(|| MmeInstructionPage {
                    registers: vec![None; MME_INSTRUCTION_PAGE_WORDS].into(),
                    decoded: vec![None; MME_INSTRUCTION_PAGE_WORDS].into(),
                });
            page.decoded[pc % MME_INSTRUCTION_PAGE_WORDS] = Some(MmeInstruction::decode(word));
            ram.count += 1;
        }
        ram
    }
    fn interpreted(
        program: &MaxwellThreeDMmeProgram,
        parameters: &[u32],
        host: &mut Host,
    ) -> Result<(), MaxwellThreeDMmeRunError<usize>> {
        let mut interpreter = MaxwellThreeDMmeInterpreter {
            program,
            instruction_page: None,
            instruction_page_index: 0,
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
        while let Some((method, value)) = interpreter.run_until_effect()? {
            crate::test_memory::ready(interpreter.host.emit_method(method, value))
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
    fn normalized(result: Result<(), MaxwellThreeDMmeRunError<usize>>) -> Result<(), String> {
        result.map_err(|error| match error {
            MaxwellThreeDMmeRunError::Execution(e) => format!("execution:{e:?}"),
            MaxwellThreeDMmeRunError::Host(e) => format!("host:{e}"),
        })
    }
    fn addi(destination: u32, source: u32, immediate: i32, result: u32, exit: bool) -> u32 {
        ((immediate as u32 & 0x3ffff) << 14)
            | (source << 11)
            | (destination << 8)
            | (result << 4)
            | 1
            | (u32::from(exit) << 7)
    }
    #[test]
    fn direct_programs_match_interpreter_parameters_methods_delays_and_partial_errors() {
        let mut programs = vec![vec![
            addi(0, 0, 0x1ffe, 2, false),
            addi(1, 1, 0, 3, false),
            addi(1, 1, 0, 3, true),
            addi(0, 1, 0, 4, false),
        ]];
        for result in 0..8 {
            let source = if matches!(result, 0 | 3 | 4) { 1 } else { 0 };
            let immediate = if source == 0 { 0x2fff } else { 0 };
            programs.push(vec![
                addi(0, 0, 0x3ffe, 2, false),
                addi(2, source, immediate, result, false),
                addi(0, 2, 0, 4, true),
                addi(0, 1, 0, 4, false),
            ]);
        }
        for words in programs {
            let program = MaxwellThreeDMmeProgram {
                instructions: Arc::new(ram(&words)),
                start_addresses: Arc::default(),
            };
            let direct = DirectProgram::compile(&program.instructions, 0).unwrap();
            for count in 1..=5 {
                let parameters: Vec<_> = (0..count)
                    .map(|i| (i as u32).wrapping_mul(0xf1234567).wrapping_add(0xdeadbeef))
                    .collect();
                for fail_at in [None, Some(0), Some(1), Some(2), Some(3)] {
                    let mut expected = Host {
                        fail_at,
                        ..Default::default()
                    };
                    let mut actual = Host {
                        fail_at,
                        ..Default::default()
                    };
                    let reference = normalized(interpreted(&program, &parameters, &mut expected));
                    assert_eq!(
                        normalized(crate::test_memory::ready(
                            direct.execute(&parameters, &mut actual)
                        )),
                        reference
                    );
                    assert_eq!(actual.sent, expected.sent);
                }
            }
        }
    }
    #[test]
    fn unproven_control_flow_register_reads_and_missing_delay_keep_isa_execution() {
        for words in [
            vec![addi(0, 0, 0, 1, true)],
            vec![7, 0x91, 0x11],
            vec![5 | (1 << 7), 0x11],
            vec![(1 << 17) | (1 << 7), 0x11],
        ] {
            assert!(DirectProgram::compile(&ram(&words), 0).is_none());
        }
    }
}
