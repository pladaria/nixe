// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
// MME instruction lowering and ABI strategy follow MacroJITx64Impl.
// https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/macro.cpp
#include "mme.h"
#include <array>
#include <asmjit/x86.h>
#include <bit>
#include <bitset>
#include <cstdio>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
using u32 = uint32_t;
using u64 = uint64_t;
using s32 = int32_t;
namespace {
using namespace asmjit::x86;
struct CompileFailure : std::runtime_error {
  u32 pc;
  CompileFailure(u32 p, const char *m) : std::runtime_error(m), pc(p) {}
};
[[noreturn]] void fail(u32 pc, const char *text) {
  throw CompileFailure(pc, text);
}
void check(asmjit::Error e) {
  if (e != asmjit::Error::kOk)
    throw std::runtime_error(asmjit::stringify_error(e));
}
struct FatalError : asmjit::ErrorHandler {
  void handle_error(asmjit::Error e, const char *,
                    asmjit::BaseEmitter *) override {
    check(e);
  }
};
struct Runtime {
  asmjit::JitRuntime jit;
};
struct Label {
  asmjit::Label id;
};
constexpr size_t MAX_CODE_SIZE = 0x10000;
template <unsigned P, unsigned B, class T> struct Field {
  u32 raw;
  T Value() const {
    if constexpr (std::is_signed_v<T>)
      return std::bit_cast<s32>(raw << (32 - P - B)) >> (32 - B);
    else
      return static_cast<T>((raw >> P) & ((u32(1) << B) - 1));
  }
  operator T() const { return Value(); }
};
namespace Macro {
constexpr std::size_t NUM_MACRO_REGISTERS = 8;
enum class Operation : u32 {
  ALU = 0,
  AddImmediate = 1,
  ExtractInsert = 2,
  ExtractShiftLeftImmediate = 3,
  ExtractShiftLeftRegister = 4,
  Read = 5,
  Unused = 6, // This operation doesn't seem to be a valid encoding.
  Branch = 7,
};

enum class ALUOperation : u32 {
  Add = 0,
  AddWithCarry = 1,
  Subtract = 2,
  SubtractWithBorrow = 3,
  // Operations 4-7 don't seem to be valid encodings.
  Xor = 8,
  Or = 9,
  And = 10,
  AndNot = 11,
  Nand = 12
};

enum class ResultOperation : u32 {
  IgnoreAndFetch = 0,
  Move = 1,
  MoveAndSetMethod = 2,
  FetchAndSend = 3,
  MoveAndSend = 4,
  FetchAndSetMethod = 5,
  MoveAndSetMethodFetchAndSend = 6,
  MoveAndSetMethodSend = 7
};

enum class BranchCondition : u32 {
  Zero = 0,
  NotZero = 1,
};

union Opcode {
  u32 raw;
  Field<0, 3, Operation> operation;
  Field<4, 3, ResultOperation> result_operation;
  Field<4, 1, BranchCondition> branch_condition;
  // If set on a branch, then the branch doesn't have a delay slot.
  Field<5, 1, u32> branch_annul;
  Field<7, 1, u32> is_exit;
  Field<8, 3, u32> dst;
  Field<11, 3, u32> src_a;
  Field<14, 3, u32> src_b;
  // The signed immediate overlaps the second source operand and the alu
  // operation.
  Field<14, 18, s32> immediate;

  Field<17, 5, ALUOperation> alu_operation;

  // Bitfield instructions data
  Field<17, 5, u32> bf_src_bit;
  Field<22, 5, u32> bf_size;
  Field<27, 5, u32> bf_dst_bit;

  u32 GetBitfieldMask() const { return (1 << bf_size) - 1; }

  s32 GetBranchTarget() const {
    return static_cast<s32>(immediate * sizeof(u32));
  }
};

} // namespace Macro

constexpr std::bitset<32> CALLEE_SAVED(0xf028);
constexpr std::bitset<32> CALLER_SAVED((1u << 10) | (1u << 11));
#ifdef _WIN32
const Gp ABI_PARAM1 = rcx, ABI_PARAM2 = rdx, ABI_PARAM3 = r8;
constexpr size_t SHADOW_SPACE = 32;
#else
const Gp ABI_PARAM1 = rdi, ABI_PARAM2 = rsi, ABI_PARAM3 = rdx;
constexpr size_t SHADOW_SPACE = 0;
#endif
size_t frame_size(std::bitset<32> regs, size_t alignment) {
  alignment -= regs.count() * 8;
  return SHADOW_SPACE + ((alignment - SHADOW_SPACE) & 15);
}
void push_abi(asmjit::x86::Assembler &a, std::bitset<32> regs,
              size_t alignment) {
  for (unsigned i = 0; i < 16; ++i)
    if (regs[i])
      check(a.push(gpq(i)));
  auto size = frame_size(regs, alignment);
  if (size)
    check(a.sub(rsp, size));
}
void pop_abi(asmjit::x86::Assembler &a, std::bitset<32> regs,
             size_t alignment) {
  auto size = frame_size(regs, alignment);
  if (size)
    check(a.add(rsp, size));
  for (int i = 15; i >= 0; --i)
    if (regs[size_t(i)])
      check(a.pop(gpq(i)));
}
const asmjit::x86::Gp STATE = asmjit::x86::rbx;
const asmjit::x86::Gp RESULT = asmjit::x86::r10d;
const asmjit::x86::Gp MAX_PARAMETER = asmjit::x86::r11;
const asmjit::x86::Gp PARAMETERS = asmjit::x86::r12;
const asmjit::x86::Gp METHOD_ADDRESS = asmjit::x86::r14d;
const asmjit::x86::Gp BRANCH_HOLDER = asmjit::x86::r15;

struct Compiler final {
  Compiler(Runtime &runtime_, std::span<const u32> code_)
      : runtime(runtime_), code(code_) {
    check(holder.init(runtime.jit.environment(), runtime.jit.cpu_features()));
    holder.set_error_handler(&error_handler);
    assembler = std::make_unique<asmjit::x86::Assembler>(&holder);
    Compile();
  }
  Runtime &runtime;
  asmjit::CodeHolder holder;
  FatalError error_handler;
  std::unique_ptr<asmjit::x86::Assembler> assembler;
  asmjit::Label label(Label &value) {
    if (!value.id.is_valid())
      value.id = assembler->new_label();
    return value.id;
  }
  void bind(Label &value) { check(assembler->bind(label(value))); }
  bool is_bound(Label &value) const {
    return value.id.is_valid() && holder.is_label_bound(value.id);
  }
  void mov_label(asmjit::x86::Gp reg, Label &target) {
    const uint8_t encoding[] = {uint8_t(0x48 | (reg.id() >= 8)),
                                uint8_t(0xb8 | (reg.id() & 7))};
    check(assembler->embed(encoding, sizeof(encoding)));
    check(assembler->embed_label(label(target), 8));
  }
  void fault(u32 kind, u32 address, u32 value) {
    assembler->mov(asmjit::x86::dword_ptr(STATE, offsetof(JITState, error)),
                   kind);
    assembler->mov(asmjit::x86::dword_ptr(STATE, offsetof(JITState, error_pc)),
                   address);
    assembler->mov(
        asmjit::x86::dword_ptr(STATE, offsetof(JITState, error_value)), value);
  }

  void Compile_ALU(Macro::Opcode opcode);
  void Compile_AddImmediate(Macro::Opcode opcode);
  void Compile_ExtractInsert(Macro::Opcode opcode);
  void Compile_ExtractShiftLeftImmediate(Macro::Opcode opcode);
  void Compile_ExtractShiftLeftRegister(Macro::Opcode opcode);
  void Compile_Read(Macro::Opcode opcode);
  void Compile_Branch(Macro::Opcode opcode);

  void Optimizer_ScanFlags();
  void Compile();
  bool Compile_NextInstruction();
  asmjit::x86::Gp Compile_FetchParameter();
  asmjit::x86::Gp Compile_GetRegister(u32 index, asmjit::x86::Gp dst);
  void Compile_ProcessResult(Macro::ResultOperation operation, u32 reg);
  void Compile_Send(asmjit::x86::Gp value);
  Macro::Opcode GetOpCode() const;

  using JITState = NixeMmeState;
  static_assert(offsetof(JITState, bank) == 0, "Maxwell3D is not at 0x0");
  using ProgramType = NixeMmeProgram;

  struct OptimizerState {
    bool can_skip_carry{};
    bool has_delayed_pc{};
    bool zero_reg_skip{};
    bool skip_dummy_addimmediate{};
    bool optimize_for_method_move{};
  };
  OptimizerState optimizer{};
  std::optional<Macro::Opcode> next_opcode{};
  ProgramType program{nullptr};
  std::array<Label, MAX_CODE_SIZE> labels;
  std::array<Label, MAX_CODE_SIZE> delay_skip;
  Label end_of_code{};
  u32 pc{};
  std::span<const u32> code;
};

void Compiler::Compile_ALU(Macro::Opcode opcode) {
  auto src_a = Compile_GetRegister(opcode.src_a, RESULT);
  auto src_b = Compile_GetRegister(opcode.src_b, eax);
  switch (opcode.alu_operation) {
  case Macro::ALUOperation::Add:
    assembler->add(src_a, src_b);
    if (!optimizer.can_skip_carry)
      assembler->setc(
          asmjit::x86::byte_ptr(STATE, offsetof(JITState, carry_flag)));
    break;
  case Macro::ALUOperation::AddWithCarry:
    assembler->bt(asmjit::x86::dword_ptr(STATE, offsetof(JITState, carry_flag)),
                  0);
    assembler->adc(src_a, src_b);
    assembler->setc(
        asmjit::x86::byte_ptr(STATE, offsetof(JITState, carry_flag)));
    break;
  case Macro::ALUOperation::Subtract:
    assembler->sub(src_a, src_b);
    if (!optimizer.can_skip_carry)
      assembler->setnc(
          asmjit::x86::byte_ptr(STATE, offsetof(JITState, carry_flag)));
    break;
  case Macro::ALUOperation::SubtractWithBorrow:
    assembler->bt(asmjit::x86::dword_ptr(STATE, offsetof(JITState, carry_flag)),
                  0);
    assembler->cmc();
    assembler->sbb(src_a, src_b);
    assembler->setnc(
        asmjit::x86::byte_ptr(STATE, offsetof(JITState, carry_flag)));
    break;
  case Macro::ALUOperation::Xor:
    assembler->xor_(src_a, src_b);
    break;
  case Macro::ALUOperation::Or:
    assembler->or_(src_a, src_b);
    break;
  case Macro::ALUOperation::And:
    assembler->and_(src_a, src_b);
    break;
  case Macro::ALUOperation::AndNot:
    assembler->not_(src_b);
    assembler->and_(src_a, src_b);
    break;
  case Macro::ALUOperation::Nand:
    assembler->and_(src_a, src_b);
    assembler->not_(src_a);
    break;
  default:
    fail(pc, "invalid MME ALU operation");
  }
  Compile_ProcessResult(opcode.result_operation, opcode.dst);
}

void Compiler::Compile_AddImmediate(Macro::Opcode opcode) {
  if (optimizer.skip_dummy_addimmediate) {
    // Games tend to use this as an exit instruction placeholder. It's to encode
    // an instruction without doing anything. In our case we can just not emit
    // anything.
    if (opcode.result_operation == Macro::ResultOperation::Move &&
        opcode.dst == 0) {
      return;
    }
  }
  // Check for redundant moves
  if (optimizer.optimize_for_method_move &&
      opcode.result_operation == Macro::ResultOperation::MoveAndSetMethod) {
    if (next_opcode.has_value()) {
      const auto next = *next_opcode;
      if (opcode.dst == 0 &&
          next.result_operation == Macro::ResultOperation::MoveAndSetMethod &&
          opcode.dst == next.dst) {
        return;
      }
    }
  }
  if (optimizer.zero_reg_skip && opcode.src_a == 0) {
    if (opcode.immediate.Value() == 0) {
      assembler->xor_(RESULT, RESULT);
    } else {
      assembler->mov(RESULT, opcode.immediate.Value());
    }
  } else {
    auto result = Compile_GetRegister(opcode.src_a, RESULT);
    if (opcode.immediate.Value() > 1) {
      assembler->add(result, opcode.immediate.Value());
    } else if (opcode.immediate.Value() == 1) {
      assembler->inc(result);
    } else if (opcode.immediate.Value() < 0) {
      assembler->sub(result, opcode.immediate.Value() * -1);
    }
  }
  Compile_ProcessResult(opcode.result_operation, opcode.dst);
}

void Compiler::Compile_ExtractInsert(Macro::Opcode opcode) {
  auto dst = Compile_GetRegister(opcode.src_a, RESULT);
  auto src = Compile_GetRegister(opcode.src_b, eax);

  const u32 mask = ~(opcode.GetBitfieldMask() << opcode.bf_dst_bit.Value());
  assembler->and_(dst, mask);
  assembler->shr(src, opcode.bf_src_bit.Value());
  assembler->and_(src, opcode.GetBitfieldMask());
  assembler->shl(src, opcode.bf_dst_bit.Value());
  assembler->or_(dst, src);

  Compile_ProcessResult(opcode.result_operation, opcode.dst);
}

void Compiler::Compile_ExtractShiftLeftImmediate(Macro::Opcode opcode) {
  const auto dst = Compile_GetRegister(opcode.src_a, ecx);
  const auto src = Compile_GetRegister(opcode.src_b, RESULT);

  assembler->shr(src, dst.r8());
  assembler->and_(src, opcode.GetBitfieldMask());
  assembler->shl(src, opcode.bf_dst_bit.Value());

  Compile_ProcessResult(opcode.result_operation, opcode.dst);
}

void Compiler::Compile_ExtractShiftLeftRegister(Macro::Opcode opcode) {
  const auto dst = Compile_GetRegister(opcode.src_a, ecx);
  const auto src = Compile_GetRegister(opcode.src_b, RESULT);

  assembler->shr(src, opcode.bf_src_bit.Value());
  assembler->and_(src, opcode.GetBitfieldMask());
  assembler->shl(src, dst.r8());

  Compile_ProcessResult(opcode.result_operation, opcode.dst);
}

void Compiler::Compile_Read(Macro::Opcode opcode) {
  if (optimizer.zero_reg_skip && opcode.src_a == 0) {
    if (opcode.immediate.Value() == 0) {
      assembler->xor_(RESULT, RESULT);
    } else {
      assembler->mov(RESULT, opcode.immediate.Value());
    }
  } else {
    auto result = Compile_GetRegister(opcode.src_a, RESULT);
    if (opcode.immediate.Value() > 1) {
      assembler->add(result, opcode.immediate.Value());
    } else if (opcode.immediate.Value() == 1) {
      assembler->inc(result);
    } else if (opcode.immediate.Value() < 0) {
      assembler->sub(result, opcode.immediate.Value() * -1);
    }
  }

  if (opcode.src_a != 0) {
    Label valid;
    assembler->cmp(RESULT, NIXE_MME_BANK_WORDS);
    assembler->long_().jb(label(valid));
    fault(2, pc, 0);
    assembler->mov(
        asmjit::x86::dword_ptr(STATE, offsetof(JITState, error_value)), RESULT);
    assembler->long_().jmp(label(end_of_code));
    bind(valid);
  } else if (u32(opcode.immediate.Value()) >= NIXE_MME_BANK_WORDS) {
    fail(pc, "MME register read is outside the canonical bank");
  }
  assembler->mov(rax, asmjit::x86::qword_ptr(STATE, offsetof(JITState, bank)));
  assembler->mov(RESULT, asmjit::x86::dword_ptr(rax, RESULT.r64(), 2));
  Compile_ProcessResult(opcode.result_operation, opcode.dst);
}

void Compiler::Compile_Send(asmjit::x86::Gp value) {
  push_abi(*assembler, CALLER_SAVED, 0);
  assembler->mov(ABI_PARAM1,
                 asmjit::x86::qword_ptr(STATE, offsetof(JITState, host)));
  assembler->mov(ABI_PARAM2.r32(), METHOD_ADDRESS);
  assembler->mov(ABI_PARAM3.r32(), value);
  assembler->mov(rax, asmjit::x86::qword_ptr(STATE, offsetof(JITState, emit)));
  assembler->call(rax);
  pop_abi(*assembler, CALLER_SAVED, 0);
  assembler->test(eax, eax);
  assembler->long_().je(label(end_of_code));

  Label dont_process{};
  // Get increment
  assembler->test(METHOD_ADDRESS, 0x3f000);
  // If zero, method address doesn't update
  assembler->short_().je(label(dont_process));

  assembler->mov(ecx, METHOD_ADDRESS);
  assembler->and_(METHOD_ADDRESS, 0xfff);
  assembler->shr(ecx, 12);
  assembler->and_(ecx, 0x3f);
  assembler->lea(eax, asmjit::x86::ptr(rcx, METHOD_ADDRESS.r64()));
  assembler->and_(eax, 0xfff);
  assembler->shl(ecx, 12);
  assembler->or_(eax, ecx);

  assembler->mov(METHOD_ADDRESS, eax);

  bind(dont_process);
}

void Compiler::Compile_Branch(Macro::Opcode opcode) {

  const s32 jump_address =
      s32(pc) + s32(opcode.GetBranchTarget() / sizeof(s32));
  if (jump_address < 0 || size_t(jump_address) >= code.size())
    fail(pc, "MME branch leaves the uploaded body");

  Label end;
  auto value = Compile_GetRegister(opcode.src_a, eax);
  assembler->cmp(value, 0); // assembler->test(value, value);
  if (optimizer.has_delayed_pc) {
    switch (opcode.branch_condition) {
    case Macro::BranchCondition::Zero:
      assembler->long_().jne(label(end));
      break;
    case Macro::BranchCondition::NotZero:
      assembler->long_().je(label(end));
      break;
    }

    if (opcode.branch_annul) {
      assembler->xor_(BRANCH_HOLDER, BRANCH_HOLDER);
      assembler->long_().jmp(label(labels[jump_address]));
    } else {
      Label handle_post_exit{};
      Label skip{};
      assembler->long_().jmp(label(skip));

      bind(handle_post_exit);
      assembler->xor_(BRANCH_HOLDER, BRANCH_HOLDER);
      assembler->long_().jmp(label(labels[jump_address]));

      bind(skip);
      mov_label(BRANCH_HOLDER, handle_post_exit);
      assembler->long_().jmp(label(delay_skip[pc]));
    }
  } else {
    switch (opcode.branch_condition) {
    case Macro::BranchCondition::Zero:
      assembler->long_().je(label(labels[jump_address]));
      break;
    case Macro::BranchCondition::NotZero:
      assembler->long_().jne(label(labels[jump_address]));
      break;
    }
  }

  bind(end);
}

void Compiler::Optimizer_ScanFlags() {
  optimizer.can_skip_carry = true;
  optimizer.has_delayed_pc = false;
  for (auto raw_op : code) {
    Macro::Opcode op{};
    op.raw = raw_op;

    if (op.operation == Macro::Operation::ALU) {
      // Scan for any ALU operations which actually use the carry flag, if they
      // don't exist in our current code we can skip emitting the carry flag
      // handling operations
      if (op.alu_operation == Macro::ALUOperation::AddWithCarry ||
          op.alu_operation == Macro::ALUOperation::SubtractWithBorrow) {
        optimizer.can_skip_carry = false;
      }
    }

    if (op.operation == Macro::Operation::Branch) {
      if (!op.branch_annul) {
        optimizer.has_delayed_pc = true;
      }
    }
  }
}

void Compiler::Compile() {
  // Matching PROTECT_RE needed for W^X systems
  // CodeHolder is writable until runtime.add installs executable code.
  labels.fill(Label());

  push_abi(*assembler, CALLEE_SAVED, 8);
  // JIT state
  assembler->mov(STATE, ABI_PARAM1);
  assembler->mov(PARAMETERS, ABI_PARAM2);
  assembler->mov(MAX_PARAMETER, ABI_PARAM3);
  assembler->xor_(RESULT, RESULT);
  assembler->xor_(METHOD_ADDRESS, METHOD_ADDRESS);
  assembler->xor_(BRANCH_HOLDER, BRANCH_HOLDER);

  assembler->mov(
      asmjit::x86::dword_ptr(STATE, offsetof(JITState, registers) + 4),
      Compile_FetchParameter());

  // Track get register for zero registers and mark it as no-op
  optimizer.zero_reg_skip = true;

  // AddImmediate tends to be used as a NOP instruction, if we detect this we
  // can completely skip the entire code path and no emit anything
  optimizer.skip_dummy_addimmediate = true;

  // Consecutive method selections can discard an overwritten zero-register
  // result while retaining writes to live MME registers.
  optimizer.optimize_for_method_move = true;

  // Check to see if we can skip emitting certain instructions
  Optimizer_ScanFlags();

  const u32 op_count = static_cast<u32>(code.size());
  for (u32 i = 0; i < op_count; i++) {
    if (i < op_count - 1) {
      pc = i + 1;
      next_opcode = GetOpCode();
    } else {
      next_opcode = {};
    }
    pc = i;
    Compile_NextInstruction();
  }

  fault(3, u32(code.size()), 0);
  bind(end_of_code);
  assembler->mov(
      asmjit::x86::qword_ptr(STATE, offsetof(JITState, next_parameter)),
      PARAMETERS);

  pop_abi(*assembler, CALLEE_SAVED, 8);
  assembler->ret();
  check(runtime.jit.add(&program, &holder));
}

bool Compiler::Compile_NextInstruction() {
  const auto opcode = GetOpCode();
  if (is_bound(labels[pc])) {
    return false;
  }

  bind(labels[pc]);

  switch (opcode.operation) {
  case Macro::Operation::ALU:
    Compile_ALU(opcode);
    break;
  case Macro::Operation::AddImmediate:
    Compile_AddImmediate(opcode);
    break;
  case Macro::Operation::ExtractInsert:
    Compile_ExtractInsert(opcode);
    break;
  case Macro::Operation::ExtractShiftLeftImmediate:
    Compile_ExtractShiftLeftImmediate(opcode);
    break;
  case Macro::Operation::ExtractShiftLeftRegister:
    Compile_ExtractShiftLeftRegister(opcode);
    break;
  case Macro::Operation::Read:
    Compile_Read(opcode);
    break;
  case Macro::Operation::Branch:
    Compile_Branch(opcode);
    break;
  default:
    fail(pc, "unsupported MME instruction encoding");
    break;
  }

  if (optimizer.has_delayed_pc) {
    if (opcode.is_exit) {
      mov_label(rax, end_of_code);
      assembler->test(BRANCH_HOLDER, BRANCH_HOLDER);
      assembler->cmove(BRANCH_HOLDER, rax);
      // Jump to next instruction to skip delay slot check
      assembler->long_().je(label(labels[pc + 1]));
    } else {
      // TODO(ogniK): Optimize delay slot branching
      Label no_delay_slot{};
      assembler->test(BRANCH_HOLDER, BRANCH_HOLDER);
      assembler->long_().je(label(no_delay_slot));
      assembler->mov(rax, BRANCH_HOLDER);
      assembler->xor_(BRANCH_HOLDER, BRANCH_HOLDER);
      assembler->jmp(rax);
      bind(no_delay_slot);
    }
    bind(delay_skip[pc]);
    if (opcode.is_exit) {
      return false;
    }
  } else {
    assembler->test(BRANCH_HOLDER, BRANCH_HOLDER);
    assembler->long_().jne(label(end_of_code));
    if (opcode.is_exit) {
      assembler->inc(BRANCH_HOLDER);
      return false;
    }
  }
  return true;
}

asmjit::x86::Gp Compiler::Compile_FetchParameter() {
  Label parameter_ok;
  assembler->cmp(PARAMETERS, MAX_PARAMETER);
  assembler->long_().jb(label(parameter_ok));
  fault(1, pc, 0);
  assembler->long_().jmp(label(end_of_code));
  bind(parameter_ok);
  assembler->mov(eax, asmjit::x86::dword_ptr(PARAMETERS));
  assembler->add(PARAMETERS, sizeof(u32));
  return eax;
}

asmjit::x86::Gp Compiler::Compile_GetRegister(u32 index, asmjit::x86::Gp dst) {
  if (index == 0) {
    // Register 0 is always zero
    assembler->xor_(dst, dst);
  } else {
    assembler->mov(dst,
                   asmjit::x86::dword_ptr(STATE, offsetof(JITState, registers) +
                                                     index * sizeof(u32)));
  }

  return dst;
}

void Compiler::Compile_ProcessResult(Macro::ResultOperation operation,
                                     u32 reg) {
  const auto SetRegister = [this](u32 reg_index,
                                  const asmjit::x86::Gp &result) {
    // Register 0 is supposed to always return 0. NOP is implemented as a store
    // to the zero register.
    if (reg_index == 0) {
      return;
    }
    assembler->mov(asmjit::x86::dword_ptr(STATE, offsetof(JITState, registers) +
                                                     reg_index * sizeof(u32)),
                   result);
  };
  const auto SetMethodAddress = [this](const asmjit::x86::Gp &reg32) {
    assembler->mov(METHOD_ADDRESS, reg32);
  };

  switch (operation) {
  case Macro::ResultOperation::IgnoreAndFetch:
    SetRegister(reg, Compile_FetchParameter());
    break;
  case Macro::ResultOperation::Move:
    SetRegister(reg, RESULT);
    break;
  case Macro::ResultOperation::MoveAndSetMethod:
    SetRegister(reg, RESULT);
    SetMethodAddress(RESULT);
    break;
  case Macro::ResultOperation::FetchAndSend:
    // Fetch parameter and send result.
    SetRegister(reg, Compile_FetchParameter());
    Compile_Send(RESULT);
    break;
  case Macro::ResultOperation::MoveAndSend:
    // Move and send result.
    SetRegister(reg, RESULT);
    Compile_Send(RESULT);
    break;
  case Macro::ResultOperation::FetchAndSetMethod:
    // Fetch parameter and use result as Method Address.
    SetRegister(reg, Compile_FetchParameter());
    SetMethodAddress(RESULT);
    break;
  case Macro::ResultOperation::MoveAndSetMethodFetchAndSend:
    // Move result and use as Method Address, then fetch and send parameter.
    SetRegister(reg, RESULT);
    SetMethodAddress(RESULT);
    Compile_Send(Compile_FetchParameter());
    break;
  case Macro::ResultOperation::MoveAndSetMethodSend:
    // Move result and use as Method Address, then send bits 12:17 of result.
    SetRegister(reg, RESULT);
    SetMethodAddress(RESULT);
    assembler->shr(RESULT, 12);
    assembler->and_(RESULT, 0b111111);
    Compile_Send(RESULT);
    break;
  default:
    fail(pc, "unsupported MME instruction encoding");
    break;
  }
}

Macro::Opcode Compiler::GetOpCode() const {
  if (pc >= code.size())
    fail(pc, "MME instruction is outside the uploaded body");
  return {code[pc]};
}

} // namespace
extern "C" void *nixe_mme_runtime_new() noexcept {
  try {
    return new Runtime;
  } catch (...) {
    return nullptr;
  }
}
extern "C" void nixe_mme_runtime_delete(void *runtime) noexcept {
  delete static_cast<Runtime *>(runtime);
}
extern "C" bool nixe_mme_compile(void *runtime, const u32 *words, size_t count,
                                 NixeMmeProgram *result,
                                 NixeMmeCompileError *error) noexcept {
  *result = nullptr;
  try {
    if (count == 0 || count >= 0x10000)
      fail(0,
           "MME body size is outside the supported instruction address space");
    auto compiler = std::make_unique<Compiler>(*static_cast<Runtime *>(runtime),
                                               std::span(words, count));
    *result = compiler->program;
    return true;
  } catch (const CompileFailure &e) {
    error->pc = e.pc;
    std::snprintf(error->message, sizeof(error->message), "%s", e.what());
  } catch (const std::exception &e) {
    error->pc = 0;
    std::snprintf(error->message, sizeof(error->message), "%s", e.what());
  }
  return false;
}
extern "C" void nixe_mme_release(void *runtime,
                                 NixeMmeProgram program) noexcept {
  static_cast<Runtime *>(runtime)->jit.release(program);
}
