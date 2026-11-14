// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
// TU104 lowering and control-flow specialization:
// https://github.com/zortos293/Lichii/blob/646b5d17135a2b2701ccc7a959afd0bbbf2f6160/src/video_core/macro_turing_jit.h
// ISA:
// https://gitlab.freedesktop.org/mesa/mesa/-/blob/main/src/nouveau/mme/mme_tu104_sim.c
#include "ampere.h"
#include <array>
#include <bit>
#include <cstdio>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace {
using namespace asmjit;
using namespace asmjit::x86;
using u32 = uint32_t;
using State = AmpereInvocation;
using Inst = AmpereInstruction;
constexpr const char *predicates[] = {
    "UUUU", "TTTT", "FFFF", "TTUU", "FFUU", "TFUU", "TUUU", "FUUU",
    "UUTT", "UUTF", "UUTU", "UUFT", "UUFF", "UUFU", "UUUT", "UUUF"};
#ifdef _WIN32
const Gp arg0 = rcx, arg1 = rdx, arg2 = r8;
#else
const Gp arg0 = rdi, arg1 = rsi, arg2 = rdx;
#endif
void check(Error error) {
  if (error != Error::kOk)
    throw std::runtime_error(stringify_error(error));
}
struct Errors : ErrorHandler {
  void handle_error(Error error, const char *, BaseEmitter *) override {
    check(error);
  }
};
struct Fault {
  Label label;
  u32 pc, error;
  std::optional<u32> value;
};
struct Compiler {
  NixeMmeRuntime &runtime;
  Errors errors;
  CodeHolder holder;
  Assembler a;
  std::span<const Inst> code;
  std::span<const uint8_t> uploaded;
  std::vector<Label> labels;
  std::vector<bool> after_end, loop_end;
  std::vector<Fault> faults;
  Label table, dispatch, success, leave;
  u32 pc = 0;
  std::vector<std::array<bool, 2>> carry_live;

  Compiler(NixeMmeRuntime &runtime_, std::span<const Inst> code_,
           std::span<const uint8_t> uploaded_)
      : runtime(runtime_), a(nullptr), code(code_), uploaded(uploaded_) {
    check(holder.init(runtime.jit.environment(), runtime.jit.cpu_features()));
    holder.set_error_handler(&errors);
    check(holder.attach(&a));
    for (size_t i = 0; i < code.size(); ++i)
      labels.push_back(a.new_label());
    table = a.new_label();
    dispatch = a.new_label();
    success = a.new_label();
    leave = a.new_label();
  }
  static Mem reg(u32 r) {
    return dword_ptr(rbx, offsetof(State, regs) + r * 4);
  }
  static Mem word(size_t offset) { return dword_ptr(rbx, offset); }
  static Mem byte(size_t offset) { return byte_ptr(rbx, offset); }
  static Mem pointer(size_t offset) { return qword_ptr(rbx, offset); }
  Label fault(u32 error, std::optional<u32> value = {}) {
    Label label = a.new_label();
    faults.push_back({label, pc, error, value});
    return label;
  }
  static bool flow(u32 op) { return op >= 23 && op <= 29; }
  static std::optional<u32> constant(const Inst &d, u32 r, u32 lane) {
    switch (r) {
    case 24:
      return 0;
    case 25:
      return u32(int32_t(int16_t(d.alu[lane].imm)));
    case 26:
      return u32(int32_t(int16_t(d.alu[1 - lane].imm)));
    case 27:
      return (u32(d.alu[0].imm) << 16) | d.alu[1].imm;
    default:
      return {};
    }
  }
  void get(Gp dst, const Inst &d, u32 r, u32 lane) {
    if (r < 24) {
      a.mov(dst, reg(r));
      return;
    }
    if (auto value = constant(d, r, lane)) {
      if (*value)
        a.mov(dst, *value);
      else
        a.xor_(dst, dst);
      return;
    }
    if (r == 28)
      a.mov(dst, r12d);
    else if (r == 29)
      a.mov(dst, r13d);
    else
      a.jmp(fault(2));
  }
  void require(u32 mask) {
    if (!mask)
      return;
    a.mov(eax, word(offsetof(State, initialized)));
    a.not_(eax);
    a.and_(eax, mask);
    // On this fault eax carries precisely the missing register bits.
    a.jnz(fault(12));
  }
  void analyze() {
    loop_end.assign(code.size(), false);
    after_end.assign(code.size(), false);
    auto mark = [&](std::vector<bool> &set, int64_t target) {
      if (target >= 0 && size_t(target) < code.size())
        set[size_t(target)] = true;
    };
    for (u32 i = 0; i < code.size(); ++i) {
      if (uploaded[i] != 7)
        continue;
      const auto &d = code[i];
      for (const auto &alu : d.alu) {
        if (alu.op == 23 && alu.imm >= 2) {
          const u32 end = i + alu.imm - 1;
          mark(loop_end, end);
          if (end < code.size() && uploaded[end] == 7 && code[end].end)
            mark(after_end, i + 1);
        }
      }
      if (!d.end)
        continue;
      mark(after_end, i + 1);
      for (const auto &alu : d.alu) {
        if (alu.op == 23)
          mark(after_end, i + alu.imm);
        else if (alu.op == 24)
          mark(after_end, i + (alu.imm & 0x7fff));
        else if (alu.op >= 25 && alu.op <= 29)
          mark(after_end, int64_t(i) + (int32_t(u32(alu.imm) << 19) >> 19));
      }
    }
  }
  static u32 predicate_mask(const Inst &d, bool value) {
    u32 result = 0;
    for (u32 i = 0; i < 4; ++i)
      if (predicates[d.predicate][i] != (value ? 'T' : 'F'))
        result |= 1u << i;
    return result;
  }
  // Carry is a full word (also the high multiply result). Propagate liveness
  // through both predicate outcomes and every possible branch/loop successor.
  // Analysis is compile-time only; unrelated macros need not preserve carry.
  void analyze_carry() {
    std::vector<std::vector<u32>> successors(code.size());
    auto edge = [&](u32 from, int64_t to) {
      if (to >= 0 && size_t(to) < code.size())
        successors[from].push_back(u32(to));
    };
    for (u32 i = 0; i < code.size(); ++i) {
      if (uploaded[i] != 7)
        continue;
      edge(i, i + 1);
      for (const auto &v : code[i].alu) {
        if (v.op == 23 && v.imm >= 2) {
          edge(i, i + v.imm);
          const u32 end = i + v.imm - 1;
          if (end < code.size())
            edge(end, i + 1);
        } else if (v.op == 24)
          edge(i, i + (v.imm & 0x7fff));
        else if (v.op >= 25 && v.op <= 29)
          edge(i, int64_t(i) + (int32_t(u32(v.imm) << 19) >> 19));
      }
    }
    std::vector<bool> input(code.size(), false);
    carry_live.resize(code.size());
    bool changed;
    do {
      changed = false;
      for (size_t k = code.size(); k-- > 0;) {
        if (uploaded[k] != 7)
          continue;
        bool output_live = false, input_live = false;
        for (u32 next : successors[k])
          output_live |= input[next];
        for (bool predicate : {false, true}) {
          bool live = output_live;
          const u32 mask = predicate_mask(code[k], predicate);
          for (int lane = 1; lane >= 0; --lane) {
            if (!(mask & (1u << lane)))
              continue;
            const u32 op = code[k].alu[lane].op;
            if (op == 0 || op == 2 || op == 4 || op == 6) {
              carry_live[k][lane] = carry_live[k][lane] || live;
              live = false;
            } else if (op == 1 || op == 3 || op == 5)
              live = true;
          }
          input_live |= live;
        }
        if (input_live && !input[k]) {
          input[k] = true;
          changed = true;
        }
      }
    } while (changed);
  }
  AmpereProgram generate() {
    analyze();
    analyze_carry();
    // Six preserved registers and 40 bytes keep both ABIs aligned at call
    // sites, including the Windows shadow space. r12/r13 hold loads, r14/r15
    // results.
    for (Gp r : {rbx, rbp, r12, r13, r14, r15})
      a.push(r);
    a.sub(rsp, 40);
    a.mov(rbx, arg0);
    a.mov(eax, word(offsetof(State, start)));
    auto initial = a.new_label();
    a.jmp(initial);
    a.bind(dispatch);
    a.sub(word(offsetof(State, budget)), 1);
    auto ready = a.new_label();
    a.jnz(ready);
    a.mov(word(offsetof(State, error_pc)), eax);
    a.mov(ebp, eax);
    a.mov(arg0, rbx);
    a.call(pointer(offsetof(State, poll)));
    a.test(eax, eax);
    auto polled = a.new_label();
    a.jnz(polled);
    a.mov(eax, 10);
    a.jmp(leave);
    a.bind(polled);
    a.mov(word(offsetof(State, budget)), 1024);
    a.mov(eax, ebp);
    a.bind(ready);
    a.bind(initial);
    a.cmp(eax, u32(code.size()));
    auto in_range = a.new_label();
    a.jb(in_range);
    a.mov(word(offsetof(State, error_pc)), eax);
    a.mov(eax, 11);
    a.jmp(leave);
    a.bind(in_range);
    a.lea(rcx, ptr(table));
    a.jmp(qword_ptr(rcx, rax, 3));

    for (pc = 0; pc < code.size(); ++pc) {
      a.bind(labels[pc]);
      if (uploaded[pc] == 7)
        instruction(code[pc]);
      else
        a.jmp(fault(11));
    }
    a.bind(success);
    a.xor_(eax, eax);
    a.bind(leave);
    a.add(rsp, 40);
    for (Gp r : {r15, r14, r13, r12, rbp, rbx})
      a.pop(r);
    a.ret();
    for (const auto &f : faults) {
      a.bind(f.label);
      a.mov(word(offsetof(State, error_pc)), f.pc);
      if (f.value)
        a.mov(word(offsetof(State, error_value)), *f.value);
      else if (f.error == 12 || f.error == 13)
        a.mov(word(offsetof(State, error_value)), eax);
      a.mov(eax, f.error);
      a.jmp(leave);
    }
    a.align(AlignMode::kData, 8);
    a.bind(table);
    for (auto label : labels)
      check(a.embed_label(label, 8));
    AmpereProgram program = nullptr;
    check(runtime.jit.add(&program, &holder));
    return program;
  }
  void instruction(const Inst &d) {
    for (u32 lane = 0; lane < 2; ++lane) {
      if (!d.loads[lane])
        continue;
      a.mov(rax, pointer(offsetof(State, consumed)));
      a.cmp(rax, pointer(offsetof(State, params_size)));
      a.jae(fault(1));
      a.mov(rcx, pointer(offsetof(State, params)));
      a.mov(lane == 0 ? r12d : r13d, dword_ptr(rcx, rax, 2));
      a.inc(rax);
      a.mov(pointer(offsetof(State, consumed)), rax);
    }
    if (!d.predicate) {
      body(d, 15);
      return;
    }
    if (d.predicate_reg == 25 || d.predicate_reg == 26) {
      a.jmp(fault(16));
      return;
    }
    if (d.predicate_reg < 24)
      require(1u << d.predicate_reg);
    get(eax, d, d.predicate_reg, 0);
    a.test(eax, eax);
    auto when_false = a.new_label();
    a.jz(when_false);
    body(d, predicate_mask(d, true));
    a.bind(when_false);
    body(d, predicate_mask(d, false));
  }
  void body(const Inst &d, u32 mask) {
    u32 required = 0, written = 0;
    for (u32 i = 0; i < 2; ++i)
      if (mask & (1u << i)) {
        for (u32 r : {d.alu[i].x, d.alu[i].y}) {
          if (r < 24)
            required |= 1u << r;
          else if (r > 29) {
            a.jmp(fault(2));
            return;
          }
        }
      }
    require(required);
    const bool has_flow =
        ((mask & 1) && flow(d.alu[0].op)) || ((mask & 2) && flow(d.alu[1].op));
    if (has_flow) {
      a.mov(word(offsetof(State, next_pc)), pc + 1);
      if (d.end)
        a.mov(byte(offsetof(State, cancel_end)), 0);
    }
    // @SPECULATIVE: preserve simultaneous operand reads as required by native
    // SDK streams; the published Mesa simulator commits lane 0 first.
    const bool early =
        (mask & 3) == 3 && d.alu[0].dst < 24 &&
        (d.alu[0].dst == d.alu[1].x || d.alu[0].dst == d.alu[1].y);
    if (early) {
      get(eax, d, d.alu[1].x, 1);
      a.mov(word(offsetof(State, scratch_x)), eax);
      get(eax, d, d.alu[1].y, 1);
      a.mov(word(offsetof(State, scratch_y)), eax);
    }
    for (u32 i = 0; i < 2; ++i) {
      bool selected = false;
      for (u32 out = 0; out < 2; ++out)
        if (mask & (4u << out))
          selected |= d.output[out][0] == i + 1 || d.output[out][1] == i + 1;
      if (!(mask & (1u << i))) {
        if (selected)
          a.xor_(i == 0 ? r14d : r15d, i == 0 ? r14d : r15d);
        continue;
      }
      const auto &alu = d.alu[i];
      const bool carry_used = carry_live[pc][i];
      // A discarded arithmetic result has no effect when its carry is dead.
      const bool pure = alu.op <= 6 || (alu.op >= 8 && alu.op <= 21);
      const bool valid_special =
          (alu.op != 1 || (i == 1 && d.alu[0].op == 0)) &&
          (alu.op != 3 || (i == 1 && d.alu[0].op == 2)) &&
          (alu.op != 5 || (alu.x == 24 && alu.y == 24)) &&
          (alu.op != 16 || (alu.imm >> 10) < 32);
      if (alu.dst == 24 && !selected && pure && valid_special && !carry_used)
        continue;
      const auto cx = constant(d, alu.x, i), cy = constant(d, alu.y, i);
      if (alu.op == 0 && !carry_used && !(i == 1 && early) && (cx || cy)) {
        if (cx && cy)
          a.mov(eax, *cx + *cy);
        else {
          get(eax, d, cx ? alu.y : alu.x, i);
          const u32 immediate = cx ? *cx : *cy;
          if (immediate == 1)
            a.inc(eax);
          else if (immediate == ~0u)
            a.dec(eax);
          else if (immediate)
            a.add(eax, immediate);
        }
      } else {
        if (i == 1 && early) {
          a.mov(ecx, word(offsetof(State, scratch_x)));
          a.mov(edx, word(offsetof(State, scratch_y)));
        } else {
          get(ecx, d, alu.x, i);
          get(edx, d, alu.y, i);
        }
        lane(d, i);
      }
      if (selected)
        a.mov(i == 0 ? r14d : r15d, eax);
      if (alu.dst < 24) {
        a.mov(reg(alu.dst), eax);
        written |= 1u << alu.dst;
      } else if (alu.dst != 24)
        a.jmp(fault(3));
    }
    if (written)
      a.or_(word(offsetof(State, initialized)), written);
    for (u32 i = 0; i < 2; ++i) {
      if (!(mask & (4u << i)))
        continue;
      if (d.output[i][0]) {
        // An overwritten selection has no observable effect. Selector
        // validation is still performed even when no value needs to be
        // materialized.
        const bool replaced =
            i == 0 && !d.output[0][1] && (mask & 8) && d.output[1][0];
        output(eax, d, d.output[i][0]);
        if (!replaced) {
          a.mov(ecx, eax);
          a.and_(ecx, 0xfff);
          a.shl(ecx, 2);
          a.mov(word(offsetof(State, method)), ecx);
          a.shr(eax, 12);
          a.and_(eax, 15);
          a.shl(eax, 2);
          a.mov(word(offsetof(State, increment)), eax);
        }
      }
      if (d.output[i][1]) {
        output(eax, d, d.output[i][1]);
        a.mov(arg2.r32(), eax);
        a.mov(arg1.r32(), word(offsetof(State, method)));
        a.mov(arg0, rbx);
        a.mov(word(offsetof(State, error_pc)), pc);
        a.call(pointer(offsetof(State, emit)));
        a.test(eax, eax);
        a.jz(fault(10));
        a.mov(eax, word(offsetof(State, increment)));
        a.add(word(offsetof(State, method)), eax);
      }
    }
    if (after_end[pc]) {
      a.cmp(byte(offsetof(State, end_next)), 0);
      a.jne(success);
    }
    if (has_flow) {
      a.cmp(byte(offsetof(State, stopped)), 0);
      a.jne(success);
    }
    if (d.end) {
      if (has_flow) {
        a.mov(al, byte(offsetof(State, cancel_end)));
        a.xor_(al, 1);
        a.mov(byte(offsetof(State, end_next)), al);
      } else
        a.mov(byte(offsetof(State, end_next)), 1);
    }
    if (loop_end[pc]) {
      auto no_loop = a.new_label();
      a.cmp(word(offsetof(State, loops)), 0);
      a.je(no_loop);
      a.cmp(word(offsetof(State, loop_end)), pc);
      a.jne(no_loop);
      a.dec(word(offsetof(State, loops)));
      a.mov(eax, word(offsetof(State, loop_start)));
      a.jmp(dispatch);
      a.bind(no_loop);
    }
    if (has_flow) {
      a.mov(eax, word(offsetof(State, next_pc)));
      if (pc + 1 < code.size()) {
        a.cmp(eax, pc + 1);
        a.jne(dispatch);
        a.jmp(labels[pc + 1]);
      } else
        a.jmp(dispatch);
    } else if (pc + 1 < code.size())
      a.jmp(labels[pc + 1]);
    else {
      ++pc;
      a.jmp(fault(11));
      --pc;
    }
  }
  void output(Gp dst, const Inst &d, u32 selector) {
    switch (selector) {
    case 1:
      a.mov(dst, r14d);
      break;
    case 2:
      a.mov(dst, r15d);
      break;
    case 3:
      a.mov(dst, r12d);
      break;
    case 4:
      a.mov(dst, r13d);
      break;
    case 5:
      a.mov(dst, d.alu[0].imm);
      break;
    case 6:
      a.mov(dst, d.alu[1].imm);
      break;
    case 8:
      a.mov(dst, d.alu[0].imm >> 12);
      break;
    case 9:
      a.mov(dst, d.alu[1].imm >> 12);
      break;
    case 10:
      a.mov(dst, (u32(d.alu[0].imm) << 16) | d.alu[1].imm);
      break;
    default:
      a.jmp(fault(9));
    }
  }
  void compare(u32 op) {
    a.cmp(ecx, edx);
    switch (op) {
    case 17:
      a.setl(al);
      break;
    case 18:
      a.setb(al);
      break;
    case 19:
      a.setle(al);
      break;
    case 20:
      a.setbe(al);
      break;
    default:
      a.sete(al);
      break;
    }
    a.movzx(eax, al);
  }
  void lane(const Inst &d, u32 i) {
    const auto &v = d.alu[i];
    const bool carry_used = carry_live[pc][i];
    switch (v.op) {
    case 0:
      // Zero and immediate operands are known at compilation, including signed
      // immediates; avoid reloading a constant for the arithmetic itself.
      if (!carry_used && constant(d, v.x, i) == std::optional<u32>(0))
        a.mov(eax, edx);
      else if (!carry_used && constant(d, v.y, i) == std::optional<u32>(0))
        a.mov(eax, ecx);
      else {
        a.mov(eax, ecx);
        a.add(eax, edx);
        if (carry_used) {
          a.setc(r8b);
          a.movzx(r8d, r8b);
          a.mov(word(offsetof(State, carry)), r8d);
        }
      }
      return;
    case 1:
      if (i != 1 || d.alu[0].op != 0) {
        a.jmp(fault(14));
        return;
      }
      a.lea(eax, ptr(rcx, rdx));
      a.add(eax, word(offsetof(State, carry)));
      return;
    case 2:
      a.mov(eax, ecx);
      a.sub(eax, edx);
      if (carry_used) {
        a.setc(r8b);
        a.movzx(r8d, r8b);
        a.mov(word(offsetof(State, carry)), r8d);
      }
      return;
    case 3:
      if (i != 1 || d.alu[0].op != 2) {
        a.jmp(fault(14));
        return;
      }
      a.mov(eax, ecx);
      a.sub(eax, edx);
      a.sub(eax, word(offsetof(State, carry)));
      return;
    case 4:
    case 6:
      if (v.op == 4) {
        a.movsxd(rax, ecx);
        a.movsxd(rdx, edx);
      } else {
        a.mov(eax, ecx);
        a.mov(edx, edx);
      }
      a.imul(rax, rdx);
      if (carry_used) {
        a.mov(rcx, rax);
        a.shr(rcx, 32);
        a.mov(word(offsetof(State, carry)), ecx);
      }
      return;
    case 5:
      if (v.x != 24 || v.y != 24) {
        a.jmp(fault(15));
        return;
      }
      a.mov(eax, word(offsetof(State, carry)));
      return;
    case 7:
      a.cmp(ecx, 0x1000);
      a.jne(fault(4));
      a.cmp(edx, 1);
      a.jne(fault(4));
      a.xor_(eax, eax);
      return;
    case 8: {
      auto zero = a.new_label(), done = a.new_label();
      a.test(ecx, ecx);
      a.jz(zero);
      a.bsr(eax, ecx);
      a.xor_(eax, 31);
      a.jmp(done);
      a.bind(zero);
      a.mov(eax, 32);
      a.bind(done);
      return;
    }
    case 9:
    case 10:
    case 11:
      a.mov(eax, ecx);
      a.mov(ecx, edx);
      if (v.op == 9)
        a.shl(eax, cl);
      else if (v.op == 10)
        a.shr(eax, cl);
      else
        a.sar(eax, cl);
      return;
    case 12:
      a.mov(eax, ecx);
      a.and_(eax, edx);
      return;
    case 13:
      a.mov(eax, ecx);
      a.and_(eax, edx);
      a.not_(eax);
      return;
    case 14:
      a.mov(eax, ecx);
      a.or_(eax, edx);
      return;
    case 15:
      a.mov(eax, ecx);
      a.xor_(eax, edx);
      return;
    case 16: {
      const u32 dest = v.imm >> 10, count = (v.imm >> 5) & 31, src = v.imm & 31;
      if (dest >= 32) {
        a.jmp(fault(5));
        return;
      }
      const u32 mask = (1u << count) - 1;
      a.mov(eax, ecx);
      a.and_(eax, ~(mask << dest));
      a.shr(edx, src);
      a.and_(edx, mask);
      a.shl(edx, dest);
      a.or_(eax, edx);
      return;
    }
    case 17:
    case 18:
    case 19:
    case 20:
    case 21:
      compare(v.op);
      a.neg(eax);
      return;
    case 22:
      a.lea(eax, ptr(rcx, rdx));
      a.and_(eax, 0xffff);
      a.cmp(eax, 0xe00);
      a.jae(fault(13));
      a.mov(rcx, pointer(offsetof(State, valid_bank)));
      a.bt(dword_ptr(rcx), eax);
      a.jnc(fault(13));
      a.mov(rcx, pointer(offsetof(State, bank)));
      a.mov(eax, dword_ptr(rcx, rax, 2));
      return;
    case 23: {
      if (v.imm < 2 || v.dst != 24 || v.y != 24) {
        a.jmp(fault(6));
        return;
      }
      a.cmp(word(offsetof(State, loops)), 0);
      a.jne(fault(6));
      auto skip = a.new_label(), done = a.new_label();
      a.test(ecx, ecx);
      a.jz(skip);
      a.dec(ecx);
      a.mov(word(offsetof(State, loops)), ecx);
      a.mov(word(offsetof(State, loop_start)), pc + 1);
      a.mov(word(offsetof(State, loop_end)), pc + v.imm - 1);
      a.jmp(done);
      a.bind(skip);
      // @SPECULATIVE: zero-count loops skip the body in native SDK macros,
      // matching the native interpreter rather than Mesa's MAX(1, count).
      a.mov(word(offsetof(State, next_pc)), pc + v.imm);
      a.bind(done);
      a.xor_(eax, eax);
      return;
    }
    case 24:
      if (v.dst != 24 || v.x != 24 || v.y != 24 || !(v.imm & 0x8000)) {
        a.jmp(fault(15));
        return;
      }
      a.mov(word(offsetof(State, next_pc)), pc + (v.imm & 0x7fff));
      a.xor_(eax, eax);
      return;
    case 25:
    case 26:
    case 27:
    case 28:
    case 29: {
      if (v.dst != 24) {
        a.jmp(fault(15));
        return;
      }
      compare(v.op - 8);
      a.cmp(eax, (v.imm & 0x8000) ? 1 : 0);
      auto not_taken = a.new_label();
      a.jne(not_taken);
      const int32_t offset = int32_t(u32(v.imm) << 19) >> 19;
      if (offset == -4096)
        a.mov(byte(offsetof(State, stopped)), 1);
      else if (int64_t(pc) + offset < 0)
        a.jmp(fault(17));
      else
        a.mov(word(offsetof(State, next_pc)), pc + u32(offset));
      // @SPECULATIVE: the last taken conditional branch determines whether
      // bit 14 cancels this instruction's END. An earlier pending END is
      // unaffected. Mesa does not model this bit; match the native reference.
      // https://github.com/zortos293/Lichii/blob/646b5d17135a2b2701ccc7a959afd0bbbf2f6160/src/video_core/macro_turing_jit.h#L626
      if (d.end && offset != -4096)
        a.mov(byte(offsetof(State, cancel_end)), (v.imm & 0x4000) ? 1 : 0);
      a.bind(not_taken);
      a.xor_(eax, eax);
      return;
    }
    case 30:
      if (v.y != 24) {
        a.jmp(fault(15));
        return;
      }
      a.cmp(rcx, pointer(offsetof(State, dram_size)));
      a.jae(fault(7));
      a.mov(rax, pointer(offsetof(State, dram)));
      a.mov(eax, dword_ptr(rax, rcx, 2));
      return;
    case 31:
      if (v.dst != 24) {
        a.jmp(fault(15));
        return;
      }
      a.cmp(rcx, pointer(offsetof(State, dram_size)));
      a.jae(fault(8));
      a.mov(rax, pointer(offsetof(State, dram)));
      a.mov(dword_ptr(rax, rcx, 2), edx);
      a.xor_(eax, eax);
      return;
    }
  }
};
} // namespace
extern "C" bool nixe_ampere_mme_compile(void *runtime,
                                        const AmpereInstruction *instructions,
                                        const uint8_t *uploaded, size_t count,
                                        AmpereProgram *result,
                                        NixeMmeCompileError *error) noexcept {
  *result = nullptr;
  try {
    if (count > 0x1000)
      throw std::runtime_error("MME instruction RAM exceeds 4096 instructions");
    Compiler compiler(*static_cast<NixeMmeRuntime *>(runtime),
                      {instructions, count}, {uploaded, count});
    *result = compiler.generate();
    return true;
  } catch (const std::exception &e) {
    error->pc = 0;
    std::snprintf(error->message, sizeof(error->message), "%s", e.what());
    return false;
  }
}
