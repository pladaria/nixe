#ifdef NIXE_PERFORMANCE_COUNTERS
#include <chrono>
#include <cstdint>
extern "C" void nixe_jit_measure_native_state(std::uint64_t nanoseconds, std::uint64_t bytes);
struct StateMeasurement {
  std::uint64_t bytes = 804;
  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  ~StateMeasurement() {
    nixe_jit_measure_native_state(std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now() - start).count(), bytes);
  }
};
#endif
// Nixe's C ABI boundary. No C++ exceptions cross into Rust.
#include <array>
#include <cstring>
#include <dynarmic/backend/exception_handler.h>
#include <dynarmic/interface/A64/a64.h>
#include <dynarmic/interface/exclusive_monitor.h>
#include <exception>
#include <memory>
#include <string>
#include "state.h"

using Dynarmic::HaltReason;
using Dynarmic::A64::Vector;
// Only copy into typed public API values; never alias a C array as std::array
// or access Dynarmic's private state. See interface/A64/a64.h in the vendor tree.
static_assert(sizeof(Vector) == 2 * sizeof(std::uint64_t));
static_assert(std::is_trivially_copyable_v<Vector>);
static_assert(sizeof(std::array<Vector, 32>) == sizeof(NixeState::vectors));
static_assert(sizeof(std::array<std::uint64_t, 31>) == sizeof(NixeState::x));
extern "C" {
struct NixeCallbacks {
  bool (*code)(void *, std::uint64_t, std::uint32_t *);
  bool (*memory)(void *, std::uint64_t, std::uint32_t, std::uint32_t,
                 std::uint64_t *, const std::uint64_t *);
  std::uint64_t (*counter)(void *);
  bool (*data_cache)(void *, std::uint64_t);
};
struct NixeExit {
  std::uint32_t kind, detail;
  std::uint64_t pc, value, ticks;
};
}

namespace {
thread_local std::string last_error;
struct Core final : Dynarmic::A64::UserCallbacks {
  NixeCallbacks callbacks;
  void *context = nullptr; // Borrowed only for the duration of Run.
  std::uint64_t tpidr = 0, tpidrro = 0;
  NixeExit exit{};
  std::uint64_t ticks_left = 0, ticks_executed = 0;
  std::unique_ptr<Dynarmic::A64::Jit> jit;

  Core(NixeCallbacks cb, Dynarmic::ExclusiveMonitor *monitor, size_t id,
       std::uintptr_t arena, size_t address_bits, std::uint32_t frequency,
       std::uint32_t ctr, std::uint32_t dczid)
      : callbacks(cb) {
    Dynarmic::A64::UserConfig conf{};
    conf.callbacks = this;
    conf.processor_id = id;
    conf.global_monitor = monitor;
    conf.tpidr_el0 = &tpidr;
    conf.tpidrro_el0 = &tpidrro;
    conf.cntfrq_el0 = frequency;
    conf.ctr_el0 = ctr;
    conf.dczid_el0 = dczid;
    conf.wall_clock_cntpct = true;
    conf.enable_cycle_counting = true;
    conf.check_halt_on_memory_access = true;
    conf.hook_hint_instructions = true;
    conf.hook_data_cache_operations = true;
    conf.fastmem_pointer = arena;
    conf.fastmem_address_space_bits = address_bits;
    conf.silently_mirror_fastmem = false;
    // Tracking/ownership faults are transient. Keep the direct access after
    // repair.
    conf.recompile_on_fastmem_failure = false;
    jit = std::make_unique<Dynarmic::A64::Jit>(conf);
  }
  void stop(std::uint32_t kind, std::uint32_t detail, std::uint64_t pc,
            std::uint64_t value = 0) {
    if (exit.kind == 0)
      exit = {kind, detail, pc, value};
    jit->HaltExecution();
  }
  bool access(std::uint64_t addr, unsigned size, unsigned op, Vector &value,
              const Vector *expected = nullptr) {
    if (callbacks.memory(context, addr, size, op, value.data(),
                         expected ? expected->data() : nullptr))
      return true;
    jit->HaltExecution(HaltReason::MemoryAbort);
    return false;
  }
  std::optional<std::uint32_t> MemoryReadCode(std::uint64_t addr) override {
    std::uint32_t word;
    if (callbacks.code(context, addr, &word))
      return word;
    return std::nullopt;
  }
#define READ(bits)                                                             \
  std::uint##bits##_t MemoryRead##bits(std::uint64_t addr) override {          \
    Vector v{};                                                                \
    access(addr, bits / 8, 0, v);                                              \
    return static_cast<std::uint##bits##_t>(v[0]);                             \
  }
  READ(8)
  READ(16) READ(32) READ(64)
#undef READ
      Vector MemoryRead128(std::uint64_t addr) override {
    Vector v{};
    access(addr, 16, 0, v);
    return v;
  }
#define WRITE(bits)                                                            \
  void MemoryWrite##bits(std::uint64_t addr, std::uint##bits##_t value)        \
      override {                                                               \
    Vector v{value, 0};                                                        \
    access(addr, bits / 8, 1, v);                                              \
  }
  WRITE(8)
  WRITE(16) WRITE(32) WRITE(64)
#undef WRITE
      void MemoryWrite128(std::uint64_t addr, Vector v) override {
    access(addr, 16, 1, v);
  }
#define EXCLUSIVE(bits)                                                        \
  bool MemoryWriteExclusive##bits(std::uint64_t addr,                          \
                                  std::uint##bits##_t value,                   \
                                  std::uint##bits##_t expected) override {     \
    Vector v{value, 0}, e{expected, 0};                                        \
    return access(addr, bits / 8, 2, v, &e) && v[0] != 0;                      \
  }
  EXCLUSIVE(8)
  EXCLUSIVE(16) EXCLUSIVE(32) EXCLUSIVE(64)
#undef EXCLUSIVE
      bool MemoryWriteExclusive128(std::uint64_t addr, Vector v,
                                   Vector e) override {
    return access(addr, 16, 2, v, &e) && v[0] != 0;
  }
  void InterpreterFallback(std::uint64_t pc, size_t) override {
    stop(1, 0, pc);
  }
  void CallSVC(std::uint32_t imm) override { stop(2, imm, jit->GetPC() - 4); }
  void ExceptionRaised(std::uint64_t pc,
                       Dynarmic::A64::Exception exception) override {
    stop(3, static_cast<unsigned>(exception), pc);
  }
  void DataCacheOperationRaised(Dynarmic::A64::DataCacheOperation op,
                                std::uint64_t value) override {
    switch (op) {
    case Dynarmic::A64::DataCacheOperation::CleanAndInvalidateByVAToPoC:
    case Dynarmic::A64::DataCacheOperation::CleanByVAToPoC:
    case Dynarmic::A64::DataCacheOperation::CleanByVAToPoU:
    case Dynarmic::A64::DataCacheOperation::CleanByVAToPoP:
    case Dynarmic::A64::DataCacheOperation::InvalidateByVAToPoC:
      // Canonical CPU-visible RAM needs no ownership transition. The block's
      // CheckHalt terminal still observes interrupts/preemption. Device-owned
      // RAM and mapping faults retain the stopped, lease-free checked path.
      if (callbacks.data_cache(context, value))
        return;
      break;
    default:
      // Set/way and zeroing operations retain the platform trap checks.
      break;
    }
    stop(4, static_cast<unsigned>(op), 0, value);
  }
  void
  InstructionCacheOperationRaised(Dynarmic::A64::InstructionCacheOperation op,
                                  std::uint64_t value) override {
    stop(5, static_cast<unsigned>(op), 0, value);
  }
  void AddTicks(std::uint64_t ticks) override {
    ticks_executed += ticks;
    ticks_left = ticks < ticks_left ? ticks_left - ticks : 0;
  }
  std::uint64_t GetTicksRemaining() override { return ticks_left; }
  std::uint64_t GetCNTPCT() override { return callbacks.counter(context); }
};
} // namespace
extern "C" {
bool nixe_dynarmic_resolve_fault(std::uint64_t pc, std::uint64_t *call,
                                 std::uint64_t *ret) noexcept {
  const auto result =
      Dynarmic::Backend::ExceptionHandler::HandleExternalFault(pc);
  if (!result)
    return false;
#if defined(__x86_64__)
  *call = result->call_rip;
  *ret = result->ret_rip;
#else
  *call = result->call_pc;
  *ret = 0;
#endif
  return true;
}
const char *nixe_dynarmic_error() noexcept { return last_error.c_str(); }
void *nixe_dynarmic_monitor_create(size_t count) noexcept {
  try {
    return new Dynarmic::ExclusiveMonitor(count);
  } catch (const std::exception &e) {
    last_error = e.what();
    return nullptr;
  }
}
void nixe_dynarmic_monitor_destroy(void *p) noexcept {
  delete static_cast<Dynarmic::ExclusiveMonitor *>(p);
}
void *nixe_dynarmic_create(NixeCallbacks cb, void *monitor, size_t id,
                           std::uintptr_t arena, size_t bits,
                           std::uint32_t frequency, std::uint32_t ctr,
                           std::uint32_t dczid) noexcept {
  try {
    return new Core(cb, static_cast<Dynarmic::ExclusiveMonitor *>(monitor), id,
                    arena, bits, frequency, ctr, dczid);
  } catch (const std::exception &e) {
    last_error = e.what();
    return nullptr;
  }
}
void nixe_dynarmic_destroy(Core *core) noexcept { delete core; }
void nixe_dynarmic_halt(Core *core) noexcept { core->jit->HaltExecution(); }
void nixe_dynarmic_clear_halt(Core *core) noexcept {
  core->jit->ClearHalt(HaltReason::UserDefined1 | HaltReason::MemoryAbort);
}
void nixe_dynarmic_clear_exclusive(Core *core) noexcept {
  core->jit->ClearExclusiveState();
}
bool nixe_dynarmic_invalidate(Core *core, std::uint64_t addr,
                              size_t size) noexcept {
  try {
    if (size == 0)
      core->jit->ClearCache();
    else
      core->jit->InvalidateCacheRange(addr, size);
    return true;
  } catch (const std::exception &e) {
    last_error = e.what();
    return false;
  }
}
// Complete transfers occur at scheduler switches or explicit snapshot
// consumers. Ordinary SVCs access only their needed registers while this
// context stays native.
// https://git.eden-emu.dev/eden-emu/eden/raw/commit/67bada77f8a43a90da2e94e89b8e7da73c256989/src/core/arm/dynarmic/arm_dynarmic_64.cpp
void nixe_dynarmic_load(Core *core, const NixeState *state) noexcept {
#ifdef NIXE_PERFORMANCE_COUNTERS
  StateMeasurement measurement;
#endif
  std::array<std::uint64_t, 31> registers;
  std::array<Vector, 32> vectors;
  std::memcpy(registers.data(), state->x, sizeof(state->x));
  std::memcpy(vectors.data(), state->vectors, sizeof(state->vectors));
  core->jit->SetRegisters(registers);
  core->jit->SetVectors(vectors);
  core->jit->SetPC(state->pc);
  core->jit->SetSP(state->sp);
  core->jit->SetPstate(state->nzcv);
  core->jit->SetFpcr(state->fpcr);
  core->jit->SetFpsr(state->fpsr);
  core->tpidr = state->tpidr;
  core->tpidrro = state->tpidrro;
}
void nixe_dynarmic_save(Core *core, NixeState *state) noexcept {
#ifdef NIXE_PERFORMANCE_COUNTERS
  StateMeasurement measurement;
#endif
  const auto registers = core->jit->GetRegisters();
  const auto vectors = core->jit->GetVectors();
  std::memcpy(state->x, registers.data(), sizeof(state->x));
  std::memcpy(state->vectors, vectors.data(), sizeof(state->vectors));
  state->pc = core->jit->GetPC();
  state->sp = core->jit->GetSP();
  state->nzcv = core->jit->GetPstate();
  state->fpcr = core->jit->GetFpcr();
  state->fpsr = core->jit->GetFpsr();
  state->tpidr = core->tpidr;
  state->tpidrro = core->tpidrro;
}
// X0-X30, SP, PC, TPIDR_EL0, TPIDRRO_EL0, NZCV, FPCR, FPSR.
// Rust validates these identifiers; only the unique stopped-context owner
// calls.
std::uint64_t nixe_dynarmic_read(Core *core, std::uint32_t reg) noexcept {
#ifdef NIXE_PERFORMANCE_COUNTERS
  StateMeasurement measurement{reg >= 35 ? 4u : 8u};
#endif
  if (reg < 31)
    return core->jit->GetRegister(reg);
  switch (reg) {
  case 31:
    return core->jit->GetSP();
  case 32:
    return core->jit->GetPC();
  case 33:
    return core->tpidr;
  case 34:
    return core->tpidrro;
  case 35:
    return core->jit->GetPstate();
  case 36:
    return core->jit->GetFpcr();
  case 37:
    return core->jit->GetFpsr();
  default:
    std::terminate();
  }
}
void nixe_dynarmic_write(Core *core, std::uint32_t reg,
                         std::uint64_t value) noexcept {
#ifdef NIXE_PERFORMANCE_COUNTERS
  StateMeasurement measurement{reg >= 35 ? 4u : 8u};
#endif
  if (reg < 31) {
    core->jit->SetRegister(reg, value);
    return;
  }
  switch (reg) {
  case 31:
    core->jit->SetSP(value);
    break;
  case 32:
    core->jit->SetPC(value);
    break;
  case 33:
    core->tpidr = value;
    break;
  case 34:
    core->tpidrro = value;
    break;
  case 35:
    core->jit->SetPstate(static_cast<std::uint32_t>(value));
    break;
  case 36:
    core->jit->SetFpcr(static_cast<std::uint32_t>(value));
    break;
  case 37:
    core->jit->SetFpsr(static_cast<std::uint32_t>(value));
    break;
  default:
    std::terminate();
  }
}
void nixe_dynarmic_read_vector(Core *core, std::uint32_t reg,
                               std::uint64_t *value) noexcept {
#ifdef NIXE_PERFORMANCE_COUNTERS
  StateMeasurement measurement{16};
#endif
  const auto vector = core->jit->GetVector(reg);
  std::memcpy(value, &vector, sizeof(vector));
}
void nixe_dynarmic_write_vector(Core *core, std::uint32_t reg,
                                const std::uint64_t *value) noexcept {
#ifdef NIXE_PERFORMANCE_COUNTERS
  StateMeasurement measurement{16};
#endif
  Vector vector;
  std::memcpy(&vector, value, sizeof(vector));
  core->jit->SetVector(reg, vector);
}
bool nixe_dynarmic_run(Core *core, void *context, std::uint64_t budget,
                       NixeExit *result) noexcept {
  try {
    core->context = context;
    core->exit = {};
    core->ticks_left = budget;
    core->ticks_executed = 0;
    core->jit->Run();
    // Return with the faulting/SVC instruction selected. Runtime applies the
    // service continuation once; normalize here without a second FFI call.
    if (core->exit.kind == 1 || core->exit.kind == 2 ||
        (core->exit.kind == 3 &&
         (core->exit.detail <= 2 || core->exit.detail == 8 ||
          core->exit.detail == 9))) {
      core->jit->SetPC(core->exit.pc);
    }
    core->exit.ticks = core->ticks_executed;
    *result = core->exit;
    core->context = nullptr;
    return true;
  } catch (const std::exception &e) {
    core->context = nullptr;
    last_error = e.what();
    return false;
  }
}
}
