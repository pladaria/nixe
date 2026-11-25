#pragma once

#include <dynarmic/interface/A64/diagnostics.h>
#include <array>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#ifdef __linux__
#include <unistd.h>
#include <sys/syscall.h>
#endif

// Diagnostic-only, per-JIT buffered records. Analysis and descriptor histories
// belong to offline tools, not the executing guest. Each stream has one writer.
class JitDiagnostics {
  using Clock = std::chrono::steady_clock;
  std::uint64_t event_limit_ = 1'048'576;
  std::unique_ptr<std::FILE, decltype(&std::fclose)> events_{nullptr, std::fclose};
  std::unique_ptr<std::FILE, decltype(&std::fclose)> snapshots_{nullptr, std::fclose};
  std::array<char, 64 * 1024> event_buffer_{};
  std::array<char, 4096> snapshot_buffer_{};
  Clock::time_point next_snapshot_{};
  std::uint64_t owner_{}, epoch_{}, event_count_{}, dropped_{};
  bool io_failed_{};

  void io_error() noexcept {
    if (!io_failed_) std::fputs("JIT diagnostics: output failed; diagnostic capture is incomplete\n", stderr);
    io_failed_ = true;
  }

  static unsigned long long u(std::uint64_t value) { return value; }
  static std::uint64_t now() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
  }
  static std::uint64_t thread_id() {
#ifdef __linux__
    return static_cast<std::uint64_t>(syscall(SYS_gettid));
#else
    return std::hash<std::thread::id>{}(std::this_thread::get_id());
#endif
  }
  void event(const char* kind, const Dynarmic::A64::Compilation& c = {}) {
    if (io_failed_ || event_count_ == event_limit_) {
      ++dropped_;
      return;
    }
    ++event_count_;
    if (std::fprintf(events_.get(), "%llu,%s,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu\n",
        u(now()), kind, u(owner_), u(epoch_), u(c.descriptor), u(c.fingerprint),
        u(c.host_address), u(c.native_bytes), u(c.guest_bytes), u(c.ir_instructions),
        u(c.translation_ns), u(c.optimization_ns), u(c.emission_ns), u(c.near_bytes), u(c.cold_bytes)) < 0) {
      io_error();
    }
  }
  static void compiled(void* opaque, const Dynarmic::A64::Compilation& c) {
    auto& self = *static_cast<JitDiagnostics*>(opaque);
    auto& s = self.counters;
    ++s.compiled;
    s.guest_bytes += c.guest_bytes;
    s.native_bytes += c.native_bytes;
    s.near_bytes += c.near_bytes;
    s.cold_bytes += c.cold_bytes;
    s.translation_ns += c.translation_ns;
    s.optimization_ns += c.optimization_ns;
    s.emission_ns += c.emission_ns;
    self.event("compile", c);
  }
  static void invalidated(void* opaque, std::uint64_t descriptor, bool fastmem) {
    auto& self = *static_cast<JitDiagnostics*>(opaque);
    if (fastmem) ++self.counters.fastmem_blocks;
    else ++self.counters.targeted_blocks;
    self.event(fastmem ? "fastmem" : "targeted", {.descriptor = descriptor});
  }
  static void cleared(void* opaque, bool capacity) {
    auto& self = *static_cast<JitDiagnostics*>(opaque);
    if (capacity) ++self.counters.capacity_clears;
    else ++self.counters.requested_clears;
    ++self.epoch_;
    self.event(capacity ? "clear_capacity" : "clear_requested");
  }
  explicit JitDiagnostics(const char* directory) {
    if (const char* value = std::getenv("NIXE_JIT_DIAGNOSTICS_MAX_EVENTS")) {
      char* end = nullptr;
      errno = 0;
      const auto limit = std::strtoull(value, &end, 10);
      if (errno != 0 || end == value || *end != '\0' || *value == '-' || limit == 0)
        throw std::runtime_error("Invalid NIXE_JIT_DIAGNOSTICS_MAX_EVENTS");
      event_limit_ = limit;
    }
    static std::atomic<std::uint64_t> next{0};
    const auto base = std::filesystem::path(directory) / ("core-" + std::to_string(next.fetch_add(1)));
    events_.reset(std::fopen((base.string() + ".events.csv").c_str(), "w"));
    snapshots_.reset(std::fopen((base.string() + ".counters.csv").c_str(), "w"));
    if (!events_ || !snapshots_) throw std::runtime_error("Cannot open JIT diagnostic files");
    std::setvbuf(events_.get(), event_buffer_.data(), _IOFBF, event_buffer_.size());
    std::setvbuf(snapshots_.get(), snapshot_buffer_.data(), _IOFBF, snapshot_buffer_.size());
    std::fputs("time_ns,event,owner_tid,epoch,descriptor,fingerprint,host_address,native_bytes,guest_bytes,ir_instructions,translation_ns,optimization_ns,emission_ns,near_bytes,cold_bytes\n", events_.get());
    std::fputs("time_ns,owner_tid,lookups,hits,compiled,guest_bytes,native_bytes,mainline_bytes,deferred_bytes,capacity_clears,requested_clears,targeted_blocks,fastmem_blocks,translation_ns,optimization_ns,emission_ns,dropped_events,io_failed,event_limit,dispatch_hits,dispatch_misses,rsb_hits,near_bytes,cold_bytes,fastmem_retained,host_calls,host_returns\n", snapshots_.get());
    counters.context = this;
    counters.compile = compiled;
    counters.invalidate = invalidated;
    counters.clear = cleared;
  }
public:
  Dynarmic::A64::Diagnostics counters{};
  static std::unique_ptr<JitDiagnostics> open() {
    const char* directory = std::getenv("NIXE_JIT_DIAGNOSTICS");
    if (!directory || !*directory) return {};
    return std::unique_ptr<JitDiagnostics>(new JitDiagnostics(directory));
  }
  void enter() {
    if (owner_ == 0) {
      owner_ = thread_id();
      event("owner");
      snapshot(true);
    }
  }
  void snapshot(bool force = false) {
    const auto time = Clock::now();
    if (!force && time < next_snapshot_) return;
    next_snapshot_ = time + std::chrono::seconds(1);
    const auto& s = counters;
    if (std::fprintf(snapshots_.get(), "%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%u,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu\n",
        u(now()), u(owner_), u(s.lookups), u(s.hits), u(s.compiled), u(s.guest_bytes),
        u(s.native_bytes), u(s.mainline_bytes), u(s.deferred_bytes), u(s.capacity_clears),
        u(s.requested_clears), u(s.targeted_blocks), u(s.fastmem_blocks), u(s.translation_ns),
        u(s.optimization_ns), u(s.emission_ns), u(dropped_), unsigned(io_failed_), u(event_limit_), u(s.dispatch_hits), u(s.dispatch_misses), u(s.rsb_hits), u(s.near_bytes), u(s.cold_bytes), u(s.fastmem_retained), u(s.host_calls), u(s.host_returns)) < 0 ||
        std::fflush(events_.get()) != 0 || std::fflush(snapshots_.get()) != 0) {
      io_error();
    }
  }
  ~JitDiagnostics() {
    snapshot(true);
    // The FILE objects must close before their setvbuf storage is destroyed.
    events_.reset();
    snapshots_.reset();
  }
};
