# Host requirements

These are required host capabilities, not a guarantee of title compatibility or
performance. RAM and GPU requirements are not specified here.

## Linux memory pages

The current tiered JIT requires **4 KiB host pages** (`getconf PAGESIZE` must
return `4096`). Its LinuxDirect memory backend maps and protects individual
4 KiB guest pages through host mappings. Hosts with 16 KiB or 64 KiB pages are
rejected; there is no automatic checked-memory fallback for the JIT.

Page size is not the only requirement: canonical backing reserves 512 GiB of
contiguous virtual space, and a 39-bit guest needs another 512 GiB plus guards
for its direct arena. These reservations do not populate that much RAM, but
require sufficient host virtual address space.

The tested Pi 5 `kernel8.img` has 4 KiB pages and only 39-bit virtual addresses;
it cannot launch guests with the current backend. Switching from the default
16 KiB kernel therefore does not suffice. ARM host development/validation is
deferred; current work focuses on AMD64. The
[portability follow-up](specs/tiered-jit/next.md) separates segmented arenas
for limited virtual space from support for larger host pages.

## x86-64

Nixe requires **LAHF/SAHF support in 64-bit mode**, advertised by
`CPUID.80000001H:ECX[0]` (`LAHF_SAHF_64`). This is an explicit CPU requirement,
not a requirement for the entire x86-64-v2 or AVX feature sets.

The tiered JIT uses LAHF/SETO to capture live native subtraction flags and
ADD/SAHF to restore them across cold polls and callbacks, without using the
host stack. SAHF installs sign, zero and carry; overflow is established
separately because SAHF leaves it unchanged. Conversions to guest NZCV happen
at architectural consumers, not at every preservation boundary. See the
[Intel instruction reference](https://cdrdv2-public.intel.com/782151/253667-sdm-vol-2b.pdf).

The JIT checks support when creating a process, before compiling or executing
code. An incompatible host fails initialization with a clear error; there is
no alternate JIT path for CPUs lacking this feature. Virtual machines must
expose the feature to the guest OS. There is no CPUID check per invocation,
fragment or link.

The tiered JIT's 128-bit guest CASP and STXP/STLXP X additionally require
**CMPXCHG16B**.
Compilation reports the missing capability explicitly on an incompatible x86
target; it does not split the atomic transaction or call a RAM helper.

## AArch64

SAHF is an x86 instruction and imposes no requirement on AArch64 hosts. Their
native boundary reads and writes the architecture's NZCV register directly.
128-bit atomics do not require LSE: the JIT uses CASPAL when available and a
validated LDAXP/STLXP loop otherwise.

## Audio output

`audout:u` uses SDL3 to open the host's default output when a guest opens
`DeviceOut`. The emulated output is currently stereo signed-16-bit PCM at 48 kHz.
SDL adapts this PCM stream to the physical device's format and sample rate.
SDL is initialized once on the application's main thread; input sampling and
SDL's audio callback run independently. No audio feeder worker is required.

The optional `nixe.toml` setting is:

```toml
[audio]
output = "stereo"
```

Omitting it selects the same stereo (2.0) mode. Other layouts, including 5.1,
are rejected until their Horizon semantics and channel routing are implemented.
The output layout and PCM format are explicit types shared by configuration and
the host adapter, so future modes do not require another playback backend.

SDL consumes PCM on demand. Empty or stopped queues supply silence without
manufacturing buffer completions. Stopping an active guest output releases its
pending buffers; closing its last session handle destroys the SDL stream.
Application shutdown closes any remaining streams before the SDL audio subsystem.
Flush/volume commands and six-channel guest buffers remain unsupported.

See [SDL3 audio streams](https://wiki.libsdl.org/SDL3/SDL_AudioStream) for host
format conversion and [SDL3 audio devices](https://wiki.libsdl.org/SDL3/CategoryAudio)
for default-device migration.
