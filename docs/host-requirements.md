# Host requirements

These are required host capabilities, not a guarantee of title compatibility or
performance. RAM and GPU requirements are not specified here.

## Linux memory pages

The Dynarmic JIT requires **4 KiB host pages** (`getconf PAGESIZE` must
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
deferred; current validation targets Linux x86-64. The direct-memory virtual
address and page-size requirements also apply to the new CPU backend.

## Native compiler

The A64 JIT is built from `vendor/dynarmic` using CMake and a C++20 compiler.
Boost headers are required; the other native dependencies are bundled.
Dynarmic detects x86 host extensions and selects its instruction sequences at
initialization. The memory backend's requirements for native atomic accesses
remain in force. Full Switch 2 instruction-set coverage is not provided by this
upstream Dynarmic revision; unsupported instructions stop explicitly.

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
