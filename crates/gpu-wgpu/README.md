# Nixe accelerated GPU backend

`nixe-gpu-wgpu` is the host adapter for the API-independent contracts in
`nixe-gpu`. It owns every `wgpu` object and must not contain Horizon commands,
Maxwell packet encodings, or Switch-specific capability policy.

The initial correct policy is intentionally conservative:

- Vulkan is the only compiled host API, while backend selection remains an
  explicit configuration value.
- Canonical guest bytes remain authoritative host memory. Buffers and images
  are device mirrors uploaded before reads and copied back after writes.
- Submissions are executed serially and report host completion separately from
  canonical-memory visibility and guest timeline publication.
- `wgpu` usage tracking implements host barriers from neutral access
  declarations; guest cache-maintenance commands remain explicit ordering
  points.
- Unsupported formats, layouts, pipeline inputs, and operation forms stop with
  a diagnostic instead of being approximated.

The CLI owns initialization and erases the concrete driver behind
`NeutralBackendRuntime` before injecting it into Horizon. Consequently the
real `nvdrv`/Maxwell path can execute accelerated work without importing
`wgpu` types or host selection policy into either console-specific crate.

The accelerated acceptance tests use redistributable synthetic shaders and
backings. They compare exact buffer contents and the expected rasterized point
against the neutral reference contract. A missing Vulkan adapter skips only
the hardware-dependent assertions; architecture and selection tests remain
active.
