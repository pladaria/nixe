# Nixe accelerated GPU backend

`nixe-gpu-wgpu` is the host adapter for the API-independent contracts in
`nixe-gpu`. It owns every `wgpu` object and must not contain Horizon commands,
Maxwell packet encodings, or Switch-specific capability policy.

The initial correct policy is intentionally conservative:

- Vulkan is the only compiled host API, while backend selection remains an
  explicit configuration value.
- Canonical memory keeps stable guest identity while content authority may
  reside on the CPU or device. Resident resources upload only CPU-dirty input
  and GPU writes return to canonical memory only at a verified CPU boundary.
- A bounded queue accepts ordered submissions asynchronously. One backend
  owner reports them on one completion timeline; canonical-memory visibility
  and guest timeline publication advance only at their required boundaries.
- `wgpu` usage tracking implements host barriers from neutral access
  declarations; guest cache-maintenance commands remain explicit ordering
  points.
- Unsupported formats, layouts, pipeline inputs, and operation forms stop with
  a diagnostic instead of being approximated.

Ordinary graphics supports resident 2D four-sample color/depth attachments and
full-subresource color resolves into single-sample images. Resolves preserve
the source and remain ordered on the GPU, including across submissions. Sample
count participates in both the pipeline cache key and its prepared-draw fast
path. Four-sample support requires a Vulkan device with standard sample locations.
Maxwell interprets supported 2x2 sample storage as four-sample attachments and
lowers matching full-surface 2D resolves through the same resident image cache.
Canonical transfers of multisample storage, other sample counts, custom sample
patterns/masks, partial resolves, and multisampled native tessellation remain
unsupported.

The CLI owns initialization and erases the concrete driver behind
`NeutralBackendRuntime` before injecting it into Horizon. Consequently the
real `nvdrv`/Maxwell path can execute accelerated work without importing
`wgpu` types or host selection policy into either console-specific crate.

The accelerated acceptance tests use redistributable synthetic shaders and
backings. They compare exact buffer contents and the expected rasterized point
against the neutral reference contract. A missing Vulkan adapter skips only
the hardware-dependent assertions; architecture and selection tests remain
active.

The multisample pixel oracle alternates 1x/4x pipelines, tests floating-point and
depth/stencil attachments, verifies fractional edge coverage and per-sample
depth rejection, and checks repeated resolves and subsequent full clears. RGBA8
and BGRA8 UNORM/sRGB cases verify channel order, linear-light sample averaging,
sRGB RGB encoding, linear alpha, and direct presentation of stored bytes.
