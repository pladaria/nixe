# GPU execution and ownership

`nixe-gpu` defines the shared execution, resource, presentation and compilation
contracts. It contains no guest instruction decoder or host graphics API.

Startup supplies `MaxwellNativeFrontend` and `NativeGpuExecution` to
`VideoSystem::with_gpu_frontend`. Horizon moves the concrete renderer into its GPU
owner thread. `ExecutionBatch` borrows an ordered native submission; immutable
commands retain canonical ranges and prepared shader, binding and graphics state.

The GPU owner consumes guest resources, resolves native hazards and schedules recording and submission. The renderer delegates owned native jobs to its Vulkan worker.
`GpuExecution::prepare` polls cold compilation without accepting a command or
publishing completion. Pending work retains its delivery permits while the
owner services visibility and host control. Acceptance captures required inputs;
completion is reported only after actual GPU work and canonical ownership effects.
Horizon then publishes the corresponding guest timeline reservation.

Canonical CPU visibility belongs to `nixe-memory`. CPU requests reach the owner
through `BackendVisibilityRequester`; the owner resolves its own demands inline
without reentering its inbox. Resource leases preserve accepted uses through
unmapping and cache eviction. Queue pressure does not silently enlarge budgets or
drop commands.

`ResidentImage` contains a typed `PresentationLease`: an image generation, content
version, producer token and last-consumer release channel. Host graphics objects
remain on their owner. Releasing a frame does not imply GPU or display completion.

## Shader compilation

`nixe-shader-recompiler` translates captured Maxwell inputs into owned SPIR-V and
resource/interface metadata. `shader_compilation` links captures with the enabled
host configuration and runtime state. `shader_identity` and `shader_encoding`
provide stable content identities and a bounded, versioned artifact codec.
`compilation` supplies bounded workers, deduplicated publication and cache budgets.
Workers never own a GPU queue or wait for jobs submitted to their own pool.

`shader_module` defines host module/binding types and synthetic topology stages.
Guest resource identities encode stage, descriptor category, index and array
element independently of emitted Vulkan binding numbers. Host presentation and
conversion shaders remain separate from guest instruction compilation.
