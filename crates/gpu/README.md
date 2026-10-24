# GPU execution and ownership

`nixe-gpu` defines the shared execution, resource, presentation and compilation
contracts. It contains no guest instruction decoder or host graphics API.

Startup supplies `MaxwellNativeFrontend` and `NativeGpuExecution` to
`VideoSystem::with_gpu_frontend`. Horizon moves the concrete renderer into its GPU
owner thread. `NativeWork` carries one consumed engine operation with physical
inputs and immutable prepared shader, binding and graphics state.

The owner consumes guest resources through `GpuExecution::consume`. It polls cold
compilation and records the effect once its dependencies are ready, delegating
owned native jobs to the Vulkan worker. A blocked channel retains its current effect while
unrelated channels and native progress can advance. External CPU visibility is
serviced between consumed effects. `progress` performs control-boundary maintenance
once before polling logical completions. Descriptor admission is released after
consumption; the separate completion ticket remains pending until actual GPU work
and canonical ownership effects finish. Horizon then publishes the corresponding
guest timeline reservation.

Canonical CPU visibility belongs to `nixe-memory`. CPU requests reach the owner
through `DeviceAccessEndpoint`. Owner-local reads either complete in RAM or retain
one exact device demand; they do not submit a request to their own endpoint.
Physical pins preserve accepted uses through unmapping and cache eviction.
Queue pressure does not silently enlarge budgets or drop commands.

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
