# GPU execution and ownership

`GpuExecution` is the renderer contract consumed by Horizon. Startup moves a concrete
renderer into `VideoSystem::with_gpu_execution`, through `NvDrvSession`, to the
monomorphized GPU owner loop. The Switch frontend remains concrete Maxwell code.
Guest GPU selection belongs to console startup; host capabilities belong to device
initialization. Neither requires a generation discriminator or a virtual call per
instruction, draw, dispatch or batch in the new renderer path.

`ExecutionBatch` borrows decoded resource definitions and an immutable ordered
command sequence. It adds no command copy, access-plan reconstruction or allocation.
A renderer implements this interface directly, without `Backend<D>`, resource
transactions or generic rollback. The current operation representation remains in
use until the shader and rendering tasks replace its remaining legacy structures.

## Required ordering and lifetimes

| Data/effect | Owner and lifetime |
| --- | --- |
| Guest registers, encodings, mappings and syncpoint ABI | Maxwell and Horizon. Revalidate mapping generations when an operation consumes guest state; retain canonical backing identity for accepted uses even if the mapping/nvmap handle is removed. |
| Prepared draw state | Frontend-created `Arc<PreparedDraw>`. IDs have a session-local slot and revision. Maxwell allocates slots monotonically with checked exhaustion; it never recycles a live logical ID. `edit_state` explicitly advances the revision during mutable preparation. Queued references prevent mutable access; copying semantic values requires assigning a new record identity. Warm argument changes reuse the same immutable record and cached metadata. |
| Ordered accesses | `ExecutionBatch::ordered_accesses` retains operation positions and exact access scopes. The canonical visibility summary does not replace command order for host hazard tracking. Vulkan barriers/layouts are renderer decisions. |
| CPU writes and host input snapshots | The frontend retains ordered canonical effects. On batch acceptance the renderer captures consumed input versions before later CPU updates can change them. Device-authoritative contents require demanded visibility, not a fabricated CPU copy. |
| Accepted host uses | The renderer retains resources/backings until real completion, independently of frontend chunk retirement. Resource retirements apply after their last accepted uses. |
| Completion | A frontend submission may have multiple ordered segments. `BackendSubmissionToken` is instance/generation scoped. The owner matches every returned token to pending segments. Host acceptance, host completion, canonical visibility and guest syncpoint publication are distinct. Only Horizon publishes its reserved guest point after all required effects complete. The final segment progresses without a following delivery. |
| Presentation | `PresentationLease` identifies an image slot/generation, content version and producer token. The owner orders the read before an overwrite or retains a GPU snapshot. Last lease drop sends one typed release to the owner. It does not establish GPU or WSI completion; the owner handles final retirement. |
| Submission failure | Terminal execution failure stops admission and wakes blocked producers. No failed or unaccepted work publishes guest success. Accepted host uses remain retained for renderer teardown. |
| Cancellation and shutdown | Stop byte admission before joining the frontend. Keep the owner available for already-running canonical visibility calls. Then cancel unaccepted work and drain accepted host uses without publishing canceled guest reservations. |

## Bounded frontend/owner handoff

The existing three-submission admission limit includes queued preparation and
pending execution. Data messages additionally carry byte permits through the inbox
and unconsumed owner chunks. Normal handoff buffering is limited to 64 MiB. One
indivisible payload larger than this limit may be admitted only when no other data
bytes are admitted; it is charged in full and blocks subsequent data admission.
This is an explicit admission policy, not a new guest texture/upload-size limit.

Maxwell streams approximately 128 backend operations or 4 MiB per prefix, retaining
a real operation for the validated final segment. Pending inline data must be
complete before a prefix is delivered. The owner groups compatible work, flushing
at its operation/byte threshold or a semantic synchronization boundary. Consumed
source steps are dropped and complete chunk permits released as effects move into
the bounded working batch or accepted renderer snapshots. Metadata byte charges
are cached in lowered work and immutable operations; unchanged draws reuse them.
Canonical guest allocations, frontend decoding scratch and renderer residency/staging
have separate lifetimes and are not claimed to fit the handoff byte limit.

CPU visibility and progress requests bypass data byte admission. Blocking a producer
never blocks the owner that must fulfill its canonical readback. Backpressure uses
completion/consumption, not per-draw request/reply messages or device-wide waits.
The owner uses `wait_for_completion_timeout` while active so it can revisit its
control inbox even when GPU work/preparation remains pending. Timeout retains all
pending tokens and permits; it is not missing work or guest completion. Data
deferred at saturated admission retains its original byte/submission permits.
Shutdown still drains accepted work with the ordinary completion wait. Only the
temporary legacy bridge implements the bounded wait through polling/sleep; native
Vulkan uses its actual timeline wait. This bridge retires in task 15.

## Temporary migration seam and shared-limit audit

`LegacyGpuExecution` is the single execution bridge to `NeutralBackendRuntime`.
Existing `with_gpu_backend` startup entry points construct it once. The wgpu oracle
continues through its old validated backend/driver stack; new renderers must use
`with_gpu_execution`. Remove the bridge and old entry points in task 15.

`ResourceDefinition` is decoded data outside that stack. The old name
`BackendResourceCreateInfo` remains an alias only for retiring driver consumers and
their tests. Legacy validation extensions and the alias retire with that backend.
`ResidentImage` can carry the new typed lease; its opaque payload variant remains
only for the current wgpu presenter. Tasks 13/15 replace/remove that variant and its
downcasts. Typed leases do not export Vulkan objects to the window thread.

| Shared representation | Disposition |
| --- | --- |
| Session-local `u64` resource identities and prepared slots | Retain compact checked identities. Current Maxwell allocation never reuses them; reusable host caches need generations, as already required by presentation/completion IDs. No speculative widening or per-command guest-model check. |
| `GpuVirtualAddress` stored as `u64` | Retain the shared value; the frontend supplies the guest address-width check. The renderer consumes resolved canonical ranges, not Maxwell page-table encodings. |
| Shader resource bindings `u8`, predicate/register layout and fixed interface tables | Retained only by the oracle and current Maxwell ISA decoders. The native compiler uses [semantic SSA](src/shader/ssa/README.md) with dense `u32` values, `u32` resource identities, explicit initial values, typed byte addresses and effects. These old tables are not universal GPU-model limits; their oracle consumers retire in tasks 15–16. |
| Eight color outputs and existing render-pass/resource descriptions | Temporary consumers are Maxwell lowering and both paths until rendering replacement. Task 12 must derive required semantic state and host limits, removing old pass reconstruction and any Maxwell encoding limit used as a universal renderer limit. |
| Logical access stages/modes/usages and backing versions | Retain required semantic effects. The renderer alone translates these into enabled host features and Vulkan synchronization. Canonical visibility stays in `nixe-memory`. |

The startup interface and typed values establish the ownership boundary. Vulkan
object allocation, generational cache lookup, queue timelines and actual WSI lease
consumption are implemented by their device, residency and presentation tasks.
