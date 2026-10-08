# Incremental graphics preparation and physical upload knowledge

Uploaded MME words are decoded once into immutable 16-word pages. Invocations
retain their uploaded version and fetch within the current page directly; branches,
delay slots, dynamic register reads and emitted-method order retain their original
semantics. Shadow registers share the direct paged register file.

Class registers use lazy 64-register pages with direct lookup of validated
12-bit method indices. Unset values, last-write provenance and repeated-command
effects remain distinct. A repeated constant-buffer binding command captures
the current selector address/size even when its command word has not changed.

Resource resolution retains immutable buffer/image components behind shared
owners. Incremental per-domain register identities select candidates and restore
the same key when transient bindings return to their previous values. Exact
copy-on-write pages reject fingerprint collisions; constant-buffer keys capture
the selector address and size at the binding command. Exact address, interpretation,
canonical mapping, CPU content epochs and descriptor dependencies decide actual
reuse. Changing one buffer binding retains unrelated components. Descriptor
handles, image interpretation and sampler interpretation can be retained
independently. Composition rechecks physical aliases whenever components change.
In-flight work owns its previous components and backend dependencies.

Prepared draws retain immutable vertex/access/dependency arrays. Pure state
validation is cached separately from resource and alias validation. Fixed-state
identity excludes vertex/index addresses, limits and index ranges. Its incremental
fingerprint returns to the same value when transient register writes are restored;
exact copy-on-write register pages reject collisions and distinguish unset values.
Up to 64 templates (also bounded by the configured pipeline-cache capacity) retain
alternating configurations. Semantic shader identities include translation IDs,
fingerprints and resource uses, rather than the owner of a temporary array.

Every template reuse installs current attachments, render-pass IDs, descriptor
tables and buffer views, while sharing validated vertex attributes. Copy-on-write
retains older in-flight bindings. Format, topology, raster, viewport and depth
changes select another template; BEGIN state and indexed mode remain consumption
checks. Retirement invalidates a prepared draw only when it consumes the retired
view or descriptor. Templates never restore their old backend bindings.

Each draw plan also retains an immutable binding snapshot. Consumed component
identity selects changes independently of the complete resource composition.
Buffer, sampled-image and sampler rebinding prepares only changed components,
preserves unrelated dependencies, and refreshes shader descriptors/accesses only
when their inputs change. Sampled images still consume materialization and copy
revision state; attachment changes use normal render-pass preparation. View,
sampler and descriptor retirement invalidate every consuming snapshot while
retaining fixed validation. Previously queued operations keep their own bindings.

GPU upload knowledge lives in the existing canonical-page mirrors. Each page
records exact known intervals; a bounded page index rejects disjoint physical
writes without scanning large images. Immutable canonical ranges cache only
conservative page-identity bounds for this rejection. Exact segments still
validate overlaps, including scatter/gather and repeated aliases. Small writes
visit only their logical subrange, avoiding scans of allocation prefixes.

Commands invalidate physical knowledge and record dirty ranges in execution
order, including distinct positions within one submission. Ordered buffer
uploads then copy their exact bytes into the mirror. GPU copies, clears,
draw/dispatch writes and image uploads invalidate overlapping knowledge. CPU
preparation of a new epoch replaces the mirror and removes its knowledge index.
Reclamation removes index entries with their mirror. No separate page authority,
CPU ownership state or optimistic completion fence is introduced.

A CPU demand waits the real last-use submissions of relevant representations;
this also covers a request racing a newer accepted producer. Canonical owner and
epoch validation still rejects stale requests. Only unknown buffer fragments
are downloaded, and image downloads cannot overwrite newer known upload bytes.
Dirty-range completion applies to the original demanded fragments after real
completion. Other prefetched pages retain mirrors without changing canonical
ownership. Fragmentation beyond 64 known intervals loses only the optimization.

Regression coverage includes selective component reuse, repeated selector
binding, old in-flight work, mapping removal, changed backend dependencies,
partial knowledge, CPU epoch reset and interval exhaustion. Physical-GPU tests
combine aliased buffers, ordered uploads, unknown GPU writes, retirement,
in-flight CPU reads and subsequent CPU writes.

GPFIFO acceptance returns its reserved guest fence before frontend interpretation.
The existing bounded frontend owner processes accepted work in order; only actual
backend completion advances fences. State mutations and later submissions retain
the frontend gate, while timeline queries and waits can proceed concurrently.
Presentation exports follow the accepted producer through that gate without
suspending QueueBuffer. Failed producers discard dependent exports and publish a
fatal error before releasing waiters. Teardown drains frontend ownership before
removing channels or mappings. Retained mapping snapshots and canonical owners
keep queued inputs alive without copying the entire guest address space.
