# Hybrid KVCR linker follow-up

This follow-up targets Ishan's `idhanani/kvcr-direct-linker` branch, not SGLang
main. It ports the reusable changes from the DeepSeek-V4.1 and Kimi K3 linker
experiments onto that branch's newer cache interfaces. It does not add model
implementations or weights.

## What changed

- Expose DeepSeek C1/C2 KV and indexer buffers in logical tree-page units. A
  smaller physical indexer page is grouped into the same logical object; absent
  C4/C128 pools are not dereferenced.
- Expose hybrid full-attention and Mamba/KDA state, including the current pool's
  transferable sibling state. Preserve pipeline-stage layer numbering and
  rank-qualified keys for TP-sharded state. Restore a tree checkpoint separately
  from the request's mutable state. Duplicate checkpoint insertion must not
  launch DMA into the slot that was just freed.
- Size sparse checkpoint capacity according to the checkpoint grid instead of
  reserving one large state object for every KV page. The byte budget remains
  per rank; rounding includes the final partial checkpoint interval.
- Optionally restore selected object spans directly from a peer into registered
  HBM, with per-layer readiness, destination-descriptor reuse, chunking, a bounded
  in-flight operation window, and preparation/assembly/submission/completion
  counters. A layer is ready only after all contributing requests and chunks
  complete. Submission/entry failures drain already submitted operations before
  reporting terminal load failure.

The companion KVCR branch is `ai-dynamo/kvcr:codex/linker-hybrid-followup`, based
on `idhanani/framework-gpu-regions`. It adds a disjoint-layout eviction index,
ordered named-span delivery, and immediate progress-thread source submission.
`deliver()` retains its existing signature; `deposit()` and `fetch()` still
represent whole objects. Local delivery supports the same projection because
KVCR may select a target-local copy before consulting its peer hint.

## Enabling the experimental path

Keep the existing linker configuration, remote hints, advertised control
endpoints, memory registration and per-rank budgets. Add these fields to
`--hicache-storage-backend-extra-config` to test direct peer restoration:

```json
{
  "direct_remote_restore": true,
  "progressive_remote_restore": true,
  "direct_remote_descriptor_cache": true,
  "direct_remote_chunk_pages": 128,
  "direct_remote_inflight_layers": 4
}
```

This JSON is an **addition**, not a complete launch configuration.
`direct_remote_restore` defaults to false: the existing claimed target-DRAM
staging path remains the default. Chunk/window zero means unbounded within the
admitted batch; the window counts delivery operations, including page chunks,
despite its historical `inflight_layers` name. A page here is one pool object
key, not necessarily one physical buffer or one token. Set
`progressive_remote_restore=false` for a wait-for-all control run.

Direct restore charges ALL_PAGES pools for the whole prefix and TRAILING_PAGES
pools only for their required tail. Lookup still intersects valid resume
boundaries, including sparse checkpoint gaps.

## Safety and review limits

This is an opt-in prototype. Query is advisory; there is **no reservation/lease
spanning query, admission and every layer's delivery**. KVCR protects sources
while each submitted operation reads them, but eviction before a later operation
claims its source can still cause failure. After admission that is a fatal layer
counter error, not a silent miss or recompute fallback. Keep this disabled for
production until a source-reservation protocol and its failure lifecycle are
reviewed. Destination storage must remain alive until all submitted DMA drains.

Both endpoints need the companion named-span delivery support. An older source
does not understand the subset protocol and will reject a partial layout.
Mamba external-linker MTP draft pools remain explicitly unsupported. The new
Mamba assembly uses the current branch's transfer-entry iterator rather than
copying the older benchmark snapshot's memory-pool implementation.

Not carried forward: already-upstream KV-hint transport/RLock fixes; batched
control (`deliver_many`) and multi-layer grouping ablations; experimental leases;
hard-coded cluster endpoints, model paths or global polling/GIL tuning; Lin's
separate HiCacheStorage adapter. These PRs cover the linker path.

## Validation and reproducibility

Tests were run in an isolated directory of the existing Linux/aarch64 GB300
container, with this SGLang checkout and the companion KVCR checkout first on
`PYTHONPATH`. No serving processes were restarted or benchmark settings changed.

```bash
PYTHONPATH=python:/path/to/kvcr/src SGLANG_RUST_BUILD_MODE=never \
python3 -m pytest \
  test/registered/unit/mem_cache/test_kvcr_direct_linker.py \
  test/registered/unit/mem_cache/test_kvcr_hybrid_followup.py \
  test/registered/unit/mem_cache/test_linker_pool_assembler.py \
  test/registered/unit/mem_cache/test_unified_cache_linker.py \
  -q -k 'not Rust'
```

The tests use real CPU tensors and controlled transports to verify bytes,
claims, geometry, checkpoint deduplication, partial completion and failure
draining. Seven Rust-backed cases are outside this invocation because a matching
native extension was not available in the isolated checkout. This is not a new
end-to-end model or performance validation of the rebased commits.

Historical experiments used 116,000-token prompts, four output tokens and serial
requests, with source population, target remote restoration, immediate HBM reuse,
and a distinct cold recompute control. DeepSeek-V4.1 used two independent TP4/EP4,
DP1 workers and 256-token pages; Kimi K3 used TP8 with DP2 attention across two
four-GPU nodes and 64-token pages. Those GPU runs motivated these changes but
used older runtime overlays. Do not attribute their timing numbers to these
new rebased commits without repeating the workload.
