# Shared HiCache Developer Guide

Shared HiCache is a default-off path for reusing KV pages that another SGLang
worker has already written to its HiCache host tier (`CPU_PINNED`). The router
sends a `cache_hints.shared_hicache` plan; the target validates it, pulls pages
from the source through NIXL, stages them into GPU KV, then schedules with a
longer cached prefix.

Core invariant: router plans are hints, not leases. The source revalidates by
hash at transfer time, and every failure must fall back to ordinary prefill.

## Fast Map

- `io_struct.py`: parses `cache_hints.shared_hicache` into `SharedHiCachePlan`.
- `plan.py`: strict plan schema, int normalization, expiry.
- `topology.py`: target/source worker, TP rank, TP shape, block-size checks.
- `scheduler_mixin.py`: local-prefix probe, TP MIN reductions, pending rids.
- `manager.py`: target-side orchestration; currently the main cleanup target.
- `target.py`: target GPU staging allocation, eviction, quarantine, insert.
- `source.py`: source request parse, hash lookup, host protection, transfer.
- `source_queue.py`: bounded source transfer worker pool.
- `control.py` / `service.py`: JSON ZMQ transfer request and completion path.
- `transfer.py`: NIXL backend and transfer-agent setup.

Do not collapse the small files back into `manager.py`. If cleaning up, move
pending target lifecycle out of `manager.py` instead.

## Request Flow

```text
router
  -> request.cache_hints.shared_hicache
  -> io_struct.py: SharedHiCachePlan
  -> scheduler_mixin.py: normal local prefix first, TP MIN sync
  -> manager.py: validate plan, derive source endpoint, allocate staging, submit
  -> service.py/source_queue.py: source worker receives async ZMQ request
  -> source.py: validate rank/hash, protect host pages, call NIXL transfer
  -> transfer.py: source HostPinned pages -> target GPU KV pages
  -> control.py/manager.py: target observes completion
  -> target.py: insert staged pages into HiRadix
  -> req.shared_hicache_hit_length extends cached prefix
```

If a request times out, gets a source reject, cannot allocate staging, or fails
insert validation, clear/drop the plan and continue prefill.

## Enable It

Requires hierarchical cache:

```bash
--enable-hierarchical-cache
```

Enable Shared HiCache in standalone SGLang:

```bash
--enable-shared-hicache
--shared-hicache-worker-id worker-a
--shared-hicache-bootstrap-port 41000
--shared-hicache-transfer-backend nixl
```

Dynamo sets `--shared-hicache-worker-id` from its endpoint connection id before
constructing the SGLang engine. For raw SGLang launches, set it manually.

Useful envs:

```bash
SGLANG_SHARED_HICACHE_FETCH_WORKERS=4
SGLANG_SHARED_HICACHE_TRANSFER_PARALLELISM=8
SGLANG_SHARED_HICACHE_NIXL_TELEMETRY=false
```

Every rank binds `tcp://<server_args.host>:<shared_hicache_bootstrap_port + tp_rank>`.
Plans carry `source_host` and `source_bootstrap_port`, so targets derive the
source rank endpoint the same way disaggregation derives bootstrap addresses.

## Plan Requirements

Minimum useful `cache_hints.shared_hicache` fields:

```json
{
  "target_worker_id": "worker-b",
  "source_worker_id": "worker-a",
  "source_host": "10.0.0.11",
  "source_bootstrap_port": 41000,
  "source_medium": "CPU_PINNED",
  "block_hashes": [111, 222],
  "kv_block_hashes": [111, 222],
  "planned_prefix_blocks": 2,
  "block_size_tokens": 64,
  "expires_at_ms": 1760000010000,
  "source_tp_rank": 0,
  "source_tp_size": 4,
  "target_tp_rank": 0,
  "target_tp_size": 4
}
```

Rules that matter in practice:

- `source_medium` must be `CPU_PINNED`.
- Source endpoints are not carried in plans; targets derive them from
  `source_host`, `source_bootstrap_port`, and source TP rank.
- `block_size_tokens` must equal the local HiRadix page size.
- `source_worker_id != target_worker_id`.
- TP rank/size must match local topology.
- `kv_block_hashes`, when present, must match `block_hashes` length.
- Batched requests need a `cache_hints` list matching batch size.
- `parallel_sample_num > 1` with Shared HiCache hints is rejected.

## Target Staging

Allocation order:

1. Full page-aligned staging allocation.
2. Evict ordinary target GPU KV and retry full allocation.
3. Stage largest page-aligned suffix that fits.
4. If nothing fits, drop the plan and prefill normally.

Page cleanup:

- Known no-write failure: free target device indices.
- Timeout / maybe-inflight transfer: quarantine target device indices.
- Insert failure after possible write: quarantine unless known safe to free.

## Source Lifetime

Before source lookup, the plan can be stale and no pin is held.

During transfer:

- Resolve host pages through the protected lookup path.
- Protect nodes with `TreeNode.protect_host()`.
- Release with `TreeNode.release_host()` in `finally`.
- Never transfer from host-index entries without protecting owner nodes.

## Host Event Gotcha

Do not publish HostPinned CPU events before write-through DMA ack. Routers treat
those events as source visibility.

Keep this regression test:

```text
test_hiradix_split_pending_write_through_publishes_cpu_prefix
```

It protects the pending write-through split case:

```text
pending host-backed node splits
  -> ack drains
  -> publish prefix fragment and both suffix fragments
```

Do not replace this with early CPU event publication or split-time DMA drains.

## Validate Locally

```bash
python3 -m pytest test/registered/hicache/test_shared_hicache.py -q
python3 -m compileall -q \
  python/sglang/srt/mem_cache/shared_hicache \
  python/sglang/srt/mem_cache/hicache_host_index.py \
  python/sglang/srt/mem_cache/hiradix_cache.py \
  python/sglang/srt/mem_cache/radix_cache.py \
  test/registered/hicache/test_shared_hicache.py
git diff --check
rg -n "G2[p]lus|g2[p]lus|Remote[G]2|remote_[g]2|remote_[w]2" python docs docs_new test
```

Key tests cover plan parsing, batched hints, TP convergence, source reject
paths, host protection, staging/eviction/partial allocation, quarantine, and the
split write-through event fix.

## Runtime Proof

A `/v1/models` response only proves the server started. Real proof needs:

- source logs: `SharedHiCache NIXL transferred`
- target logs: `Shared HiCache staged`
- nonzero `cached_tokens_details.shared_hicache` when requested
- nonzero `sglang:cached_tokens_total{cache_source="shared_hicache"}`
- nonzero `sglang:shared_hicache_tokens_total{outcome="hit",reason_code="ok"}`
- no `target_staging_alloc_failed`, transfer timeout, queue-full, rank reject,
  or direct-transfer failure logs

## Debug First

No hits:

- Was hierarchical cache enabled on both workers?
- Is Shared HiCache enabled and is `worker_id` set?
- Did the target request actually include `cache_hints.shared_hicache`?
- Is the plan expired?
- Do source/target worker ids and TP ranks match reality?
- Does the source have HostPinned events for the requested hash chain?

Source request missing:

- Does the plan have the right source host and bootstrap port?
- Are ZMQ TCP ports reachable from target to source?

Target falls back:

- Check staging counters first.
- Check hash mismatch / non-contiguous page rejection logs.
- Check quarantine counters for timeout or maybe-inflight transfers.

Perf looks bad:

- First prove effective reuse with cached-token and staging metrics.
- Then inspect transfer setup/wait time.
- Do not tune NIXL tail before proving source visibility and target staging.
