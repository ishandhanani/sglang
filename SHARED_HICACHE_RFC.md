# RFC: Shared HiCache

## Summary

Shared HiCache lets one SGLang worker reuse another worker's HiCache host-tier
KV blocks when an external router provides an explicit plan.

The first supported path is:

```text
source worker CPU_PINNED HiCache pages
  -> NIXL direct transfer
  -> target worker GPU KV pages
  -> target radix-cache insert
```

This is the concrete implementation linked from the higher-level
[Programmatic KV Cache RFC](PROGRAMMATIC_KV_CACHE_RFC.md).

## Motivation

Dynamo can observe KV-cache placement globally through SGLang KV events. It can
know that worker A has a prefix in HostPinned memory while worker B is the
better target for load, placement, or admission.

Without Shared HiCache, routing to worker B means worker B recomputes the
prefix. With Shared HiCache, Dynamo routes to worker B and sends a peer reuse
plan. SGLang then pulls the reusable host KV from worker A into worker B's GPU
KV cache before prefill.

## Non-Goals

Shared HiCache is not:

- a generic `HiCacheStorage` backend;
- a public user-facing API;
- a replacement for local prefix matching;
- a requirement that the backend obey every router plan;
- a Mooncake Store path;
- support for every model/topology in the first PR.

The current PR is a default-off, NIXL-backed, peer-worker reuse path.

## Request Hint

Dynamo sends Shared HiCache through the generic `cache_hints` envelope:

```json
{
  "cache_hints": {
    "shared_hicache": {
      "plan_version": 1,
      "plan_id": "router-generated-id",
      "request_id": "request-id",
      "target_worker_id": "target-worker-uuid",
      "source_worker_id": "source-worker-uuid",
      "source_host": "10.0.0.11",
      "source_bootstrap_port": 41000,
      "source_medium": "CPU_PINNED",
      "block_hashes": [123, 456, 789],
      "kv_block_hashes": [123, 456, 789],
      "planned_prefix_blocks": 3,
      "start_block_index": 0,
      "block_size_tokens": 64,
      "created_at_ms": 1760000000000,
      "expires_at_ms": 1760000001000,
      "source_tp_rank": 0,
      "source_tp_size": 4,
      "target_tp_rank": 0,
      "target_tp_size": 4
    }
  }
}
```

SGLang normalizes this into `SharedHiCachePlan`. The plan intentionally does
not carry concrete source endpoints. Targets derive the source TP-rank control
endpoint from:

```text
tcp://<source_host>:<source_bootstrap_port + source_tp_rank>
```

This mirrors disaggregation: runtime metadata advertises a worker host plus a
bootstrap port, and each engine rank owns its rank-offset port.

Strict validation:

- `cache_hints` must be a dict;
- batched requests must provide one hint object per request;
- `parallel_sample_num > 1` is rejected;
- integer fields must be actual integers;
- worker id fields must be non-empty strings;
- stale `source_endpoint` payloads are rejected; use
  `source_host/source_bootstrap_port`;
- `source_medium` must be `CPU_PINNED`;
- target worker id must match local worker id;
- source and target workers must differ;
- plan version, expiry, block size, TP rank, and TP size must match.

## Request Flow

```text
1. Worker A processes a request.
2. Worker A writes reusable prefix KV into HiCache CPU_PINNED pages.
3. Worker A publishes CPU_PINNED KV events.
4. Dynamo updates its global KV index.
5. A later request arrives.
6. Dynamo chooses worker B as target and attaches cache_hints.shared_hicache.
7. Worker B validates the plan and probes local prefix coverage.
8. Worker B derives Worker A's source TP endpoint from `source_host`,
   `source_bootstrap_port`, and `source_tp_rank`.
9. Worker B allocates page-aligned target GPU KV staging pages.
10. Worker B sends a ZMQ transfer request to Worker A.
11. Worker A resolves block hashes to live HiCache host pages.
12. Worker A protects source host pages against eviction.
13. Worker A uses NIXL to write source host pages into target GPU pages.
14. Worker B receives completion notification.
15. Worker B verifies contiguous expected block hashes.
16. Worker B inserts staged GPU pages into the local radix cache.
17. Worker B schedules the request with a longer cached prefix.
18. Worker A releases source host protection.
```

## Source-Side Contract

The source worker must be authoritative for source bytes.

Required behavior:

- resolve `block_hashes` against live HiCache host pages;
- protect accepted host nodes before transfer;
- reject stale or missing pages;
- reject if source medium is not `CPU_PINNED`;
- reject if plan topology or worker id is incompatible;
- release protection on success, failure, timeout, or cancellation;
- keep host eviction from reclaiming protected pages.

The core invariant is protect-vs-evict atomicity. Source can reject, but it
cannot accept and then allow eviction to invalidate the backing pages while the
target is reading.

## Target-Side Contract

The target worker owns allocation, safety, and cache insertion.

Required behavior:

- allocate page-aligned target GPU KV blocks;
- evict local GPU KV first when needed for staging capacity;
- fall back to partial page-aligned staging when possible;
- clip requested hashes and expected pages to granted staging capacity;
- quarantine target pages on indeterminate direct-transfer failures;
- free target pages on ordinary misses or validation failures;
- verify returned pages match the expected contiguous hash suffix;
- insert staged device pages into the local radix cache;
- report `shared_hicache` cached tokens.

## Scheduler Contract

The scheduler must keep Shared HiCache from breaking TP-rank convergence.

Current behavior:

- probe local prefix before starting remote transfer;
- use TP-wide MIN reduction for status and prefix length;
- skip the request while transfer is pending;
- clamp each rank to the common prefix length;
- fall back to local prefill when any rank rejects the hint.

This matches the disagg-style polling pattern: lower status means less advanced
and dominates.

## Failure Semantics

Shared HiCache is fail-open for normal misses:

- no hint -> local behavior;
- invalid hint -> local behavior;
- expired plan -> local behavior;
- source route unavailable -> local behavior;
- source missing pages -> local behavior;
- source cannot protect pages -> local behavior;
- target staging allocation unavailable -> local behavior;
- local cache already covers the requested prefix -> local behavior.

Indeterminate direct-transfer failures are handled differently. If target GPU
pages may still be written after a timeout or backend error, the target
quarantines those pages instead of returning them directly to the allocator.

## Configuration

SGLang flags:

```bash
--enable-hierarchical-cache
--enable-shared-hicache
--shared-hicache-transfer-backend nixl
--shared-hicache-worker-id <worker-id>           # standalone/manual launch
--shared-hicache-bootstrap-port <base-port>      # standalone/manual launch
```

Rules:

- Shared HiCache requires `--enable-hierarchical-cache`.
- Worker id is an arbitrary non-empty string. Dynamo sets it from the Dynamo
  endpoint connection id; standalone launches must pass
  `--shared-hicache-worker-id`.
- Every rank binds
  `tcp://<server_args.host>:<shared_hicache_bootstrap_port + tp_rank>`.
- For cross-process or cross-host reuse, launch SGLang/Dynamo with a bind host
  reachable by the advertised `source_host`, normally `--host 0.0.0.0`.
- Plans carry `source_host` and `source_bootstrap_port`; there is no Shared
  HiCache route registry and no request-carried `source_endpoint`.
- Dynamo publishes Shared HiCache runtime metadata under
  `sglang_shared_hicache` as JSON containing `source_host` and
  `source_bootstrap_port`.
- If `--shared-hicache-bootstrap-port` is omitted under Dynamo, Dynamo derives
  the base port as `DYN_SYSTEM_PORT + 20000`; if `DYN_SYSTEM_PORT` is absent, it
  falls back to the disaggregation bootstrap port reserved for that worker.
- Same-host multi-worker tests must space `DYN_SYSTEM_PORT` by at least TP width
  so the rank-offset port ranges do not overlap.
- The supported source medium is `CPU_PINNED`.
- Runtime parallelism and timeout are controlled by `SGLANG_SHARED_HICACHE_*`
  env vars. Current defaults are `SGLANG_SHARED_HICACHE_FETCH_WORKERS=8` and
  `SGLANG_SHARED_HICACHE_TIMEOUT_SECS=1.0`.
- High source-transfer worker counts such as
  `SGLANG_SHARED_HICACHE_FETCH_WORKERS=8` require sufficient memlock for NIXL/UCX
  registration. The validated 4K fetch8 gate used inherited unlimited memlock.

## Current Validation

### Cache-Hints Retest

SGLang `f07f09a45`, Dynamo `650ea95c8`:

- validated API rename from direct `shared_hicache_plan` to
  `cache_hints.shared_hicache`;
- focused SGLang pytest: `18 passed`;
- full MiniMax TP4 ramp on H100 NVL:
  - source c32: `192/192`;
  - target c8/c16/c32/c64: all `192/192`;
- nonzero Shared HiCache request, token, transfer-byte, and cached-token metrics;
- zero direct-transfer failures and source-transfer timeouts.

### Shared HiCache + Router vs Mooncake Store

SGLang `d33602478`, Dynamo `650ea95c8`:

| Metric | Shared HiCache + router | Mooncake Store |
|---|---:|---:|
| Remote cache-read tokens | 46080 | 37728 |
| Remote target avg latency | 2858.5 ms | 3902.9 ms |
| Remote target p95 latency | 3373.3 ms | 5462.6 ms |
| E2E reuse workflow | 9.723 s | 11.693 s |
| Direct staged inserts | 48 | n/a |

Setup:

- MiniMax-M2.7;
- two TP4 workers on one 8x H100 NVL host;
- 4096-token exact prefix overlap;
- source concurrency 12;
- target concurrency 8;
- `HICACHE_RATIO=4`;
- `SGLANG_SHARED_HICACHE_FETCH_WORKERS=8`;
- `--max-total-tokens 49152`;
- all phases `12/12` HTTP 200;
- zero direct-transfer failures, source-transfer timeouts, staging allocation
  failures, transfer timeouts, or tracebacks.

### Host/Bootstrap Hard Cut

SGLang PR branch `5a8b8d367`, Dynamo `a7a2f98d8`:

- removed request-carried `source_endpoint`, `{tp_rank}` endpoint templating,
  and the intermediate Shared HiCache route registry;
- route lookup now derives the source rank endpoint from
  `source_host/source_bootstrap_port/source_tp_rank`;
- Dynamo auto-populates worker id and shared bootstrap metadata; the 4K harness
  launched `dynamo.sglang` with `--host 0.0.0.0` and no manual
  `--shared-hicache-bootstrap-port`;
- focused SGLang pytest: `17 passed`;
- MiniMax-M2.7 TP4 exact-4096 gate on TRY-67676:
  - artifact:
    `/tmp/dynamo_shared_hicache_host_bootstrap_4k_prlimit_20260603T110010Z`;
  - cold target: `12/12` HTTP 200, avg `5909.44 ms`, p95 `7627.32 ms`;
  - source populate: `12/12` HTTP 200, avg `3956.52 ms`, p95 `4098.07 ms`;
  - source write: `12/12` HTTP 200, avg `908.46 ms`, p95 `912.91 ms`;
  - remote target: `12/12` HTTP 200, avg `3794.96 ms`, p95 `4565.18 ms`;
  - remote target cache-read: `46080`;
  - remote-vs-cold speedup: `35.78%` avg latency, `40.15%` p95 latency;
  - final target metrics: `112` Shared HiCache requests, `44` OK hits,
    `134336` Shared HiCache tokens, `10.63 GB` transfer bytes, and `33584`
    `shared_hicache` cached tokens;
  - `56` NIXL transfer logs, `48` direct staged inserts;
  - zero direct-transfer failures, source-transfer timeouts, transfer timeouts,
    queue-full logs, or tracebacks.

The failed no-registry rerun immediately before this was not caused by port
derivation itself. Dynamo advertised the node IP, while SGLang had been bound to
loopback. The validated run made the launch contract explicit with
`--host 0.0.0.0` and inherited unlimited memlock via
`sudo prlimit --pid <shell-pid> --memlock=unlimited:unlimited`.

## Current PR Shape

Major implementation surfaces:

- request parsing: `python/sglang/srt/managers/io_struct.py`;
- plan schema: `python/sglang/srt/mem_cache/shared_hicache/plan.py`;
- scheduler integration:
  `python/sglang/srt/mem_cache/shared_hicache/scheduler_mixin.py`;
- target allocation/insertion:
  `python/sglang/srt/mem_cache/shared_hicache/target.py`;
- source resolution/protection:
  `python/sglang/srt/mem_cache/shared_hicache/source.py`;
- NIXL transfer:
  `python/sglang/srt/mem_cache/shared_hicache/transfer.py`;
- ZMQ control plane and endpoint derivation:
  `python/sglang/srt/mem_cache/shared_hicache/service.py`;
- manager/orchestration:
  `python/sglang/srt/mem_cache/shared_hicache/manager.py`;
- metrics: `python/sglang/srt/observability/metrics_collector.py`;
- tests: `test/registered/hicache/test_shared_hicache.py`.
