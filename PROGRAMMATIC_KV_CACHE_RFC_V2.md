# RFC: Programmatic KV Cache for Agentic Workloads

Agent workloads make the value of a KV block predictable from above the engine, but that value is invisible to request-local LRU. We propose exposing a narrow, orchestrator-initiated hint surface so an external router can pass cache intent to SGLang without making invasive changes to the engine scheduler and cache manager. In this way, SGLang keeps ownership of scheduling and memory and is free to clip, defer, or reject any hint. 

The first concrete hint we propose is Shared HiCache. This allows for a router to request KV cache to be moved from a workers G2 to another workers G1 directly within HiCache without a need for a secondary memory pool. This scaffolding paves the way for various other KV trasnfer mechansism that directly improve performance of agentic infernece. 

---

## Motivation

Agentic inference creates KV-cache patterns that are visible above the engine but invisible to request-local policy. Some examples include:

- **Stable prefixes dominate a turn.** System prompt, skills/guidance, repo and workspace context, task instructions, and prior tool outputs all carry over; the new request often appends only a small suffix. Missing on the prefix forces an expensive prefill.
- **Tool-call latency spans orders of magnitude.** A local shell command is milliseconds; an external service wait is minutes. The right KV policy depends on that gap, and request-local policy cannot see it.
- **Context is not append-only.** Deep-agent loops compress, summarize, and refine history and swap skills/tool definitions mid-trajectory. The engine keeps KV for context the agent might have already discarded - zombie cache.
- **Subagent lifecycle is a signal.** When a subagent spawns, the main-agent KV should be immediately evictable and prefetched back on close. When a subagent returns a summary, its scratch KV is dead and should not linger.
- **Concurrency causes KV thrash.** Interleaved agents evict each other's still-live prefixes under load.

## Problem Statement

The orchestrator knows the structure request-local policy cannot: which sessions are live, which token ranges are shared prefixes vs unique tails, which tool gap is 10ms vs 10min, when a subagent opened and closed. The engine sees a block hash and a refcount.

Current shape:

```text
request arrives
  -> router chooses target worker
  -> target worker checks local cache
       hit:  reuse
       miss: recompute or apply local offload policy
```

This is insufficient when: another worker already holds the prefix; the workload knows a request will resume soon; a session has ended and its KV should be demoted/freed; the orchestrator wants to protect high-value KV; or local policy cannot tell a short tool gap from a long one.

The missing abstraction is a narrow, observable surface where the orchestrator biases the cache manager without owning scheduler or memory internals. Letting an external system manipulate cache internals directly is the wrong design: it is brittle, it duplicates scheduler policy outside the engine, and it forces a refactor of the scheduler/cache-manager boundary. Hints keep ownership inside SGLang and let the orchestrator soft-influence behavior at request and lifecycle boundaries.

---

## Design Principles

1. **Orchestrator owns policy; engine executes.** The agent-graph / workflow intelligence lives outside. The engine understands priority, TTL, session membership, tier - nothing about why.
2. **Zero overhead when unused.** No allocations, heap ops, or timestamp checks for KV that carries no hint. Un-hinted workloads behave exactly like today.
3. **Hints are soft, bounded, and safe to reject.** The engine may accept, clip, defer, or ignore. Every hint is observable. Nothing a client says can pin memory unboundedly or deadlock the scheduler.
4. **Router-initiated by default.** Workloads can still emit intent, but the router is where workload context merges with global KV placement, worker load, health, and admission. The router has: a global KV index from events, built-in HA/fault-tolerance, existing overlap/load routing + admission control, and (with the harness<->orchestrator work) trajectory awareness, not just request awareness.

---

## Hint Taxonomy (conceptual)

The hint taxonomy at a glance:

| Hint | Intent | One-line |
|---|---|---|
| **Priority / Retention** | bias eviction order | some token ranges are worth more than others when eviction is unavoidable |
| **Pin / Retain** | bounded protection | keep a high-value prefix resident for a TTL |
| **Prefetch / Onboard** | move KV hotter early | warm GPU before the continuation lands (e.g. during long tool call) |
| **Session Lifecycle** | treat a session's KV as a unit | open/close a session; bulk-demote or free its KV deterministically |
| **Demote / Offload** | move KV colder, not gone | spill to host/disk/external during long pauses instead of recompute. Moves together with Prefetch/Onboard |
| **Share** | reuse peer KV | natively pull a prefix from another worker / shared tier |

Per hint: what it means, when it fires, the closest external prior art, and the sglang/dynamo machinery we already have to build it. No request schemas here by design - those belong in per-hint implementation RFCs.

### Priority / Retention

Bias eviction order rather than hard-pin: some token ranges are worth more than others when eviction is unavoidable. The orchestrator attaches a relative priority (optionally with a retention duration) to a token range, and under memory pressure the engine evicts low-priority KV first. This is the softest hint - it never reserves memory, it only reorders the victim list. Prior art: vLLM #37003 `RetentionDirective` (priority 0-100 + TTL) and TRT-LLM `KvCacheRetentionConfig` (priority 0-100, duration_ms, decode-time priority). Ours: #21045 replaces a hard TTL pin with a priority-biased retention duration in radix-cache eviction; dynamo #7384 injects `retention_seconds`.

### Pin / Retain

Keep a high-value prefix resident, or protected from ordinary eviction, for a **bounded** TTL. For: expensive retrieved context, shared planner state, a short tool-call gap where recompute dominates latency. Examples include the Continuum style of pinning on tool boundaries and how Anthropic does `cache_control {ttl}`.

### Prefetch / Onboard

Move KV into a hotter tier *before* it is needed: warm GPU for a likely next-turn prefix; pull shared KV into a freshly selected worker; reload main-agent KV during a subagent's close.

### Session Lifecycle

Tag KV by session/subagent so the cache applies lifecycle-aware policy. Open a session; every block produced under it carries the tag; on close, the whole session's KV is demoted or freed as a unit.

### Demote / Offload

Move KV to a colder tier instead of dropping it: long external tool call, paused trajectory, low-priority-but-reusable subagent state, memory pressure where recompute is expensive. Demote moves together with Prefetch/Onboard - the KV offloaded during a pause is the KV warmed back before the continuation resumes.

- **Prior art:** vLLM #39305 (selective offload - store only part of a prompt, since 40-60% of KV is never reused) and #38260 (multi-tier `TieringManager`: one primary CPU tier + secondary tiers, "secondary tiers own their evictions," TP-invariant canonical CPU layout). TRT-LLM gates offload via `secondaryOffloadMinPriority` - i.e. priority and tiering compose.
- **Ours:** sglang HiCache already owns L1/L2/L3; demotion should name a target tier (host/disk/external) when it can.

### Share

Reuse a prefix that already lives on another worker or a shared tier: route a continuation to a less-loaded worker but pull the prefix from the old one; share a common prefix across sibling subagents; warm a scale-up worker.

- **Ours / shipped:** this is `cache_hints.shared_hicache` (`SHARED_HICACHE_RFC.md`)
  - source `CPU_PINNED` HiCache -> NIXL -> target GPU -> radix insert, with the router providing the peer-reuse plan. It is the existence proof for the whole model: the machinery to move KV natively between workers is what every other hint also needs.

---

## Prior Art & Positioning

### Landscape

| Effort | Layer | Core idea | Relation to us |
|---|---|---|---|
| vLLM #37003 (vMaroon/IBM, llm-d) | engine | per-range `RetentionDirective` (priority 0-100, TTL) + `retention_scope`; dual-structure evictor; "the API is the product" | same policy/mechanism split; we keep mechanism radix-native and session-first |
| vLLM #37168 (xinrunxue/ascend) | engine | active invalidation (`POST /release_kv_cache`), session-aware refcounting, Aging/Fresh two-zone | our `session_control` close == active invalidation; our radix tag == their refcount, simpler |
| vLLM agentic-api #18 (cyr...) | orchestrator | Session Cache Manager maintaining `session_id -> [block_hash]` map; 3 ways to reconstruct the mapping | the design we explicitly *avoid* - see Two Schools |
| vLLM #39305 (ruocco) | engine/connector | selective offload of a prompt prefix (token offset) | our Demote/Offload hint, narrower |
| vLLM #38260 (dannyharnik) | engine | multi-tier `TieringManager`, secondary tiers own evictions | the tiering substrate Offload/Demote steers |
| TRT-LLM `KvCacheRetentionConfig` | engine | canonical token-range retention API (priority 0-100, duration_ms, decode priority, offload min-priority) | the cross-engine prior art a single Dynamo payload could target |
| KVFlow / Continuum / Tail-Optimized / MARCONI / KVCache-in-the-Wild | research | workflow-aware / TTL / SLO / cost-aware / workload-aware eviction | the evidence base; each maps to one hint (see Motivation table) |

### Two schools of "session awareness" (and where we sit)

- **Orchestrator-maintains-the-map** (agentic-api #18): the agent layer keeps `session_id -> [block_hash]`. Its own RFC names the fatal friction: *"no shared identifier exists between the two sides."* It must then reconstruct vLLM's internal rolling block hash (coupling to a non-public algorithm), or get block metadata back in the response, or push `session_id` into the engine anyway.
- **Engine-tags-the-block** (vLLM #37168 refcount; **our #27058 radix-native**): the engine tags each radix node with the session(s) that produced it. No shared hash to reconstruct, no mapping to keep in sync. Open/close is a tag-policy change; close bulk-frees by tag.

We are router-initiated like vLLM #37003, but session-first and radix-native. Tagging radix nodes with their session sidesteps the "no shared identifier" problem agentic-api #18 is stuck on: there is no internal block hash to reconstruct and no external mapping to keep in sync. We treat session lifecycle (open/close) as the primary lever and priority/TTL as the secondary one; most of the vLLM cluster does the reverse. Our #7665 result backs this - for subagent KV, lifecycle reclamation beats priority eviction. Throughout, the engine keeps full ownership of scheduling and memory; the orchestrator only biases, which is why every hint is safe to reject and cannot deadlock the scheduler. Shared HiCache already proves the worker-to-worker KV-movement machinery, which is the hard part the other RFCs still have to build.

---

## How this builds on our existing sglang / dynamo work

The v2 is not greenfield. Map of the building blocks already in flight:

| Capability | sglang | dynamo |
|---|---|---|
| Session lifecycle (radix-native) | #27058 (radix-native), #27024/#27025 (why pinned slots fail), #22273/#21875 (leak fixes) | #7377/#7665 (`session_control` open/close), pi-provider#4 (per-subagent sessions) |
| Priority / retention | #21045 (priority retention duration) | #7384 (retention_seconds injection) |
| Pin / TTL | #18941 (TTL prefix pinning, `/hicache/pin_prefix`) | #6213 (`nvext.cache_control` TTL), #6571 (Anthropic top-level `cache_control`) |
| Agent identifiers / ATIF | #24656 (agent_hints/sglext design feedback) | #8789 (`agent_context`), #9140 (ATIF alignment) |
| Share / peer reuse | `cache_hints.shared_hicache` (shipped) | router peer-reuse plan |
| KV events / observability | #16030, #18205/#18209 (tier-transition events), #26974/#26976 (tier metrics) | KV index, overlap scores #9538 |
| Program-level scheduling | - | #9448 (thunderagent_router) |

Sequencing: Shared HiCache is shipped and proves the model. The spine of the first programmatic-KV milestone is session lifecycle - radix-native sessions (#27058) plus `session_control` open/close - with priority/TTL pinning (#18941, #21045) as the next layer. Prefetch/onboard and multi-tier demotion follow.

---

## Request Path vs Control Path

Two delivery channels, by lifetime of the intent:

- **Request-scoped** (the `cache_hints` envelope on a request): priority, pin, prefetch-for-this-turn, share. Travels with the request the hint is about.
- **Lifecycle / out-of-band** (a lightweight control path between requests): session open, session close, "evict main-agent KV now," explicit invalidation after compaction. These do not belong on any one request.

vLLM #37168 puts active invalidation on `POST /release_kv_cache`; our equivalent is `session_control` open/close (#7377/#7665). v2 keeps both channels but does not fix their wire format here.

The control path is router-to-engine only, never user-facing. This bounds the threat model: only the session owner (the router) can open, close, or invalidate a session, so there is no DoS-by-flush - the concern both #37168 and agentic-api #18 raise.

---

## Open Questions

1. **Session identifier source.** How does the engine learn session boundaries - `session_control` open/close (ours), `cache_salt` (#37168), or `session_id` passthrough (#18 Path 3)? We lean on explicit open/close; confirm.
2. **Who computes TTL.** Client passes expected tool latency / deadline vs. engine infers from tool-duration history (Continuum). Explicit is simpler and avoids cold-start; inferred is more robust to lying clients.
3. **SLO-aware eviction.** Do we let a client pass a TTFT budget so eviction can be tail-optimized (Tail-Optimized's TEL-safe/unsafe split)? Powerful but needs a next-turn-length estimate.
4. **Cross-engine compatibility.** How wire-compatible do we stay with TRT-LLM / vLLM token-range directives so one Dynamo payload targets all three? (This doc intentionally does not commit a schema - decision deferred.)
5. **Session migration.** agentic-api #18 wants session KV to follow a session across instances (their goal G3). In scope for us, or explicitly out?
6. **Hybrid / SSM models.** MARCONI's no-partial-rollback constraint: do retention/ offload hints behave correctly for recurrent layers, or are they attention-only for now?
7. **Workload-category signal.** KVCache-in-the-Wild shows reuse is predictable per (request-type x turn) category. Worth letting clients tag a category that the engine's default policy can use, or leave it to explicit priority?
8. **Prefetch/evict pairing semantics.** When a subagent spawns, what is the exact contract: evict-then-prefetch, and what guarantees the main-agent KV is back before the continuation schedules?

---

## Non-Goals

- Not a public user-facing cache API; the producer is the router/orchestrator.
- Not direct orchestrator manipulation of cache-manager internals; hints only bias.
- Not a replacement for local prefix matching or for sglang's LRU - it augments and reorders them.
- Not a guarantee the engine obeys any hint; accept/clip/defer/reject is always allowed.
- Not a commitment, in this doc, to any concrete request schema or field names.
- Not (yet) cross-instance session migration, hybrid/SSM correctness, or SLO-budgeted eviction - those are tracked as Open Questions.

---

## References

External RFCs / APIs:
- vLLM #37003 - Context-Aware KV-Cache Retention API (Prioritized Evictions): https://github.com/vllm-project/vllm/issues/37003 (impl PR #38514; fork moreh-dev/vllm#10 aligns to TRT-LLM)
- vLLM #37168 - Active Coordination and Two-Zone Scheduling for Long-Running Agents: https://github.com/vllm-project/vllm/issues/37168 (impl vllm-ascend#6722)
- vLLM agentic-api #18 - Session-aware KV cache management: https://github.com/vllm-project/agentic-api/issues/18
- vLLM #39305 - Selective KV Cache offload: https://github.com/vllm-project/vllm/issues/39305 (impl PR #39983)
- vLLM #38260 - Multi-tier KV offloading via the offloading connector: https://github.com/vllm-project/vllm/issues/38260
- TensorRT-LLM `KvCacheRetentionConfig` / `TokenRangeRetentionConfig` (token_start/token_end/priority 0-100/duration_ms; default 35; decode_retention_priority; secondaryOffloadMinPriority)

Research:
- KVCache in the Wild (Alibaba traces): https://arxiv.org/abs/2506.02634
- Continuum (KV cache TTL for multi-turn agents): https://arxiv.org/abs/2511.02230
- Tail-Optimized Caching for LLM Inference: https://arxiv.org/abs/2510.15152
- KVFlow (workflow-aware prefix caching): https://arxiv.org/abs/2507.07400
- MARCONI (prefix caching for hybrid LLMs): https://arxiv.org/abs/2411.19379

Our prior work (sglang):
- #24656 (agent-aware KV phase 1 / API feedback), #21846 (distributed KV roadmap), #27058 (radix-native sessions), #27024 / #27025 (streaming-session deadlock + bound), #22273 / #21875 (streaming-session leak fixes), #18941 (TTL prefix pinning), #21045 (priority retention duration).

Our prior work (dynamo):
- #7665 / #7377 / #7384 (session_control + ephemeral KV routing), pi-dynamo-provider#4 (per-subagent sessions), #6213 / #6571 (Anthropic-style cache_control), #8789 / #9140 (agent_context / ATIF), #9448 (thunderagent_router program scheduler).

Companion docs:
- `PROGRAMMATIC_KV_CACHE_RFC.md` (v1 envelope + hint sketch)
- `SHARED_HICACHE_RFC.md` (first concrete hint)
