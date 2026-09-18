"""Real-GPU validation driver for the KVCR direct linker.

Launches SGLang servers, drives prompt sequences over HTTP, and writes a JSON
report with per-request ``cached_tokens``, greedy outputs, and the linker's
stats lines scraped from the server logs. Every scenario compares restored
runs against a control so byte movement is judged by output equality and
served tokens, not by counters alone.

Scenarios:

  roundtrip  One worker. Prompt A, enough distinct prompts to evict A from the
             GPU, then A again: with the linker the replay is served from KVCR
             (cached_tokens > 0) and reproduces the control output.
  peer       Two workers with separate KVCR tiers. The source serves A and
             offloads it; the cold target replays A with an explicit kv.fetch
             hint, then controls: no hint, stale hint, dead peer.

Example:
  python kvcr_validate.py roundtrip --model Qwen/Qwen3-0.6B --gpus 0 \
      --workdir /tmp/kvcr-val --report roundtrip_tp1.json
  python kvcr_validate.py peer --model Qwen/Qwen3-0.6B --gpus 0,1 --tp 1 \
      --workdir /tmp/kvcr-val --report peer_tp1.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from sglang.srt.mem_cache.utils import get_storage_hash_str, hash_str_to_int64

STATS_RE = re.compile(r"KVCRDirectLinker stats rank=(\d+): (.*)$")


def _post(url: str, payload: dict, timeout: float = 900.0) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def _get(url: str, timeout: float = 10.0) -> int:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.status


class Server:
    def __init__(
        self,
        *,
        name: str,
        model: str,
        port: int,
        gpus: str,
        tp: int,
        workdir: Path,
        page_size: int,
        max_total_tokens: int,
        linker_config: dict | None,
        extra_args: list[str],
        dp_rank: int | None = None,
    ) -> None:
        self.name = name
        self.port = port
        self.base = f"http://127.0.0.1:{port}"
        self.log_path = workdir / f"{name}.log"
        self.linker_config = linker_config
        command = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            model,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--tp-size",
            str(tp),
            "--page-size",
            str(page_size),
            "--max-total-tokens",
            str(max_total_tokens),
            "--log-level",
            "info",
            *extra_args,
        ]
        if linker_config is not None:
            command += [
                "--enable-unified-cache-external-linker",
                "--unified-cache-external-linker-backend",
                "kvcr",
                "--hicache-storage-backend-extra-config",
                json.dumps(linker_config),
            ]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus)
        self.command = command
        self.dp_rank = dp_rank
        self.log = open(self.log_path, "w")
        self.process = subprocess.Popen(
            command, stdout=self.log, stderr=subprocess.STDOUT, env=env
        )

    def wait_ready(self, timeout_s: float = 900.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"{self.name} exited; see {self.log_path}")
            try:
                if _get(f"{self.base}/health") == 200:
                    return
            except (urllib.error.URLError, ConnectionError, OSError):
                pass
            time.sleep(1.0)
        raise TimeoutError(f"{self.name} did not become healthy")

    def generate(
        self,
        token_ids: list[int],
        max_new_tokens: int,
        kv_hints=None,
        extra_fields: dict | None = None,
        stream: bool = False,
    ) -> dict:
        payload = {
            "input_ids": token_ids,
            "sampling_params": {"temperature": 0.0, "max_new_tokens": max_new_tokens},
        }
        if kv_hints is not None:
            payload["kv_hints"] = kv_hints
        if self.dp_rank is not None:
            # DP-attention ranks own separate caches; pin every request to one.
            payload["routed_dp_rank"] = self.dp_rank
        if extra_fields:
            payload.update(extra_fields)
        if not stream:
            return _post(f"{self.base}/generate", payload)
        # Streaming separates time to first token (prepare + restore + prefill)
        # from the decode tail; the final chunk carries the full text and meta.
        payload["stream"] = True
        request = urllib.request.Request(
            f"{self.base}/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        started = time.monotonic()
        first = None
        last = None
        with urllib.request.urlopen(request, timeout=900.0) as response:
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                body = line[len("data:") :].strip()
                if body == "[DONE]":
                    break
                if first is None:
                    first = time.monotonic()
                last = json.loads(body)
        if last is None:
            raise RuntimeError("streamed generate returned no chunks")
        last["ttft_s"] = (first or time.monotonic()) - started
        return last

    def scheduler_pid(self) -> int | None:
        """PID of this server's scheduler subprocess, if it is running."""
        try:
            out = subprocess.run(
                ["pgrep", "-f", "sglang::scheduler", "-P", str(self.process.pid)],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.split()
        except OSError:
            return None
        return int(out[0]) if out else None

    def flush(self) -> None:
        # /flush_cache answers with plain text, not JSON, and refuses (400)
        # while a request is still finishing; retry briefly.
        request = urllib.request.Request(
            f"{self.base}/flush_cache",
            data=b"{}",
            headers={"Content-Type": "application/json"},
        )
        deadline = time.monotonic() + 30.0
        while True:
            try:
                with urllib.request.urlopen(request, timeout=120.0) as response:
                    response.read()
                return
            except urllib.error.HTTPError as error:
                if error.code != 400 or time.monotonic() >= deadline:
                    raise
                time.sleep(0.5)

    def stats(self) -> dict[str, dict[str, str]]:
        """Latest stats line per rank from the log."""
        latest: dict[str, dict[str, str]] = {}
        with open(self.log_path, errors="replace") as handle:
            for line in handle:
                match = STATS_RE.search(line)
                if match is None:
                    continue
                fields = dict(pair.split("=", 1) for pair in match.group(2).split())
                latest[match.group(1)] = fields
        return latest

    def log_matches(self, pattern: str) -> list[str]:
        regex = re.compile(pattern)
        with open(self.log_path, errors="replace") as handle:
            return [line.rstrip() for line in handle if regex.search(line)]

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)
            try:
                self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=30)
        self.log.close()


# Natural-text token stream for prompts (set by --natural-prompt). Random-token
# prompts leave FP8 MoE models with near-uniform logits, so greedy outputs can
# differ between two recomputes of the same input and equality checks become
# meaningless; a repeated passage gives every run a large argmax margin.
_NATURAL: list[int] | None = None
_PASSAGE = (
    "The first electronic computers filled entire rooms and were programmed by "
    "rewiring their circuits. Stored-program machines replaced the wiring with "
    "instructions kept in memory, and high-level languages let people describe "
    "a computation without naming a single register. Operating systems then "
    "shared one machine among many users, networks connected those machines, "
    "and the same ideas now run on chips small enough to lose in a pocket. "
)


def _load_natural_tokens(model: str, needed: int) -> list[int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    ids = tokenizer.encode(_PASSAGE, add_special_tokens=False)
    return (ids * (needed // len(ids) + 2))[:needed]


def _prompt(rng: random.Random, length: int, vocab: int) -> list[int]:
    if _NATURAL is None:
        return [rng.randrange(1000, vocab) for _ in range(length)]
    # A random head keeps prompts from sharing prefixes (fillers must evict the
    # prompt, not extend it); the tail is natural text from a random offset.
    head = [rng.randrange(1000, vocab) for _ in range(8)]
    start = rng.randrange(0, max(1, len(_NATURAL) - length))
    return head + _NATURAL[start : start + length - len(head)]


def _hint(source_control: str, token_ids: list[int], page_size: int) -> dict:
    pages = len(token_ids) // page_size * page_size
    hashes = get_storage_hash_str(token_ids[:pages], None, page_size=page_size)
    return {
        "protocol_version": "0.1",
        "message_id": uuid.uuid4().hex,
        "actions": [
            {
                "action_id": uuid.uuid4().hex,
                "action_type": "kv.fetch",
                "action_version": "1.0",
                "payload": {
                    "source_control_endpoint": source_control,
                    "block_hashes": [hash_str_to_int64(h) for h in hashes],
                },
            }
        ],
    }


def _effective_page_size(server: Server, requested: int) -> int:
    """The page size the server runs with; some models force their own.

    A hint hashed over the wrong page size never matches the source's keys and
    shows up as `miss_no_candidates`, so hints must follow the server.
    """
    for line in server.log_matches(r"'page_size': \d+"):
        match = re.search(r"'page_size': (\d+)", line)
        if match:
            actual = int(match.group(1))
            if actual != requested:
                print(
                    f"{server.name}: server runs page_size={actual}, "
                    f"not the requested {requested}; hints use {actual}",
                    flush=True,
                )
            return actual
    return requested


def _summary(result: dict) -> dict:
    meta = result["meta_info"]
    summary = {
        "text": result["text"],
        "cached_tokens": meta.get("cached_tokens"),
        "prompt_tokens": meta.get("prompt_tokens"),
        "e2e_latency_s": meta.get("e2e_latency"),
    }
    if "ttft_s" in result:
        summary["ttft_s"] = result["ttft_s"]
    return summary


def _linker_config(args, *, control_port: int | None) -> dict:
    config = {
        "local_dram_bytes_per_worker": args.dram_gib << 30,
        "preparation_deadline_ms": args.deadline_ms,
        "fetch_chunk_pages": 64,
        # Scheduler processes may be killed before close() logs, so log often.
        "stats_log_interval_s": 2.0,
    }
    if control_port is not None:
        config.update(
            enable_remote_hint=True,
            control_port=control_port,
            control_advertise_host="127.0.0.1",
        )
    if args.linker_config_json:
        config.update(json.loads(args.linker_config_json))
    return config


def scenario_roundtrip(args, workdir: Path) -> dict:
    """Local offload, forced GPU eviction, restore; control run without linker."""
    rng = random.Random(args.seed)
    prompt = _prompt(rng, args.prompt_tokens, args.vocab)
    fillers = [
        _prompt(rng, args.filler_tokens, args.vocab)
        for _ in range(args.max_total_tokens // args.filler_tokens + 4)
    ]
    report: dict = {"scenario": "roundtrip", "tp": args.tp, "runs": {}}
    for label, linker in (
        ("control", None),
        ("linker", _linker_config(args, control_port=None)),
    ):
        server = Server(
            name=f"roundtrip_{label}",
            model=args.model,
            port=args.port,
            gpus=args.gpus,
            tp=args.tp,
            workdir=workdir,
            page_size=args.page_size,
            max_total_tokens=args.max_total_tokens,
            linker_config=linker,
            extra_args=args.extra,
            dp_rank=args.dp_rank,
        )
        try:
            server.wait_ready()
            # A server's very first request can differ numerically (kernel
            # JIT and warmup paths); keep it out of the compared runs.
            server.generate(_prompt(random.Random(args.seed + 1), 256, args.vocab), 1)
            time.sleep(args.settle_s)
            first = server.generate(prompt, args.max_new_tokens)
            time.sleep(args.settle_s)
            for filler in fillers:
                server.generate(filler, 1)
            time.sleep(args.settle_s)
            replay = server.generate(prompt, args.max_new_tokens)
            time.sleep(args.settle_s)
            report["runs"][label] = {
                "first": _summary(first),
                "replay": _summary(replay),
                "stats": server.stats(),
                "host_pool_lines": server.log_matches(r"HiCache|host memory|hicache"),
                "linker_startup": server.log_matches(r"KVCRDirectLinker rank="),
                "command": server.command,
            }
        finally:
            server.stop()
    control, linker = report["runs"]["control"], report["runs"]["linker"]
    report["checks"] = {
        "control_replay_recomputed": control["replay"]["cached_tokens"] in (0, None),
        "linker_replay_restored": (linker["replay"]["cached_tokens"] or 0) > 0,
        "outputs_identical": control["first"]["text"]
        == control["replay"]["text"]
        == linker["first"]["text"]
        == linker["replay"]["text"],
        # The comparison that judges the restore: restored bytes must produce
        # what a recompute of the same prompt produces, on both servers.
        "linker_replay_matches_recompute": linker["replay"]["text"]
        == linker["first"]["text"]
        == control["replay"]["text"],
        "no_hicache_host_pool": not any(
            "host" in line.lower() and "alloc" in line.lower()
            for line in linker["host_pool_lines"]
        ),
    }
    return report


def scenario_peer(args, workdir: Path) -> dict:
    """Peer reuse with cold target GPU and KVCR, plus no/stale/dead-peer controls."""
    rng = random.Random(args.seed)
    prompt = _prompt(rng, args.prompt_tokens, args.vocab)
    stale = _prompt(rng, args.prompt_tokens, args.vocab)
    gpus = args.gpus.split(",")
    per_worker = max(1, len(gpus) // 2)
    source_gpus = ",".join(gpus[:per_worker])
    target_gpus = ",".join(gpus[per_worker : 2 * per_worker])
    source_control = args.control_port
    target_control = args.control_port + 100
    report: dict = {"scenario": "peer", "tp": args.tp, "runs": {}}
    source = Server(
        name="peer_source",
        model=args.model,
        port=args.port,
        gpus=source_gpus,
        tp=args.tp,
        workdir=workdir,
        page_size=args.page_size,
        max_total_tokens=args.max_total_tokens,
        linker_config=_linker_config(args, control_port=source_control),
        extra_args=args.extra,
        dp_rank=args.dp_rank,
    )
    # With DP attention the server derives a block of TCP ports from its port
    # (port + 233 onwards), so two servers on adjacent ports collide; keep the
    # target well clear of the source.
    target = Server(
        name="peer_target",
        model=args.model,
        port=args.port + 100,
        gpus=target_gpus,
        tp=args.tp,
        workdir=workdir,
        page_size=args.page_size,
        max_total_tokens=args.max_total_tokens,
        linker_config=_linker_config(args, control_port=target_control),
        extra_args=args.extra,
        dp_rank=args.dp_rank,
    )
    try:
        source.wait_ready()
        target.wait_ready()
        # Each server's very first request can differ numerically (kernel JIT
        # and warmup paths); keep it out of the compared runs.
        warm = _prompt(random.Random(args.seed + 1), 256, args.vocab)
        source.generate(warm, 1)
        target.generate(warm, 1)
        time.sleep(args.settle_s)
        control = source.generate(prompt, args.max_new_tokens)
        time.sleep(args.settle_s)
        source_endpoint = f"tcp://127.0.0.1:{source_control}"
        page = _effective_page_size(source, args.page_size)
        runs = {"control_source": _summary(control)}

        def cold_target_run(label: str, kv_hints, expect_hit: bool) -> None:
            target.flush()
            time.sleep(0.5)
            started = time.monotonic()
            result = target.generate(prompt, args.max_new_tokens, kv_hints=kv_hints)
            runs[label] = _summary(result)
            runs[label]["wall_s"] = time.monotonic() - started
            runs[label]["expect_hit"] = expect_hit

        cold_target_run("hinted", _hint(source_endpoint, prompt, page), True)
        # Warm target: a second hinted replay must be served from the local
        # tier, not presented as peer reuse.
        cold_target_run("hinted_again", _hint(source_endpoint, prompt, page), True)
        cold_target_run("no_hint", None, False)
        cold_target_run("stale_hint", _hint(source_endpoint, stale, page), False)
        cold_target_run(
            "dead_peer",
            _hint(f"tcp://127.0.0.1:{args.control_port + 500}", prompt, page),
            False,
        )
        if args.peer_prompts > 0:
            # Steady state: distinct prompts served once on the source, then
            # hinted on the target without flushing in between, so first
            # contact (metadata exchange) is separated from per-transfer cost.
            # Recompute baselines use other prompts of the same length so the
            # target's own tier cannot serve them.
            steady_rng = random.Random(args.seed + 2)
            served = [
                _prompt(steady_rng, args.prompt_tokens, args.vocab)
                for _ in range(args.peer_prompts)
            ]
            fresh = [
                _prompt(steady_rng, args.prompt_tokens, args.vocab)
                for _ in range(args.peer_prompts)
            ]
            source_texts = [
                _summary(source.generate(tokens, args.max_new_tokens))["text"]
                for tokens in served
            ]
            time.sleep(args.settle_s)
            target.flush()
            time.sleep(0.5)
            profiler = None
            if args.pyspy_steady:
                # Sample every thread of the target scheduler while the steady
                # block runs; the raw collapsed stacks attribute per thread.
                pid = target.scheduler_pid()
                profile_path = workdir / "target_steady.pyspy"
                duration = max(10, int(args.peer_prompts * 2 * 1.5) + 5)
                if pid is not None:
                    profiler = subprocess.Popen(
                        [
                            str(Path(sys.executable).parent / "py-spy"),
                            "record",
                            "--pid",
                            str(pid),
                            "--threads",
                            "--idle",
                            "--rate",
                            "200",
                            "--duration",
                            str(duration),
                            "--format",
                            "raw",
                            "-o",
                            str(profile_path),
                        ],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    report["target_steady_profile"] = str(profile_path)
                    time.sleep(1.0)
            steady = []
            for tokens, source_text in zip(served, source_texts):
                started = time.monotonic()
                result = target.generate(
                    tokens,
                    args.max_new_tokens,
                    kv_hints=_hint(source_endpoint, tokens, page),
                    stream=True,
                )
                entry = _summary(result)
                entry["wall_s"] = time.monotonic() - started
                entry["matches_source"] = entry["text"] == source_text
                steady.append(entry)
            recompute = []
            for tokens in fresh:
                started = time.monotonic()
                entry = _summary(
                    target.generate(tokens, args.max_new_tokens, stream=True)
                )
                entry["wall_s"] = time.monotonic() - started
                recompute.append(entry)
            runs["steady_hinted"] = steady
            runs["steady_recompute"] = recompute
            if profiler is not None:
                profiler.wait(timeout=120)
        time.sleep(args.settle_s)
        report["runs"] = runs
        report["source_stats"] = source.stats()
        report["target_stats"] = target.stats()
        report["commands"] = {"source": source.command, "target": target.command}
    finally:
        source.stop()
        target.stop()
    runs = report["runs"]
    texts = {label: run["text"] for label, run in runs.items() if isinstance(run, dict)}
    steady = runs.get("steady_hinted", [])
    report["checks"] = {
        "steady_all_restored": all((e["cached_tokens"] or 0) > 0 for e in steady),
        "steady_all_match_source": all(e["matches_source"] for e in steady),
        "steady_hinted_e2e_s": [e["e2e_latency_s"] for e in steady],
        "steady_hinted_ttft_s": [e.get("ttft_s") for e in steady],
        "steady_recompute_e2e_s": [
            e["e2e_latency_s"] for e in runs.get("steady_recompute", [])
        ],
        "steady_recompute_ttft_s": [
            e.get("ttft_s") for e in runs.get("steady_recompute", [])
        ],
        "hinted_restored": (runs["hinted"]["cached_tokens"] or 0) > 0,
        "no_hint_recomputed": runs["no_hint"]["cached_tokens"] in (0, None),
        "stale_hint_recomputed": runs["stale_hint"]["cached_tokens"] in (0, None),
        "dead_peer_recomputed": runs["dead_peer"]["cached_tokens"] in (0, None),
        "dead_peer_bounded_wait_s": runs["dead_peer"]["wall_s"],
        "outputs_identical": len(set(texts.values())) == 1,
        # The comparison that judges the transfer: bytes restored from the
        # peer must produce what the target's own recompute produces.
        "hinted_matches_recompute": texts["hinted"]
        == texts["no_hint"]
        == texts["control_source"],
    }
    return report


def scenario_pd(args, workdir: Path) -> dict:
    """Prefill worker with the linker plus a plain decode worker.

    Requests carry the bootstrap fields a PD load balancer adds and are posted
    to both workers, as the balancer does. The prefill worker must restore the
    replayed prompt after GPU eviction while the decode worker keeps receiving
    its KV through the ordinary prefill-to-decode transfer.
    """
    from concurrent.futures import ThreadPoolExecutor

    rng = random.Random(args.seed)
    prompt = _prompt(rng, args.prompt_tokens, args.vocab)
    fillers = [
        _prompt(rng, args.filler_tokens, args.vocab)
        for _ in range(args.max_total_tokens // args.filler_tokens + 4)
    ]
    gpus = args.gpus.split(",")
    bootstrap_port = args.control_port + 1000
    report: dict = {"scenario": "pd", "tp": args.tp, "runs": {}}
    for label, linker in (
        ("control", None),
        ("linker", _linker_config(args, control_port=None)),
    ):
        prefill = Server(
            name=f"pd_prefill_{label}",
            model=args.model,
            port=args.port,
            gpus=gpus[0],
            tp=args.tp,
            workdir=workdir,
            page_size=args.page_size,
            max_total_tokens=args.max_total_tokens,
            linker_config=linker,
            extra_args=[
                *args.extra,
                "--disaggregation-mode",
                "prefill",
                "--disaggregation-transfer-backend",
                "nixl",
                "--disaggregation-bootstrap-port",
                str(bootstrap_port),
            ],
        )
        decode = Server(
            name=f"pd_decode_{label}",
            model=args.model,
            port=args.port + 1,
            gpus=gpus[1],
            tp=args.tp,
            workdir=workdir,
            page_size=args.page_size,
            max_total_tokens=args.max_total_tokens,
            linker_config=None,
            extra_args=[
                *args.extra,
                "--disaggregation-mode",
                "decode",
                "--disaggregation-transfer-backend",
                "nixl",
            ],
        )
        pool = ThreadPoolExecutor(max_workers=2)

        def pd_generate(token_ids: list[int], max_new_tokens: int) -> dict:
            fields = {
                "bootstrap_host": "127.0.0.1",
                "bootstrap_port": bootstrap_port,
                "bootstrap_room": random.randrange(1 << 62),
            }
            prefill_future = pool.submit(
                prefill.generate, token_ids, max_new_tokens, None, fields
            )
            decode_future = pool.submit(
                decode.generate, token_ids, max_new_tokens, None, fields
            )
            return {
                "prefill": _summary(prefill_future.result()),
                "decode": _summary(decode_future.result()),
            }

        try:
            prefill.wait_ready()
            decode.wait_ready()
            first = pd_generate(prompt, args.max_new_tokens)
            time.sleep(args.settle_s)
            for filler in fillers:
                pd_generate(filler, 1)
            time.sleep(args.settle_s)
            replay = pd_generate(prompt, args.max_new_tokens)
            time.sleep(args.settle_s)
            report["runs"][label] = {
                "first": first,
                "replay": replay,
                "prefill_stats": prefill.stats(),
                "commands": {"prefill": prefill.command, "decode": decode.command},
            }
        finally:
            pool.shutdown(wait=False)
            prefill.stop()
            decode.stop()
    control, linker = report["runs"]["control"], report["runs"]["linker"]
    report["checks"] = {
        "control_replay_recomputed": control["replay"]["prefill"]["cached_tokens"]
        in (0, None),
        "linker_replay_restored": (linker["replay"]["prefill"]["cached_tokens"] or 0)
        > 0,
        "decode_outputs_identical": control["first"]["decode"]["text"]
        == control["replay"]["decode"]["text"]
        == linker["first"]["decode"]["text"]
        == linker["replay"]["decode"]["text"],
    }
    return report


_SCENARIOS = {
    "roundtrip": scenario_roundtrip,
    "peer": scenario_peer,
    "pd": scenario_pd,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", choices=sorted(_SCENARIOS))
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument(
        "--dp-rank",
        type=int,
        default=None,
        help="pin requests to one attention-DP rank",
    )
    parser.add_argument("--port", type=int, default=30100)
    parser.add_argument("--control-port", type=int, default=25100)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--max-total-tokens", type=int, default=8192)
    parser.add_argument("--prompt-tokens", type=int, default=3000)
    parser.add_argument("--filler-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--vocab", type=int, default=30000)
    parser.add_argument("--dram-gib", type=int, default=4)
    parser.add_argument("--deadline-ms", type=int, default=5000)
    parser.add_argument("--settle-s", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--linker-config-json", default=None, help="JSON merged into the linker config"
    )
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument(
        "--natural-prompt",
        action="store_true",
        help="build prompts from a repeated passage instead of random tokens",
    )
    parser.add_argument(
        "--peer-prompts",
        type=int,
        default=0,
        help="peer scenario: distinct prompts hinted back-to-back without flushes",
    )
    parser.add_argument(
        "--pyspy-steady",
        action="store_true",
        help="peer scenario: py-spy the target scheduler during the steady block",
    )
    # Unknown flags are forwarded to sglang.launch_server verbatim.
    args, args.extra = parser.parse_known_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    if args.natural_prompt:
        global _NATURAL
        _NATURAL = _load_natural_tokens(
            args.model, args.prompt_tokens + args.filler_tokens + 4096
        )
    report = _SCENARIOS[args.scenario](args, workdir)
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(json.dumps(report["checks"], indent=2))
    return (
        0
        if all(v is True for k, v in report["checks"].items() if isinstance(v, bool))
        else 1
    )


if __name__ == "__main__":
    sys.exit(main())
