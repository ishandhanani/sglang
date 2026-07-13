# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Lifecycle management for Dynamo's native SGLang gRPC sidecar."""

import logging
import multiprocessing as mp

from sglang.srt.utils.common import kill_itself_when_parent_died, kill_process_tree
from sglang.srt.utils.network import NetworkAddress
from sglang.srt.utils.watchdog import SubprocessWatchdog

logger = logging.getLogger(__name__)

_SIDECAR_JOIN_TIMEOUT_SECS = 10


def _loopback_host(host: str) -> str:
    if not host or host == "0.0.0.0":
        return "127.0.0.1"
    if host == "::":
        return "::1"
    return host


def build_dynamo_sidecar_args(server_args) -> list[str]:
    endpoint = NetworkAddress(
        _loopback_host(server_args.host), server_args.grpc_port
    ).to_url()
    return ["--sglang-endpoint", endpoint]


def _run_dynamo_sidecar(args: list[str]) -> None:
    kill_itself_when_parent_died()
    try:
        from dynamo.sglang_sidecar.main import main as run_dynamo_sidecar
    except ImportError as e:
        raise RuntimeError(
            "--enable-sidecar requires Dynamo with the dynamo.sglang_sidecar "
            "entrypoint installed in the SGLang environment."
        ) from e

    run_dynamo_sidecar(args)


class DynamoSidecar:
    def __init__(self, proc, args: list[str]):
        self.proc = proc
        self.args = args
        self._watchdog = SubprocessWatchdog(
            processes=[proc], process_names=["dynamo_sidecar"]
        )

    def start(self) -> None:
        self.proc.start()
        self._watchdog.start()
        logger.info("Dynamo sidecar started pid=%s args=%s", self.proc.pid, self.args)

    def stop(self) -> None:
        self._watchdog.stop()
        if self.proc.is_alive():
            self.proc.terminate()
            self.proc.join(timeout=_SIDECAR_JOIN_TIMEOUT_SECS)
        else:
            self.proc.join(timeout=0)

        if self.proc.is_alive():
            logger.warning("Dynamo sidecar did not terminate; killing process tree")
            kill_process_tree(self.proc.pid, wait_timeout=_SIDECAR_JOIN_TIMEOUT_SECS)


def start_dynamo_sidecar(server_args) -> DynamoSidecar:
    sidecar_args = build_dynamo_sidecar_args(server_args)
    proc = mp.get_context("spawn").Process(
        name="dynamo_sidecar",
        target=_run_dynamo_sidecar,
        args=(sidecar_args,),
    )
    sidecar = DynamoSidecar(proc, sidecar_args)
    sidecar.start()
    return sidecar
