import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import zmq

from sglang.srt.managers.scheduler_components.output_sender import SenderWrapper
from sglang.srt.server_args import PortArgs
from sglang.srt.utils.network import get_zmq_socket, restrict_zmq_peers

logger = logging.getLogger(__name__)

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "[::1]"})


def bind_hint_socket(
    context: zmq.Context, endpoint: str, dp_rank: int, num_dp_ranks: int
) -> zmq.Socket:
    """Bind the --hint-zmq-endpoint ROUTER socket of one KV cache.

    ``dp_rank`` is the index of the KV cache this scheduler serves (the
    attention-DP rank, or the pure-DP replica): a TCP endpoint binds its port
    plus ``dp_rank``, an ipc endpoint gets ``-<dp_rank>`` appended when there
    is more than one. The socket is unauthenticated, so a TCP endpoint that
    is not loopback requires SGLANG_ZMQ_ALLOWED_PEERS to name its peers;
    without it the bind is refused (same remediation as CVE-2026-3060).
    """
    tcp = re.fullmatch(r"tcp://(.+):(\d+)", endpoint)
    if tcp:
        host, port = tcp.group(1), int(tcp.group(2))
        if host not in _LOOPBACK_HOSTS:
            from sglang.srt.environ import envs

            if not envs.SGLANG_ZMQ_ALLOWED_PEERS.get():
                raise ValueError(
                    f"--hint-zmq-endpoint {endpoint} is reachable from the network "
                    "but SGLANG_ZMQ_ALLOWED_PEERS is empty. Set it to the hint "
                    "controller's addresses, or bind a loopback address."
                )
        endpoint = f"tcp://{host}:{port + dp_rank}"
    elif num_dp_ranks > 1:
        endpoint = f"{endpoint}-{dp_rank}"
    socket = context.socket(zmq.ROUTER)
    socket.setsockopt(zmq.LINGER, 0)
    if tcp:
        restrict_zmq_peers(context, socket)
    socket.bind(endpoint)
    logger.info("Hint socket bound at %s (dp_rank %d)", endpoint, dp_rank)
    return socket


if TYPE_CHECKING:
    from sglang.test.scripted_runtime.tokenizer_recv_proxy import (
        ScriptedTokenizerRecvProxy,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class SchedulerIpcChannels:
    recv_from_tokenizer: Union[zmq.Socket, "ScriptedTokenizerRecvProxy"]
    recv_from_rpc: Optional[zmq.Socket]
    # --hint-zmq-endpoint, on the same ranks that pull from the tokenizer.
    hint_socket: Optional[zmq.Socket]
    send_to_tokenizer: SenderWrapper
    send_to_detokenizer: SenderWrapper
    send_metrics_from_scheduler: Optional[zmq.Socket]

    @classmethod
    def create(
        cls,
        *,
        port_args: PortArgs,
        is_rank_zero: bool,
        skip_tokenizer_init: bool,
        metrics_enabled: bool,
        enable_scripted_runtime: bool,
        hint_zmq_endpoint: Optional[str] = None,
        hint_dp_rank: int = 0,
        num_dp_ranks: int = 1,
    ) -> "SchedulerIpcChannels":
        context = zmq.Context(2)
        hint_socket = None

        if is_rank_zero:
            recv_from_tokenizer = get_zmq_socket(
                context, zmq.PULL, port_args.scheduler_input_ipc_name, False
            )
            if hint_zmq_endpoint:
                hint_socket = bind_hint_socket(
                    context, hint_zmq_endpoint, hint_dp_rank, num_dp_ranks
                )
            if enable_scripted_runtime:
                from sglang.test.scripted_runtime.tokenizer_recv_proxy import (
                    ScriptedTokenizerRecvProxy,
                )

                recv_from_tokenizer = ScriptedTokenizerRecvProxy(
                    underlying=recv_from_tokenizer
                )
            recv_from_rpc = get_zmq_socket(
                context, zmq.DEALER, port_args.rpc_ipc_name, False
            )

            send_to_tokenizer_raw = get_zmq_socket(
                context, zmq.PUSH, port_args.tokenizer_ipc_name, False
            )
            if skip_tokenizer_init:
                # No decode work: send outputs straight to the tokenizer side
                # (MultiTokenizerRouter fans out when tokenizer_worker_num > 1).
                send_to_detokenizer_raw = get_zmq_socket(
                    context, zmq.PUSH, port_args.tokenizer_ipc_name, False
                )
            else:
                # Send to the DetokenizerManager
                send_to_detokenizer_raw = get_zmq_socket(
                    context, zmq.PUSH, port_args.detokenizer_ipc_name, False
                )

            send_to_tokenizer = SenderWrapper(send_to_tokenizer_raw)
            send_to_detokenizer = SenderWrapper(send_to_detokenizer_raw)
        else:
            recv_from_tokenizer = None
            recv_from_rpc = None
            send_to_tokenizer = SenderWrapper(None)
            send_to_detokenizer = SenderWrapper(None)

        if metrics_enabled:
            send_metrics_from_scheduler = get_zmq_socket(
                context, zmq.PUSH, port_args.metrics_ipc_name, False
            )
        else:
            send_metrics_from_scheduler = None

        return cls(
            recv_from_tokenizer=recv_from_tokenizer,
            recv_from_rpc=recv_from_rpc,
            hint_socket=hint_socket,
            send_to_tokenizer=send_to_tokenizer,
            send_to_detokenizer=send_to_detokenizer,
            send_metrics_from_scheduler=send_metrics_from_scheduler,
        )
