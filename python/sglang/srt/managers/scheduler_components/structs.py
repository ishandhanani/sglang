"""Data containers (R6 clusters) owned by ``Scheduler``.

This module collects small frozen dataclasses that pack groups of related
sibling fields into single attributes on ``Scheduler``. The dataclasses
hold no instance behavior beyond what ``@dataclass`` generates; factory
classmethods (e.g. ``SchedulerIpcChannels.create``) group construction
logic together with the type it builds. Collaborator-class behavior
lives in its own module under ``scheduler_components/``.
"""

from dataclasses import dataclass
from typing import Optional

import zmq

from sglang.srt.managers.scheduler_components.output_sender import SenderWrapper
from sglang.srt.server_args import PortArgs
from sglang.srt.utils.network import get_zmq_socket


@dataclass(frozen=True, slots=True, kw_only=True)
class SchedulerIpcChannels:
    """Bundle of zmq channels owned by ``Scheduler``.

    Packs the five IPC-related fields that ``init_ipc_channels`` builds:

    - ``recv_from_tokenizer`` / ``recv_from_rpc``: PULL / DEALER sockets, only
      populated on the rank-0 PP+attn-TP+attn-CP scheduler instance (None elsewhere).
    - ``send_to_tokenizer`` / ``send_to_detokenizer``: PUSH sockets wrapped in
      :class:`SenderWrapper`; on non-leader ranks the wrapper holds ``None``.
    - ``send_metrics_from_scheduler``: optional PUSH socket emitting scheduler
      metrics when ``current_scheduler_metrics_enabled`` is set.

    Holding these together collapses five attributes on ``Scheduler`` into one.
    """

    recv_from_tokenizer: Optional[zmq.Socket]
    recv_from_rpc: Optional[zmq.Socket]
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
    ) -> "SchedulerIpcChannels":
        """Build the channel bundle from rank flags and port spec.

        ``is_rank_zero`` reflects the PP+attn-TP+attn-CP rank-0 predicate
        on the owning scheduler instance; non-leader ranks get None
        sockets wrapped where the field is a ``SenderWrapper``.
        ``skip_tokenizer_init`` toggles whether detokenizer messages go
        straight back to the TokenizerManager IPC.  ``metrics_enabled``
        gates construction of the optional scheduler-metrics PUSH socket.
        """
        context = zmq.Context(2)

        if is_rank_zero:
            recv_from_tokenizer = get_zmq_socket(
                context, zmq.PULL, port_args.scheduler_input_ipc_name, False
            )
            recv_from_rpc = get_zmq_socket(
                context, zmq.DEALER, port_args.rpc_ipc_name, False
            )

            send_to_tokenizer_raw = get_zmq_socket(
                context, zmq.PUSH, port_args.tokenizer_ipc_name, False
            )
            if skip_tokenizer_init:
                # Directly send to the TokenizerManager
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
            send_to_tokenizer=send_to_tokenizer,
            send_to_detokenizer=send_to_detokenizer,
            send_metrics_from_scheduler=send_metrics_from_scheduler,
        )
