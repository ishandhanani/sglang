# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional

import msgspec

from sglang.srt.utils.msgspec_utils import msgspec_struct_pydantic_core_schema

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.unified_cache.session_ref_tracker import (
        SessionCacheEvictResult,
        UnifiedSessionRefTracker,
    )
    from sglang.srt.observability.metrics_collector import RadixCacheMetricsCollector

logger = logging.getLogger(__name__)

KV_HINT_PROTOCOL_VERSION = "0.1"
KV_HINT_DEREF_V1 = "kv_hint.deref.v1"
KV_HINT_DEMOTE_V1 = "kv_hint.demote.v1"
KV_HINT_PREFETCH_V1 = "kv_hint.prefetch.v1"
_MAX_ACTIONS_PER_ENVELOPE = 64
_MAX_ACTION_ID_LENGTH = 512
_MAX_SESSION_ID_LENGTH = 512
_DEREF_NEXT_REQUEST_LIMIT = 8192
_DEREF_ACTION_LIMIT = 8192


class KvHintStruct(msgspec.Struct, kw_only=True):
    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        return msgspec_struct_pydantic_core_schema(cls, handler)


class KvHintAction(KvHintStruct):
    action_id: str
    action_type: str
    action_version: str
    payload: dict[str, Any] = msgspec.field(default_factory=dict)


class KvDerefPayload(KvHintStruct):
    pass


class KvDemotePayload(KvHintStruct):
    session_id: str
    session_generation: Optional[int] = None


class KvPrefetchPayload(KvHintStruct):
    pass


class KvHints(KvHintStruct):
    protocol_version: str
    message_id: str
    actions: list[KvHintAction] = msgspec.field(default_factory=list)


class KvHintManager:
    """Dispatches typed router actions at cache-owned lifecycle hooks."""

    def __init__(
        self,
        session_refs: Optional[UnifiedSessionRefTracker] = None,
        metrics_collector: Optional[RadixCacheMetricsCollector] = None,
        demote_session_to_storage: Optional[
            Callable[[str, str, Optional[int]], dict[str, Any]]
        ] = None,
        storage_enabled: Optional[Callable[[], bool]] = None,
    ) -> None:
        self._session_refs = session_refs
        self._metrics_collector = metrics_collector
        self._demote_session_to_storage = demote_session_to_storage
        self._storage_enabled = storage_enabled or (lambda: False)
        self._deref_next_sessions: OrderedDict[str, int] = OrderedDict()
        self._deref_next_requests: OrderedDict[str, str] = OrderedDict()
        self._applied_deref_actions: OrderedDict[tuple[str, str], None] = OrderedDict()

    def capabilities(self) -> list[str]:
        capabilities = []
        if self._session_refs is not None:
            capabilities.append(KV_HINT_DEREF_V1)
        if self._storage_enabled():
            capabilities.append(KV_HINT_PREFETCH_V1)
            if (
                self._session_refs is not None
                and self._demote_session_to_storage is not None
            ):
                capabilities.append(KV_HINT_DEMOTE_V1)
        return capabilities

    def on_request(self, req: Req, hints: KvHints) -> None:
        """Accept advisory actions after the request's session is resolved."""
        if hints.protocol_version != KV_HINT_PROTOCOL_VERSION:
            logger.warning(
                "Ignoring KV hints with unsupported protocol version=%s",
                hints.protocol_version,
            )
            return
        if len(hints.actions) > _MAX_ACTIONS_PER_ENVELOPE:
            logger.warning(
                "Ignoring KV hints with too many actions=%s", len(hints.actions)
            )
            return

        accepted = False
        for action in hints.actions:
            if not self._valid_action_id(action):
                continue
            if action.action_type == "kv.deref" and action.action_version == "1.0":
                accepted |= self._accept_deref(req, action)
            elif action.action_type == "kv.demote" and action.action_version == "1.0":
                accepted |= self._apply_demote(action)
            elif action.action_type == "kv.prefetch" and action.action_version == "1.0":
                accepted |= self._accept_prefetch(req, action)
            else:
                logger.debug(
                    "Ignoring unsupported KV action action_id=%s action_type=%s action_version=%s",
                    action.action_id,
                    action.action_type,
                    action.action_version,
                )

        if accepted:
            req.kv_hints = hints

    def on_request_success(self, req: Req, *, has_reusable_leaf: bool) -> None:
        """Apply the successful request's lifecycle action, then track normal leaves."""
        if self._session_refs is None:
            return

        deref = self._deref_action(req)
        if deref is not None:
            action_key = self._deref_action_key(req, deref)
            if action_key is not None and action_key in self._applied_deref_actions:
                self._applied_deref_actions.move_to_end(action_key)
                self._record_duplicate_deref()
                logger.info(
                    "Skipped duplicate KV DEREF session_id=%s generation=%s action_id=%s",
                    req.session_id,
                    req.session_generation,
                    deref.action_id,
                )
                return

            start_time = time.perf_counter()
            result = self._session_refs.evict_radix_session(
                req.session_id, req.session_generation
            )
            self._record_deref_result(start_time, result)
            if result.status == "evicted" and result.generation is not None:
                if action_key is not None:
                    self._remember_deref_action(action_key)
                self._remember_deref_session(req.session_id, result.generation)
            self._log_deref_result(req, result)
            return

        if has_reusable_leaf:
            self._session_refs.register_session_ref(req)

    def on_request_match(self, req: Optional[Req]) -> None:
        """Track the first matching request after a successful DEREF."""
        if req is None or req.session_id is None or req.session_generation is None:
            return

        generation = self._deref_next_sessions.get(req.session_id)
        if generation != req.session_generation:
            return

        self._deref_next_sessions.move_to_end(req.session_id)
        self._remember_deref_request(req.rid, req.session_id)

    def on_request_prefill_ready(self, req: Req) -> None:
        """Record the next DEREF request after its L3 prefetch resolves."""
        session_id = self._deref_next_requests.pop(req.rid, None)
        if session_id is None:
            return

        self._deref_next_sessions.pop(session_id, None)
        if self._metrics_collector is None:
            return

        self._metrics_collector.record_kv_hint_deref_next_request(
            input_tokens=len(req.full_untruncated_fill_ids),
            device_tokens=len(req.prefix_indices),
            host_tokens=req.host_hit_length,
            storage_tokens=req.storage_hit_length,
        )

    def _accept_deref(self, req: Req, action: KvHintAction) -> bool:
        if self._session_refs is None:
            logger.warning("Ignoring KV DEREF because session radix cache is disabled")
            return False
        if req.session_id is None or req.session_generation is None:
            logger.warning("Ignoring KV DEREF without a radix-native session")
            return False
        if self._decode_payload(action, KvDerefPayload) is None:
            return False
        req.kv_hint_deref_action_id = action.action_id
        logger.info(
            "Accepted KV DEREF session_id=%s generation=%s action_id=%s",
            req.session_id,
            req.session_generation,
            action.action_id,
        )
        return True

    def _apply_demote(self, action: KvHintAction) -> bool:
        if self._session_refs is None or self._demote_session_to_storage is None:
            logger.warning("Ignoring KV DEMOTE because session radix cache is disabled")
            return False
        if not self._storage_enabled():
            logger.warning("Ignoring KV DEMOTE because HiCache storage is disabled")
            return False
        payload = self._decode_payload(action, KvDemotePayload)
        if (
            payload is None
            or not payload.session_id
            or len(payload.session_id) > _MAX_SESSION_ID_LENGTH
        ):
            logger.warning(
                "Ignoring malformed KV DEMOTE action_id=%s", action.action_id
            )
            return False
        if payload.session_generation is not None and payload.session_generation < 0:
            logger.warning(
                "Ignoring malformed KV DEMOTE generation action_id=%s", action.action_id
            )
            return False

        try:
            result = self._demote_session_to_storage(
                action.action_id,
                payload.session_id,
                payload.session_generation,
            )
        except Exception:
            logger.exception(
                "KV DEMOTE failed open action_id=%s session_id=%s",
                action.action_id,
                payload.session_id,
            )
            return False

        logger.info(
            "Applied KV DEMOTE action_id=%s session_id=%s state=%s selected_tokens=%s message=%s",
            action.action_id,
            payload.session_id,
            result.get("state"),
            result.get("selected_tokens", 0),
            result.get("message", ""),
        )
        return True

    def _accept_prefetch(self, req: Req, action: KvHintAction) -> bool:
        if not self._storage_enabled():
            logger.warning("Ignoring KV PREFETCH because HiCache storage is disabled")
            return False
        if self._decode_payload(action, KvPrefetchPayload) is None:
            return False
        req.kv_hint_force_prefetch = True
        logger.info(
            "Accepted KV PREFETCH request_id=%s action_id=%s", req.rid, action.action_id
        )
        return True

    @staticmethod
    def _valid_action_id(action: KvHintAction) -> bool:
        if action.action_id and len(action.action_id) <= _MAX_ACTION_ID_LENGTH:
            return True
        logger.warning(
            "Ignoring malformed KV action action_type=%s action_version=%s",
            action.action_type,
            action.action_version,
        )
        return False

    @staticmethod
    def _decode_payload(
        action: KvHintAction, payload_type: type[KvHintStruct]
    ) -> Optional[KvHintStruct]:
        try:
            return msgspec.convert(action.payload, type=payload_type, strict=True)
        except (msgspec.ValidationError, TypeError):
            logger.warning(
                "Ignoring malformed KV action payload action_id=%s action_type=%s",
                action.action_id,
                action.action_type,
            )
            return None

    @staticmethod
    def _deref_action(req: Req) -> Optional[KvHintAction]:
        hints = req.kv_hints
        if hints is None:
            return None
        for action in hints.actions:
            if (
                action.action_id == getattr(req, "kv_hint_deref_action_id", None)
                and action.action_type == "kv.deref"
                and action.action_version == "1.0"
            ):
                return action
        return None

    def _remember_deref_session(self, session_id: str, generation: int) -> None:
        self._deref_next_sessions[session_id] = generation
        self._deref_next_sessions.move_to_end(session_id)
        while len(self._deref_next_sessions) > _DEREF_NEXT_REQUEST_LIMIT:
            self._deref_next_sessions.popitem(last=False)

    def _remember_deref_request(self, request_id: str, session_id: str) -> None:
        self._deref_next_requests[request_id] = session_id
        self._deref_next_requests.move_to_end(request_id)
        while len(self._deref_next_requests) > _DEREF_NEXT_REQUEST_LIMIT:
            self._deref_next_requests.popitem(last=False)

    @staticmethod
    def _deref_action_key(req: Req, deref: KvHintAction) -> Optional[tuple[str, str]]:
        if not deref.action_id or req.session_id is None:
            return None
        return (req.session_id, deref.action_id)

    def _remember_deref_action(self, action_key: tuple[str, str]) -> None:
        self._applied_deref_actions[action_key] = None
        self._applied_deref_actions.move_to_end(action_key)
        while len(self._applied_deref_actions) > _DEREF_ACTION_LIMIT:
            self._applied_deref_actions.popitem(last=False)

    def _record_deref_result(
        self, start_time: float, result: SessionCacheEvictResult
    ) -> None:
        if self._metrics_collector is None:
            return
        self._metrics_collector.record_kv_hint_deref(
            status=result.status,
            duration_seconds=time.perf_counter() - start_time,
            indexed_component_leaves=result.indexed_component_leaves,
        )

    def _record_duplicate_deref(self) -> None:
        if self._metrics_collector is None:
            return
        self._metrics_collector.record_kv_hint_deref(
            status="duplicate",
            duration_seconds=0.0,
            indexed_component_leaves=0,
        )

    @staticmethod
    def _log_deref_result(req: Req, result: SessionCacheEvictResult) -> None:
        logger.info(
            "Applied KV DEREF session_id=%s status=%s generation=%s indexed_component_leaves=%s",
            req.session_id,
            result.status,
            result.generation,
            result.indexed_component_leaves,
        )
