# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

from enum import Enum
from typing import List, Optional

import msgspec

from sglang.srt.utils.msgspec_utils import msgspec_struct_pydantic_core_schema

KV_HINT_DEREF_V1 = "kv_hint.deref.v1"


def supported_kv_hint_capabilities(
    *, enable_session_radix_cache: bool
) -> List[str]:
    capabilities = []
    if enable_session_radix_cache:
        capabilities.append(KV_HINT_DEREF_V1)
    return capabilities


class DerefApplyOn(str, Enum):
    CURRENT_SUCCESS = "current_success"
    NEXT_SUCCESS = "next_success"


class KvHintStruct(msgspec.Struct, kw_only=True):
    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        return msgspec_struct_pydantic_core_schema(cls, handler)


class DerefHint(KvHintStruct):
    apply_on: DerefApplyOn


class KvTransferPlan(KvHintStruct):
    source_control_endpoint: str
    block_hashes: List[int]


class MigrateHint(KvHintStruct):
    transfer_plan: KvTransferPlan


class KvHints(KvHintStruct):
    deref: Optional[DerefHint] = None
    migrate: Optional[MigrateHint] = None
