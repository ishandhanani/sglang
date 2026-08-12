import types
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.kv_hints import (  # noqa: E402
    DerefApplyOn,
    DerefHint,
    KvHints,
    KvTransferPlan,
    MigrateHint,
    supported_kv_hint_capabilities,
)
from sglang.srt.managers.io_struct import GenerateReqInput  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestKvHints(unittest.TestCase):
    def test_request_dict_converts_to_typed_envelope(self):
        request = GenerateReqInput(
            text="hello",
            kv_hints={
                "deref": {"apply_on": "next_success"},
            },
        )

        self.assertEqual(
            request.kv_hints.deref.apply_on, DerefApplyOn.NEXT_SUCCESS
        )

    def test_capabilities_follow_enabled_handlers(self):
        self.assertEqual(
            supported_kv_hint_capabilities(enable_session_radix_cache=True),
            ["kv_hint.deref.v1"],
        )
        self.assertEqual(
            supported_kv_hint_capabilities(enable_session_radix_cache=False), []
        )

    def test_schedules_deref_with_explicit_timing(self):
        scheduler = types.SimpleNamespace(enable_session_radix_cache=True)
        req = types.SimpleNamespace(
            session_id="session-a", session_generation=7, deref_apply_on=None
        )
        hints = KvHints(deref=DerefHint(apply_on=DerefApplyOn.NEXT_SUCCESS))

        Scheduler._apply_kv_hints(scheduler, hints, req)

        self.assertEqual(req.deref_apply_on, DerefApplyOn.NEXT_SUCCESS)

    def test_deref_without_session_fails_open(self):
        scheduler = types.SimpleNamespace(enable_session_radix_cache=True)
        req = types.SimpleNamespace(
            session_id=None, session_generation=None, deref_apply_on=None
        )
        hints = KvHints(deref=DerefHint(apply_on=DerefApplyOn.CURRENT_SUCCESS))

        Scheduler._apply_kv_hints(scheduler, hints, req)

        self.assertIsNone(req.deref_apply_on)

    def test_unsupported_migrate_has_no_scheduler_side_effect(self):
        scheduler = types.SimpleNamespace(enable_session_radix_cache=True)
        req = types.SimpleNamespace(
            session_id="session-a", session_generation=7, deref_apply_on=None
        )
        hints = KvHints(
            migrate=MigrateHint(
                transfer_plan=KvTransferPlan(
                    source_control_endpoint="tcp://source:23280",
                    block_hashes=[11, 22],
                )
            )
        )

        Scheduler._apply_kv_hints(scheduler, hints, req)

        self.assertIsNone(req.deref_apply_on)


if __name__ == "__main__":
    unittest.main()
