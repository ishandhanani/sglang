import types
import unittest
from unittest.mock import MagicMock, call

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler  # noqa: E402

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestSessionCacheRouterHint(unittest.TestCase):
    def test_applies_valid_actions(self):
        tree_cache = MagicMock()
        tree_cache.set_session_cache_priority.side_effect = [
            types.SimpleNamespace(
                status="updated", generation=7, indexed_component_leaves=2
            ),
            types.SimpleNamespace(
                status="unchanged", generation=8, indexed_component_leaves=3
            ),
        ]
        scheduler = types.SimpleNamespace(
            enable_session_radix_cache=True,
            tree_cache=tree_cache,
        )

        Scheduler._apply_router_session_cache_actions(
            scheduler,
            {
                "source_control_endpoint": "tcp://source:1234",
                "session_cache_actions": [
                    {
                        "session_id": "session-a",
                        "cache_priority": "evictable",
                        "session_generation": 7,
                    },
                    {
                        "session_id": "session-b",
                        "cache_priority": "protected",
                    },
                ],
            },
        )

        self.assertEqual(
            tree_cache.set_session_cache_priority.call_args_list,
            [
                call("session-a", protected=False, generation=7),
                call("session-b", protected=True, generation=None),
            ],
        )

    def test_malformed_actions_fail_open(self):
        tree_cache = MagicMock()
        scheduler = types.SimpleNamespace(
            enable_session_radix_cache=True,
            tree_cache=tree_cache,
        )

        Scheduler._apply_router_session_cache_actions(
            scheduler,
            {
                "session_cache_actions": [
                    None,
                    {"session_id": "", "cache_priority": "evictable"},
                    {
                        "session_id": "session-a",
                        "cache_priority": "invalid",
                    },
                    {
                        "session_id": "session-b",
                        "cache_priority": "protected",
                        "session_generation": True,
                    },
                ]
            },
        )

        tree_cache.set_session_cache_priority.assert_not_called()


if __name__ == "__main__":
    unittest.main()
