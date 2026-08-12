import types
import unittest
from unittest.mock import MagicMock, call

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler  # noqa: E402

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestSessionCacheRouterHint(unittest.TestCase):
    def test_schedules_session_eviction_after_request_completion(self):
        tree_cache = MagicMock()
        scheduler = types.SimpleNamespace(
            enable_session_radix_cache=True,
            tree_cache=tree_cache,
        )
        req = types.SimpleNamespace(
            session_id="session-a",
            session_generation=7,
            evict_session_after_finish=False,
            defer_session_eviction_after_finish=False,
        )

        Scheduler._schedule_router_session_eviction(
            scheduler,
            {"evict_session": True, "defer_session_eviction": True},
            req,
        )

        tree_cache.evict_radix_session.assert_not_called()
        self.assertEqual(req.session_generation, 7)
        self.assertTrue(req.evict_session_after_finish)
        self.assertTrue(req.defer_session_eviction_after_finish)

    def test_session_final_eviction_remains_immediate(self):
        scheduler = types.SimpleNamespace(
            enable_session_radix_cache=True,
            tree_cache=MagicMock(),
        )
        req = types.SimpleNamespace(
            session_id="session-a",
            session_generation=7,
            evict_session_after_finish=False,
            defer_session_eviction_after_finish=False,
        )

        Scheduler._schedule_router_session_eviction(
            scheduler, {"evict_session": True}, req
        )

        self.assertTrue(req.evict_session_after_finish)
        self.assertFalse(req.defer_session_eviction_after_finish)

    def test_session_eviction_hint_fails_open(self):
        tree_cache = MagicMock()
        scheduler = types.SimpleNamespace(
            enable_session_radix_cache=True,
            tree_cache=tree_cache,
        )
        req = types.SimpleNamespace(
            session_id="session-a",
            session_generation=7,
            evict_session_after_finish=False,
            defer_session_eviction_after_finish=False,
        )

        for hint in (None, {}, {"evict_session": False}, {"evict_session": "true"}):
            Scheduler._schedule_router_session_eviction(scheduler, hint, req)

        tree_cache.evict_radix_session.assert_not_called()
        self.assertFalse(req.evict_session_after_finish)
        self.assertFalse(req.defer_session_eviction_after_finish)

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

    def test_applies_storage_demotions(self):
        tree_cache = MagicMock()
        tree_cache.demote_session_to_storage.return_value = {
            "state": "pending",
            "selected_tokens": 128,
            "message": "storage publish queued",
        }
        scheduler = types.SimpleNamespace(
            enable_session_radix_cache=True,
            tree_cache=tree_cache,
        )

        Scheduler._apply_router_session_storage_demotions(
            scheduler,
            {
                "session_storage_demotions": [
                    {
                        "operation_id": "demote-1",
                        "session_id": "session-a",
                        "session_generation": 7,
                    }
                ]
            },
        )

        tree_cache.demote_session_to_storage.assert_called_once_with(
            "demote-1", "session-a", generation=7
        )

    def test_prefetch_hint_forces_storage_lookup(self):
        tree_cache = MagicMock()
        tree_cache.is_backuped.return_value = True
        tree_cache.is_root.return_value = False
        tree_cache.get_prefix_hash_values.return_value = ["prefix"]
        tree_cache.get_last_hash_value.return_value = "last"
        tree_cache.hicache_storage_pass_prefix_keys = True
        scheduler = types.SimpleNamespace(
            enable_hicache_storage=True,
            tree_cache=tree_cache,
        )
        req = MagicMock()
        req.router_hint = {"prefetch_from_storage": True}
        req.last_host_node = 3
        req.prefix_indices = [1, 2]
        req.host_hit_length = 0
        req.full_untruncated_fill_ids = [10, 11, 12, 13]
        req._compute_max_prefix_len.return_value = 4
        req.rid = "request-1"

        Scheduler._prefetch_kvcache(scheduler, req)

        tree_cache.prefetch_from_storage.assert_called_once_with(
            "request-1",
            3,
            [12, 13],
            "last",
            ["prefix"],
            force=True,
        )


if __name__ == "__main__":
    unittest.main()
