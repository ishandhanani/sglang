import importlib.metadata
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[4]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


register_cpu_ci = load_module(
    "ci_register", REPO_ROOT / "python/sglang/test/ci/ci_register.py"
).register_cpu_ci
register_cpu_ci(est_time=1, suite="base-a-test-cpu")
helper = load_module("install_sglang", REPO_ROOT / "docker/install_sglang.py")


class TestInstallSglang(unittest.TestCase):
    def check(self, requirements, **options):
        return helper.can_reuse_build_environment(
            {
                "build-system": {
                    "build-backend": "setuptools.build_meta",
                    "requires": requirements,
                    **options,
                }
            }
        )

    def test_matching_torch_cuda_local_version_reuses_environment(self):
        with patch.object(
            helper.importlib.metadata, "version", return_value="2.13.0+cu130"
        ):
            self.assertTrue(self.check(["torch==2.13.0"]))
            self.assertFalse(self.check(["torch==2.12.0"]))

    def test_missing_dependency_uses_isolation(self):
        with patch.object(
            helper.importlib.metadata,
            "version",
            side_effect=importlib.metadata.PackageNotFoundError("missing"),
        ):
            self.assertFalse(self.check(["missing>=1"]))
            self.assertTrue(self.check(['missing>=1; python_version < "2"']))

    def test_unverifiable_requirements_or_backend_use_isolation(self):
        self.assertFalse(self.check(["backend @ https://example.com/backend.whl"]))
        self.assertFalse(self.check(["backend[extra]>=1"]))
        self.assertFalse(self.check(["setuptools"], **{"backend-path": ["."]}))
        self.assertFalse(self.check(["maturin"], **{"build-backend": "maturin"}))
        self.assertFalse(self.check([]))


if __name__ == "__main__":
    unittest.main()
