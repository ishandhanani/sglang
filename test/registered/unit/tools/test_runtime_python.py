import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

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
helper = load_module("runtime_python", REPO_ROOT / "docker/runtime_python.py")


class TestRuntimePython(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.packages = self.root / "site-packages"
        self.cuda = self.root / "cuda"

    def write(self, path, content=b"\x7fELFidentical-payload", mode=0o644):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        path.chmod(mode)
        return path

    def test_cuda_reuses_only_identical_content_and_permissions(self):
        library = self.write(self.cuda / "targets/test/lib/libexample.so.13.0")
        wheel = self.write(self.packages / "nvidia/cu13/lib/libexample.so.13")
        different = self.write(
            self.packages / "nvidia/cu13/lib/libdifferent.so",
            b"\x7fELFdifferent-payload",
        )
        mode = self.write(self.packages / "nvidia/cu13/lib/libmode.so", mode=0o755)
        before = {p: p.read_bytes() for p in (wheel, different, mode)}
        result = helper.deduplicate_cuda(self.packages, self.cuda)
        self.assertEqual(len(result), 1)
        self.assertTrue(wheel.is_symlink())
        self.assertFalse(os.path.isabs(os.readlink(wheel)))
        self.assertEqual(wheel.resolve(), library)
        self.assertFalse(different.is_symlink())
        self.assertFalse(mode.is_symlink())
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        self.assertEqual(helper.deduplicate_cuda(self.packages, self.cuda), [])

    def test_library_aliases_preserve_paths_bytes_and_directory(self):
        original = self.write(self.packages / "package/lib/libexample.so.1")
        alias = self.write(original.with_name("libexample.so"))
        other = self.write(self.packages / "other/lib/libexample.so")
        different = self.write(
            original.with_name("libexample.so.2"), b"\x7fELFdifferent"
        )
        result = helper.deduplicate_library_aliases(self.packages)
        self.assertEqual(len(result), 1)
        self.assertTrue(original.samefile(alias))
        self.assertFalse(alias.is_symlink())
        self.assertFalse(original.samefile(other))
        self.assertFalse(original.samefile(different))
        self.assertEqual(helper.deduplicate_library_aliases(self.packages), [])

    def test_non_elf_and_existing_symlinks_are_unchanged(self):
        self.write(self.cuda / "targets/test/lib/libexample.so", b"not-an-elf")
        wheel = self.write(
            self.packages / "nvidia/cu13/lib/libexample.so", b"not-an-elf"
        )
        alias = wheel.with_name("libexample.so.1")
        alias.symlink_to(wheel.name)
        self.assertEqual(helper.deduplicate_cuda(self.packages, self.cuda), [])
        self.assertEqual(helper.deduplicate_library_aliases(self.packages), [])
        self.assertFalse(wheel.is_symlink())
        self.assertEqual(os.readlink(alias), wheel.name)

    def test_editable_metadata_is_separate_without_excluding_dependencies(self):
        metadata = self.write(
            self.packages / "sglang-1.2.3.dist-info/METADATA", b"sglang"
        )
        pth = self.write(self.packages / "__editable__.sglang-1.2.3.pth", b"/workspace")
        finder = self.write(
            self.packages / "__editable___sglang_1_2_3_finder.py", b"finder"
        )
        dependency = self.write(
            self.packages / "sglang_kernel-1.2.3.dist-info/METADATA", b"kernel"
        )
        destination = self.root / "metadata"
        bytecode = self.write(
            self.packages
            / "__pycache__/__editable___sglang_1_2_3_finder.cpython-312.pyc",
            b"compiled finder",
        )
        helper.separate_editable_metadata(self.packages, destination)
        self.assertEqual(
            (destination / metadata.parent.name / metadata.name).read_bytes(), b"sglang"
        )
        self.assertEqual((destination / pth.name).read_bytes(), b"/workspace")
        self.assertEqual((destination / finder.name).read_bytes(), b"finder")
        self.assertEqual(dependency.read_bytes(), b"kernel")
        self.assertFalse(metadata.exists())
        self.assertEqual(
            (destination / "__pycache__" / bytecode.name).read_bytes(),
            b"compiled finder",
        )

    def test_missing_editable_install_fails(self):
        with self.assertRaisesRegex(RuntimeError, "editable SGLang"):
            helper.separate_editable_metadata(self.packages, self.root / "metadata")

    def test_metadata_copy_preserves_read_only_input(self):
        metadata = self.write(
            self.packages / "sglang-1.2.3.dist-info/METADATA", b"sglang"
        )
        pth = self.write(self.packages / "__editable__.sglang-1.2.3.pth", b"/workspace")
        bytecode = self.write(
            self.packages
            / "__pycache__/__editable___sglang_1_2_3_finder.cpython-312.pyc",
            b"compiled finder",
        )
        destination = self.root / "metadata"
        helper.separate_editable_metadata(self.packages, destination, copy=True)
        for source in (metadata, pth, bytecode):
            copied = destination / source.relative_to(self.packages)
            self.assertEqual(source.read_bytes(), copied.read_bytes())
            self.assertEqual(helper.file_key(source), helper.file_key(copied))


if __name__ == "__main__":
    unittest.main()
