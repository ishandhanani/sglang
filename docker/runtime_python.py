"""Prepare runtime Python layers without removing packages or kernel artifacts."""

import argparse
import hashlib
import json
import os
import shutil
import stat
from collections import defaultdict
from pathlib import Path


def file_key(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        return None
    return (
        info.st_size,
        stat.S_IMODE(info.st_mode),
        info.st_uid,
        info.st_gid,
        tuple((name, os.getxattr(path, name)) for name in sorted(os.listxattr(path))),
    )


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").digest()


def is_elf(path):
    with path.open("rb") as stream:
        return stream.read(4) == b"\x7fELF"


def deduplicate_cuda(site_packages, cuda_root):
    """Reuse OS CUDA libraries only when the installed wheel has identical bytes.

    Leave different versions untouched. The final runtime inherits the exact
    same CUDA base as this staging image; wheel paths and RECORD hashes remain
    valid. Relative symlinks also work when the image filesystem is relocated.
    """
    candidates = defaultdict(list)
    for path in sorted(cuda_root.resolve().glob("targets/*/lib/*.so*")):
        key = file_key(path)
        if key is not None:
            candidates[key].append(path)

    hashes = {}
    linked = []
    for path in sorted((site_packages / "nvidia").rglob("*.so*")):
        key = file_key(path)
        if key not in candidates or not is_elf(path):
            continue
        checksum = digest(path)
        for candidate in candidates[key]:
            if candidate not in hashes:
                hashes[candidate] = digest(candidate)
            if checksum != hashes[candidate]:
                continue
            path.unlink()
            path.symlink_to(os.path.relpath(candidate, path.parent))
            linked.append(
                {"path": str(path), "target": str(candidate), "bytes": key[0]}
            )
            break
    return linked


def deduplicate_library_aliases(site_packages):
    """Restore shared storage for identical versioned library aliases in a wheel.

    Wheels can contain separate copies of libfoo.so, libfoo.so.0 and
    libfoo.so.0.1. Keep every pathname and byte, and retain each library's
    original directory for $ORIGIN lookup. Do not link across Python packages.
    """
    candidates = defaultdict(list)
    for path in sorted(site_packages.rglob("*.so*")):
        key = file_key(path)
        if key is not None:
            candidates[(path.parent, path.name.split(".so", 1)[0], key)].append(path)

    linked = []
    for (_, _, key), paths in candidates.items():
        if len(paths) < 2:
            continue
        by_hash = {}
        for path in paths:
            if not is_elf(path):
                continue
            checksum = digest(path)
            canonical = by_hash.setdefault(checksum, path)
            if path.samefile(canonical):
                continue
            path.unlink()
            os.link(canonical, path)
            linked.append(
                {"path": str(path), "target": str(canonical), "bytes": key[0]}
            )
    return linked


def separate_editable_metadata(site_packages, destination):
    """Keep source/version-dependent files out of the large dependency layer."""
    distributions = list(site_packages.glob("sglang-*.dist-info"))
    editable = list(site_packages.glob("__editable__*sglang*"))
    if len(distributions) != 1 or not any(path.suffix == ".pth" for path in editable):
        raise RuntimeError("expected one installed editable SGLang distribution")
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(distributions + editable):
        if (destination / path.name).exists():
            raise RuntimeError(f"metadata destination already exists: {path.name}")
        shutil.move(path, destination / path.name)
    # A Python invocation after pip's editable install may have compiled the
    # generated finder. Keep that version-dependent cache with its source.
    for path in sorted((site_packages / "__pycache__").glob("__editable__*sglang*")):
        cache = destination / "__pycache__"
        cache.mkdir(exist_ok=True)
        shutil.move(path, cache / path.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-packages", type=Path, required=True)
    parser.add_argument("--cuda-root", type=Path, default=Path("/usr/local/cuda"))
    parser.add_argument("--metadata-dir", type=Path, required=True)
    args = parser.parse_args()
    cuda = deduplicate_cuda(args.site_packages, args.cuda_root)
    aliases = deduplicate_library_aliases(args.site_packages)
    separate_editable_metadata(args.site_packages, args.metadata_dir)
    print(json.dumps({"cuda_libraries": cuda, "library_aliases": aliases}, indent=2))


if __name__ == "__main__":
    main()
