"""Reuse the image's build dependencies when they satisfy the selected source."""

import importlib.metadata
import subprocess
import sys
import tomllib
from pathlib import Path

from packaging.requirements import Requirement


def can_reuse_build_environment(pyproject):
    build = pyproject.get("build-system", {})
    if build.get("build-backend") != "setuptools.build_meta":
        return False
    if build.get("backend-path") or not build.get("requires"):
        return False
    for declaration in build["requires"]:
        requirement = Requirement(declaration)
        if requirement.marker and not requirement.marker.evaluate():
            continue
        # Version metadata cannot prove that a direct URL has the same content.
        if requirement.url or requirement.extras:
            return False
        try:
            version = importlib.metadata.version(requirement.name)
        except importlib.metadata.PackageNotFoundError:
            return False
        if not requirement.specifier.contains(version, prereleases=True):
            return False
    return True


def main():
    project = sys.argv[1]
    with Path("python/pyproject.toml").open("rb") as stream:
        reuse = can_reuse_build_environment(tomllib.load(stream))
    command = [sys.executable, "-m", "pip", "install", "--no-deps"]
    if reuse:
        command += ["--no-build-isolation", "--check-build-dependencies"]
    print(
        f"SGLang build isolation: {'reused environment' if reuse else 'isolated'}",
        flush=True,
    )
    # A real compilation failure must fail the build, not silently retry it.
    subprocess.run([*command, "-e", project], check=True)


if __name__ == "__main__":
    main()
