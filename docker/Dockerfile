# syntax=docker/dockerfile:1.19

ARG CUDA_VERSION=13.0.3
ARG BUILD_TYPE=all
ARG BRANCH_TYPE=remote
ARG SGL_KERNEL_VERSION=0.4.7
ARG SGL_VERSION
ARG USE_LATEST_SGLANG=0
ARG SGL_NCCL_VERSION=2.30.7
ARG INSTALL_FLASHINFER_JIT_CACHE=0
ARG FLASHINFER_VERSION=0.6.18
ARG MOONCAKE_VERSION=0.3.13
ARG MSCCLPP_VERSION=sglang-v0.10.1
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04 AS base

# CI refreshes unpinned OS/Python dependencies daily; local builds opt in.
ARG DEPENDENCY_CACHE_EPOCH

ARG TARGETARCH
ARG GDRCOPY_VERSION=2.5.1
ARG PIP_DEFAULT_INDEX
ARG UBUNTU_MIRROR
ARG GITHUB_ARTIFACTORY=github.com

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    GDRCOPY_HOME=/usr/src/gdrdrv-${GDRCOPY_VERSION}/

# Add GKE default lib and bin locations
ENV PATH="${PATH}:/usr/local/nvidia/bin" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"

# Replace Ubuntu sources if specified
RUN if [ -n "$UBUNTU_MIRROR" ]; then \
    sed -i "s|http://.*archive.ubuntu.com|$UBUNTU_MIRROR|g" /etc/apt/sources.list && \
    sed -i "s|http://.*security.ubuntu.com|$UBUNTU_MIRROR|g" /etc/apt/sources.list; \
fi

# Python setup (combined with apt update to reduce layers)
# Ubuntu 24.04 ships Python 3.12 in main, so we no longer need the deadsnakes
# PPA. Dropping it avoids transient Launchpad 504s in `add-apt-repository`.
RUN --mount=type=cache,target=/var/cache/apt,id=base-apt-${TARGETARCH},sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache \
    && apt update && apt install -y --no-install-recommends wget software-properties-common \
    && apt install -y --no-install-recommends python3.12-full python3.12-dev \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 2 \
    && update-alternatives --set python3 /usr/bin/python3.12 \
    # Fix for apt-add-repository
    && cd /usr/lib/python3/dist-packages/ \
    && ln -s apt_pkg.cpython-312-*-linux-gnu.so apt_pkg.so \
    && rm -rf /var/lib/apt/lists/*

# create virtual env for sglang to avoid conflict with system python packages
RUN python3 -m venv /opt/sglang
ENV PATH="/opt/sglang/bin:${PATH}"

# Install system dependencies (organized by category for better caching)
RUN --mount=type=cache,target=/var/cache/apt,id=base-apt-${TARGETARCH},sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    # Core system utilities
    ca-certificates \
    software-properties-common \
    netcat-openbsd \
    kmod \
    unzip \
    openssh-server \
    curl \
    wget \
    lsof \
    locales \
    # Build essentials (needed for framework stage)
    build-essential \
    cmake \
    perl \
    patchelf \
    ccache \
    git-lfs \
    # MPI and NUMA
    libopenmpi-dev \
    libnuma1 \
    libnuma-dev \
    numactl \
    # transformers multimodal VLM
    ffmpeg \
    # InfiniBand/RDMA
    libibverbs-dev \
    libibverbs1 \
    libibumad3 \
    librdmacm1 \
    libnl-3-200 \
    libnl-route-3-200 \
    libnl-route-3-dev \
    libnl-3-dev \
    ibverbs-providers \
    infiniband-diags \
    perftest \
    # Development libraries
    libgoogle-glog-dev \
    libgtest-dev \
    libjsoncpp-dev \
    libunwind-dev \
    libboost-all-dev \
    libssl-dev \
    libgrpc-dev \
    libgrpc++-dev \
    libprotobuf-dev \
    protobuf-compiler \
    protobuf-compiler-grpc \
    pybind11-dev \
    libhiredis-dev \
    libcurl4-openssl-dev \
    libczmq4 \
    libczmq-dev \
    libfabric-dev \
    linux-libc-dev \
    # Package building tools
    devscripts \
    debhelper \
    fakeroot \
    dkms \
    check \
    libsubunit0 \
    libsubunit-dev \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Replace pip global cache if specified
RUN if [ -n "${PIP_DEFAULT_INDEX}" ]; then \
    python3 -m pip config set global.index-url ${PIP_DEFAULT_INDEX}; \
fi

# GDRCopy installation
RUN mkdir -p /tmp/gdrcopy && cd /tmp \
    && curl --retry 3 --retry-delay 2 -fsSL -o v${GDRCOPY_VERSION}.tar.gz \
        https://${GITHUB_ARTIFACTORY}/NVIDIA/gdrcopy/archive/refs/tags/v${GDRCOPY_VERSION}.tar.gz \
    && tar -xzf v${GDRCOPY_VERSION}.tar.gz && rm v${GDRCOPY_VERSION}.tar.gz \
    && cd gdrcopy-${GDRCOPY_VERSION}/packages \
    && CUDA=/usr/local/cuda ./build-deb-packages.sh \
    && dpkg -i gdrdrv-dkms_*.deb libgdrapi_*.deb gdrcopy-tests_*.deb gdrcopy_*.deb \
    && cd / && rm -rf /tmp/gdrcopy

# Fix DeepEP IBGDA symlink
RUN ln -sf /usr/lib/$(uname -m)-linux-gnu/libmlx5.so.1 /usr/lib/$(uname -m)-linux-gnu/libmlx5.so

# Set up locale
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

########################################################
########## PARALLEL BUILDER STAGES ####################
########################################################
#
# These stages run IN PARALLEL via BuildKit:
#
#   base
#     |
#     +-- rust_builder --> torch_deps --> hpc_ops_builder
#     |                \-> gateway_builder
#     +-- flashinfer_cache (independent binary wheels)
#     +-- devtools_builder (independent)
#     |
#     v
#   framework (combines all artifacts)
#

########################################################
# PARALLEL STAGE 1: Torch/Deps Builder (starts from base)
########################################################
FROM base AS rust_builder

# Share the compiler installation; source edits only rerun compilation.
ENV PATH="/root/.cargo/bin:${PATH}"
RUN curl --proto '=https' --tlsv1.2 --retry 3 --retry-delay 2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --no-modify-path --profile minimal \
    && rustc --version && cargo --version

FROM rust_builder AS torch_deps

ARG CUDA_VERSION
ARG BUILD_TYPE
ARG SGL_KERNEL_VERSION
ARG FLASHINFER_VERSION
ARG GITHUB_ARTIFACTORY
ENV FLASHINFER_VERSION=${FLASHINFER_VERSION}

WORKDIR /sgl-workspace

# Install sgl-kernel (from pre-built wheel)
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip setuptools wheel html5lib six \
    && case "$CUDA_VERSION" in \
        13.0.3) ;; \
        *) echo "Unsupported CUDA version: $CUDA_VERSION" && exit 1 ;; \
    esac \
    # --no-deps prevents pip from pulling torch from default PyPI
    && python3 -m pip install sglang-kernel==${SGL_KERNEL_VERSION} --force-reinstall --no-deps

# The stub resolves dependencies only; setup.py and Rust compilation belong to
# the final editable install. Rust/proto edits must not reinstall every package.
COPY python/pyproject.toml /tmp/sglang_deps/python/pyproject.toml

# Install sglang dependencies (torch, transformers, etc.).
# Generate constraints.txt to prevent reinstalling these deps in later stages.
RUN --mount=type=cache,target=/root/.cache/pip \
    case "$CUDA_VERSION" in \
        13.0.3) CUINDEX=130 ;; \
        *) echo "Unsupported CUDA version: $CUDA_VERSION" && exit 1 ;; \
    esac \
    && cd /tmp/sglang_deps/python \
    && mkdir -p sglang \
    && touch sglang/__init__.py \
    && echo '__version__ = "0.0.0"' > sglang/version.py \
    && touch README.md \
    && touch LICENSE \
    && python3 -m pip install --extra-index-url https://download.pytorch.org/whl/cu${CUINDEX} ".[${BUILD_TYPE}]" \
    && cd /sgl-workspace \
    && rm -rf /tmp/sglang_deps \
    && pip freeze | grep -v "^sglang==" > /sgl-workspace/constraints.txt

# distro resolves to the apt python3-distro under /usr/lib/python3, which the runtime
# stage does not COPY; force a pip copy into /usr/local so it survives the stage split.
RUN python3 -m pip install --ignore-installed --no-deps distro

# Cosmos3 guardrails default ON: sglang enables them whenever `cosmos_guardrail`
# is importable (Cosmos3Pipeline.create_pipeline_stages); opt out at runtime with
# SGLANG_DISABLE_COSMOS3_GUARDRAILS=1. Installed with --no-deps because the
# package metadata conflicts with our pins in ways that don't matter at runtime
# (imageio>=2.37 vs our ==2.36) or would be harmful (opencv-python — also pulled
# by retinaface-py — would shadow our opencv-python-headless cv2). The deps not
# already in the image are installed explicitly; the find_spec check mirrors
# sglang's availability probe plus cosmos_guardrail's own imports.
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --no-deps \
        cosmos-guardrail==0.3.1 \
        retinaface-py==0.0.2 \
    && python3 -m pip install \
        "better-profanity==0.7.0" \
        "nltk==3.9.1" \
        "peft==0.18.1" \
    && python3 -c "import importlib.util as u; missing = [m for m in ('cosmos_guardrail', 'retinaface', 'better_profanity', 'nltk', 'cv2', 'peft', 'sentencepiece') if u.find_spec(m) is None]; assert not missing, f'guardrail deps missing: {missing}'"

########################################################
# PARALLEL STAGE 2: HPC-Ops Builder (needs torch_deps)
########################################################
FROM torch_deps AS hpc_ops_builder

# HPC-Ops (https://github.com/Tencent/hpc-ops, MIT): fused attention / MoE /
# RoPE kernels from the Tencent Hunyuan AI Infra team, consumed by the opt-in
# hpc_ops attention and MoE runner backends.
ARG HPC_OPS_COMMIT=ab1a402724635507037426068f6cddc3d30dc0a8

WORKDIR /build

ARG CMAKE_BUILD_PARALLEL_LEVEL
# The kernels target Hopper (sm90a) only, so skip non-x86_64 images.
# setup.py derives the version from `git rev-parse`, so keep the .git dir
# (a source zip archive would not build).
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/ccache,id=ccache-${TARGETARCH}-${CUDA_VERSION} \
    mkdir -p /wheels && \
    if [ "$(uname -m)" = "x86_64" ]; then \
        git clone https://github.com/Tencent/hpc-ops.git && \
        cd hpc-ops && \
        git checkout ${HPC_OPS_COMMIT} && \
        CMAKE_CUDA_COMPILER_LAUNCHER=ccache CMAKE_CXX_COMPILER_LAUNCHER=ccache \
        CCACHE_COMPILERCHECK=content python3 setup.py bdist_wheel -d /wheels; \
    fi

########################################################
# PARALLEL STAGE 3: FlashInfer Cache (independent binary wheels)
########################################################
FROM base AS flashinfer_cache

ARG CUDA_VERSION
ARG INSTALL_FLASHINFER_JIT_CACHE
ARG FLASHINFER_VERSION

# Stage jit-cache/cubin artifacts into /flashinfer_jit_output for clean COPY later
RUN --mount=type=cache,target=/root/.cache/pip \
    case "$CUDA_VERSION" in \
        13.0.3) CUINDEX=130 ;; \
        *) echo "Unsupported CUDA version: $CUDA_VERSION" && exit 1 ;; \
    esac \
    && mkdir -p /flashinfer_jit_output \
    # flashinfer-cubin is CUDA-version-agnostic, unlike jit-cache, so its index-url has no cu${CUINDEX} suffix
    && python3 -m pip install --no-deps --no-compile --target /flashinfer_jit_output \
        flashinfer-cubin==${FLASHINFER_VERSION} --index-url https://flashinfer.ai/whl \
    && if [ "$INSTALL_FLASHINFER_JIT_CACHE" = "1" ]; then \
        python3 -m pip install --no-deps --no-compile --target /flashinfer_jit_output \
            flashinfer-jit-cache==${FLASHINFER_VERSION} --index-url https://flashinfer.ai/whl/cu${CUINDEX}; \
    fi

########################################################
# PARALLEL STAGE 4: Dev Tools Builder (starts from base)
########################################################
FROM base AS devtools_builder

ARG GITHUB_ARTIFACTORY

WORKDIR /tools

# Minimal apt deps needed for oh-my-zsh install in this stage
# Full dev apt packages (gdb, vim, tmux, nsight, etc.) are installed in the framework stage
RUN --mount=type=cache,target=/var/cache/apt,id=devtools-apt-${TARGETARCH},sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends zsh git \
    && rm -rf /var/lib/apt/lists/*

# Download CLI tools (each in its own layer for parallel downloads)
RUN curl --retry 3 --retry-delay 2 -LSso /tools/diff-so-fancy \
        https://${GITHUB_ARTIFACTORY}/so-fancy/diff-so-fancy/releases/download/v1.4.4/diff-so-fancy \
    && chmod +x /tools/diff-so-fancy

RUN curl --retry 3 --retry-delay 2 -LSso /tools/clang-format \
        https://${GITHUB_ARTIFACTORY}/muttleyxd/clang-tools-static-binaries/releases/download/master-32d3ac78/clang-format-16_linux-amd64 \
    && chmod +x /tools/clang-format

RUN curl --retry 3 --retry-delay 2 -fsSL -o /tmp/clangd.zip \
        https://${GITHUB_ARTIFACTORY}/clangd/clangd/releases/download/18.1.3/clangd-linux-18.1.3.zip \
    && unzip -q /tmp/clangd.zip -d /tmp \
    && cp /tmp/clangd_18.1.3/bin/* /tools/ \
    && mkdir -p /tools/lib && cp -r /tmp/clangd_18.1.3/lib/* /tools/lib/ \
    && rm -rf /tmp/clangd.zip /tmp/clangd_18.1.3

RUN CMAKE_VERSION=3.31.1 \
    && ARCH=$(uname -m) \
    && CMAKE_INSTALLER="cmake-${CMAKE_VERSION}-linux-${ARCH}" \
    && curl --retry 3 --retry-delay 2 -fsSL -o "/tmp/${CMAKE_INSTALLER}.tar.gz" \
        "https://${GITHUB_ARTIFACTORY}/Kitware/CMake/releases/download/v${CMAKE_VERSION}/${CMAKE_INSTALLER}.tar.gz" \
    && tar -xzf "/tmp/${CMAKE_INSTALLER}.tar.gz" -C /tmp \
    && cp -r "/tmp/${CMAKE_INSTALLER}/bin/"* /tools/ \
    && mkdir -p /tools/share && cp -r "/tmp/${CMAKE_INSTALLER}/share/"* /tools/share/ \
    && rm -rf "/tmp/${CMAKE_INSTALLER}" "/tmp/${CMAKE_INSTALLER}.tar.gz"

RUN curl --proto '=https' --tlsv1.2 --retry 3 --retry-delay 2 -sSf https://just.systems/install.sh | \
    sed "s|https://github.com|https://${GITHUB_ARTIFACTORY}|g" | \
    bash -s -- --tag 1.42.4 --to /tools

# Install oh-my-zsh and plugins
RUN sh -c "$(curl --retry 3 --retry-delay 2 -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended \
    && git clone --depth 1 https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-/root/.oh-my-zsh/custom}/plugins/zsh-autosuggestions \
    && git clone --depth 1 https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-/root/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting

########################################################
# PARALLEL STAGE 5: Gateway Builder (starts from base)
########################################################
# Builds sgl-model-gateway in isolation so Python-only changes
# don't trigger a full Rust recompilation.
FROM rust_builder AS gateway_builder

WORKDIR /build

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install maturin

# Install the gateway's pinned toolchain before copying frequently edited code.
# rustup otherwise downloads it inside every invalidated compilation layer.
COPY sgl-model-gateway/rust-toolchain.toml /build/sgl-model-gateway/rust-toolchain.toml
RUN cd /build/sgl-model-gateway && rustup show

# Copy ONLY the gateway source (not the full repo).
COPY --link sgl-model-gateway /build/sgl-model-gateway

# Cached artifacts never enter the builder's image layers. Copy final outputs
# out of the mount before RUN exits so cache eviction cannot change the image.
ARG CARGO_BUILD_JOBS
RUN --mount=type=cache,target=/root/.cargo/registry,id=cargo-registry \
    --mount=type=cache,target=/root/.cargo/git,id=cargo-git \
    --mount=type=cache,target=/build/sgl-model-gateway/target,id=gateway-target-${TARGETARCH},sharing=locked \
    cd /build/sgl-model-gateway/bindings/python \
    && ulimit -n 65536 && maturin build --release --features vendored-openssl --out /build/gateway_wheels \
    && cd /build/sgl-model-gateway \
    && cargo build --release --bin sgl-model-gateway --features vendored-openssl \
    && cp target/release/sgl-model-gateway /build/sgl-model-gateway-bin

########################################################
########## Final Framework Image ######################
########################################################
#
# Combines all artifacts from parallel builder stages
#
FROM torch_deps AS framework

ARG CUDA_VERSION
ARG GITHUB_ARTIFACTORY

WORKDIR /sgl-workspace

# =============================================================================
# Copy artifacts from parallel builders
# =============================================================================

# Copy HPC-Ops wheel and install (empty on non-x86_64; kernels are sm90a-only)
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,from=hpc_ops_builder,source=/wheels,target=/tmp/wheels/hpc-ops \
    if ls /tmp/wheels/hpc-ops/*.whl >/dev/null 2>&1; then \
        pip install --no-deps /tmp/wheels/hpc-ops/*.whl; \
    fi

# Copy flashinfer cubin (always) and jit-cache (if installed) packages
COPY --link --from=flashinfer_cache /flashinfer_jit_output/ /opt/sglang/lib/python3.12/site-packages/

# Copy dev tools
COPY --from=devtools_builder /tools/diff-so-fancy /usr/local/bin/
COPY --from=devtools_builder /tools/clang-format /usr/local/bin/
COPY --from=devtools_builder /tools/clangd /usr/local/bin/
COPY --from=devtools_builder /tools/lib /usr/local/lib/
COPY --from=devtools_builder /tools/cmake /usr/local/bin/
COPY --from=devtools_builder /tools/ctest /usr/local/bin/
COPY --from=devtools_builder /tools/cpack /usr/local/bin/
COPY --from=devtools_builder /tools/share/cmake-3.31 /usr/local/share/cmake-3.31
COPY --from=devtools_builder /tools/just /usr/local/bin/
COPY --from=devtools_builder /root/.oh-my-zsh /root/.oh-my-zsh

# Install dev apt packages (need to re-run since we're in a different stage)
RUN --mount=type=cache,target=/var/cache/apt,id=framework-apt-${TARGETARCH},sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    gdb \
    ninja-build \
    vim \
    tmux \
    htop \
    zsh \
    tree \
    silversearcher-ag \
    cloc \
    pkg-config \
    bear \
    less \
    rdma-core \
    openssh-server \
    gnuplot \
    infiniband-diags \
    perftest \
    ibverbs-providers \
    libibumad3 \
    libibverbs1 \
    libnl-3-200 \
    libnl-route-3-200 \
    librdmacm1 \
    && rm -rf /var/lib/apt/lists/*

# Install NVIDIA development tools
RUN --mount=type=cache,target=/var/cache/apt,id=framework-apt-${TARGETARCH},sharing=locked \
    apt update -y \
    && apt install -y --no-install-recommends gnupg \
    && echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu2004/$(if [ "$(uname -m)" = "aarch64" ]; then echo "arm64"; else echo "amd64"; fi) /" | tee /etc/apt/sources.list.d/nvidia-devtools.list \
    && apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/$(if [ "$(uname -m)" = "aarch64" ]; then echo "arm64"; else echo "x86_64"; fi)/7fa2af80.pub \
    && apt update -y \
    && apt install -y --no-install-recommends nsight-systems-cli \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Python packages and tools (before source copy for better caching)
# =============================================================================

# Install Mooncake
ARG MOONCAKE_VERSION
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install mooncake-transfer-engine-cuda13==${MOONCAKE_VERSION}

# Install MSCCL++ Python dependencies and package (builds extension via CMake through pip)
ARG MSCCLPP_VERSION
ARG CMAKE_BUILD_PARALLEL_LEVEL
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/ccache,id=ccache-${TARGETARCH}-${CUDA_VERSION} \
    git clone --depth=1 --branch ${MSCCLPP_VERSION} https://${GITHUB_ARTIFACTORY}/microsoft/mscclpp.git /tmp/mscclpp \
    && case "${CUDA_VERSION}" in \
           13.*) \
               CMAKE_ARGS="-DMSCCLPP_BYPASS_GPU_CHECK=ON -DMSCCLPP_USE_CUDA=ON -DMSCCLPP_GPU_ARCHS=80,90,100,100a,103,103a" \
               CMAKE_CUDA_COMPILER_LAUNCHER=ccache CMAKE_CXX_COMPILER_LAUNCHER=ccache \
               CCACHE_COMPILERCHECK=content python3 -m pip install "/tmp/mscclpp[cuda13]"; \
               ;; \
           *) \
               echo "Unsupported CUDA version for MSCCL++: ${CUDA_VERSION}" && exit 1; \
               ;; \
       esac \
    && rm -rf /tmp/mscclpp

# Install essential Python packages (use constraints to prevent conflicts)
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -c /sgl-workspace/constraints.txt \
    datamodel_code_generator \
    pre-commit \
    pytest \
    black \
    isort \
    icdiff \
    uv \
    wheel \
    scikit-build-core \
    py-spy \
    cubloaty \
    google-cloud-storage \
    pandas \
    matplotlib \
    tabulate \
    termplotlib \
    "runai-model-streamer[s3,gcs,azure]>=0.15.7"

# PyTorch 2.13 requires nvidia-nccl-cu13==2.29.7, so keep that version while
# resolving dependencies and generating constraints.txt. DeepEP needs 2.30.7
# at runtime; apply the override only after the constrained dependency solve.
ARG SGL_NCCL_VERSION
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --force-reinstall --no-deps \
        "nvidia-nccl-cu13==${SGL_NCCL_VERSION}"

# The `nixl` stub package is needed (it owns the `nixl` import path) but
# unconditionally requires nixl-cu12, so install it with --no-deps and pair it
# with the nixl-cu13 binary to avoid shipping wrong-CUDA libs.
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install nixl nixl-cu13 --no-deps \
    && python3 -m pip install "cuda-python>=13,<14"

# Install the latest available AI Dynamo prerelease from NVIDIA's package index.
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --pre --extra-index-url https://pypi.nvidia.com/ ai-dynamo

# Add yank script
COPY --chown=root:root --chmod=755 docker/configs/yank /usr/local/bin/yank

# These configs are optional; users can override them by mounting their own files
COPY docker/configs/opt/.vimrc /opt/sglang/.vimrc
COPY docker/configs/opt/.tmux.conf /opt/sglang/.tmux.conf
COPY docker/configs/opt/.gitconfig /opt/sglang/.gitconfig

# Configure development environment
COPY docker/configs/.zshrc /root/.zshrc

# Fix Trivy-reported CVEs
# pip:             urllib3 (CVE-2025-43859), pillow (CVE-2026-25990)
# binutils family: CVE-2025-{1147,1148,3198,5244,5245,7545,7546,8225,11082,11083,11412,11413,11414,11494,11839,11840}
# libgnutls30t64:  CVE-2025-{9820,14831}
# libpam:          CVE-2024-10963
# libsqlite3-0:    CVE-2025-{6965,7709}
# libtasn1-6:      CVE-2025-13151
# dpkg:            CVE-2025-6297
RUN python3 -m pip install --upgrade "urllib3>=2.6.3" "pillow>=12.1.1"
RUN --mount=type=cache,target=/var/cache/apt,id=framework-apt-${TARGETARCH},sharing=locked \
    apt-get update && apt-get install -y --only-upgrade \
    binutils binutils-common binutils-x86-64-linux-gnu libbinutils \
    libctf0 libctf-nobfd0 libgprofng0 libsframe1 \
    libgnutls30t64 \
    libpam-modules libpam-modules-bin libpam-runtime libpam0g \
    libsqlite3-0 libtasn1-6 \
    dpkg dpkg-dev libdpkg-perl \
    && rm -rf /var/lib/apt/lists/*

# Cache the small build backends so editable installs can reuse the installed
# Torch instead of extracting another copy into a temporary build environment.
# install_sglang.py falls back to isolation if the selected source needs others.
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install "setuptools-rust>=1.11" "setuptools-scm>=8.0"

# Install pre-built gateway artifacts from parallel builder
COPY --link --from=gateway_builder /build/sgl-model-gateway-bin /opt/sglang/bin/sgl-model-gateway
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,from=gateway_builder,source=/build/gateway_wheels,target=/tmp/gateway_wheels \
    python3 -m pip install --force-reinstall /tmp/gateway_wheels/*.whl

# =============================================================================
# Copy sglang source and do editable install (LAST for better caching)
# =============================================================================

# Copy local source if building from local
FROM scratch AS local_src
COPY . /src

FROM framework AS framework_final

ARG BRANCH_TYPE
ARG BUILD_TYPE
ARG CUDA_VERSION
ARG SGL_VERSION
ARG USE_LATEST_SGLANG

WORKDIR /sgl-workspace

RUN --mount=type=bind,from=local_src,source=/src,target=/tmp/local_src \
    if [ "$BRANCH_TYPE" = "local" ]; then \
        cp -r /tmp/local_src /sgl-workspace/sglang; \
    elif [ "$USE_LATEST_SGLANG" = "1" ]; then \
        git clone --depth=1 https://github.com/sgl-project/sglang.git /sgl-workspace/sglang; \
    elif [ -z "$SGL_VERSION" ]; then \
        echo "ERROR: SGL_VERSION must be set when USE_LATEST_SGLANG=0 and BRANCH_TYPE!=local" && exit 1; \
    else \
        git clone --depth=1 --branch v${SGL_VERSION} https://github.com/sgl-project/sglang.git /sgl-workspace/sglang; \
    fi

# Editable install (fast - dependencies already installed via constraints)
# Clean up __pycache__/tests/pyc in same RUN to avoid writing ~28k files to layer
ARG CARGO_BUILD_JOBS
ARG MAX_JOBS
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cargo/registry,id=cargo-registry \
    --mount=type=cache,target=/root/.cargo/git,id=cargo-git \
    --mount=type=cache,target=/sgl-workspace/sglang/rust/target,id=sglang-target-${TARGETARCH},sharing=locked \
    --mount=type=cache,target=/sgl-workspace/sglang/rust/sglang-radix-tree/target,id=sglang-radix-target-${TARGETARCH},sharing=locked \
    --mount=type=bind,source=docker/install_sglang.py,target=/tmp/install_sglang.py \
    cd /sgl-workspace/sglang \
    && python3 /tmp/install_sglang.py "python[${BUILD_TYPE}]" \
    && kernels lock python \
    && ( success=0; \
         # aarch64: kernels-community/sgl-flash-attn3 ships no arm variants; JIT-compile at runtime.
         # Remove this branch once arm cubins are published upstream.
         if [ "$(uname -m)" = "aarch64" ]; then \
             echo "Skipping kernels-community/sgl-flash-attn3 cubin download on aarch64 (no variants published upstream); kernels will be JIT-compiled at runtime"; \
             success=1; \
         else \
             for i in 1 2 3; do \
                 echo "Attempt $i/3: downloading sgl-kernel cubins..."; \
                 if kernels download python > /tmp/kernels-download.log 2>&1; then \
                     cat /tmp/kernels-download.log; success=1; break; \
                 fi; \
                 cat /tmp/kernels-download.log; \
                 # A missing variant cannot be fixed by retrying the same revision. \
                 if grep -Fq "Cannot find a build variant for this system" /tmp/kernels-download.log; then \
                     break; \
                 fi; \
                 [ "$i" = "3" ] && break; \
                 echo "sgl-kernel cubin download failed, retrying in 30s..." && sleep 30; \
             done; \
             # x86: if no prebuilt sgl-flash-attn3 variant matches this torch+CUDA \
             # combo, fall back to runtime JIT instead of failing the build, \
             # mirroring the aarch64 branch. \
             if [ "$success" != "1" ]; then \
                 echo "WARNING: no matching sgl-flash-attn3 cubin variant for this torch+CUDA; kernels will be JIT-compiled at runtime"; \
                 success=1; \
             fi; \
         fi; \
         [ "$success" = "1" ] ) \
    && rm -f /tmp/kernels-download.log \
    && mkdir -p /root/.cache/huggingface /root/.cache/sglang \
    && ( if [ -f python/kernels.lock ]; then mv python/kernels.lock /root/.cache/sglang/; fi ) \
    && ( find /opt/sglang/lib/python3.12/site-packages -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true )

ENV SGLANG_RUST_BUILD_MODE=never


# Set workspace directory
WORKDIR /sgl-workspace/sglang

# Keep build provenance at the end so metadata changes do not invalidate build layers.
ARG SGLANG_BUILD_COMMIT=unknown
ARG SGLANG_BUILD_URL=
ARG SGLANG_IMAGE_TAG=local/sglang:dev
ENV SGLANG_BUILD_COMMIT=${SGLANG_BUILD_COMMIT:-unknown} \
    SGLANG_BUILD_URL=${SGLANG_BUILD_URL:-} \
    SGLANG_IMAGE_TAG=${SGLANG_IMAGE_TAG:-local/sglang:dev}
LABEL org.opencontainers.image.source="https://github.com/sgl-project/sglang" \
      org.opencontainers.image.revision="${SGLANG_BUILD_COMMIT}" \
      org.opencontainers.image.version="${SGLANG_IMAGE_TAG}" \
      org.opencontainers.image.url="${SGLANG_BUILD_URL}" \
      ai.sglang.build.commit="${SGLANG_BUILD_COMMIT}" \
      ai.sglang.build.url="${SGLANG_BUILD_URL}" \
      ai.sglang.image.tag="${SGLANG_IMAGE_TAG}"

########################################################
########## Runtime Image ##############################
########################################################
#
# PURPOSE: Production runtime environment with JIT support
#
# This stage creates a production-ready image containing:
# - Pre-installed SGLang and CUDA dependencies
# - Full CUDA toolchain for JIT compilation (DeepGEMM, Triton, FlashInfer)
# - Optimized for inference workloads and deployment
# - Smaller than framework (no dev tools like vim, tmux, nsight, etc.)
#
# Use this stage when you need:
# - Production deployment of SGLang
# - JIT compilation support for FP8/microscaling kernels
# - Ready-to-run inference server environment
#
# Install the devel image's CUDA packages without its Nsight Compute dependency.
# Starting from runtime keeps Nsight out of the inherited image layers.
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu24.04 AS runtime_base

ARG DEPENDENCY_CACHE_EPOCH

ARG CUDA_VERSION
ARG TARGETARCH
ARG GDRCOPY_VERSION=2.5.1

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    GDRCOPY_HOME=/usr/src/gdrdrv-${GDRCOPY_VERSION}/ \
    SGLANG_RUST_BUILD_MODE=never \
    LIBRARY_PATH=/usr/local/cuda/lib64/stubs

# Add GKE default lib and bin locations + CUDA compiler paths for FlashInfer JIT
ENV PATH="${PATH}:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/cuda/nvvm/bin" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"

# Keep the full CUDA compilation toolchain and development libraries for JIT.
# These are the packages installed by NVIDIA's devel image, excluding Nsight.
RUN --mount=type=cache,target=/var/cache/apt,id=runtime-apt-${TARGETARCH},sharing=locked \
    CUDA_PKG_VERSION=$(echo "${CUDA_VERSION}" | cut -d. -f1,2 | tr . -) \
    && rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update && apt-get install -y --no-install-recommends \
    cuda-cudart-dev-${CUDA_PKG_VERSION} \
    cuda-command-line-tools-${CUDA_PKG_VERSION}=${NV_CUDA_LIB_VERSION} \
    cuda-minimal-build-${CUDA_PKG_VERSION}=${NV_CUDA_LIB_VERSION} \
    cuda-libraries-dev-${CUDA_PKG_VERSION}=${NV_CUDA_LIB_VERSION} \
    cuda-nvml-dev-${CUDA_PKG_VERSION} \
    libnpp-dev-${CUDA_PKG_VERSION}=${NV_LIBNPP_VERSION} \
    libcusparse-dev-${CUDA_PKG_VERSION}=${NV_LIBCUSPARSE_VERSION} \
    libcublas-dev-${CUDA_PKG_VERSION}=${NV_LIBCUBLAS_VERSION} \
    libcudnn9-dev-cuda-13=${NV_CUDNN_VERSION} \
    libcudnn9-headers-cuda-13=${NV_CUDNN_VERSION} \
    && rm -rf /var/lib/apt/lists/*

# Install runtime dependencies.
# Python 3.12 ships in Ubuntu 24.04 main, so no deadsnakes PPA needed.
RUN --mount=type=cache,target=/var/cache/apt,id=runtime-apt-${TARGETARCH},sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends --allow-change-held-packages \
    # Python runtime
    python3.12-full \
    python3.12-dev \
    wget \
    # Core system utilities
    ca-certificates \
    netcat-openbsd \
    curl \
    ffmpeg \
    git \
    # Runtime libraries
    libopenmpi3 \
    libnuma1 \
    libibverbs1 \
    libibumad3 \
    librdmacm1 \
    libnl-3-200 \
    libnl-route-3-200 \
    ibverbs-providers \
    libgoogle-glog0v6t64 \
    libunwind8 \
    libboost-system1.83.0 \
    libboost-thread1.83.0 \
    libboost-filesystem1.83.0 \
    libgrpc++1.51t64 \
    libprotobuf32t64 \
    libhiredis1.1.0 \
    libcurl4 \
    libczmq4 \
    libfabric1 \
    libssl-dev \
    # RDMA runtime
    rdma-core \
    infiniband-diags \
    perftest \
    # Build tools for JIT compilation
    ninja-build \
    # NCCL packages needed for pynccl_allocator JIT compilation (-lnccl)
    libnccl2 \
    libnccl-dev \
    # GPG key verification
    gnupg2 \
    linux-libc-dev \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 2 \
    && update-alternatives --set python3 /usr/bin/python3.12 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# create virtual env for sglang to avoid conflict with system python packages
RUN python3 -m venv /opt/sglang \
    # Keep pip launchers; the framework supplies the actual pip package/version.
    && rm -rf /opt/sglang/lib/python3.12/site-packages/pip \
        /opt/sglang/lib/python3.12/site-packages/pip-*.dist-info
ENV PATH="/opt/sglang/bin:${PATH}"

# Set up locale
RUN --mount=type=cache,target=/var/cache/apt,id=runtime-apt-${TARGETARCH},sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends locales \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

# Fix Trivy-reported CVEs (see framework stage for full CVE list)
RUN --mount=type=cache,target=/var/cache/apt,id=runtime-apt-${TARGETARCH},sharing=locked \
    apt-get update && apt-get install -y --only-upgrade \
    binutils binutils-common binutils-x86-64-linux-gnu libbinutils \
    libctf0 libctf-nobfd0 libgprofng0 libsframe1 \
    libgnutls30t64 \
    libpam-modules libpam-modules-bin libpam-runtime libpam0g \
    libsqlite3-0 libtasn1-6 \
    dpkg dpkg-dev libdpkg-perl \
    && rm -rf /var/lib/apt/lists/*

# Prepare the Python payload against the exact OS libraries used by runtime.
# Only byte-identical shared libraries are deduplicated; every package, GPU
# architecture, source/header and prebuilt kernel remains available.
FROM runtime_base AS runtime_python
COPY --link --from=framework_final \
    --exclude=sglang-*.dist-info \
    --exclude=__editable__*sglang* \
    --exclude=**/__editable__*sglang* \
    /opt/sglang/lib/python3.12/site-packages /opt/sglang/lib/python3.12/site-packages
RUN --mount=type=bind,source=docker/runtime_python.py,target=/tmp/runtime_python.py \
    python3 -B /tmp/runtime_python.py --mode dependencies \
        --site-packages /opt/sglang/lib/python3.12/site-packages

# Extract only metadata through a bind mount: changing a source version must
# not create another complete site-packages blob in the registry build cache.
FROM runtime_base AS runtime_metadata
RUN --mount=type=bind,from=framework_final,source=/opt/sglang/lib/python3.12/site-packages,target=/framework-python-packages \
    --mount=type=bind,source=docker/runtime_python.py,target=/tmp/runtime_python.py \
    python3 -B /tmp/runtime_python.py --mode metadata \
        --site-packages /framework-python-packages \
        --metadata-dir /runtime-python-metadata

FROM runtime_base AS runtime

# Independent layers let source/version edits reuse the large dependency blob.
# The small editable-install metadata is copied separately at its original path.
COPY --link --from=runtime_python /opt/sglang/lib/python3.12/site-packages /opt/sglang/lib/python3.12/site-packages
COPY --link --from=runtime_metadata /runtime-python-metadata/ /opt/sglang/lib/python3.12/site-packages/

# Keep installed extensions and sources, excluding Cargo's intermediate output.
COPY --link --from=framework_final --exclude=**/target /sgl-workspace /sgl-workspace

# Copy sgl-model-gateway binary
COPY --link --from=framework_final /opt/sglang/bin/sgl-model-gateway /opt/sglang/bin/sgl-model-gateway

# Copy sglang binary
COPY --link --from=framework_final /opt/sglang/bin/sglang /opt/sglang/bin/sglang

# Copy py-spy binary
COPY --link --from=framework_final /opt/sglang/bin/py-spy /opt/sglang/bin/py-spy

# Copy cache for kernels from kernels community
# Ordinary COPY preserves /root's restrictive permissions from the base image.
COPY --from=framework_final /root/.cache/huggingface /root/.cache/huggingface
COPY --from=framework_final /root/.cache/sglang /root/.cache/sglang

# Copy GDRCopy runtime libraries (but not the build artifacts)
COPY --link --from=framework_final /usr/lib/*-linux-gnu/libgdrapi.so* /usr/lib/
COPY --link --from=framework_final /usr/bin/gdrcopy_* /usr/bin/
COPY --link --from=framework_final /usr/src/gdrdrv-${GDRCOPY_VERSION} /usr/src/gdrdrv-${GDRCOPY_VERSION}

# Fix DeepEP IBGDA symlink in runtime
RUN ln -sf /usr/lib/$(uname -m)-linux-gnu/libmlx5.so.1 /usr/lib/$(uname -m)-linux-gnu/libmlx5.so \
    && test -s /usr/lib/libgdrapi.so.2 \
    && ldconfig

WORKDIR /sgl-workspace/sglang

# Keep build provenance at the end so metadata changes do not invalidate build layers.
ARG SGLANG_BUILD_COMMIT=unknown
ARG SGLANG_BUILD_URL=
ARG SGLANG_IMAGE_TAG=local/sglang:dev
ENV SGLANG_BUILD_COMMIT=${SGLANG_BUILD_COMMIT:-unknown} \
    SGLANG_BUILD_URL=${SGLANG_BUILD_URL:-} \
    SGLANG_IMAGE_TAG=${SGLANG_IMAGE_TAG:-local/sglang:dev}
LABEL org.opencontainers.image.source="https://github.com/sgl-project/sglang" \
      org.opencontainers.image.revision="${SGLANG_BUILD_COMMIT}" \
      org.opencontainers.image.version="${SGLANG_IMAGE_TAG}" \
      org.opencontainers.image.url="${SGLANG_BUILD_URL}" \
      ai.sglang.build.commit="${SGLANG_BUILD_COMMIT}" \
      ai.sglang.build.url="${SGLANG_BUILD_URL}" \
      ai.sglang.image.tag="${SGLANG_IMAGE_TAG}"

# Default command
CMD ["/bin/bash"]
