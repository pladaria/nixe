FROM ubuntu:24.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG MESA_VERSION=26.0.6

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bison \
        build-essential \
        ca-certificates \
        flex \
        git \
        libdrm-dev \
        libexpat1-dev \
        libzstd-dev \
        ninja-build \
        pkg-config \
        python3-mako \
        python3-packaging \
        python3-venv \
        python3-yaml \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/meson \
    && /opt/meson/bin/pip install --no-cache-dir mako meson packaging PyYAML

COPY mesa-tools-nouveau.patch /tmp/mesa-tools-nouveau.patch

RUN git clone --depth 1 --branch "mesa-${MESA_VERSION}" \
        https://gitlab.freedesktop.org/mesa/mesa.git /opt/mesa \
    && git -C /opt/mesa apply /tmp/mesa-tools-nouveau.patch \
    && /opt/meson/bin/meson setup /opt/mesa/build /opt/mesa \
        -Dtools=nouveau \
        -Dgallium-drivers=[] \
        -Dvulkan-drivers=[] \
        -Dplatforms=[] \
        -Dopengl=false \
        -Degl=disabled \
        -Dgbm=disabled \
        -Dglx=disabled \
        -Dllvm=disabled \
    && /opt/meson/bin/meson compile -C /opt/mesa/build nv_push_dump \
    && install -m 0755 \
        /opt/mesa/build/src/nouveau/headers/nv_push_dump \
        /usr/local/bin/nv_push_dump

FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libdrm2 \
        libexpat1 \
        libunwind8 \
        libzstd1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/bin/nv_push_dump /usr/local/bin/nv_push_dump

WORKDIR /work
ENTRYPOINT ["/usr/local/bin/nv_push_dump"]
