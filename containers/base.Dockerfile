# Add an entrypoint to the image to allow for easy use of the container on the commandline
# This options needs to be disabled for the best compatibility with Nextflow.
ARG USE_ENTRYPOINT=false
ARG MAKEJOBS=4

# Install pixelator dependencies in a separate stage to improve caching
FROM registry.fedoraproject.org/fedora-minimal:42 AS runtime-base
RUN microdnf install -y \
    python3.13 \
    sqlite \
    zlib \
    libdeflate \
    libstdc++ \
    procps-ng \
    gzip \
    tar \
    && microdnf clean all

# This is needed to easily run other python scripts inside the pixelator container
# eg. samplesheet checking in nf-core/pixelator
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.13 1


FROM runtime-base AS builder-base

RUN microdnf install -y \
    python3.13 \
    python3.13-devel \
    wget \
    git \
    sqlite-devel \
    zlib-devel \
    libdeflate-devel \
    gcc \
    gcc-c++ \
    make \
    cmake \
    autoconf \
    automake \
    libtool \
    nasm \
    && microdnf clean all

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.13 1

# Disable Python downloads, because we want to use the system interpreter
ENV UV_PYTHON_DOWNLOADS=0
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

# Build Fastp and isal from source
FROM builder-base AS build-fastp

RUN git clone https://github.com/intel/isa-l.git

WORKDIR /isa-l
RUN ./autogen.sh
RUN ./configure --prefix=/usr/local --libdir=/usr/local/lib64
RUN make
RUN make install
WORKDIR /

RUN git clone https://github.com/OpenGene/fastp.git
WORKDIR /fastp
RUN git checkout v1.1.0
RUN make -j${MAKEJOBS}
RUN make install

FROM builder-base AS uv-deps-install-amd64
ENV UV_PYTHON_DOWNLOADS=0

WORKDIR /pixelator
COPY uv.lock pyproject.toml README.md ./
COPY .git .git

## We need to define the ARG we will be using in each build stage
ARG TARGETPLATFORM
ARG TARGETARCH
ARG TARGETVARIANT

ENV ANNOY_TARGET_VARIANT="${TARGETVARIANT:-v3}"
ENV UV_PYTHON_DOWNLOADS=0
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# We need to control the annoy compile options. By default annoy will use march=native which causes issues when
# building on more modern host than the target platform.

# This will build a command like uv pip install annoy==1.17.0 using the version of annoy that
# is specified in the uv.lock file
RUN if [ -n "$ANNOY_TARGET_VARIANT" ]; then \
    export ANNOY_COMPILER_ARGS="-D_CRT_SECURE_NO_WARNINGS,-DANNOYLIB_MULTITHREADED_BUILD,-march=x86-64-$ANNOY_TARGET_VARIANT"; \
    fi;
RUN uv pip install --system $(uv export --format 'requirements.txt' --no-hashes | grep annoy)

FROM builder-base AS uv-deps-install-arm64
ENV UV_PYTHON_DOWNLOADS=0
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

WORKDIR /pixelator
COPY poetry.lock pyproject.toml README.md /pixelator/
COPY .git /pixelator/.git

RUN uv pip install --system $(uv export --format 'requirements.txt' --no-hashes | grep annoy)

# ------------------------------------------
# -- Build the pixelator package
# ------------------------------------------
FROM builder-base AS build-pixelator

ARG VERSION_OVERRIDE

WORKDIR /pixelator
COPY . /pixelator
COPY .git /pixelator/.git

RUN if [ ! -z "${VERSION_OVERRIDE}" ]; then \
    echo "Overriding version to ${VERSION_OVERRIDE}"; \
    SETUPTOOLS_SCM_PRETEND_VERSION="${VERSION_OVERRIDE}" uv sync --system --reinstall-package pixelator ; \
    else uv sync --system --reinstall-package pixelator ; \
    fi

# ------------------------------------------
# -- Build the runtime environment for amd64
# ------------------------------------------

FROM runtime-base AS runtime-amd64

# Copy both fastp executable and isa-l library
COPY --from=build-fastp /usr/local/ /usr/local/
COPY --from=uv-deps-install-amd64 /usr/local/ /usr/local/

# ------------------------------------------
# -- Build the runtime environment for arm64
# ------------------------------------------

FROM runtime-base AS runtime-arm64

# Copy both fastp executable and isa-l library
COPY --from=build-fastp /usr/local/ /usr/local/
COPY --from=uv-deps-install-arm64 /usr/local/ /usr/local/

# ------------------------------------------
# -- Build the final image
# ------------------------------------------

FROM runtime-${TARGETARCH} AS runtime-final

# Make sure that the python packages are available in the system path
# We add this explicitly since nextflow often runs with PYTHONNOUSERSITE set
# to fix interference with conda and this can cause problems.
COPY --from=build-pixelator /usr/local/ /usr/

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.13 1
# Fastp will also build isal and we need to make that available
RUN ldconfig /usr/local/lib64
