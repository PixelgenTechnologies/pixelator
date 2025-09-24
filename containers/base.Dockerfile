# Add an entrypoint to the image to allow for easy use of the container on the commandline
# This options needs to be disabled for the best compatibility with Nextflow.
ARG USE_ENTRYPOINT=false
ARG MAKEJOBS=4

# Install pixelator dependencies in a separate stage to improve caching
FROM registry.fedoraproject.org/fedora-minimal:42 AS runtime-base
RUN microdnf install -y \
    python3.12 \
    git \
    sqlite \
    zlib \
    libdeflate \
    libstdc++ \
    procps-ng \
    gzip \
    && microdnf clean all

ENV PIPX_BIN_DIR="/usr/local/bin"
RUN python3.12 -m ensurepip
RUN pip3.12 install --upgrade pip

# Poetry no longer used; dependencies are built via hatchling in the build stage

# This is needed to easily run other python scripts inside the pixelator container
# eg. samplesheet checking in nf-core/pixelator
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1


FROM runtime-base AS builder-base

RUN microdnf install -y \
    python3.12 \
    python3.12-devel \
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
RUN make -j${MAKEJOBS}
RUN make install

# ------------------------------------------
# -- Build the pixelator package
# ------------------------------------------
FROM runtime-base AS build-pixelator

ARG VERSION_OVERRIDE

WORKDIR /pixelator
COPY . /pixelator
COPY .git /pixelator/.git

# Build sdist using hatchling + hatch-vcs
RUN python3.12 -m pip install --no-cache-dir build hatchling hatch-vcs
RUN if [ ! -z "${VERSION_OVERRIDE}" ]; then \
    echo "Overriding version to ${VERSION_OVERRIDE}"; \
    SETUPTOOLS_SCM_PRETEND_VERSION="${VERSION_OVERRIDE}" python3.12 -m build --sdist; \
    else python3.12 -m build --sdist; \
    fi

RUN cp -r /pixelator/dist/ /dist/ && \
    rm -rf /pixelator

# ------------------------------------------
# -- Install Python deps in build image (with compilers)
# ------------------------------------------
FROM builder-base AS deps-install

# Build-arg passthrough for Annoy target variant
ARG TARGETPLATFORM
ARG TARGETARCH
ARG TARGETVARIANT
ENV ANNOY_TARGET_VARIANT="${TARGETVARIANT:-v3}"

# Copy built sdist and install to a relocatable prefix
COPY --from=build-pixelator /dist /dist
RUN if [ "$TARGETARCH" = "amd64" ] && [ -n "$ANNOY_TARGET_VARIANT" ]; then \
    export ANNOY_COMPILER_ARGS="-D_CRT_SECURE_NO_WARNINGS,-DANNOYLIB_MULTITHREADED_BUILD,-march=x86-64-$ANNOY_TARGET_VARIANT"; \
    echo "Compiling Python deps for $TARGETPLATFORM/$ANNOY_TARGET_VARIANT (amd64 with x86-64 flags)"; \
    else \
    unset ANNOY_COMPILER_ARGS; \
    echo "Compiling Python deps without x86-64 variant flags on $TARGETARCH"; \
    fi && \
    pip3.12 install -I --prefix=/runtime /dist/*.tar.gz && \
    rm -rf /dist

# ------------------------------------------
# -- Build the runtime environment for amd64
# ------------------------------------------

FROM runtime-base AS runtime-amd64

COPY --from=build-fastp /usr/local/ /usr/local/
COPY --from=deps-install /runtime/ /usr/

# ------------------------------------------
# -- Build the runtime environment for arm64
# ------------------------------------------

FROM runtime-base AS runtime-arm64

COPY --from=build-fastp /usr/local/ /usr/local/
COPY --from=deps-install /runtime/ /usr/

# ------------------------------------------
# -- Build the final image
# ------------------------------------------

FROM runtime-${TARGETARCH} AS runtime-final

# Ensure Python packages are available in system path and isal
RUN ldconfig /usr/local/lib64

# Nothing to install here; deps already copied into this image in runtime-ARCH stages

RUN pip3.12 cache purge
