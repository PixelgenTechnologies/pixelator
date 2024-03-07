# Add an entrypoint to the image to allow for easy use of the container on the commandline
# This options needs to be disabled for the best compatibility with Nextflow.
ARG USE_ENTRYPOINT=false
ARG MAKEJOBS=4

# Install pixelator dependencies in a separate stage to improve caching
FROM registry.fedoraproject.org/fedora-minimal:39 as runtime-base
RUN microdnf install -y \
        python3.11 \
        git \
        sqlite \
        zlib \
        libdeflate \
        libstdc++ \
        procps-ng \
     && microdnf clean all

ENV PIPX_BIN_DIR="/usr/local/bin"
RUN python3.11 -m ensurepip
RUN pip3.11 install --upgrade pip
RUN pip3.11 install pipx
RUN pipx install poetry
RUN poetry self add poetry-plugin-export
RUN poetry self add "poetry-dynamic-versioning[plugin]"

# This is needed to easily run other python scripts inside the pixelator container
# eg. samplesheet checking in nf-core/pixelator
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1


FROM runtime-base as builder-base

RUN microdnf install -y \
        python3.11 \
        python3.11-devel \
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


# Build Fastp from source
FROM builder-base as build-fastp

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

FROM builder-base as poetry-deps-install-amd64

WORKDIR /pixelator
COPY poetry.lock pyproject.toml ./
COPY .git .git
RUN poetry export --output requirements.txt --without-hashes --no-interaction --no-ansi

## We need to define the ARG we will be using in each build stage
ARG TARGETPLATFORM
ARG TARGETARCH
ARG TARGETVARIANT

ENV ANNOY_TARGET_VARIANT="${TARGETVARIANT:-v3}"

# We need to control the annoy compile options. By default annoy will use march=native which causes issues when
# building on more modern host than the target platform.

RUN if [ -n "$ANNOY_TARGET_VARIANT" ]; then \
    export ANNOY_COMPILER_ARGS="-D_CRT_SECURE_NO_WARNINGS,-DANNOYLIB_MULTITHREADED_BUILD,-march=x86-64-$ANNOY_TARGET_VARIANT"; \
    echo "Building Annoy for explicit target $TARGETPLATFORM/$ANNOY_TARGET_VARIANT"; \
    pip3.11 install --prefix=/runtime -r requirements.txt; \
   else \
        echo "Building Annoy without implicit target $TARGETPLATFORM"; \
        pip3.11 install --prefix=/runtime -r requirements.txt; \
    fi \
    && rm requirements.txt


FROM runtime-base as  poetry-deps-install-arm64

WORKDIR /pixelator
COPY poetry.lock pyproject.toml /pixelator/
COPY .git /pixelator/.git

RUN poetry export --output requirements.txt --without-hashes --no-interaction --no-ansi
RUN pip3.11 install --prefix=/runtime -r requirements.txt && rm requirements.txt

# ------------------------------------------
# -- Build the pixelator package
# ------------------------------------------
FROM runtime-base as build-pixelator

ARG VERSION_OVERRIDE

WORKDIR /pixelator
COPY . /pixelator
COPY .git /pixelator/.git

RUN poetry config virtualenvs.create false
RUN if [ ! -z "${VERSION_OVERRIDE}" ]; then \
        echo "Overriding version to ${VERSION_OVERRIDE}"; \
        POETRY_DYNAMIC_VERSIONING_OVERRIDE="pixelgen-pixelator=${VERSION_OVERRIDE}" poetry build -f sdist; \
    else poetry build -f sdist; \
    fi

RUN cp -r /pixelator/dist/ /dist/ && \
    rm -rf /pixelator

# ------------------------------------------
# -- Build the runtime environment for amd64
# ------------------------------------------

FROM runtime-base as runtime-amd64

# Copy both fastp executable and isa-l library
COPY --from=build-fastp /usr/local/ /usr/local/
COPY --from=poetry-deps-install-amd64 /runtime/ /usr/local/

# ------------------------------------------
# -- Build the runtime environment for arm64
# ------------------------------------------

FROM runtime-base as runtime-arm64

# Copy both fastp executable and isa-l library
COPY --from=build-fastp /usr/local/ /usr/local/
COPY --from=poetry-deps-install-arm64 /runtime/ /usr/local/

# ------------------------------------------
# -- Build the final image
# ------------------------------------------

FROM runtime-${TARGETARCH} as runtime-final

# Make sure that the python packages are available in the system path
# We add this explicitly since nextflow often runs with PYTHONNOUSERSITE set
# to fix interference with conda and this can cause problems.
# Fastp will also build isal and we need to make that available
ENV PYTHONPATH="$PYTHONPATH:/usr/local/lib/python3.11/site-packages:/usr/local/lib64/python3.11/site-packages"
RUN ldconfig /usr/local/lib64

COPY --from=build-pixelator /dist /dist
RUN ls -alh /dist/
RUN pip3.11 install /dist/*.tar.gz
RUN rm -rf /dist

RUN pip3.11 cache purge
