FROM registry.fedoraproject.org/fedora-minimal:42 AS runtime-base
# procps-ng provides `ps`, which Nextflow needs to collect task metrics.
RUN microdnf install -y \
    procps-ng \
    && microdnf clean all

# ------------------------------------------
# -- Build the pixelator package
# ------------------------------------------
FROM runtime-base AS build-pixelator

# git is needed by hatch-vcs to derive the package version
RUN microdnf install -y git && microdnf clean all

# TODO: Remove once pixelgen-pixelator-core is installed from a released wheel
# again. Until then it is built from a git branch, which needs a Rust toolchain
# and a C linker.
RUN microdnf install -y rust cargo gcc && microdnf clean all

COPY --from=ghcr.io/astral-sh/uv:0.13 /uv /bin/uv

# Pin the patch so the image does not float when uv's default 3.14 moves.
# python-build-standalone ships python, python3 and python3.14 in this prefix.
ENV UV_PYTHON_INSTALL_DIR=/opt/python
RUN uv python install 3.14.7 \
    && ln -sfn "$(dirname "$(dirname "$(uv python find 3.14.7)")")" /opt/python/prefix

# Install pixelator into that interpreter, so `python3 -c "import pixelator"`
# works even when Nextflow sets PYTHONNOUSERSITE.
ENV PATH="/opt/python/prefix/bin:${PATH}"
ENV UV_PYTHON=/opt/python/prefix/bin/python3.14
ENV UV_PYTHON_DOWNLOADS=0
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/opt/python/prefix

ARG VERSION_OVERRIDE

WORKDIR /pixelator
COPY . /pixelator

RUN if [ ! -z "${VERSION_OVERRIDE}" ]; then \
    echo "Overriding version to ${VERSION_OVERRIDE}"; \
    SETUPTOOLS_SCM_PRETEND_VERSION="${VERSION_OVERRIDE}" uv sync --no-editable --no-dev --reinstall-package pixelator ; \
    else uv sync --no-editable --no-dev --reinstall-package pixelator ; \
    fi

# ------------------------------------------
# -- Build the final image
# ------------------------------------------
# This must stay the last stage: docker-bake.hcl uses it as the `pixelator-base` context.
FROM runtime-base AS runtime-final

COPY --from=build-pixelator /opt/python /opt/python

# The interpreter bin is first on PATH, so `python`, `python3` and `python3.14`
# all resolve to it. /usr/bin links cover callers that invoke those paths directly
# (Nextflow, nf-core samplesheet checks, and the prod image entrypoint).
ENV PATH="/opt/python/prefix/bin:${PATH}"
RUN ln -sfn /opt/python/prefix/bin/python /usr/bin/python \
    && ln -sfn /opt/python/prefix/bin/python3 /usr/bin/python3 \
    && ln -sfn /opt/python/prefix/bin/python3.14 /usr/bin/python3.14 \
    && ln -sfn /opt/python/prefix/bin/pixelator /usr/bin/pixelator
