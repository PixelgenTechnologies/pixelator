FROM registry.fedoraproject.org/fedora-minimal:42 AS runtime-base
# procps-ng provides `ps`, which Nextflow needs to collect task metrics.
RUN microdnf install -y \
    python3.13 \
    procps-ng \
    && microdnf clean all

# This is needed to easily run other python scripts inside the pixelator container
# eg. samplesheet checking in nf-core/pixelator
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.13 1

# ------------------------------------------
# -- Build the pixelator package
# ------------------------------------------
FROM runtime-base AS build-pixelator

# git is needed by hatch-vcs to derive the package version
RUN microdnf install -y git && microdnf clean all

COPY --from=ghcr.io/astral-sh/uv:0.13 /uv /bin/uv

# Use the system interpreter and compile bytecode at install time
ENV UV_PYTHON_DOWNLOADS=0
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT="/usr/"

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

# Make sure that the python packages are available in the system path
# We add this explicitly since nextflow often runs with PYTHONNOUSERSITE set
# to fix interference with conda and this can cause problems.
COPY --from=build-pixelator /usr/local/ /usr/
