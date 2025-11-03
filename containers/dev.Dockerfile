FROM pixelator-base

# Add an entrypoint to the image to allow for easy use of the container on the commandline
# This options needs to be disabled for the best compatibility with Nextflow.
ARG USE_ENTRYPOINT=false
ARG MAKEJOBS=4

LABEL org.opencontainers.image.vendor="Pixelgen Technologies AB"
LABEL org.opencontainers.image.base.name="registry.fedoraproject.org/fedora-minimal:42"
LABEL org.opencontainers.image.licenses="MIT"

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

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"


WORKDIR /pixelator
COPY ./ /pixelator
COPY .git /pixelator/.git

RUN uv sync
