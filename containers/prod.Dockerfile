# Add an entrypoint to the image to allow for easy use of the container on the commandline
# This options needs to be disabled for the best compatibility with Nextflow.
ARG USE_ENTRYPOINT=false
ARG MAKEJOBS=4

FROM pixelator-base AS runtime

LABEL org.opencontainers.image.vendor="Pixelgen Technologies AB"
LABEL org.opencontainers.image.base.name="registry.fedoraproject.org/fedora-minimal:42"
LABEL org.opencontainers.image.licenses="MIT"

# Install pixelator dependencies in a separate stage to improve caching
FROM runtime AS entrypoint-true
ENTRYPOINT [ "/usr/bin/pixelator" ]

FROM runtime AS entrypoint-false
ENTRYPOINT []

FROM entrypoint-${USE_ENTRYPOINT} AS final

WORKDIR /
RUN mkdir /data
WORKDIR /data
