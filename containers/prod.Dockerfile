# Add an entrypoint to the image to allow for easy use of the container on the commandline
# This options needs to be disabled for the best compatibility with Nextflow.
ARG USE_ENTRYPOINT=false
ARG MAKEJOBS=4

FROM pixelator-base as runtime

LABEL name = "pixelator"
LABEL license = "MIT"
LABEL vendor = "Pixelgen Technologies AB"


# Install pixelator dependencies in a separate stage to improve caching
FROM runtime AS entrypoint-true
ENTRYPOINT [ "/usr/local/bin/pixelator" ]

FROM runtime AS entrypoint-false
ENTRYPOINT []

FROM entrypoint-${USE_ENTRYPOINT} as final

WORKDIR /
RUN mkdir /data
WORKDIR /data
