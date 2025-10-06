FROM pixelator-base

# Add an entrypoint to the image to allow for easy use of the container on the commandline
# This options needs to be disabled for the best compatibility with Nextflow.
ARG USE_ENTRYPOINT=false
ARG MAKEJOBS=4

LABEL org.opencontainers.image.vendor="Pixelgen Technologies AB"
LABEL org.opencontainers.image.base.name="registry.fedoraproject.org/fedora-minimal:42"
LABEL org.opencontainers.image.licenses="MIT"

WORKDIR /pixelator
COPY ./ /pixelator
COPY .git /pixelator/.git

RUN poetry export --output requirements.txt --without-hashes --no-interaction --no-ansi --only dev

RUN pip3.12 install -r requirements.txt && rm requirements.txt
