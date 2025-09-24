FROM pixelator-base

# Add an entrypoint to the image to allow for easy use of the container on the commandline
# This options needs to be disabled for the best compatibility with Nextflow.
ARG USE_ENTRYPOINT=false
ARG MAKEJOBS=4

LABEL org.opencontainers.image.vendor="Pixelgen Technologies AB"
LABEL org.opencontainers.image.base.name="registry.fedoraproject.org/fedora-minimal:39"
LABEL org.opencontainers.image.licenses="MIT"

WORKDIR /pixelator
COPY ./ /pixelator
COPY .git /pixelator/.git

# Install uv and export dev-only dependencies to a requirements file, then install with pip
RUN python3.12 -m pip install --no-cache-dir uv \
  && uv export --only-group dev --output-file requirements_dev.txt \
  && pip3.12 install --no-cache-dir -r requirements_dev.txt \
  && rm -f requirements_dev.txt
