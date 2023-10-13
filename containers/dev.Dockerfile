# Add an entrypoint to the image to allow for easy use of the container on the commandline
# This options needs to be disabled for the best compatibility with Nextflow.
ARG USE_ENTRYPOINT=false
ARG MAKEJOBS=4

FROM pixelator-base
WORKDIR /pixelator
RUN poetry export --output requirements.txt --without-hashes --no-interaction --no-ansi --with dev
RUN pip3.11 install -r requirements.txt && rm requirements.txt
