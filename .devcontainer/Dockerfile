# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.245.2/containers/python-3/.devcontainer/base.Dockerfile

# [Choice] Python version (use -bullseye variants on local arm64/Apple Silicon): 3, 3.10, 3.9, 3.8, 3.7, 3.6, 3-bullseye, 3.10-bullseye, 3.9-bullseye, 3.8-bullseye, 3.7-bullseye, 3.6-bullseye, 3-buster, 3.10-buster, 3.9-buster, 3.8-buster, 3.7-buster, 3.6-buster
ARG VARIANT="3.11-bullseye"
FROM mcr.microsoft.com/vscode/devcontainers/python:0-${VARIANT}

# [Choice] Node.js version: none, lts/*, 16, 14, 12, 10
ARG NODE_VERSION="none"
RUN if [ "${NODE_VERSION}" != "none" ]; then su vscode -c "umask 0002 && . /usr/local/share/nvm/nvm.sh && nvm install ${NODE_VERSION} 2>&1"; fi

RUN apt-get update && \
    export DEBIAN_FRONTEND=noninteractive && \
    sudo apt install -y gcc g++ pigz libhdf5-dev vim

USER vscode
ENV PATH="${PATH}:/home/vscode/.local/bin/:/home/vscode/micromamba/bin"
ENV ENV_NAME="base"
ENV MAMBA_ROOT_PREFIX="/home/vscode/conda"
ENV MAMBA_EXE="/home/vscode/micromamba"
RUN curl -L micro.mamba.pm/install.sh > /tmp/micromamba_install.sh && \
    BIN_FOLDER="${HOME}/.local/bin}" INIT_YES="yes" CONDA_FORGE_YES="yes" yes | bash /tmp/micromamba_install.sh
RUN /home/vscode/.local/bin//micromamba shell init -p "$MAMBA_ROOT_PREFIX" -s bash > /dev/null && \
    echo "micromamba activate \$ENV_NAME" >> ~/.bashrc

COPY .devcontainer/mamba_environment.yaml /tmp/mamba_environment.yaml
RUN micromamba install -y -f /tmp/mamba_environment.yaml
