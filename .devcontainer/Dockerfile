FROM mcr.microsoft.com/devcontainers/python:3.14-bookworm

USER root

# Install basic tools
RUN apt-get update && apt-get install -y \
    curl \
    git \
    wget \
    gpg \
    ca-certificates \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 22 LTS from NodeSource
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install latest GitHub CLI from official repository
RUN mkdir -p -m 755 /etc/apt/keyrings \
    && curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
       -o /etc/apt/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
       > /etc/apt/sources.list.d/github-cli.list \
    && apt-get update \
    && apt-get install -y gh \
    && rm -rf /var/lib/apt/lists/*

# Install uv as vscode user
USER vscode

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Claude Code
RUN curl -fsSL https://claude.ai/install.sh | bash

# Install Codex
RUN curl -fsSL https://chatgpt.com/codex/install.sh | sh

ENV PATH="/home/vscode/.local/bin:$PATH"