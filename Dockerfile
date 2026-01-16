FROM nvidia/cuda:13.0.0-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="$HOME/.local/bin:$PATH"

COPY ./ ./

RUN uv pip install --system -e .

# note: we stay root user for profiling access later
CMD ["/bin/bash"]