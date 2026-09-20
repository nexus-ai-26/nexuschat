FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=0 UV_LINK_MODE=copy

RUN apt-get update -qy
RUN apt-get install -qyy -o APT::Install-Recommends=false -o APT::Install-Suggests=false ca-certificates \
    git wget

WORKDIR /app

COPY uv.lock pyproject.toml .python-version ./

RUN --mount=type=cache,id=s/f4598b4a-be3f-4a55-a0ba-2a26c86be730-/root/.cache/uv,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY . /app

FROM python:3.13-slim-bookworm

COPY --from=builder --chown=app:app /app /app

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src:${PYTHONPATH:-}"

WORKDIR /app

CMD alembic upgrade head && python app/main.py
