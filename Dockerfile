FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    MPLCONFIGDIR=/tmp/matplotlib

RUN useradd --create-home --uid 10001 app
WORKDIR /app

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN uv sync --frozen --no-dev

COPY configs ./configs
COPY data/synthetic_helpcenter ./data/synthetic_helpcenter
RUN chown -R app:app /app
USER app

# Offline demo: run the example matrix, then write the report into runs/example/.
CMD ["sh", "-c", "rag-eval run configs/example.yaml && rag-eval report"]
