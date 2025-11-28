FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir --upgrade pip uv

RUN python -m venv /app/venv

ENV VIRTUAL_ENV=/app/venv

ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy metadata first for Docker layer caching
COPY pyproject.toml ./

RUN uv pip install --no-cache-dir .

COPY server/server.py .

EXPOSE 8000

CMD ["python", "server.py", "--transport", "http", "--host", "0.0.0.0"]
