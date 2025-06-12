FROM python:3.11-slim

WORKDIR /app

# Install uv for faster package installation
RUN pip install uv

RUN uv venv

RUN uv pip install "fastmcp[cli]" zeep "python-dotenv"

COPY server/server.py .

# Expose the port the server runs on.
# Our script defaults to port 8000 for HTTP mode.
EXPOSE 8000

# Define the entry point for the container.
# This runs your script when the container starts.
ENTRYPOINT ["python", "server.py"]

# Default command to run.
# This sets the server to run in HTTP mode and listen on all interfaces,
# which is necessary for Docker port mapping to work.
CMD ["--transport", "http", "--host", "0.0.0.0"]