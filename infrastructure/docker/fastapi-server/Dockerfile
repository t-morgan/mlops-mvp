# --- Build Stage ---
FROM python:3.11-slim AS builder

# Install the build-time system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends gcc python3-dev

WORKDIR /app

# Copy and install all Python requirements.
COPY requirements/core.txt ./requirements/
COPY requirements/api.txt ./requirements/
RUN pip install --no-cache-dir -r requirements/api.txt


# --- Final Runtime Stage ---
FROM python:3.11-slim

WORKDIR /app

# Copy the installed python packages from the builder stage.
COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy our application source code
COPY src/ ./src/

# Command to run the Uvicorn server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]