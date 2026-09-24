FROM python:3.12-slim

# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ---------------------------------------------------------
# Working directory
# ---------------------------------------------------------

WORKDIR /app

# ---------------------------------------------------------
# System dependencies
# ---------------------------------------------------------

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------
# Python dependencies
# ---------------------------------------------------------

COPY requirements/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# ---------------------------------------------------------
# Copy project
# ---------------------------------------------------------

COPY app /app/app
COPY src /app/src
COPY config /app/config
COPY models /app/models

# ---------------------------------------------------------
# Create logs directory
# ---------------------------------------------------------

RUN mkdir -p /app/logs

# ---------------------------------------------------------
# Port
# ---------------------------------------------------------

EXPOSE 8000

# ---------------------------------------------------------
# Start FastAPI
# ---------------------------------------------------------

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]