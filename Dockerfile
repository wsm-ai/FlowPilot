FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN python -m pip install --no-cache-dir --disable-pip-version-check \
    -r requirements.txt

RUN groupadd --system flowpilot \
    && useradd --system --gid flowpilot --create-home \
        --home-dir /home/flowpilot flowpilot \
    && mkdir -p /app/data \
    && chown flowpilot:flowpilot /app/data

COPY app ./app

USER flowpilot

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
