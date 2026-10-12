FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# libGL / glib are needed by opencv and mediapipe
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# CPU-only torch wheel (the default PyPI wheel pulls CUDA libraries, about 2 GB)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY pyproject.toml ./
COPY src ./src
COPY models ./models
RUN pip install --no-cache-dir --no-deps .

# Run as a non-root user (Hugging Face Spaces also expects port 7860)
RUN useradd --create-home app
USER app
EXPOSE 7860
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"
CMD ["uvicorn", "fall_detection.app:app", "--host", "0.0.0.0", "--port", "7860"]
