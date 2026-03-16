FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl unzip \
    && rm -rf /var/lib/apt/lists/*

# HuggingFace requires a non-root user with ID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy requirements first (for Docker layer caching)
COPY --chown=user requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy app files
COPY --chown=user README.md Dockerfile *.py ./

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]