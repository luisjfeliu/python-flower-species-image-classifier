FROM us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13.py310:latest

WORKDIR /app

ARG REPO_URL
ARG REPO_REF=main
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN test -n "$REPO_URL"
RUN git clone "$REPO_URL" . \
    && git checkout "$REPO_REF"
RUN pip install --no-cache-dir -r requirements.txt

# Default to training; override or add args at job submission time.
ENTRYPOINT ["python", "main.py"]
CMD ["train"]
