FROM us-central1-docker.pkg.dev/k8s-staging-images/agent-sandbox/python-runtime-sandbox:latest-main
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libxcb1 libsm6 libxext6 libxrender1 libgl1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "pymunk<7" gym-pusht opencv-python-headless numpy
USER 1000
