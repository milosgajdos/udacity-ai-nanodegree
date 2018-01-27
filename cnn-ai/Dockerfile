FROM continuumio/anaconda3:4.4.0

COPY requirements/dog-linux.yml /tmp/dog-linux.yml

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y build-essential && \
            conda env create -f /tmp/dog-linux.yml && \
            source activate dog-project && \
            source deactivate && \
            rm -rf /var/lib/apt/lists/*
