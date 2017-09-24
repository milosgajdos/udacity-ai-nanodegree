FROM continuumio/anaconda3:4.4.0

COPY aind-environment-unix.yml /tmp/aind-environment-unix.yml

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y build-essential && \
            conda env create -f /tmp/aind-environment-unix.yml && \
            source activate aind && \
            pip install \
            git+https://github.com/hmmlearn/hmmlearn.git \
            pygame && \
            source deactivate && \
            conda create --name nb35 python=3.5 jupyter nb_conda && \
            rm -rf /var/lib/apt/lists/*
