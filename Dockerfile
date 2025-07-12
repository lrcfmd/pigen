# Use CUDA base image to support both CPU and GPU
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Install conda
ENV CONDA_DIR /opt/conda
RUN apt-get update && apt-get install -y wget bzip2 && \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    bash /miniconda.sh -b -p $CONDA_DIR && \
    rm /miniconda.sh && \
    ln -s $CONDA_DIR/etc/profile.d/conda.sh /etc/profile.d/conda.sh

ENV PATH=$CONDA_DIR/bin:$PATH

# Copy environment file and create env
COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml && conda clean -a

# Set conda environment as default
ENV CONDA_DEFAULT_ENV=pigen
ENV PATH=$CONDA_DIR/envs/pigen/bin:$PATH

# Copy your code into container
WORKDIR /app
COPY . /app

# Install your package
RUN pip install -e .

# Default command
CMD ["/bin/bash"]
