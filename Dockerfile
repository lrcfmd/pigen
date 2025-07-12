# Start from miniconda base image
FROM continuumio/miniconda3

# Set working directory inside container
WORKDIR /app

# Copy environment file
COPY environment.yml .

# Create conda environment named pigen and activate it
RUN conda env create -f environment.yml

# Make sure conda is initialized in every shell
RUN echo "source activate pigen" > ~/.bashrc
ENV PATH /opt/conda/envs/pigen/bin:$PATH

# Copy the rest of your code
COPY . .

# Install your package in editable mode
RUN pip install -e .

# Default command
CMD ["/bin/bash"]

