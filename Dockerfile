# Use the official Onyxia Python image (verify the version you need)
FROM inseefrlab/onyxia-jupyter-python

# 1. Switch to root to install system dependencies
USER root

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    mpich \
    libmpich-dev \
    libfftw3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. Switch back to the Onyxia default user (1000)
# This is crucial for Onyxia/Kubernetes permissions
USER 1000
WORKDIR /home/onyxia

# 3. Install mpi4py and other libraries
# We compile mpi4py from source to link it with the system MPICH
RUN MPICC=mpicc pip install --no-cache-dir --no-binary=mpi4py mpi4py && \
    pip install --no-cache-dir numba pyFFTW

