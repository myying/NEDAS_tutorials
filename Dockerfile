FROM python:3.13-slim-bookworm AS builder

# Stage 1: build environment
# Install basic build tools, MPICH, FFTW, Python, and dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    bash wget git gnupg2 ca-certificates build-essential \
    mpich libmpich-dev libfftw3-dev\
    python3 python3-pip python3-dev python3-venv && \
    rm -rf /var/lib/apt/lists/*

# Add Intel's repository and install just the IFX compiler
RUN wget -qO - https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | apt-key add - && \
    echo "deb https://apt.repos.intel.com/oneapi all main" > /etc/apt/sources.list.d/oneAPI.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    intel-oneapi-compiler-fortran && \
    rm -rf /var/lib/apt/lists/*

# Install NEDAS and required Python packages
RUN git clone https://github.com/nansencenter/NEDAS.git /opt/NEDAS
WORKDIR /opt/NEDAS

RUN python3 -m venv /opt/py
ENV PATH=/opt/py/bin:$PATH
RUN pip3 install -r /opt/NEDAS/requirements.txt && \
    pip3 install --no-cache-dir numba pyFFTW jupyter && \
    MPICC=mpicc pip3 install --no-cache-dir --no-binary=mpi4py mpi4py

# Build qg model
ENV PATH=/opt/intel/oneapi/compiler/latest/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/latest/lib
ENV FFTW_DIR=/lib/x86_64-linux-gnu
ENV LIBM_DIR=/lib/x86_64-linux-gnu
WORKDIR /opt/NEDAS/NEDAS/models/qg/fortran/src
RUN make

# Stage 2: runtime environment
FROM python:3.13-slim-bookworm

# install the dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    bash git vim ca-certificates mpich && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user with the same UID/GID as the host user
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} appgroup && \
    useradd -u ${USER_ID} -g appgroup -d /home/appuser -s /bin/bash appuser

COPY --from=builder /opt/py /opt/py
COPY --from=builder /opt/NEDAS /home/appuser/NEDAS
COPY --from=builder \
    /opt/intel/oneapi/compiler/latest/lib/libimf.so \
    /opt/intel/oneapi/compiler/latest/lib/libintlc.so \
    /opt/intel/oneapi/compiler/latest/lib/libsvml.so \
    /opt/intel/oneapi/compiler/latest/lib/libirng.so \
    /opt/intel/oneapi/compiler/latest/lib/libifcoremt.so \
    /usr/local/lib/
COPY --from=builder /lib/x86_64-linux-gnu/libfftw3.so /lib/x86_64-linux-gnu/

RUN chown -R ${USER_ID}:${GROUP_ID} /home/appuser /opt/py

# Switch to the new user
USER appuser

ENV PATH=/opt/py/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/lib:/lib/x86_64-linux-gnu
ENV SHELL=/bin/bash

RUN pip install -e /home/appuser/NEDAS

# clone the tutorials repo
RUN git clone https://github.com/myying/NEDAS_tutorials.git /home/appuser/NEDAS_tutorials

WORKDIR /home/appuser/NEDAS_tutorials

EXPOSE 8888

# Default command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser"]

