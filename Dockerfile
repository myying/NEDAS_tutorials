FROM python:3.13-slim

# Install MPICH + build tools
RUN apt-get update && apt-get install -y \
    mpich \
    libmpich-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Jupyter + mpi4py
RUN pip install --no-cache-dir jupyter notebook mpi4py

# Create a non-root user for Onyxia compatibility
RUN useradd -m -s /bin/bash onyxia
RUN mkdir -p /home/onyxia/work
RUN chown -R onyxia:onyxia /home/onyxia

USER onyxia
WORKDIR /home/onyxia/work

# Copy init script and ensure it's executable (if you have sudo/root needs, do this before USER)
COPY --chown=onyxia:onyxia onyxia-init.sh /opt/onyxia-init.sh
COPY --chown=onyxia:onyxia onyxia-set-repositories.sh /opt/onyxia-set-repositories.sh
RUN chmod +x /opt/onyxia-init.sh
RUN chmod +x /opt/onyxia-set-repositories.sh

# Expose Jupyter port
EXPOSE 8888

# Start Jupyter - removed --allow-root since we are using the 'onyxia' user
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser"]
