#
# Build an image for OSKAR CI.
#
FROM nvidia/cuda:12.6.3-base-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository -y ppa:apptainer/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    apptainer-suid \
    build-essential \
    casacore-dev \
    clang-tidy \
    cmake \
    cuda-cudart-dev-12-6 \
    cuda-nvcc-12-6 \
    git \
    lcov \
    libcufft-dev-12-6 \
    libhdf5-dev \
    nvidia-opencl-dev \
    python3 \
    python3-pip \
    python3-venv \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create a virtual environment and enable it by setting the PATH.
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python packages.
RUN pip3 install numpy fastcov sphinx sphinx-rtd-theme
