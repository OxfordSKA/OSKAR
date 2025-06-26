#
# Build OSKAR and its Python interface.
#
# First install required development packages.
#
FROM nvidia/cuda:12.8.1-base-ubuntu24.04 AS build
ENV DEBIAN_FRONTEND=noninteractive
# ENV CUDA_ARCH="ALL;9.0;10.0;12.0"
ENV CUDA_ARCH="ALL"
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    build-essential \
    casacore-dev \
    cmake \
    cuda-cudart-dev-12-8 \
    libcufft-dev-12-8 \
    cuda-nvcc-12-8 \
    git \
    libhdf5-dev \
    python3-dev \
    python3-pip \
    python3-venv
WORKDIR /home/build/harp_beam
RUN git clone https://gitlab.com/quentingueuning/harp_beam.git harp_beam.git && \
    cmake harp_beam.git/ -DCUDA_ARCH="${CUDA_ARCH};8.0;8.6" && \
    make -j4 && make install
WORKDIR /home/build/oskar
RUN git clone https://github.com/OxfordSKA/OSKAR.git OSKAR.git && \
    cmake OSKAR.git/ -DCUDA_ARCH="${CUDA_ARCH};9.0;10.0;12.0" -DBUILD_TESTING=OFF && \
    make -j16 && make install

# Create a virtual environment and enable it by setting the PATH.
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip3 install -U astropy numpy matplotlib setuptools && \
    pip3 install 'git+https://github.com/OxfordSKA/OSKAR.git@master#egg=oskarpy&subdirectory=python'

# Copy into a minimal image.
FROM nvidia/cuda:12.8.1-base-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    libcasa-tables7t64 \
    libcufft-12-8 \
    libgomp1 \
    libhdf5-103-1t64 \
    python3 \
    && apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/*
COPY --from=build /usr/local/bin/oskar* /usr/local/bin/
COPY --from=build /usr/local/lib /usr/local/lib/
COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
