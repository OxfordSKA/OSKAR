#
# Build OSKAR and its Python interface.
#
# First install required development packages.
#
FROM nvidia/cuda:11.4.3-base-ubuntu20.04 AS build
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_ARCH="ALL;8.0;8.6"
RUN rm /etc/apt/sources.list.d/cuda.list && \
    apt-key del 7fa2af80 && \
    apt-get update && apt-get install -y --no-install-recommends wget && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    casacore-dev \
    cmake \
    cuda-cudart-dev-11-4 \
    libcufft-dev-11-4 \
    cuda-nvcc-11-4 \
    git \
    libhdf5-dev \
    python3-dev \
    python3-pip
WORKDIR /home/build/harp_beam
RUN git clone https://gitlab.com/quentingueuning/harp_beam.git harp_beam.git && \
    cmake harp_beam.git/ -DCUDA_ARCH="${CUDA_ARCH}" && \
    make -j4 && make install
WORKDIR /home/build/oskar
RUN git clone https://github.com/OxfordSKA/OSKAR.git OSKAR.git && \
    cmake OSKAR.git/ -DCUDA_ARCH="${CUDA_ARCH}" -DBUILD_TESTING=OFF && \
    make -j4 && make install
RUN pip3 install -U astropy numpy matplotlib setuptools && \
    pip3 install 'git+https://github.com/OxfordSKA/OSKAR.git@master#egg=oskarpy&subdirectory=python'

# Copy into a minimal image.
FROM nvidia/cuda:11.4.3-base-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN rm /etc/apt/sources.list.d/cuda.list && \
    apt-key del 7fa2af80 && \
    apt-get update && apt-get install -y --no-install-recommends wget && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libcasa-tables4 \
    libcufft-11-4 \
    libgomp1 \
    libhdf5-103 \
    python3 \
    && apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/*
COPY --from=build /usr/local/bin/oskar* /usr/local/bin/
COPY --from=build /usr/local/lib /usr/local/lib/
COPY --from=build /usr/local/lib/python3.8 /usr/local/lib/python3.8/
