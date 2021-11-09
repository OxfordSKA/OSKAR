#
# Build OSKAR and its Python interface.
#
# First install required development packages.
#
FROM nvidia/cuda:11.4.2-base-ubuntu20.04 AS build
ENV DEBIAN_FRONTEND=noninteractive
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
WORKDIR /home/build
RUN git clone https://github.com/OxfordSKA/OSKAR.git OSKAR.git && \
    cmake OSKAR.git/ -DCUDA_ARCH="ALL;8.0;8.6" -DBUILD_TESTING=OFF && \
    make -j4 && make install
RUN pip3 install -U astropy numpy matplotlib setuptools && \
    pip3 install 'git+https://github.com/OxfordSKA/OSKAR.git@master#egg=oskarpy&subdirectory=python'

# Copy into a minimal image.
FROM nvidia/cuda:11.4.2-base-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive
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
