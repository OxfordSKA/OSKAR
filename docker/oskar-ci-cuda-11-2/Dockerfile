#
# Build an image for OSKAR CI.
# Install the casacore Tables library and its dependencies.
#
FROM nvidia/cuda:11.2.2-base-ubuntu20.04 AS build
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    bison \
    build-essential \
    cmake \
    flex \
    gfortran \
    liblapack-dev
WORKDIR /home/build
ADD https://github.com/casacore/casacore/archive/v3.2.0.tar.gz casacore.tar.gz
RUN mkdir casacore-src && \
    tar zxf casacore.tar.gz -C casacore-src --strip-components 1 && \
    cmake casacore-src/ -DMODULE=tables -DBUILD_TESTING=OFF -DBUILD_PYTHON=OFF \
    -DUSE_FFTW3=OFF -DUSE_OPENMP=ON -DUSE_HDF5=OFF -DUSE_THREADS=ON && \
    make -j4 && make install

# Copy casacore into a new image with other build dependencies installed.
FROM nvidia/cuda:11.2.2-base-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    cuda-cudart-dev-11-2 \
    cuda-nvcc-11-2 \
    git \
    libcufft-dev-11-2 \
    libhdf5-dev \
    && apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/*
COPY --from=build /usr/local/lib /usr/local/lib/
COPY --from=build /usr/local/include/casacore /usr/local/include/casacore/
