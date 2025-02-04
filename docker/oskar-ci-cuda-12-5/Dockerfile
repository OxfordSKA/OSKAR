#
# Build an image for OSKAR CI.
# Install the casacore Tables library and its dependencies.
#
FROM nvidia/cuda:12.5.1-base-ubuntu22.04 AS build
ENV DEBIAN_FRONTEND=noninteractive

# Install Singularity build dependencies.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    golang \
    libssl-dev \
    uuid-dev \
    libgpgme11-dev \
    squashfs-tools \
    libseccomp-dev \
    pkg-config \
    wget

# Compile Singularity.
WORKDIR /build/singularity
RUN export VERSION=3.8.5 && \
    wget https://github.com/apptainer/singularity/releases/download/v${VERSION}/singularity-${VERSION}.tar.gz && \
    tar -xzf singularity-${VERSION}.tar.gz  && \
    cd singularity-${VERSION} && \
    ./mconfig && make -C ./builddir && make -C ./builddir install

# Install casacore dependencies.
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    bison \
    flex \
    gfortran \
    liblapack-dev
WORKDIR /build/casacore
ADD https://github.com/casacore/casacore/archive/v3.2.0.tar.gz casacore.tar.gz
RUN mkdir casacore-src && \
    tar zxf casacore.tar.gz -C casacore-src --strip-components 1 && \
    cmake casacore-src/ -DMODULE=tables -DBUILD_TESTING=OFF -DBUILD_PYTHON=OFF \
    -DUSE_FFTW3=OFF -DUSE_OPENMP=ON -DUSE_HDF5=OFF -DUSE_THREADS=ON && \
    make -j4 && make install

# Copy casacore and Singularity into a new image with other
# build dependencies installed.
FROM nvidia/cuda:12.5.1-base-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    build-essential \
    clang-tidy \
    cmake \
    cuda-cudart-dev-12-5 \
    cuda-nvcc-12-5 \
    git \
    lcov \
    libcufft-dev-12-5 \
    libhdf5-dev \
    nvidia-opencl-dev \
    python3 \
    python3-pip \
    python3-sphinx \
    libssl-dev \
    uuid-dev \
    libgpgme-dev \
    squashfs-tools \
    libseccomp-dev \
    pkg-config && \
    apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python packages.
RUN pip3 install numpy sphinx-rtd-theme

# Copy casacore and Singularity from build stage.
COPY --from=build /usr/local/bin/run-singularity /usr/local/bin/
COPY --from=build /usr/local/bin/singularity /usr/local/bin/
COPY --from=build /usr/local/etc /usr/local/etc/
COPY --from=build /usr/local/lib /usr/local/lib/
COPY --from=build /usr/local/libexec /usr/local/libexec/
COPY --from=build /usr/local/var /usr/local/var/
COPY --from=build /usr/local/include/casacore /usr/local/include/casacore/
