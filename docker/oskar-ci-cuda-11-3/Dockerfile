#
# Build an image for OSKAR CI.
# Install the casacore Tables library and its dependencies.
#
FROM nvidia/cuda:11.3.1-base-ubuntu20.04 AS build
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
WORKDIR /build/casacore
ADD https://github.com/casacore/casacore/archive/v3.2.0.tar.gz casacore.tar.gz
RUN mkdir casacore-src && \
    tar zxf casacore.tar.gz -C casacore-src --strip-components 1 && \
    cmake casacore-src/ -DMODULE=tables -DBUILD_TESTING=OFF -DBUILD_PYTHON=OFF \
    -DUSE_FFTW3=OFF -DUSE_OPENMP=ON -DUSE_HDF5=OFF -DUSE_THREADS=ON && \
    make -j4 && make install

# Install Singularity build dependencies.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
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
RUN go get -u github.com/golang/dep/cmd/dep
WORKDIR /build/singularity
RUN export VERSION=3.8.4 && \
    wget https://github.com/sylabs/singularity/releases/download/v${VERSION}/singularity-ce-${VERSION}.tar.gz && \
    tar -xzf singularity-ce-${VERSION}.tar.gz  && \
    cd singularity-ce-${VERSION} && \
    ./mconfig && make -C ./builddir && make -C ./builddir install

# Copy casacore and Singularity into a new image with other
# build dependencies installed.
FROM nvidia/cuda:11.3.1-base-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    build-essential \
    clang-tidy \
    cuda-cudart-dev-11-3 \
    cuda-nvcc-11-3 \
    git \
    lcov \
    libcufft-dev-11-3 \
    libhdf5-dev \
    nvidia-opencl-dev \
    python3 \
    python3-pip \
    python3-sphinx \
    libssl-dev \
    uuid-dev \
    libgpgme11-dev \
    squashfs-tools \
    libseccomp-dev \
    pkg-config && \
    apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/*

# Need to install CMake >= 3.21.0 for JUnit test output.
# Following instructions at https://apt.kitware.com/
RUN apt-get update && apt-get -qq install gpg wget && \
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null && \
    echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ focal main' | tee /etc/apt/sources.list.d/kitware.list >/dev/null && \
    apt-get update && \
    rm /usr/share/keyrings/kitware-archive-keyring.gpg && \
    apt-get -qq install kitware-archive-keyring cmake && \
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
