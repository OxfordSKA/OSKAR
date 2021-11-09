#
# Build a base image for OSKAR.
# Install the casacore Tables library and its dependencies.
#
##############################################################################
# This image is no longer required, as we can just use the Ubuntu package
# libcasa-tables4 instead (which pulls in only libcasa-casa4).
##############################################################################
#
FROM nvidia/cuda:11.4.2-base-ubuntu20.04 AS build
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

# Copy into a minimal image.
# Also include other runtime library dependencies here (e.g. cuFFT),
# to avoid having to keep reinstalling these later.
FROM nvidia/cuda:11.4.2-base-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    libcufft-11-4 \
    libgomp1 \
    libhdf5-103 \
    && apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/*
COPY --from=build /usr/local/lib /usr/local/lib/
COPY --from=build /usr/local/include/casacore /usr/local/include/casacore/
