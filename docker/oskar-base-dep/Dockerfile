#
# Build a base image for OSKAR.
# Install the casacore Measurement Set library and its dependencies.
#
FROM nvidia/cuda:10.1-base-ubuntu18.04 AS build
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y --no-install-recommends \
    bison \
    build-essential \
    cmake \
    flex \
    gfortran \
    libcfitsio-dev \
    liblapack-dev \
    wcslib-dev
WORKDIR /home/build
ADD https://github.com/casacore/casacore/archive/v3.2.0.tar.gz casacore.tar.gz
RUN mkdir casacore-src && \
    tar zxf casacore.tar.gz -C casacore-src --strip-components 1 && \
    cmake casacore-src/ -DMODULE=ms -DBUILD_TESTING=OFF -DBUILD_PYTHON=OFF \
    -DUSE_FFTW3=OFF -DUSE_OPENMP=ON -DUSE_HDF5=OFF -DUSE_THREADS=ON && \
    make -j4 && make install

# Copy into a minimal image.
# Also include other runtime library dependencies here (e.g. cuFFT),
# to avoid having to keep reinstalling these later.
FROM nvidia/cuda:10.1-base-ubuntu18.04
RUN apt update && apt install -y --no-install-recommends \
    cuda-cufft-10-1 \
    gosu \
    libblas3 \
    libgfortran4 \
    libgomp1 \
    liblapack3 \
    libquadmath0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
COPY --from=build /usr/local/lib /usr/local/lib/
COPY --from=build /usr/local/include/casacore /usr/local/include/casacore/

# Use entrypoint script to create a user on the fly and avoid running as root.
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]
