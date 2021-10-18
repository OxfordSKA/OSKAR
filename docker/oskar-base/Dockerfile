#
# Build OSKAR using the base image.
#
FROM artefact.skao.int/oskar-base-dep AS build
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    cuda-cudart-dev-11-4 \
    libcufft-dev-11-4 \
    cuda-nvcc-11-4 \
    git \
    libhdf5-dev
WORKDIR /home/build
RUN git clone https://github.com/OxfordSKA/OSKAR.git OSKAR.git && \
    cmake OSKAR.git/ -DCUDA_ARCH="ALL;8.0;8.6" -DBUILD_TESTING=OFF && \
    make -j4 && make install

# Copy into a minimal image.
FROM artefact.skao.int/oskar-base-dep
COPY --from=build /usr/local/bin/oskar* /usr/local/bin/
COPY --from=build /usr/local/lib /usr/local/lib/
COPY --from=build /usr/local/include/oskar /usr/local/include/oskar/
