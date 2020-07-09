#
# Build OSKAR using the base image.
#
FROM fdulwich/oskar-base-dep AS build
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    cuda-cudart-dev-10-1 \
    cuda-cufft-dev-10-1 \
    cuda-nvcc-10-1 \
    git \
    liblapack-dev
WORKDIR /home/build
RUN git clone https://github.com/OxfordSKA/OSKAR.git OSKAR.git && \
    cmake OSKAR.git/ && \
    make -j4 && make install

# Copy into a minimal image.
FROM fdulwich/oskar-base-dep
COPY --from=build /usr/local/bin/oskar* /usr/local/bin/
COPY --from=build /usr/local/lib /usr/local/lib/
COPY --from=build /usr/local/include/oskar /usr/local/include/oskar/
