#
# Build the OSKAR Python interface using the base image.
#
FROM fdulwich/oskar-base AS build
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    python3-dev \
    python3-pip
RUN pip3 install -U astropy numpy matplotlib setuptools && \
    pip3 install 'git+https://github.com/OxfordSKA/OSKAR.git@master#egg=oskarpy&subdirectory=python'

# Copy into a minimal image.
FROM fdulwich/oskar-base
RUN apt-get update && apt-get install -y --no-install-recommends python3 \
    && apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/*
COPY --from=build /usr/local/lib/python3.6 /usr/local/lib/python3.6/
