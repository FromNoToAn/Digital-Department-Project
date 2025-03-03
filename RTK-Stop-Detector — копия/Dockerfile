FROM ubuntu:22.04

# Install system requirements
RUN apt-get update && apt-get install -y \
    wget \
    python3-dev \
    python3-pip \
    build-essential \
    libssl-dev \
    libffi-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# FFmpeg dependencies
RUN apt-get update && apt-get -y install \
    autoconf \
    automake \
    cmake \
    git \
    git-core \
    libass-dev \
    libfreetype6-dev \
    libgnutls28-dev \
    libmp3lame-dev \
    libsdl2-dev \
    libtool \
    libva-dev \
    libvdpau-dev \
    libvorbis-dev \
    libxcb1-dev \
    libxcb-shm0-dev \
    libxcb-xfixes0-dev \
    meson \
    ninja-build \
    pkg-config \
    texinfo \
    yasm \
    zlib1g-dev \
    libunistring-dev \
    && rm -rf /var/lib/apt/lists/*

# FFmpeg requirements
RUN apt-get update  && apt-get -y install \
    libaom-dev \
    libdav1d-dev \
    libx264-dev \
    libx265-dev \
    libnuma-dev \
    libvpx-dev \
    libfdk-aac-dev \
    libopus-dev \
    libdav1d-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone ffmpeg
RUN git clone --depth 1 -b n7.0.1 https://github.com/FFmpeg/FFmpeg
COPY ./install/h2645_parse.c /FFmpeg/libavcodec/h2645_parse.c

# Set envs for ffmpeg
ENV PATH="/FFmpeg/bin:$PATH"
ENV PKG_CONFIG_PATH="/FFmpeg/ffmpeg_build/lib/pkgconfig"

# Compile ffmpeg
RUN /FFmpeg/configure \
    --prefix="/FFmpeg/ffmpeg_build" \
    --pkg-config-flags="--static" \
    --extra-cflags="-I/FFmpeg/ffmpeg_build/include" \
    --extra-ldflags="-L/FFmpeg/ffmpeg_build/lib" \
    --extra-libs="-lpthread -lm" \
    --ld="g++" \
    --bindir="/FFmpeg/bin" \
    --enable-gpl \
    --enable-gnutls \
    --enable-libaom \
    --enable-libass \
    --enable-libfdk-aac \
    --enable-libfreetype \
    --enable-libmp3lame \
    --enable-libopus \
    --enable-libdav1d \
    --enable-libvorbis \
    --enable-libvpx \
    --enable-libx264 \
    --enable-libx265 \
    --enable-nonfree

# Install ffmpeg
ENV PATH="/FFmpeg/bin:$PATH"
RUN make -j$(nproc) && \
    make install && \
    hash -r

# Expose the Flask port
EXPOSE 9004

# Set the workdir
WORKDIR /detector-api

# Copy requirements
COPY ./requirements.txt ./requirements.txt

# Install requirements
RUN pip3 install -r ./requirements.txt

# Copy project
COPY ./ ./

# Run project
#NTRYPOINT [ "bash", "run.sh"]
ENTRYPOINT ["python3", "stopDetector.py"]
