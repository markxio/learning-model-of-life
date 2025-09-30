#!/bin/bash

## Run apt update and install with -y and noninteractive frontend
#apt update && apt install -y --no-install-recommends \
#    wget \
#    git \
#    unzip \
#    zip \
#    libgl1 \
#    libglib2.0-0 \
#    ffmpeg \
#    tzdata \ # Explicitly install tzdata if needed \
#    cmake

apt-get update && apt-get install -y git python3-pip python3-tomli
