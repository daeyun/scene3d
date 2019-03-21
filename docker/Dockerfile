# chase-ci

FROM nvidia/cuda:10.0-devel-ubuntu18.04

RUN apt-get -y update && apt-get -y dist-upgrade && apt-get -y autoremove && apt-get -y autoclean

# https://serverfault.com/a/949998
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y tzdata && dpkg-reconfigure --frontend noninteractive tzdata

RUN apt-get -y update && apt-get install -y build-essential cmake wget curl sudo git autotools-dev pkg-config autoconf libopenmpi-dev vim tree zip g++ zlib1g-dev unzip libeigen3-dev \
        libassimp-dev software-properties-common libgles2-mesa-dev freeglut3-dev mesa-utils-extra libglfw3-dev libosmesa6-dev libboost-all-dev \
        libcgal-dev tmux ninja-build libgoogle-glog-dev libgflags-dev rsync tree htop locate whois libomp-dev libblosc-dev libsqlite3-dev assimp-utils libboost-all-dev \
        libglm-dev libgtest-dev libopencv-dev

# "locate" needs updatedb
RUN updatedb

#RUN apt-get install -y nvidia-cuda-toolkit

CMD tail -f /dev/null
