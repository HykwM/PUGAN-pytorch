FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-devel AS BASE

# Install Ubuntu software
RUN sed -i 's@archive.ubuntu.com@ftp.jaist.ac.jp/pub/Linux@g' /etc/apt/sources.list
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    locales \
    sudo \
    tzdata \
    x11-apps \
    && apt-get clean \
    && rm -rf /var/cache/apt/archives/* \
    && rm -rf /var/lib/apt/lists/*

# Set locale
RUN locale-gen ja_JP.UTF-8 en_US.UTF-8 \
    && echo 'LC_ALL=en_US.UTF-8' > /etc/default/locale \
    && echo 'LANG=en_US.UTF-8' >> /etc/default/locale
ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8 \
    TZ=Asia/Tokyo

# Set X11 env
ENV QT_X11_NO_MITSHM=1 \
    HOME=/tmp \
    XAUTHORITY=/home/hostuser/.Xauthority

# Install python library
RUN mkdir /src
COPY requirements.txt /src/
COPY pointnet2/ /src/
WORKDIR /src
RUN pip install -r requirements.txt 
RUN python setup.py install
RUN pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl 

# Fix sudo
RUN echo 'Defaults env_keep += "PATH"' >> /etc/sudoers \
    && sed -i "/secure_path/d" /etc/sudoers