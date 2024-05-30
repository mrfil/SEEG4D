ARG CUDA_VERSION=11.6.1-runtime-ubuntu20.04
FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}

WORKDIR /workspace

ARG DEBIAN_FRONTEND="noninteractive"
ARG mne_v=v0.23.0
ARG MNE_USER="mne_user"
ARG HOME_DIR="/home/${MNE_USER}"
ENV MNE_USER=${MNE_USER}
ENV HOME_DIR=${HOME_DIR}

RUN chmod 777 -R /opt
VOLUME /var/mne_test

ENV FSLDIR="/opt/fsl-6.0.5.1" \
    PATH="/opt/fsl-6.0.5.1/bin:$PATH" \
    FSLOUTPUTTYPE="NIFTI_GZ" \
    FSLMULTIFILEQUIT="TRUE" \
    FSLTCLSH="/opt/fsl-6.0.5.1/bin/fsltclsh" \
    FSLWISH="/opt/fsl-6.0.5.1/bin/fslwish" \
    FSLLOCKDIR="" \
    FSLMACHINELIST="" \
    FSLREMOTECALL="" \
    FSLGECUDAQ="cuda.q"


RUN export DEBIAN_FRONTEND=noninteractive && apt-get update && \
    apt-get install --no-install-recommends --yes \
        wget \
        unzip \
        ca-certificates \
        software-properties-common && \
    apt-get clean && apt-get autoremove


RUN apt-key del 7fa2af80 && \
    apt-get install -y wget && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb

RUN sed -i '/developer\.download\.nvidia\.com\/compute\/cuda\/repos/d' /etc/apt/sources.list && \
    rm /etc/apt/sources.list.d/cuda.list 

RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt install -y python3.9
RUN apt install -y python3.9-distutils
RUN apt install -y python3.10
RUN apt install -y python3.10-distutils
RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test

RUN apt-get update -qq \
    && apt-get install -y -q --no-install-recommends \
           libopenblas-dev \
           bzip2 \
           ca-certificates \
           cmake \
           curl \
           dcm2niix \
           locales \
           unzip \
           bc \
           dc \
           file \
           git \
           subversion \
           cmake \
           build-essential \
           libfontconfig1 \
           libfreetype6 \
           libgl1-mesa-dev \
           libgl1-mesa-dri \
           libglu1-mesa-dev \
           libglu1-mesa \
           libgomp1 \
           libice6 \
           libxcursor1 \
           libxft2 \
           libxinerama1 \
           libxrandr2 \
           libxrender1 \
           libxt6 \
           tcsh \
           nano \
	       vim \
	       libfontconfig \
           libfreetype6:amd64 \
           libx11-dev \
           libxxf86vm-dev \
           libxcursor-dev \
           libxrandr-dev \
           libxinerama-dev \
           libegl-dev \
           libwayland-dev \
           wayland-protocols \
           libxkbcommon-dev \
           libdbus-1-dev \
           linux-libc-dev \
           libxi-dev \
           libxmu-dev \
           libxext6 \
           libxcb-randr0-dev \
           libxcb-xtest0-dev \ 
           libxcb-xinerama0-dev \
           libxcb-shape0-dev \
           libxcb-xkb-dev \
           libqt5gui5 \
           libxss1 \
           libxft2 \
           python-tk \
           libvtk6-dev \ 
           libjpeg62 \
           libffi-dev \
           sudo \
           wget \
           xvfb \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
    && dpkg-reconfigure --frontend=noninteractive locales \
    && update-locale LANG="en_US.UTF-8" \
    && chmod 777 /opt && chmod a+s /opt 


RUN echo "Downloading FSL ..." \
    && mkdir -p /opt/fsl-6.0.5.1 \
    && curl -fsSL --retry 5 https://fsl.fmrib.ox.ac.uk/fsldownloads/fsl-6.0.5.1-centos7_64.tar.gz \
    | tar -xz -C /opt/fsl-6.0.5.1 --strip-components 1 \
    && echo "Installing FSL conda environment ..." \
    && bash /opt/fsl-6.0.5.1/etc/fslconf/fslpython_install.sh -f /opt/fsl-6.0.5.1


RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3.9 get-pip.py
RUN python3.10 get-pip.py

RUN pip3.9 install --no-cache-dir -U install setuptools
RUN pip3.9 install cytoolz && \ 
    pip3.9 install dask && \
    pip3.9 install lz4 && \
    pip3.9 install pandas && \
    pip3.9 install tini && \
    pip3.9 install numpy && \ 
    pip3.9 install scipy && \ 
    pip3.9 install matplotlib && \
    pip3.9 install numba && \
    pip3.9 install xlrd && \ 
    pip3.9 install scikit-learn && \ 
    pip3.9 install h5py && \
    pip3.9 install pillow && \
    pip3.9 install statsmodels && \ 
    pip3.9 install jupyter && \ 
    pip3.9 install joblib && \ 
    pip3.9 install psutil && \ 
    pip3.9 install numexpr && \ 
    pip3.9 install imageio && \
    pip3.9 install tqdm && \ 
    pip3.9 install spyder-kernels && \
    pip3.9 install imageio-ffmpeg && \
    pip3.9 install vtk && \ 
    pip3.9 install pyvista && \ 
    pip3.9 install pyvistaqt && \ 
    pip3.9 install qdarkstyle && \ 
    pip3.9 install pyqt5 && \
    pip3.9 install darkdetect && \
    pip3.9 install mayavi && \
    pip3.9 install PySurfer && \
    pip3.9 install dipy && \
    pip3.9 install nibabel && \
    pip3.9 install nilearn && \
    pip3.9 install python-picard && \
    pip3.9 install mne && \ 
    pip3.9 install mffpy && \
    pip3.9 install ipywidgets && \
    pip3.9 install s3fs && \
    pip3.9 install bokeh && \
    pip3.9 install ipyvtklink && \
    pip3.9 install nipype && \ 
    pip3.9 install scikit-image && \
    pip3.9 install fslpy

RUN pip3.10 install --no-cache-dir -U install setuptools
RUN pip3.10 install cytoolz && \ 
    pip3.10 install dask && \
    pip3.10 install lz4 && \
    pip3.10 install pandas && \
    pip3.10 install tini && \
    pip3.10 install numpy && \ 
    pip3.10 install scipy && \ 
    pip3.10 install matplotlib && \
    pip3.10 install numba && \
    pip3.10 install xlrd && \ 
    pip3.10 install scikit-learn && \ 
    pip3.10 install h5py && \
    pip3.10 install pillow && \
    pip3.10 install statsmodels && \ 
    pip3.10 install jupyter && \ 
    pip3.10 install joblib && \ 
    pip3.10 install psutil && \ 
    pip3.10 install numexpr && \ 
    pip3.10 install imageio && \
    pip3.10 install tqdm && \ 
    pip3.10 install spyder-kernels && \
    pip3.10 install imageio-ffmpeg && \
    pip3.10 install pyvista && \ 
    pip3.10 install pyvistaqt && \ 
    pip3.10 install qdarkstyle && \ 
    pip3.10 install darkdetect && \
    pip3.10 install dipy && \
    pip3.10 install nibabel && \
    pip3.10 install nilearn && \
    pip3.10 install python-picard && \
    pip3.10 install mffpy && \
    pip3.10 install ipywidgets && \
    pip3.10 install s3fs && \
    pip3.10 install bokeh && \
    pip3.10 install ipyvtklink && \
    pip3.10 install nipype && \ 
    pip3.10 install scikit-image && \
    pip3.10 install bpy

ENV NVIDIA_DISABLE_REQUIRE=1