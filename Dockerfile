# This Dockerfile creates a base dev environment for running grl package code.

# use nvidia/cuda image
FROM nvidia/cuda:11.1.1-devel-ubuntu20.04

# set bash as current shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

# apt install conda dependencies, openspiel dependencies, and general utils
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 \
    git virtualenv clang cmake curl python3 python3-dev python3-pip python3-setuptools python3-wheel python3-tk \
    tmux nano htop

# install conda
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh -O ~/anaconda.sh && \
        /bin/bash ~/anaconda.sh -b -p /opt/conda && \
        rm ~/anaconda.sh && \
        ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
        echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
        find /opt/conda/ -follow -type f -name '*.a' -delete && \
        find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
        /opt/conda/bin/conda clean -afy

# set path to conda
ENV PATH /opt/conda/bin:$PATH

# create conda env and set it as default
COPY environment.yml /grl/
RUN conda update conda && \
    conda env create -f /grl/environment.yml && \
    echo "conda activate grl" >> ~/.bashrc
ENV PATH=/opt/conda/envs/grl/bin:$PATH \
    CONDA_DEFAULT_ENV=$grl

# copy remaining files required for docker setup
COPY .git /grl/.git/
COPY dependencies /grl/dependencies/
COPY .gitmodules setup.py /grl/

# install openspiel dependency
RUN cd /grl && \
    git submodule update --init --recursive && \
    cd dependencies/open_spiel && \
    ./install.sh && \
    pip install -e .

# add optional tmux config not required for running code
# https://github.com/gpakosz/.tmux
RUN cd /root && \
   git clone https://github.com/gpakosz/.tmux.git && \
   ln -s -f .tmux/.tmux.conf && \
   cp .tmux/.tmux.conf.local . && \
   echo "tmux_conf_theme_root_attr='bold'" >> /root/.tmux.conf.local && \
   echo "set -g mouse on" >> /root/.tmux.conf.local

# copy all repo files (except those listed in .dockerignore)
COPY . /grl
WORKDIR /grl

# install grl python package
RUN pip install -e .
