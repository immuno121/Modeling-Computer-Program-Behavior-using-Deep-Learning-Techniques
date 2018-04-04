#!/bin/bash
#Modules
module initrm cudnn/5.0
module initadd cuda75 cudnn/5.1

# Install Miniconda
curl -O https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
chmod a+x Miniconda2-latest-Linux-x86_64.sh
./Miniconda2-latest-Linux-x86_64.sh
rm Miniconda2-latest-Linux-x86_64.sh
source ~/.bashrc
conda upgrade --all
conda install anaconda-client
conda env create shasvat/tf-py2

echo -e "Installation completed! \nrun \"source ~/.bashrc ; source activate tf-py2\" before submitting TensorFlow jobs."
