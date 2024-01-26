#!/bin/bash

# source this file to setup the environment
# ANTs
module purge
module load GCC/10.2.0  OpenMPI/4.0.5
module load ANTs/2.3.5

# python env
source /home/gaoc11/basic_env/bin/activate