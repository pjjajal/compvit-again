# compvit again


*This repo is copied from [compvit](www.github.com/pjjajal/compvit) repo.*

This repository contains the code for *Compressive Vision Transformers*. 

## Setup Instructions

This repository uses `anaconda` to create its python environment. 
I recommend you install a [miniforge](https://conda-forge.org/miniforge/) distribution.

Once you have `anaconda` installed, setup the environment using the following:
```sh
$ conda create -n compvit python=3.9
$ conda activate compvit
$ pip install -r requirements.txt
$ conda install ipykernel # use this if you will be looking at the notebooks.
$ mim install mmcv-full
$ mim install mmcvsegmentation
```

(This should work, hopefully.)

## Repository Outline
