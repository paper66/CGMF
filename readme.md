# CGMF

![](https://img.shields.io/badge/python-3.8.12-green)![](https://img.shields.io/badge/pytorch-1.8.2-green)![](https://img.shields.io/badge/cudatoolkit-10.2-green)![](https://img.shields.io/badge/cudnn-7.6.5-green)

This folder provides a reference implementation of **CGMF** and four publicly available datasets in the paper:

## Basic Usage

### Requirements

The code was tested with `python 3.8.12`, `pytorch 1.8.2`, `cudatookkit 10.2, and `cudnn 7.6.5`. Install the dependencies via [Anaconda](https://www.anaconda.com/):

```shell
# create virtual environment
conda create --name CGMF python=3.8

# activate environment
conda activate CGMF

# install pytorch & cudatoolkit
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts

# install other requirements
conda install numpy pandas
pip install optuna torchdiffeq torchcde
```

### Run the code

```shell
cd code-and-data

# unzip the dataset
gzip -dk solar_AL.txt.gz
gzip -dk traffic.txt.gz
gzip -dk electricity.txt.gz
gzip -dk exchange_rate.txt.gz

# run the model CGMF
python main.py
```

## Folder Structure

```latex
└── code-and-data
    ├── data # Four publicly available datasets
    ├── model # Source code of CGMF
    ├── config.py # Basic parameter settings
    ├── data.py # Data loading and pre-processing modules
    ├── evaluate.py # Model evaluation module
    ├── main.py # Run model for training, validation, and test
    ├── train.py # Model training module
    ├── utils.py # Defination of auxiliary functions for model running
    └── README.md # This document
```
