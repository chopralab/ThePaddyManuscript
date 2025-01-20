# Paddy-Arche

Paddy is a Python package developed as an extension of the Paddy Field Algorithm (PFA), a genetic global optimization algorithm proposed by Premaratne et al. (2009). This work showcases the modifications and extended formulations of the PFA developed by members of Chopra-Lab. The experiments include:
- Numeric optimization
- Hyperparameter optimization of a multilayer perceptron
- Targeted molecule generation via the junction tree variational autoencoder (Jaakkola et al., 2019).

This repository contains both the source code for Paddy and the experiments used for benchmarking.

---

# Project Structure

.
├── gramacy
├── JTVAE
├── minmax
├── mlp
├── paddy
├── README.md
└── requirements

## Directory Description

- `gramacy`, `JTVAE`, `minmax`, and `mlp` are the directories containing files used benchmarks testing and evaluation
- `paddy` contains the Paddy algorithm repo and its documentation
- `requirements` holds environment configuration files for each benchmark

## Benchmark Testing Compute

JTVAE was tested on a CentOS Linux, Version 7 (Core) machine using Conda 4.6.11, build version 3.17.9. These requirement files should work for `linux-64` distributions running Conda. Additional information regarding Conda usage can be found [here](https://docs.conda.io/).

Experiments including MinMax Gramacy Lee and MLP Hyperparameter Optimization run on Purdue's HPC system, Gilbreth, specifically on node K. Node K features the following specifications:
- **Cores per Node:** 64
- **Memory per Node:** 512 GB
- **GPUs per Node:** 2 A100 (80 GB)
---


Conda environments can be created with the command:
```bash
conda env create -f name.yml
```
Where `name` is the name of the requirement file. All `.yml` files are located in the "Requirements" folder.

---

## Downloading Paddy

To download Paddy and this repository, run the following commands:
```bash
git clone https://github.com/chopralab/Paddy_Manuscript_Repo
git submodule init
git submodule update
```

---

## Experiments

JTVAE scripts should be run from the `Paddy_Manuscript_Repo` root directory.

MinMax, Gramacy Lee, and MLP benchmarks were executed on Purdue's High-Performance Computing (HPC) cluster, Gilbreth. To reproduce these experiments, you'll need to modify the shell scripts with your specific HPC configurations before running on the backend.
Each benchmark's implementation, including Python files and shell scripts, can be found in their respective directories under the python_files folder.

### Specific Experiment Details

#### MinMax
Run using a Conda environment created by running:
conda env create -f requirements/MinMaxGramacy.yml

Execute the benchmark scripts:
```bash
python minmax/python_files/<script_name>.py
```

#### Interpolation
Run using a Conda environment created by running: 
conda env create -f requirements/MinMaxGramacy.yml

Execute the benchmark scripts:
```bash
python gramacy/benchmark/python_files/<script_name>.py
```

#### MLP Hyperparameter Optimization
Run using a Conda environment created by running: 
conda env create -f requirements/MLP_environemnt.yml.yml

Execute the benchmark scripts:
```bash
python mlp/python_files/<script_name>.py
```

#### JTVAE
Run using a Conda environment created from `JTVAE.yml`. Example command:
```bash
python JTVAE/Paddy/Paddy_Tversky_Gen.py
```
Directories for Hyperopt, Paddy, and Random are named accordingly.

## Results Logs

Due to GitHub's file size limitations, benchmark logs have been compressed into tar files. To access individual benchmark results:

1. Download the respective benchmark's tar file
2. Extract the contents to view detailed logs:
tar -xvf <benchmark_name>_logs.tar


## Results Visualization

To reproduce the manuscript figures:

1. Each benchmark directory contains a plots folder with visualization scripts
2. Required data files are stored in the respective benchmark's tar file under the logs directory
3. Extract the tar file to access the data needed for plotting:
tar -xvf <benchmark_name>_logs.tar

## Plotting Results
The visualization process is organized by benchmark and algorithm:

Navigate to the `plots` directory within each benchmark folder to find:

1. Algorithm-specific Jupyter notebooks for data parsing and visualization
2. Each notebook is named according to its corresponding algorithm and plot type

To generate plots:

1. Extract the log data from the respective benchmark's tar file
2. Open the relevant Jupyter notebook
3. Run all cells to generate the figures

#### JTVAE
A 3-D scatter plot of the sampling space can be visualized using the `UMAP.yml` environment.

---

## Citation

If you use Paddy in your research, please cite the original PFA paper by Premaratne et al. (2009) and the relevant works from Chopra-Lab.
