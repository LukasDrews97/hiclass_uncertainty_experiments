# Experiments: Uncertainty Quantification for Local Hierarchical Classification through Model Calibration

## Requirements
- Linux
- Python 3.11

## Installation using pipenv
### Install pipenv
```bash
pip install --user pipenv
```
### Create environment from Pipfile
```bash
pipenv install
```
### Activate environment
```bash
pipenv shell
```

## Installation using conda
### Create environment 
```bash
conda env create -f conda_env.yml
```
### Activate environment
```bash
conda activate hiclass_experiments
```


## Run Experiments
### Open subdirectory
```bash
cd Consumer_Complaint_Database
```

### Download training data from external sources
```bash
dvc update -R .
```

### Start pipeline
```bash
dvc repro -v
```
