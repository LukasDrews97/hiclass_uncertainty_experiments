# Experiments: Uncertainty Quantification for Local Hierarchical Classification through Model Calibration

## Requirements
- Linux
- Python 3.11

## Installation
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
