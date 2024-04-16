import os
import pandas as pd

# create folders
os.makedirs("results/", exist_ok=True)
os.makedirs("results/train/", exist_ok=True)
os.makedirs("results/calibration/", exist_ok=True)
os.makedirs("results/benchmark/", exist_ok=True)

# create csv files
train_headers = ['model', 'base_classifier', 'n_jobs', 'random_state', 'train_split', 'train_time']
train_df = pd.DataFrame(columns=train_headers)
train_df.to_csv('results/benchmark/train.csv', index=False)

calibration_headers = ['model', 'base_classifier', 'n_jobs', 'calibration_method', 'random_state', 'train_split', 'cal_split', 'cal_time']
calibration_df = pd.DataFrame(columns=calibration_headers)
calibration_df.to_csv('results/benchmark/calibration.csv', index=False)

eval_headers = [
    'model', 'base_classifier', 'random_state', 'train_split', 'cal_split', 
    'calibration_method', 'probability_combiner', 'precision', 'recall', 
    'f1', 'p_pre', 'p_rec', 'p_f1', 'brier_score', 'log_loss', 'ece', 'sce', 'ace'
]
eval_df = pd.DataFrame(columns=eval_headers)
eval_df.to_csv('results/benchmark/evaluation.csv', index=False)
