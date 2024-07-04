import os
import pandas as pd
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", action="store", required=False, type=str, default="./")
    args = vars(parser.parse_args())
    path = args["path"]

    # create folders
    os.makedirs(path+"results/", exist_ok=True)
    os.makedirs(path+"results/train/", exist_ok=True)
    os.makedirs(path+"results/calibration/", exist_ok=True)
    os.makedirs(path+"results/benchmark/", exist_ok=True)
    os.makedirs(path+"results/benchmark/params/", exist_ok=True)
    os.makedirs(path+"results/benchmark/predictions/", exist_ok=True)
    os.makedirs(path+"results/benchmark/memory/", exist_ok=True)
    os.makedirs(path+"results/benchmark/memory/train/", exist_ok=True)
    os.makedirs(path+"results/benchmark/memory/calibration/", exist_ok=True)
    os.makedirs(path+"results/benchmark/memory/evaluation/", exist_ok=True)

    # create csv files
    train_headers = ['model', 'base_classifier', 'n_jobs', 'random_state', 'train_split', 'train_time']
    train_df = pd.DataFrame(columns=train_headers)
    train_df.to_csv(path+'results/benchmark/train.csv', index=False)

    calibration_headers = ['model', 'base_classifier', 'n_jobs', 'calibration_method', 'random_state', 'train_split', 'cal_split', 'cal_time']
    calibration_df = pd.DataFrame(columns=calibration_headers)
    calibration_df.to_csv(path+'results/benchmark/calibration.csv', index=False)

    eval_headers = [
        'model', 'base_classifier', 'random_state', 'train_split', 'cal_split', 
        'calibration_method', 'probability_combiner', 'noise', 'precision', 'recall', 
        'f1', 'p_pre', 'p_rec', 'p_f1', 'brier_score_avg', 'log_loss_avg', 'ece_avg', 'sce_avg', 'ace_avg', 
        'brier_score_ll', 'log_loss_ll', 'ece_ll', 'sce_ll', 'ace_ll',
        'brier_score_all', 'log_loss_all', 'ece_all', 'sce_all', 'ace_all', 
        'acc_ll', 'acc_proba_ll', 'conf_ll', 'conf_proba_ll', 'count_ll', 'count_ll_proba', 'acc_avg', 
        'acc_avg_proba', 'conf_avg', 'conf_avg_proba', 'count_avg_ll', 'count_avg_proba'
    ]
    eval_df = pd.DataFrame(columns=eval_headers)
    eval_df.to_csv(path+'results/benchmark/evaluation.csv', index=False)
