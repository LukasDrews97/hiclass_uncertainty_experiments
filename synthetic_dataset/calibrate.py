import pickle
import pandas as pd
import time
import os
import sys
from argparse import ArgumentParser
from joblib import parallel_backend
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath('../'))
from utils import load_data, calculate_relative_cal_split

def run(random_state, train_split, cal_split, train_model_name, cal_model_name, calibration_method, path, noise, args):
    X, y = load_data("synthetic_dataset", noise)

    # Split training and test subsets
    _, X_temp, _, y_temp = train_test_split(
        X, y, test_size=(1-train_split), random_state=random_state
    )

    _, X_cal, _, y_cal = train_test_split(X_temp, y_temp, test_size=cal_split, random_state=random_state)

    # write ground truth calibration labels to csv
    labels_name = path+"results/benchmark/predictions/calibration_labels.csv"
    if not os.path.isfile(labels_name):
        pd.DataFrame(y_cal).to_csv(labels_name)


    with open(train_model_name, 'rb') as pickle_file:
        pipeline = pickle.load(pickle_file)

        # manually set calibration method
        pipeline["model"].calibration_method = calibration_method

        training_time = 0
        if pipeline["model"].calibration_method:
            start_time = time.time()
            with parallel_backend("threading", n_jobs=args["n_jobs"]):
                pipeline.calibrate(X_cal, y_cal)
            end_time = time.time()
            training_time = end_time - start_time
        
        result_df = pd.read_csv(path+"results/benchmark/calibration.csv")

        if args["calibration_method"] is None:
            args["calibration_method"] = "none"

        row = [{
            'model': args["model"],
            'base_classifier': args["base_classifier"],
            'calibration_method': args["calibration_method"],
            'n_jobs': args["n_jobs"],
            'random_state': random_state,
            'train_split': train_split,
            'cal_split': args["cal_split"],
            'cal_time': training_time
        }]

        result_df = pd.concat([result_df, pd.DataFrame(row)], ignore_index=True)
        result_df.to_csv(path+"results/benchmark/calibration.csv", index=False)
        
        pipeline["model"]._clean_up()
        with open(cal_model_name, 'wb') as pickle_file:
            pickle.dump(pipeline, pickle_file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base_classifier", action="store", required=True, type=str)
    parser.add_argument("--model", action="store", required=True, type=str)
    parser.add_argument("--calibration_method", action="store", required=True, type=str)
    parser.add_argument("--n_jobs", action="store", required=True, type=int)
    parser.add_argument("--random_state", action="store", required=True, type=int)
    parser.add_argument("--train_split", action="store", required=True, type=float)
    parser.add_argument("--cal_split", action="store", required=True, type=float)
    parser.add_argument("--path", action="store", required=False, type=str, default="./")
    parser.add_argument("--noise", action="store", required=False, type=float, default=0.0)


    args = vars(parser.parse_args())

    path = args["path"]
    train_split = args["train_split"]
    cal_split = calculate_relative_cal_split(train_split, args["cal_split"])

    train_model_name = f'{path}results/train/train_{args["model"]}_{args["base_classifier"]}_{args["random_state"]}_{args["noise"]}.sav'
    cal_model_name = f'{path}results/calibration/calibrate_{args["model"]}_{args["base_classifier"]}_{args["calibration_method"]}_{args["random_state"]}_{args["noise"]}.sav'
    
    if args["calibration_method"] == "none":
        args["calibration_method"] = None

    run(
        args["random_state"],
        train_split,
        cal_split,
        train_model_name,
        cal_model_name,
        calibration_method=args["calibration_method"],
        args=args,
        path=path,
        noise=args["noise"]
    )