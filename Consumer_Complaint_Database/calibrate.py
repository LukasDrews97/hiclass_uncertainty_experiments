import pickle
import pandas as pd
import time
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from utils import load_data, calculate_relative_cal_split

def run(random_state, train_split, cal_split, train_model_name, cal_model_name, calibration_method, args):
    X, y = load_data()

    # Split training and test subsets
    _, X_temp, _, y_temp = train_test_split(
        X, y, test_size=(1-train_split), random_state=random_state
    )

    _, X_cal, _, y_cal = train_test_split(X_temp, y_temp, test_size=cal_split, random_state=random_state)


    with open(train_model_name, 'rb') as pickle_file:
        pipeline = pickle.load(pickle_file)

        # manually set calibration method
        pipeline["model"].calibration_method = calibration_method

        training_time = 0
        if pipeline["model"].calibration_method:
            start_time = time.time()
            pipeline.calibrate(X_cal, y_cal)
            end_time = time.time()
            training_time = end_time - start_time
        
        result_df = pd.read_csv("results/benchmark/calibration.csv")

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
        result_df.to_csv("results/benchmark/calibration.csv", index=False)
        
        pipeline["model"]._clean_up()
        pickle.dump(pipeline, open(cal_model_name, 'wb'))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base_classifier", action="store", required=True, type=str)
    parser.add_argument("--model", action="store", required=True, type=str)
    parser.add_argument("--calibration_method", action="store", required=True, type=str)
    parser.add_argument("--n_jobs", action="store", required=True, type=int)
    parser.add_argument("--random_state", action="store", required=True, type=int)
    parser.add_argument("--train_split", action="store", required=True, type=float)
    parser.add_argument("--cal_split", action="store", required=True, type=float)

    args = vars(parser.parse_args())

    train_split = args["train_split"]
    cal_split = calculate_relative_cal_split(train_split, args["cal_split"])

    train_model_name = f'results/train/train_{args["model"]}_{args["base_classifier"]}_{args["random_state"]}.sav'
    cal_model_name = f'results/calibration/calibrate_{args["model"]}_{args["base_classifier"]}_{args["calibration_method"]}_{args["random_state"]}.sav'
    
    if args["calibration_method"] == "None":
        args["calibration_method"] = None

    run(
        args["random_state"],
        train_split,
        cal_split,
        train_model_name,
        cal_model_name,
        calibration_method=args["calibration_method"],
        args=args
    )