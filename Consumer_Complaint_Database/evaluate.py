import pickle
import pandas as pd
import json
import numpy as np
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from hiclass.metrics import (
    precision,
    recall,
    f1,
    multiclass_brier_score,
    log_loss,
    expected_calibration_error,
    static_calibration_error,
    adaptive_calibration_error
)
from hiclass.probability_combiner import (
    MultiplyCombiner,
    ArithmeticMeanCombiner,
    GeometricMeanCombiner
)

from utils import load_data, calculate_relative_cal_split

def run(random_state, train_split, cal_split, cal_model_name, args, path):
    X, y = load_data()

    # Split training and test subsets
    _, X_temp, _, y_temp = train_test_split(
        X, y, test_size=(1-train_split), random_state=random_state
    )

    X_test, _, y_test, _ = train_test_split(X_temp, y_temp, test_size=cal_split, random_state=random_state)

    # write ground truth test labels to csv
    pd.DataFrame(y_test).to_csv(path+"results/benchmark/predictions/test_labels.csv")

    def get_predictions_from_proba(model, proba):
        labels = [model.classes_[level][np.argmax(proba[level], axis=1)] for level in range(model.max_levels_)]
        return np.column_stack(labels)

    with open(cal_model_name, 'rb') as pickle_file:
        pipeline = pickle.load(pickle_file)

        # save pipeline params
        params = {key:str(value) for key, value in pipeline.get_params().items()}
        params_name = f'{path}results/benchmark/params/{args["model"]}_{args["base_classifier"]}_{args["calibration_method"]}_{args["random_state"]}.json'
        with open(params_name, 'w') as params_file:
            json.dump(params, params_file, indent=4)


        pipeline_preds = pipeline.predict(X_test)
        pipeline_probs = pipeline.predict_proba(X_test)

        preds_name = f'{path}results/benchmark/predictions/preds_{args["model"]}_{args["base_classifier"]}_{args["calibration_method"]}_{args["random_state"]}.npy'
        np.save(preds_name, pipeline_preds, allow_pickle=False)

        combiners = [
        ("none", None),
        ("multiply", MultiplyCombiner(classifier=pipeline['model'])), 
        ("arithmetic", ArithmeticMeanCombiner(classifier=pipeline['model'])), 
        ("geometric", GeometricMeanCombiner(classifier=pipeline['model'])), 
        ]

        result_df = pd.read_csv(path+"results/benchmark/evaluation.csv")

        for key, combiner in combiners:
            if combiner is None:
                combined_probs = pipeline_probs
            else:
               combined_probs = combiner.combine(pipeline_probs)
            
            probs_name = f'{path}results/benchmark/predictions/probs_{args["model"]}_{args["base_classifier"]}_{args["calibration_method"]}_{args["random_state"]}_{key}.npz'
            np.savez_compressed(probs_name, **{"lvl_"+str(lvl):arr for lvl, arr in enumerate(combined_probs)})

            pre = precision(y_test, pipeline_preds)
            rec = recall(y_test, pipeline_preds)
            f1_ = f1(y_test, pipeline_preds)

            # compute classification metrics from probabilities
            pipeline_proba_preds = get_predictions_from_proba(pipeline["model"], combined_probs)
            pre_p = precision(y_test, pipeline_proba_preds)
            rec_p = recall(y_test, pipeline_proba_preds)
            f1_p = f1(y_test, pipeline_proba_preds)

            avg_brier_score = multiclass_brier_score(pipeline['model'], y_test, combined_probs)
            avg_log_loss = log_loss(pipeline['model'], y_test, combined_probs)
            avg_ece = expected_calibration_error(pipeline['model'], y_test, combined_probs, pipeline_preds)
            avg_sce = static_calibration_error(pipeline['model'], y_test, combined_probs, pipeline_preds)
            avg_ace = adaptive_calibration_error(pipeline['model'], y_test, combined_probs, pipeline_preds)

            row = [{
                'model': args["model"],
                'base_classifier': args["base_classifier"],
                'random_state': random_state,
                'train_split': train_split,
                'cal_split': args["cal_split"],
                'calibration_method': args["calibration_method"],
                'probability_combiner': key,
                'precision': pre,
                'recall': rec,
                'f1': f1_,
                'p_pre': pre_p,
                'p_rec': rec_p,
                'p_f1': f1_p,
                'brier_score': avg_brier_score,
                'log_loss': avg_log_loss,
                'ece': avg_ece,
                'sce': avg_sce,
                'ace': avg_ace
            }]

            result_df = pd.concat([result_df, pd.DataFrame(row)], ignore_index=True)
            result_df.to_csv(path+"results/benchmark/evaluation.csv", index=False)
            

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base_classifier", action="store", required=True, type=str)
    parser.add_argument("--model", action="store", required=True, type=str)
    parser.add_argument("--calibration_method", action="store", required=True, type=str)
    parser.add_argument("--random_state", action="store", required=True, type=int)
    parser.add_argument("--train_split", action="store", required=True, type=float)
    parser.add_argument("--cal_split", action="store", required=True, type=float)
    parser.add_argument("--path", action="store", required=False, type=str, default="./")

    args = vars(parser.parse_args())

    path = args["path"]
    cal_model_name = f'{path}results/calibration/calibrate_{args["model"]}_{args["base_classifier"]}_{args["calibration_method"]}_{args["random_state"]}.sav'
    train_split = args["train_split"]
    cal_split = calculate_relative_cal_split(train_split, args["cal_split"])


    run(
        random_state=args["random_state"], 
        train_split=train_split, 
        cal_split=cal_split, 
        cal_model_name=cal_model_name,
        args=args,
        path=path
    )