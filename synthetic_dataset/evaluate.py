import pickle
import pandas as pd
import json
import os
import sys
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

sys.path.append(os.path.abspath('../'))
from FlatClassifier import FlatClassifier
from utils import load_data, calculate_relative_cal_split, create_reliability_diagram

def run(random_state, train_split, cal_split, cal_model_name, args, path, noise):
    X, y = load_data("synthetic_dataset", noise)

    # Split training and test subsets
    _, X_temp, _, y_temp = train_test_split(
        X, y, test_size=(1-train_split), random_state=random_state
    )

    X_test, _, y_test, _ = train_test_split(X_temp, y_temp, test_size=cal_split, random_state=random_state)

    # write ground truth test labels to csv
    labels_name = path+"results/benchmark/predictions/test_labels.csv"
    if not os.path.isfile(labels_name):
        pd.DataFrame(y_test).to_csv(labels_name)

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

        #preds_name = f'{path}results/benchmark/predictions/preds_{args["model"]}_{args["base_classifier"]}_{args["calibration_method"]}_{args["random_state"]}.npy'
        #np.save(preds_name, pipeline_preds, allow_pickle=False)

        combiners = [
        ("none", None),
        ("multiply", MultiplyCombiner(classifier=pipeline['model'])), 
        ("arithmetic", ArithmeticMeanCombiner(classifier=pipeline['model'])), 
        ("geometric", GeometricMeanCombiner(classifier=pipeline['model'])), 
        ]

        if isinstance(pipeline["model"], FlatClassifier):
            combiners = [("none", None)]

        result_df = pd.read_csv(path+"results/benchmark/evaluation.csv")

        for key, combiner in combiners:
            if combiner is None:
                combined_probs = pipeline_probs
            else:
                combined_probs = combiner.combine(pipeline_probs)
            
            #probs_name = f'{path}results/benchmark/predictions/probs_{args["model"]}_{args["base_classifier"]}_{args["calibration_method"]}_{args["random_state"]}_{key}.npz'
            #np.savez_compressed(probs_name, **{"lvl_"+str(lvl):arr for lvl, arr in enumerate(combined_probs)})

            pre = precision(y_test, pipeline_preds)
            rec = recall(y_test, pipeline_preds)
            f1_ = f1(y_test, pipeline_preds)

            # compute classification metrics from probabilities
            if isinstance(pipeline["model"], FlatClassifier):
                pipeline_proba_preds = pipeline.predict(X_test, from_proba=True)
            else:
                pipeline_proba_preds = get_predictions_from_proba(pipeline["model"], combined_probs)

            # save preds from proba
            #preds_proba_name = f'{path}results/benchmark/predictions/preds_proba_{args["model"]}_{args["base_classifier"]}_{args["calibration_method"]}_{args["random_state"]}_{key}.npz'
            #np.save(preds_proba_name, pipeline_proba_preds, allow_pickle=False)

            # compute last level reliability diagram
            last_level_acc, last_level_conf, _, last_level_counts = create_reliability_diagram(pipeline["model"], y_test, combined_probs, pipeline_preds, level=pipeline["model"].max_levels_-1)
            # compute last level reliability diagram for proba preds
            last_level_acc_proba, last_level_conf_proba, _, last_level_counts_proba = create_reliability_diagram(pipeline["model"], y_test, combined_probs, pipeline_proba_preds, level=pipeline["model"].max_levels_-1)


            # compute averaged reliability diagram for pipeline preds
            if isinstance(pipeline["model"], FlatClassifier):
                avg_acc, avg_conf, avg_counts = last_level_acc, last_level_conf, last_level_counts

            else:
                acc, conf, counts = [], [], []
                for level in range(pipeline["model"].max_levels_):
                    acc_, conf_, _, counts_ = create_reliability_diagram(pipeline["model"], y_test, combined_probs, pipeline_preds, level=level)
                    acc.append(acc_)
                    conf.append(conf_)
                    counts.append(counts_)
    
                avg_acc = np.sum(acc, axis=0) / pipeline["model"].max_levels_
                avg_conf = np.sum(conf, axis=0) / pipeline["model"].max_levels_
                avg_counts = np.sum(counts, axis=0) / pipeline["model"].max_levels_

            # compute averaged reliability diagram for pipeline proba preds
            if isinstance(pipeline["model"], FlatClassifier):
                avg_acc_proba, avg_conf_proba, avg_counts_proba = last_level_acc_proba, last_level_conf_proba, last_level_counts_proba

            else:
                acc, conf, counts = [], [], []
                for level in range(pipeline["model"].max_levels_):
                    acc_, conf_, _, counts_ = create_reliability_diagram(pipeline["model"], y_test, combined_probs, pipeline_proba_preds, level=level)
                    acc.append(acc_)
                    conf.append(conf_)
                    counts.append(counts_)
    
                avg_acc_proba = np.sum(acc, axis=0) / pipeline["model"].max_levels_
                avg_conf_proba = np.sum(conf, axis=0) / pipeline["model"].max_levels_
                avg_counts_proba = np.sum(counts, axis=0) / pipeline["model"].max_levels_


            pre_p = precision(y_test, pipeline_proba_preds)
            rec_p = recall(y_test, pipeline_proba_preds)
            f1_p = f1(y_test, pipeline_proba_preds)

            last_level = pipeline_preds.ndim - 1

            brier_score_ll = multiclass_brier_score(pipeline['model'], y_test, combined_probs, level=last_level)
            log_loss_ll = log_loss(pipeline['model'], y_test, combined_probs, level=last_level)
            ece_ll = expected_calibration_error(pipeline['model'], y_test, combined_probs, pipeline_preds, level=last_level)
            sce_ll = static_calibration_error(pipeline['model'], y_test, combined_probs, pipeline_preds, level=last_level)
            ace_ll = adaptive_calibration_error(pipeline['model'], y_test, combined_probs, pipeline_preds, level=last_level)

            if isinstance(combined_probs, list):
                avg_brier_score = multiclass_brier_score(pipeline['model'], y_test, combined_probs)
                avg_log_loss = log_loss(pipeline['model'], y_test, combined_probs)
                avg_ece = expected_calibration_error(pipeline['model'], y_test, combined_probs, pipeline_preds)
                avg_sce = static_calibration_error(pipeline['model'], y_test, combined_probs, pipeline_preds)
                avg_ace = adaptive_calibration_error(pipeline['model'], y_test, combined_probs, pipeline_preds)
            else:
                avg_brier_score = brier_score_ll
                avg_log_loss = log_loss_ll
                avg_ece = ece_ll
                avg_sce = sce_ll
                avg_ace = ace_ll


            scores_to_list = lambda scores: f'[{"|".join([str(score) for score in scores])}]'
            if isinstance(combined_probs, list):
                brier_score_all = multiclass_brier_score(pipeline['model'], y_test, combined_probs, agg=None)
                log_loss_all = log_loss(pipeline['model'], y_test, combined_probs, agg=None)
                ece_all = expected_calibration_error(pipeline['model'], y_test, combined_probs, pipeline_preds, agg=None)
                sce_all = static_calibration_error(pipeline['model'], y_test, combined_probs, pipeline_preds, agg=None)
                ace_all = adaptive_calibration_error(pipeline['model'], y_test, combined_probs, pipeline_preds, agg=None)
            else:
                brier_score_all = [brier_score_ll]
                log_loss_all = [log_loss_ll]
                ece_all = [ece_ll]
                sce_all = [sce_ll]
                ace_all = [ace_ll]


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
                'brier_score_avg': avg_brier_score,
                'log_loss_avg': avg_log_loss,
                'ece_avg': avg_ece,
                'sce_avg': avg_sce,
                'ace_avg': avg_ace,
                'brier_score_ll': brier_score_ll,
                'log_loss_ll': log_loss_ll,
                'ece_ll': ece_ll,
                'sce_ll': sce_ll,
                'ace_ll': ace_ll,
                'brier_score_all': scores_to_list(brier_score_all),
                'log_loss_all': scores_to_list(log_loss_all),
                'ece_all': scores_to_list(ece_all),
                'sce_all': scores_to_list(sce_all),
                'ace_all': scores_to_list(ace_all),
                'acc_ll': scores_to_list(last_level_acc),
                'acc_proba_ll': scores_to_list(last_level_acc_proba),
                'conf_ll': scores_to_list(last_level_conf),
                'conf_proba_ll': scores_to_list(last_level_conf_proba),
                'count_ll': scores_to_list(last_level_counts),
                'count_ll_proba': scores_to_list(last_level_counts_proba),
                'acc_avg': scores_to_list(avg_acc),
                'acc_avg_proba': scores_to_list(avg_acc_proba),
                'conf_avg': scores_to_list(avg_conf),
                'conf_avg_proba': scores_to_list(avg_conf_proba),
                'count_avg': scores_to_list(avg_counts),
                'count_avg_proba': scores_to_list(avg_counts_proba)
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
    parser.add_argument("--noise", action="store", required=False, type=float, default=0.0)

    args = vars(parser.parse_args())

    path = args["path"]
    cal_model_name = f'{path}results/calibration/calibrate_{args["model"]}_{args["base_classifier"]}_{args["calibration_method"]}_{args["random_state"]}_{args["noise"]}.sav'
    train_split = args["train_split"]
    cal_split = calculate_relative_cal_split(train_split, args["cal_split"])


    run(
        random_state=args["random_state"], 
        train_split=train_split, 
        cal_split=cal_split, 
        cal_model_name=cal_model_name,
        args=args,
        path=path,
        noise=args["noise"]
    )