from argparse import ArgumentParser
#from hiclass import LocalClassifierPerNode, LocalClassifierPerParentNode, LocalClassifierPerLevel
from hiclass import Pipeline
import pickle
import time
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

'''
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
'''

sys.path.append(os.path.abspath('../'))
#from FlatClassifier import FlatClassifier
from utils import load_data, create_base_classifier, create_model

def run(model, random_state, train_split, model_name, path):
    X, y = load_data("consumer_complaints")

    # Split training and test subsets
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=(1-train_split), random_state=random_state
    )

    # write ground truth training labels to csv
    labels_name = path+"results/benchmark/predictions/training_labels.csv"
    if not os.path.isfile(labels_name):
        pd.DataFrame(y_train).to_csv(labels_name)


    pipeline = Pipeline([
    ('count', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('model', model),
    ])

    start_time = time.time()
    pipeline.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time

    result_df = pd.read_csv(path+"results/benchmark/train.csv")

    row = [{
        'model': args["model"],
        'base_classifier': args["base_classifier"],
        'n_jobs': args["n_jobs"],
        'random_state': random_state,
        'train_split': train_split,
        'train_time': training_time
    }]

    result_df = pd.concat([result_df, pd.DataFrame(row)], ignore_index=True)
    result_df.to_csv(path+"results/benchmark/train.csv", index=False)

    with open(model_name, 'wb') as pickle_file:
        pickle.dump(pipeline, pickle_file)
    

'''
def create_base_classifier(args):
    classifier_name = args["base_classifier"]
    if classifier_name == "LogisticRegression":
        return LogisticRegression(n_jobs=args["n_jobs"], random_state=args["random_state"], max_iter=10000)
    elif classifier_name == "MultinomialNB":
        return MultinomialNB()
    elif classifier_name == "RandomForestClassifier":
        return RandomForestClassifier(n_jobs=args["n_jobs"], random_state=args["random_state"], n_estimators=100, max_depth=6)
    elif classifier_name == "CatBoostClassifier":
        return CatBoostClassifier(thread_count=args["n_jobs"], random_seed=args["random_state"], allow_writing_files=False, silent=True, n_estimators=100, max_depth=6)
    elif classifier_name == "KNeighborsClassifier":
        return KNeighborsClassifier(n_jobs=args["n_jobs"])
    elif classifier_name == "DecisionTreeClassifier":
        return DecisionTreeClassifier(random_state=args["random_state"])
    elif classifier_name == "LGBMClassifier":
        return LGBMClassifier(n_jobs=args["n_jobs"], random_state=args["random_state"], n_estimators=100, max_depth=6)
'''
'''
def create_model(args, base_classifier):
    model_name = args["model"]
    if model_name == "LocalClassifierPerNode":
        return LocalClassifierPerNode(
            local_classifier=base_classifier,
            n_jobs=args["n_jobs"],
            bert=False,
            calibration_method="cvap",
            probability_combiner=None,
            return_all_probabilities=True
        )
    elif model_name == "LocalClassifierPerParentNode":
        return LocalClassifierPerParentNode(
            local_classifier=base_classifier,
            n_jobs=args["n_jobs"],
            bert=False,
            calibration_method="cvap",
            probability_combiner=None,
            return_all_probabilities=True
        )
    elif model_name == "LocalClassifierPerLevel":
        return LocalClassifierPerLevel(
            local_classifier=base_classifier,
            n_jobs=args["n_jobs"],
            bert=False,
            calibration_method="cvap",
            probability_combiner=None,
            return_all_probabilities=True
        )
    elif model_name == "FlatClassifier":
        return FlatClassifier(
            local_classifier=base_classifier,
            n_jobs=args["n_jobs"],
            calibration_method="cvap"
        )
'''


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base_classifier", action="store", required=True, type=str)
    parser.add_argument("--model", action="store", required=True, type=str)
    parser.add_argument("--n_jobs", action="store", required=True, type=int)
    parser.add_argument("--random_state", action="store", required=True, type=int)
    parser.add_argument("--train_split", action="store", required=True, type=float)
    parser.add_argument("--cal_split", action="store", required=True, type=float)
    parser.add_argument("--path", action="store", required=False, type=str, default="./")

    args = vars(parser.parse_args())

    path = args["path"]
    base_classifier = create_base_classifier(args)
    model = create_model(args, base_classifier)

    model_name = f'{path}results/train/train_{args["model"]}_{args["base_classifier"]}_{args["random_state"]}.sav'

    run(model=model,
        random_state=args["random_state"],
        train_split=args["train_split"],
        model_name=model_name,
        path=path
    )
