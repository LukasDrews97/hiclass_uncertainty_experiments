from argparse import ArgumentParser
from hiclass import LocalClassifierPerNode, LocalClassifierPerParentNode, LocalClassifierPerLevel
from hiclass import Pipeline
import pickle
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


from catboost import CatBoostClassifier
from utils import load_data



def run(model, random_state, train_split, model_name):
    X, y = load_data()

    # Split training and test subsets
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=(1-train_split), random_state=random_state
    )

    pipeline = Pipeline([
    ('count', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('model', model),
    ])

    start_time = time.time()
    pipeline.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time

    result_df = pd.read_csv("results/benchmark/train.csv")

    row = [{
        'model': args["model"],
        'base_classifier': args["base_classifier"],
        'n_jobs': args["n_jobs"],
        'random_state': random_state,
        'train_split': train_split,
        'train_time': training_time
    }]

    result_df = pd.concat([result_df, pd.DataFrame(row)], ignore_index=True)
    result_df.to_csv("results/benchmark/train.csv", index=False)

    pickle.dump(pipeline, open(model_name, 'wb'))


def create_base_classifier(args):
    classifier_name = args["base_classifier"]
    if classifier_name == "LogisticRegression":
        return LogisticRegression(n_jobs=args["n_jobs"], random_state=args["random_state"])
    elif classifier_name == "MultinomialNB":
        return MultinomialNB()
    elif classifier_name == "RandomForestClassifier":
        return RandomForestClassifier(n_jobs=args["n_jobs"], random_state=args["random_state"])
    elif classifier_name == "CatBoostClassifier":
        return CatBoostClassifier(thread_count=args["n_jobs"], random_seed=args["random_state"], allow_writing_files=False, silent=True)
    elif classifier_name == "KNeighborsClassifier":
        return KNeighborsClassifier(n_jobs=args["n_jobs"])
    elif classifier_name == "DecisionTreeClassifier":
        return DecisionTreeClassifier(random_state=args["random_state"])

def create_model(args, base_classifier):
    model_name = args["model"]
    if model_name == "LocalClassifierPerNode":
        return LocalClassifierPerNode(
            local_classifier=base_classifier,
            n_jobs=args["n_jobs"],
            bert=False,
            calibration_method=None,
            probability_combiner=None,
            return_all_probabilities=True
        )
    elif model_name == "LocalClassifierPerParentNode":
        return LocalClassifierPerParentNode(
            local_classifier=base_classifier,
            n_jobs=args["n_jobs"],
            bert=False,
            calibration_method=None,
            probability_combiner=None,
            return_all_probabilities=True
        )
    elif model_name == "LocalClassifierPerLevel":
        return LocalClassifierPerLevel(
            local_classifier=base_classifier,
            n_jobs=args["n_jobs"],
            bert=False,
            calibration_method=None,
            probability_combiner=None,
            return_all_probabilities=True
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base_classifier", action="store", required=True, type=str)
    parser.add_argument("--model", action="store", required=True, type=str)
    parser.add_argument("--n_jobs", action="store", required=True, type=int)
    parser.add_argument("--random_state", action="store", required=True, type=int)
    parser.add_argument("--train_split", action="store", required=True, type=float)
    parser.add_argument("--cal_split", action="store", required=True, type=float)

    args = vars(parser.parse_args())

    base_classifier = create_base_classifier(args)
    model = create_model(args, base_classifier)

    model_name = f'results/train/train_{args["model"]}_{args["base_classifier"]}_{args["random_state"]}.sav'

    run(model=model,
        random_state=args["random_state"],
        train_split=args["train_split"],
        model_name=model_name
    )
