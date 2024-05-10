import pandas as pd
from hiclass import LocalClassifierPerNode, LocalClassifierPerParentNode, LocalClassifierPerLevel
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from FlatClassifier import FlatClassifier

def load_data(name):
    if name == "consumer_complaints":
        return _load_consumer_complaints()

def _load_consumer_complaints():
    data = pd.read_csv(
    "data/complaints.csv.zip", 
    compression="zip", 
    header=0, 
    low_memory=False, 
    usecols=["Consumer complaint narrative", "Product", "Sub-product"])

    data.dropna(
        subset=["Consumer complaint narrative", "Product", "Sub-product"], inplace=True
    )

    data.reset_index(drop=True, inplace=True)
    X = data["Consumer complaint narrative"].to_numpy()
    y = data[["Product", "Sub-product"]].to_numpy()

    #X, y = X[:8_000], y[:8_000]
    return X, y

def calculate_relative_cal_split(train_split, cal_split):
    return cal_split * (1 / (1 - train_split))

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

