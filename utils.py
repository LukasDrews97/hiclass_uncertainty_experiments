import ast
import pandas as pd
import numpy as np
from hiclass import LocalClassifierPerNode, LocalClassifierPerParentNode, LocalClassifierPerLevel
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from hiclass.metrics import _prepare_data
from sklearn.preprocessing import LabelEncoder

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from FlatClassifier import FlatClassifier

def load_data(name, seed=100):
    if name == "consumer_complaints":
        return _load_consumer_complaints()
    if name == "synthetic_dataset":
        return _load_synthetic_dataset()
    if name == "synthetic_dataset_noise":
        return _load_synthetic_dataset(seed, add_noise=True, multiplier=0.5)

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

def _load_synthetic_dataset(seed=100, add_noise=False, multiplier=0.5):
    data = pd.read_csv("data/platypus_diseases.csv")
    if add_noise:
        rng = np.random.default_rng(seed=seed)
        fever_std = data['fever'].std()
        size_std = data["size"].std()
        
        data['fever'] += rng.normal(0, multiplier*fever_std, size=(len(data)))
        data['size'] += rng.normal(0, multiplier*size_std, size=(len(data)))
        
        data['stomach pain'] = add_noise_categorical(data, "stomach pain", rng, multiplier)
        data['skin rash'] = add_noise_categorical(data, "skin rash", rng, multiplier)
        data['cough'] = add_noise_categorical(data, "cough", rng, multiplier)
        data['sniffles'] = add_noise_categorical(data, "sniffles", rng, multiplier)
        data['headache'] = add_noise_categorical(data, "headache", rng, multiplier)
    
    x_columns = ['fever', 'diarrhea', 'stomach pain', 'skin rash', 'cough', 'sniffles', 'short breath', 'headache', 'size']
    y_column = 'label'

    X = data[x_columns].to_numpy()
    y = data['label'].apply(lambda labels: ast.literal_eval(labels)).tolist()
    y = np.array([synthetic_make_label_2_level(l) for l in y])

    return X, y


def add_noise_categorical(data, name, rng, factor):
    std = data[name].std()
    min_v = data[name].min()
    max_v = data[name].max()
    data = data[name].copy()
    data += rng.normal(0, factor*std, size=(len(data)))
    data = np.clip(data, min_v, max_v)
    data = np.clip(np.round(data), min_v, max_v)
    return data

def synthetic_make_label_2_level(label_list):
    if label_list[0] == 'Allergy':
        return [label_list[0], label_list[-1]]
    else:
        return [label_list[0], label_list[1]]

def calculate_relative_cal_split(train_split, cal_split):
    return cal_split * (1 / (1 - train_split))

def create_base_classifier(args):
    classifier_name = args["base_classifier"]
    if classifier_name == "LogisticRegression":
        return LogisticRegression(n_jobs=args["n_jobs"], random_state=args["random_state"], max_iter=10000)
    elif classifier_name == "MultinomialNB":
        return MultinomialNB()
    elif classifier_name == "RandomForestClassifier":
        return RandomForestClassifier(n_jobs=args["n_jobs"], random_state=args["random_state"], n_estimators=100)
    elif classifier_name == "CatBoostClassifier":
        return CatBoostClassifier(thread_count=args["n_jobs"], random_seed=args["random_state"], allow_writing_files=False, silent=True, n_estimators=50, max_depth=5)
    elif classifier_name == "KNeighborsClassifier":
        return KNeighborsClassifier(n_jobs=args["n_jobs"])
    elif classifier_name == "DecisionTreeClassifier":
        return DecisionTreeClassifier(random_state=args["random_state"])
    elif classifier_name == "LGBMClassifier":
        return LGBMClassifier(n_jobs=args["n_jobs"], random_state=args["random_state"], n_estimators=100)

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



def create_reliability_diagram(classifier, y_true, y_prob, y_pred, level, n_bins=10):
    if isinstance(y_prob, np.ndarray):
        y_prob = [y_prob]
        y_prob_level = 0
    else:
        y_prob_level = level
    assert isinstance(y_prob, list)
    
    y_true_score_indices = np.array([classifier.class_to_index_mapping_[level].get(label, -1) for label in y_true[:, level]])
    #y_true_scores = y_prob[level][list(range(len(y_prob[level]))), y_true_score_indices]
    y_true_scores = np.where(y_true_score_indices == -1, 0, y_prob[y_prob_level][np.arange(len(y_prob[y_prob_level])), y_true_score_indices])
    
    y_true, y_pred, labels, y_prob = _prepare_data(classifier, y_true, y_prob[y_prob_level], level, y_pred)

    n_samples, n_classes = y_prob.shape
    
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    y_true_encoded = label_encoder.transform(y_true)
    y_pred_encoded = label_encoder.transform(y_pred)

    #for k in range(n_classes):
    class_scores = y_true_scores
    #class_scores = y_prob[:, k]
    stacked = np.column_stack([class_scores, y_pred_encoded, y_true_encoded])

    # create bins
    _, bin_edges = np.histogram(stacked, bins=n_bins, range=(0, 1))
    bin_indices = np.digitize(stacked, bin_edges)[:, 0]

    # add bin index to each data point
    data = np.column_stack([stacked, bin_indices])

    # create bin mask
    masks = (data[:, -1, None] == range(1, n_bins + 1)).T
    # create actual bins
    bins = [data[masks[i]] for i in range(n_bins)]

    acc = np.zeros(n_bins)
    conf = np.zeros(n_bins)
    for i in range(n_bins):
        acc[i] = (
            1 / (bins[i].shape[0]) * np.sum((bins[i][:, 1] == bins[i][:, 2]))
            if bins[i].shape[0] != 0
            else 0
        )
        conf[i] = (
            1 / (bins[i].shape[0]) * np.sum(bins[i][:, 0])
            if bins[i].shape[0] != 0
            else 0
        )


    bin_counts = np.array([len(bins[i]) for i in range(len(bins))])
    
    return acc, conf, bins, bin_counts

