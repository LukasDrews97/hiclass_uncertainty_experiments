import pandas as pd

def load_data():
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

    X, y = X[:8_000], y[:8_000]
    return X, y

def calculate_relative_cal_split(train_split, cal_split):
    return cal_split * (1 / (1 - train_split))
