import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def load_data() -> pd.DataFrame:
    return pd.read_csv('ames.csv')


def remove_outliers(X : pd.DataFrame, y : pd.Series) -> None:
    """
    Remove outliers do dataset baseado no IQR (Interquartile Range).
    """

    
    q1 = y.quantile(0.25)
    q3 = y.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    mask = (y >= lower_bound) & (y <= upper_bound)

    return X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)