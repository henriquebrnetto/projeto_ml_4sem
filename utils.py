import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import pandas as pd

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


def aggregate_categorical_importances(importances_df):
    aggregated = {}

    for feature, importance in importances_df['Importance'].items():
        if feature.startswith('cat__'):
            # Remove category suffix by splitting from the right only once
            base_feature = 'cat__' + '_'.join(feature.replace('cat__', '').rsplit('_', 1)[0:1])
        else:
            base_feature = feature  # Numerical or clog features stay the same

        if base_feature not in aggregated:
            aggregated[base_feature] = 0
        aggregated[base_feature] += importance

    return pd.DataFrame.from_dict(aggregated, orient='index', columns=['Aggregated_Importance']) \
        .sort_values(by='Aggregated_Importance', ascending=False)

def plot_Xtrain(X_train, filename):
    n_bins = np.floor(np.sqrt(X_train.shape[0])).astype(int).item()
    
    result = X_train \
        .select_dtypes(include='number') \
        .hist(bins=n_bins, figsize=(20, 15))

    for subplot in result.flatten():
        column = subplot.get_title()
        if not column:
            continue
        subplot.set_ylabel('Frequência')

    file = './graphs/'+filename
    plt.savefig(file)
    print(f'Gráfico salvo em {file}')

    return
