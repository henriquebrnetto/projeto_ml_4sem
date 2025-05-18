import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import pandas as pd

def load_data() -> pd.DataFrame:
    return pd.read_csv('ames.csv')


def get_column_subsets(X : pd.DataFrame) -> tuple:
    right_tail = ['Lot.Frontage', 'Lot.Area', 'Mas.Vnr.Area', 'BsmtFin.SF.1',
                'BsmtFin.SF.2','Bsmt.Unf.SF', 'Total.Bsmt.SF', 'X1st.Flr.SF',
                'X2nd.Flr.SF', 'Gr.Liv.Area', 'Garage.Area', 'Wood.Deck.SF',
                'Open.Porch.SF', 'Enclosed.Porch', 'Screen.Porch', 'X3Ssn.Porch',
                'Tot.Porch.SF']

    numerical = [col for col in X.select_dtypes(include='number').columns.tolist() \
                  if col not in right_tail]
    categorical = X.select_dtypes(include='object').columns.tolist()
    
    return right_tail, numerical, categorical

def split_by_prefix(features: list) -> tuple:
    """
    Splits a list of feature names into three lists based on their prefixes:
    'cat__' for categorical, 'num__' for numerical, and 'clog_' for right-skewed data.
    Returns a tuple: (categorical, numerical, right_skewed)
    """
    right_skewed = [f.split('__')[1] for f in features if f.startswith('clog__')]
    numerical = [f.split('__')[1] for f in features if f.startswith('num__')]
    categorical = [f.split('__')[1] for f in features if f.startswith('cat__')]

    return right_skewed, numerical, categorical


def get_outlier_bounds(y : pd.Series):
    q1 = y.quantile(0.25)
    q3 = y.quantile(0.75)
    iqr = q3 - q1

    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def remove_outliers(X : pd.DataFrame, y : pd.Series) -> None:
    """
    Remove outliers do dataset baseado no IQR (Interquartile Range).
    """

    y_ = y.copy()
    y_ = pd.Series(np.log1p(y_))
    lower_bound, upper_bound = get_outlier_bounds(y_)

    mask = (y_ >= lower_bound) & (y_ <= upper_bound)

    return X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)


def aggregate_categorical_importances(importances_df : pd.DataFrame) -> pd.DataFrame:
    aggregated = {}

    for feature, importance in importances_df['Importance'].items():
        if feature.startswith('cat__'):
            # Remove category suffix by splitting from the right only once
            base_feature = 'cat__' + '_'.join(feature.replace('cat__', '').rsplit('_', 1)[0:1])
        else:
            base_feature = feature  # Numerical or log features stay the same

        if base_feature not in aggregated:
            aggregated[base_feature] = 0
        aggregated[base_feature] += importance

    return pd.DataFrame.from_dict(aggregated, orient='index', columns=['Aggregated_Importance']) \
        .sort_values(by='Aggregated_Importance', ascending=False)


def plot_distribution(X : pd.DataFrame, filename : str) -> None:
    n_bins = np.floor(np.sqrt(X.shape[0])).astype(int).item()
    
    result = X \
        .select_dtypes(include='number') \
        .hist(bins=n_bins, figsize=(20, 15))

    for subplot in result.flatten():
        column = subplot.get_title()
        if not column:
            continue
        subplot.set_ylabel('Frequência')

    file = './graphs/'+filename
    plt.tight_layout()
    # plt.subplots_adjust(wspace=0.4, hspace=0.4, left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.savefig(file)
    plt.close('all')
    print(f'Gráfico salvo em {file}')


def y_hist(y: pd.Series, filename : str) -> None:
    n_bins = np.floor(np.sqrt(y.shape[0])).astype(int).item()
    y_log = pd.Series(np.log1p(y))

    _, axes = plt.subplots(1, 2, figsize=(20, 7))

    # Original Histogram
    lower_bound, upper_bound = get_outlier_bounds(y)

    y.hist(bins=n_bins, ax=axes[0])
    axes[0].axvline(lower_bound, color='red', linestyle='--')
    axes[0].axvline(upper_bound, color='red', linestyle='--')
    axes[0].set_title('Distribuição Original')
    axes[0].set_ylabel('Frequência')

    
    # Log Histogram
    lower_bound, upper_bound = get_outlier_bounds(y_log)

    y_log.hist(bins=n_bins, ax=axes[1])
    axes[1].axvline(lower_bound, color='red', linestyle='--')
    axes[1].axvline(upper_bound, color='red', linestyle='--')
    axes[1].set_title('Distribuição Log-Transformada')
    axes[1].set_ylabel('Frequência')

    file = './graphs/' + filename
    plt.tight_layout()
    plt.savefig(file)
    plt.close('all')
    print(f'Gráfico salvo em {file}')