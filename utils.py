import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def load_data() -> pd.DataFrame:
    return pd.read_csv('ames.csv')