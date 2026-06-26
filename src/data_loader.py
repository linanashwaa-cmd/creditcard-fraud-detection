import pandas as pd
from src.config import DATA_PATH


def load_data():
    """
    Load the dataset from the dataset folder.
    """
    df = pd.read_csv(DATA_PATH)
    return df


def inspect_data(df):
    """
    Display basic information about the dataset.
    """

    print("\n" + "=" * 60)
    print("FIRST FIVE ROWS")
    print("=" * 60)
    print(df.head())

    print("\n" + "=" * 60)
    print("DATASET SHAPE")
    print("=" * 60)
    print(df.shape)

    print("\n" + "=" * 60)
    print("COLUMN NAMES")
    print("=" * 60)
    print(df.columns)

    print("\n" + "=" * 60)
    print("DATA TYPES")
    print("=" * 60)
    print(df.dtypes)

    print("\n" + "=" * 60)
    print("MISSING VALUES")
    print("=" * 60)
    print(df.isnull().sum())

    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(df.describe())

    print("\n" + "=" * 60)
    print("CLASS DISTRIBUTION")
    print("=" * 60)
    print(df["Class"].value_counts())