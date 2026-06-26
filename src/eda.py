import matplotlib.pyplot as plt
import seaborn as sns


def plot_class_distribution(df):

    plt.figure(figsize=(7,5))

    sns.countplot(x="Class", data=df)

    plt.title("Class Distribution")

    plt.xlabel("Transaction Type")

    plt.ylabel("Count")

    plt.xticks([0,1],["Non-Fraud","Fraud"])

    plt.show()


def plot_amount_distribution(df):

    plt.figure(figsize=(8,5))

    sns.histplot(df["Amount"], bins=50)

    plt.title("Transaction Amount Distribution")

    plt.xlabel("Amount")

    plt.ylabel("Frequency")

    plt.show()


def correlation_heatmap(df):

    plt.figure(figsize=(16,12))

    sns.heatmap(df.corr(), cmap="coolwarm")

    plt.title("Correlation Heatmap")

    plt.show()