from src.data_loader import load_data, inspect_data
from src.eda import plot_class_distribution
from src.eda import plot_amount_distribution
from src.eda import correlation_heatmap
from src.preprocessing import preprocess_data
from src.train_model import train_models
from src.evaluate import evaluate_model

def main():

    print("=" * 70)
    print("        CREDIT CARD FRAUD DETECTION USING MACHINE LEARNING")
    print("=" * 70)

    df = load_data()

    inspect_data(df)
    plot_class_distribution(df)
    plot_amount_distribution(df)
    correlation_heatmap(df)

    X_train, X_test, y_train, y_test = preprocess_data(df)
    print("\nPreprocessing completed successfully!")
    print("Training Data :", X_train.shape)
    print("Testing Data :", X_test.shape)

    best_model = train_models(
        X_train,
        X_test,
        y_train,
        y_test
    )

    evaluate_model(
        best_model,
        X_test,
        y_test
    )

if __name__ == "__main__":
    main()