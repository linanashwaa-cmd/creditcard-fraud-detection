from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

from src.config import SCALER_PATH


def preprocess_data(df):

    # Separate Features and Target
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Scale only the Amount column
    scaler = StandardScaler()
    X["Amount"] = scaler.fit_transform(X[["Amount"]])

    # Save the scaler
    joblib.dump(scaler, SCALER_PATH)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test