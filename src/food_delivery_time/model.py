import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


def prepare_features(df: pd.DataFrame):

    df = df.copy()

    # Separate target
    y = df["Delivery_Time_min"]

    # Drop ID and target from features
    X = df.drop(columns=["Order_ID", "Delivery_Time_min"])

    # One-hot encode categoricals
    X = pd.get_dummies(X, drop_first=True)

    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_simple_model(df: pd.DataFrame):

    X_train, X_test, y_train, y_test = prepare_features(df)

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)

    return model, mae
