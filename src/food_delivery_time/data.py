from pathlib import Path

import pandas as pd


def load_data(path: str | Path) -> pd.DataFrame:
    """
    Load the food delivery dataset from CSV.

    Expected columns (from your Food_Delivery_Times.csv):
    - Order_ID
    - Distance_km
    - Weather
    - Traffic_Level
    - Time_of_Day
    - Vehicle_Type
    - Preparation_Time_min
    - Courier_Experience_yrs
    - Delivery_Time_min
    """
    df = pd.read_csv(path)

    # Ensure numeric columns have correct types
    numeric_cols = [
        "Distance_km",
        "Preparation_Time_min",
        "Courier_Experience_yrs",
        "Delivery_Time_min",
    ]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    return df
