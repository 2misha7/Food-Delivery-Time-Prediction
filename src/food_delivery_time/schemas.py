from typing import Optional

from pydantic import BaseModel


class TrainRow(BaseModel):
    Order_ID: int
    Distance_km: float
    Weather: Optional[str] = None
    Traffic_Level: Optional[str] = None
    Time_of_Day: Optional[str] = None
    Vehicle_Type: Optional[str] = None
    Preparation_Time_min: float
    Courier_Experience_yrs: Optional[float] = None
    Delivery_Time_min: float  # target


class PredictRow(BaseModel):
    Order_ID: int
    Distance_km: float
    Weather: Optional[str] = None
    Traffic_Level: Optional[str] = None
    Time_of_Day: Optional[str] = None
    Vehicle_Type: Optional[str] = None
    Preparation_Time_min: float
    Courier_Experience_yrs: Optional[float] = None


class ContinueTrainRequest(BaseModel):
    model_name: str
    new_model_name: str
    train_input: list[TrainRow]


class PredictRequest(BaseModel):
    model_name: str
    input: list[PredictRow]


class TrainMetrics(BaseModel):
    mae: float
    mae_full: float
    rows: int
    features: int
