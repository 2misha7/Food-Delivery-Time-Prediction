from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi import Request
from fastapi.exceptions import RequestValidationError

from src.food_delivery_time.schemas import (
    ContinueTrainRequest,
    PredictRequest,
    TrainMetrics,
)
from src.food_delivery_time.service import (
    continue_train_from_rows,
    predict_from_rows,
    list_models,
)

app = FastAPI()
MODEL_DIR = Path("models")


@app.get("/")
def root():
    return {"status": "ok"}


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content={
            "message": "Invalid input JSON",
            "errors": exc.errors(),
        },
    )



@app.exception_handler(FileNotFoundError)
async def model_not_found_handler(request: Request, exc: FileNotFoundError):
    return JSONResponse(status_code=404, content={"message": str(exc)})


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(status_code=400, content={"message": str(exc)})


@app.post("/continue-train", response_model=TrainMetrics)
def continue_train_endpoint(req: ContinueTrainRequest):
    metrics = continue_train_from_rows(
        model_dir=MODEL_DIR,
        model_name=req.model_name,
        train_rows=[r.model_dump() for r in req.train_input],
        new_model_name=req.new_model_name,
    )
    return metrics


@app.post("/predict")
def predict_endpoint(req: PredictRequest):
    preds = predict_from_rows(
        model_dir=MODEL_DIR,
        model_name=req.model_name,
        input_rows=[r.model_dump() for r in req.input],
    )
    return {"predictions": preds}


@app.get("/models")
def models_endpoint():
    return list_models(MODEL_DIR)
