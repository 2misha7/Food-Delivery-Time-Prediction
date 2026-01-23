This project uses a GCS bucket as a DVC remote.



To run dvc pull, you must configure Google credentials (either via a shared service account key or gcloud auth application-default login).

Otherwise, you can download the dataset directly from https://www.kaggle.com/datasets/denkuznetz/food-delivery-time-prediction and place it in data/.”


Sprint 2

We send labeled training data as JSON. Pydantic validates schema/types.
Backend converts JSON → pandas DataFrame.
Model is trained and saved as a new version name.

The API supports any dataset size; for testing and demonstration we used subsets. The same endpoint can be used with the full dataset without code changes.

how to use pycaret: 

  - Train and save best model: python -m food_delivery_time.automl_cli train data/deliverytime.csv --target Delivery_Time_min --model-name my-best-model
  - Predict: python -m food_delivery_time.automl_cli predict my-best-model data/deliverytime.csv --output predictions.csv
  - Score on labeled data: python -m food_delivery_time.automl_cli score my-best-model data/deliverytime.csv --target Delivery_Time_min

how to use optuna:

 python -m food_delivery_time.optuna_cli data/deliverytime.csv --target Delivery_Time_min --n-trials 50 --model-name optuna-best   


 Sprint 4

 Implemented a local Ansible-based deployment for a Dockerized FastAPI service
 How to run app from Ansible ansible-playbook -i inventory.yaml playbook.yaml -K
 How to run app from Docker docker run --rm -p 8000:8000 food-delivery
