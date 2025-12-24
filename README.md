This project uses a GCS bucket as a DVC remote.



To run dvc pull, you must configure Google credentials (either via a shared service account key or gcloud auth application-default login).

Otherwise, you can download the dataset directly from https://www.kaggle.com/datasets/denkuznetz/food-delivery-time-prediction and place it in data/.”


Sprint 2

We send labeled training data as JSON. Pydantic validates schema/types.
Backend converts JSON → pandas DataFrame.
Model is trained and saved as a new version name.

The API supports any dataset size; for testing and demonstration we used subsets. The same endpoint can be used with the full dataset without code changes.
