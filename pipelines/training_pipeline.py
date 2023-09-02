from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.evaluation import evaluation_model
from steps.model_train import model_train

@pipeline
def train_pipeline(data_path: str):
    df=ingest_df(data_path)
    clean_df(df)
    model_train(df)
    evaluation_model(df)

