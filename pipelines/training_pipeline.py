from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.model_train import model_train
from steps.evaluation import evaluation
from zenml import pipeline



@pipeline(enable_cache=True)
def train_pipeline(data_path:str):
    """
    Args:
        ingest_data: DataClass
        clean_data: DataClass
        model_train: DataClass
        evaluation: DataClass
    Returns:
        mse: float
        rmse: float
    """
    df = ingest_data()
    x_train, x_test, y_train, y_test = clean_data(df)
    model = model_train(x_train, x_test, y_train, y_test)
    r2_score, rmse = evaluation(model, x_test, y_test)
    return r2_score, rmse


# this pipeline created for traning purpose
