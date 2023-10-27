from pipelines.training_pipeline import train_pipeline
import logging

if __name__== "__main__":
    # Run the pipeline
    train_pipeline(data_path="/Users/anujvaghani0/Developer/customer-satisfaction-MLOps/data/olist_customers_dataset.csv")
    logging.info("Pipeline run completed")
#testing
