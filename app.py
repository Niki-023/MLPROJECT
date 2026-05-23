from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
import sys
from src.mlproject.components.data_ingestion import DataIngestion
#from src.mlproject.components.data_ingestion import DataIngestionConfig
from src.mlproject.components.data_transformation import DataTransformationConfig, DataTranformation
from src.mlproject.components.model_trainer import ModelTrainerConfig,ModelTrainer




if __name__ =="__main__":
    logging.info("The execution has started")

    try: 
        #data_ingestion_config = DataIngestionConfig()
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.Initiate_data_ingestion()


        #data_transformation_config = DataTransformationConfig()
        data_transformation = DataTranformation()
        train_arr, test_arr,_ =data_transformation.initate_data_transformation(train_data_path,test_data_path)

        ## Model Training
        model_trainer = ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_arr,test_arr))


    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)
