import sys
import os

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging
from src.exception import CustomException
from src.utils import read_yaml

class TrainingPipeline:
    def __init__(self, config_path="config.yml"):
        try:
            self.config_path = config_path
            self.config = read_yaml(self.config_path)
            logging.info("TrainingPipeline initialized.")
        except Exception as e:
            raise CustomException(e, sys)

    def start_data_ingestion(self):
        try:
            logging.info("Starting Data Ingestion Phase")
            di_config = self.config['data_ingestion']
            
            data_ingestion = DataIngestion(
                zip_path=di_config['zip_path'],
                extract_path=di_config['extract_path'],
                data_dir=di_config['data_dir'],
                output_root=di_config['output_root']
            )
            data_ingestion.initiate_data_pipeline()
            
            # The output root is where train/val/test folders are saved
            return di_config['output_root']
            
        except Exception as e:
            raise CustomException(e, sys)

    def start_data_transformation(self, output_root):
        try:
            logging.info("Starting Data Transformation Phase")
            
            train_dir = os.path.join(output_root, 'train')
            val_dir = os.path.join(output_root, 'val')
            test_dir = os.path.join(output_root, 'test')
            
            data_transformation = DataTransformation(
                train_dir=train_dir,
                val_dir=val_dir,
                test_dir=test_dir,
                config_path=self.config_path
            )
            
            train_loader, val_loader, test_loader = data_transformation.initiate_data_transformation()
            
            return train_loader, val_loader, test_loader
            
        except Exception as e:
            raise CustomException(e, sys)

    def start_model_training(self, train_loader, val_loader):
        try:
            logging.info("Starting Model Training Phase")
            
            model_trainer = ModelTrainer(config_path=self.config_path)
            
            # We pass the val_loader as the test_loader parameter for validation during training
            results = model_trainer.initiate_model_trainer(train_loader, val_loader)
            
            return results
            
        except Exception as e:
            raise CustomException(e, sys)

    def run_pipeline(self):
        try:
            logging.info("STARTING FULL TRAINING PIPELINE")

            
            # 1. Data Ingestion
            output_root = self.start_data_ingestion()
            
            # 2. Data Transformation
            train_loader, val_loader, test_loader = self.start_data_transformation(output_root)
            
            # 3. Model Training
            results = self.start_model_training(train_loader, val_loader)

            logging.info("TRAINING PIPELINE COMPLETED!")
            
            return results
            
        except Exception as e:
            logging.error("Pipeline failed!")
            raise CustomException(e, sys)

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()
