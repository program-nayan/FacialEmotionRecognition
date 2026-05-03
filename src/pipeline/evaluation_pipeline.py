import sys
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import classification_report
import joblib
from pathlib import Path

from src.components.model_trainer import ModelTrainer
from src.components.data_transformation import DataTransformation
from src.components.visualization import Visualizer
from src.logger import logging
from src.exception import CustomException
from src.utils import read_yaml, predict_custom_image
from torchvision import transforms

class ModelEvaluation:
    """
    Independent pipeline for evaluating the trained model, generating reports,
    plotting metrics, and testing on custom images.
    """
    def __init__(self, config_path="config.yml"):
        try:
            self.config_path = config_path
            self.config = read_yaml(config_path)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            self.best_model_path = os.path.join("artifacts", "best_model_checkpoint.pth")
            self.checkpoint_path = os.path.join("artifacts", "checkpoint.pth")
            self.rf_model_path = os.path.join("artifacts", "rf_model.joblib")
            
            logging.info(f"ModelEvaluation initialized. Device: {self.device}")
        except Exception as e:
            raise CustomException(e, sys)

    def _load_best_model(self, num_classes):
        """Loads the best model state dict into the architecture."""
        try:
            trainer = ModelTrainer(config_path=self.config_path)
            model = trainer._get_model(num_classes)
            
            if os.path.exists(self.best_model_path):
                model.load_state_dict(torch.load(self.best_model_path, map_location=self.device))
                logging.info(f"Successfully loaded best model from {self.best_model_path}")
            else:
                logging.warning(f"Best model checkpoint not found at {self.best_model_path}")
                
            return model
        except Exception as e:
            raise CustomException(e, sys)

    def run_full_evaluation(self):
        """Runs the complete evaluation process: metrics plotting and test data assessment."""
        try:
            logging.info("Starting Full Evaluation Process")
            
            # 1. Plot Training & Test Metrics from history
            if os.path.exists(self.checkpoint_path):
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
                results = checkpoint.get('results', None)
                if results:
                    Visualizer.plot_metrics(results)
                    logging.info("Training history plots generated.")
            
            # 2. Evaluate on Test Set
            output_root = self.config['data_ingestion']['output_root']
            data_transformation = DataTransformation(
                train_dir=os.path.join(output_root, 'train'),
                val_dir=os.path.join(output_root, 'val'),
                test_dir=os.path.join(output_root, 'test'),
                config_path=self.config_path
            )
            # We only need the test loader
            _, _, test_loader = data_transformation.initiate_data_transformation()
            
            # Explicitly add Grayscale transformation as requested for evaluation
            test_loader.dataset.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                test_loader.dataset.transform
            ])
            
            num_classes = len(test_loader.dataset.classes)
            class_names = test_loader.dataset.classes
            model = self._load_best_model(num_classes)
            model.eval()
            
            all_preds = []
            all_labels = []
            
            with torch.inference_mode():
                for images, labels in tqdm(test_loader, desc="Evaluating on Test Set"):
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    preds = torch.argmax(outputs, dim=1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            # Plot Confusion Matrix
            Visualizer.plot_confusion_matrix(all_labels, all_preds, classes=class_names)
            
            # Print Classification Report
            print("\nClassification Report:")
            print(classification_report(all_labels, all_preds, target_names=class_names))
            
            logging.info("Full evaluation completed successfully.")
            
        except Exception as e:
            raise CustomException(e, sys)

    def test_custom_image(self, image_path):
        """Predicts the emotion of a single custom image."""
        try:
            logging.info(f"Testing custom image: {image_path}")
            
            # Need loaders to get class names and transform
            output_root = self.config['data_ingestion']['output_root']
            data_transformation = DataTransformation(
                train_dir=os.path.join(output_root, 'train'),
                val_dir=os.path.join(output_root, 'val'),
                test_dir=os.path.join(output_root, 'test'),
                config_path=self.config_path
            )
            _, _, test_loader = data_transformation.initiate_data_transformation()
            class_names = test_loader.dataset.classes
            
            # Get the test transform object
            _, base_transform = data_transformation._get_data_transformer_object()
            
            # Prepend Grayscale transformation for custom image testing
            transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                base_transform
            ])
            
            model = self._load_best_model(len(class_names))
            
            # Load RF model if available
            rf_model = None
            if os.path.exists(self.rf_model_path):
                rf_model = joblib.load(self.rf_model_path)
                
            predict_custom_image(image_path, model, class_names, transform, self.device, rf_model)
            
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    eval_pipeline = ModelEvaluation()
    
    # Run full evaluation by default
    full_eval = input("Do you want to run full evaluation? (yes/no): ")
    if full_eval.lower() == "yes":
        eval_pipeline.run_full_evaluation()
    else:
        pass
    
    # Example: How to test a custom image (uncomment to use)
    while True:
        input_image_path = Path(input("Please enter the image path : "))
        if not input_image_path.exists():
            logging.error(f"Image not found: {input_image_path}")
            continue
        eval_pipeline.test_custom_image(input_image_path)
        
        another = input("Do you want to test another image? (yes/no): ")
        if another.lower() == "no":
            break
