import sys
import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from src.logger import logging
from src.exception import CustomException
from src.utils import read_yaml

class DataTransformation:
    """
    Class for handling image transformations and creating DataLoaders using configurations from a YAML file.
    """
    def __init__(self, train_dir, val_dir, test_dir, config_path="config.yml"):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        
        try:
            # Load hyperparameters from config.yml
            self.config_all = read_yaml(config_path)
            self.config = self.config_all['data_transformation']
            
            self.batch_size = self.config.get('batch_size', 32)
            self.img_size = self.config.get('img_size', 224)
            # Force 0 on Windows for stability, else use config value
            self.num_workers = 0 if os.name == 'nt' else self.config.get('num_workers', 4)
            
            logging.info(f"Loaded config: Batch Size={self.batch_size}, Image Size={self.img_size}, Num Workers={self.num_workers}")
        except Exception as e:
            raise CustomException(e, sys)

    def _get_data_transformer_object(self):
        """
        Returns training and testing transformation objects.
        """
        try:
            logging.info("Initializing data transformation objects")
            
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(self.img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            test_transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            return train_transform, test_transform
            
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self):
        """
        Applies transformations to datasets and returns DataLoaders.
        """
        try:
            logging.info("Starting data transformation and loader creation")
            
            train_transform, test_transform = self._get_data_transformer_object()

            # Loading datasets from directories
            train_data = datasets.ImageFolder(self.train_dir, transform=train_transform)
            val_data = datasets.ImageFolder(self.val_dir, transform=test_transform)
            test_data = datasets.ImageFolder(self.test_dir, transform=test_transform)

            logging.info(f"Loaded {len(train_data)} training images.")
            logging.info(f"Loaded {len(val_data)} validation images.")
            logging.info(f"Loaded {len(test_data)} test images.")

            # Creating DataLoaders using config values
            train_loader = DataLoader(
                train_data, 
                batch_size=self.batch_size, 
                shuffle=True, 
                num_workers=self.num_workers
            )
            val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
            test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

            logging.info("DataLoaders created successfully")

            return train_loader, val_loader, test_loader
            
        except Exception as e:
            raise CustomException(e, sys)
