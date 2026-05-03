import sys
import zipfile
import os
import shutil
import random
import pandas as pd
from PIL import Image
from src.logger import logging
from src.exception import CustomException 
import torch
from torchvision import datasets
from torch.utils.data import random_split
from tqdm import tqdm

class DataIngestion:
    def __init__(self, zip_path, extract_path, data_dir, output_root):
        self.zip_path = zip_path
        self.extract_path = extract_path
        self.data_dir = data_dir
        self.output_root = output_root
        
        # Placeholders for datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def _extract_zip(self):
        logging.info("Data Ingestion started")
        try:
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.extract_path)
            logging.info(f"Extracted to {os.path.abspath(self.extract_path)}")
        except Exception as e:
            raise CustomException(e, sys)
    
    def _split_data(self):
        logging.info("Data split started")
        try:
            full_dataset = datasets.ImageFolder(root=self.data_dir, transform=None)
            train_size = int(0.7 * len(full_dataset))
            val_size = int(0.15 * len(full_dataset))
            test_size = len(full_dataset) - train_size - val_size
            
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                full_dataset, [train_size, val_size, test_size]
            )
            
            logging.info(f"Split completed: Train({len(self.train_dataset)}), Val({len(self.val_dataset)}), Test({len(self.test_dataset)})")
        except Exception as e:
            raise CustomException(e, sys)
        
    def _integrity_check(self):       
        logging.info("Starting integrity check...")
        try:
            dims = []
            corrupt_files = []
            all_images = []
            
            for root, _, files in os.walk(self.data_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        all_images.append(os.path.join(root, file))

            if not all_images:
                logging.warning("No images found for integrity check.")
                return

            sample_check = random.sample(all_images, min(500, len(all_images)))
            for img_path in tqdm(sample_check, desc="Checking integrity"):
                try:
                    with Image.open(img_path) as img:
                        dims.append(img.size)
                except Exception:
                    corrupt_files.append(img_path)
            
            logging.info(f"Corrupt files found: {len(corrupt_files)}")
            if dims:
                dim_df = pd.DataFrame(dims, columns=['Width', 'Height'])
                logging.info(f"Image stats:\n{dim_df.describe()}")
            
            if corrupt_files:
                logging.warning(f"Corrupt files: {corrupt_files}")
        except Exception as e:
            raise CustomException(e, sys)

    def _create_split_dirs(self, dataset, split_name):
        logging.info(f"Organizing {split_name} set...")
        try:
            for i in tqdm(range(len(dataset)), desc=f"Creating {split_name} set"):
                path, label_idx = dataset.dataset.samples[dataset.indices[i]]
                class_name = dataset.dataset.classes[label_idx]
                
                target_dir = os.path.join(self.output_root, split_name, class_name)
                os.makedirs(target_dir, exist_ok=True)
                shutil.copy(path, os.path.join(target_dir, os.path.basename(path)))
        except Exception as e:
            raise CustomException(e, sys)

    def _delete_raw_data(self):
        try:
            if os.path.exists(self.data_dir):
                shutil.rmtree(self.data_dir)
                logging.info(f"Successfully deleted raw data: {self.data_dir}")
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_pipeline(self):
        try:
            # Check if data is already processed
            processed_checks = [
                os.path.join(self.output_root, 'train'),
                os.path.join(self.output_root, 'val'),
                os.path.join(self.output_root, 'test')
            ]
            
            if all(os.path.exists(path) for path in processed_checks):
                logging.info(f"Processed data already exists at {self.output_root}. Skipping ingestion.")
                return

            self._extract_zip()
            self._integrity_check()
            self._split_data()
            
            if self.train_dataset and self.val_dataset and self.test_dataset:
                self._create_split_dirs(dataset=self.train_dataset, split_name='train')
                self._create_split_dirs(dataset=self.val_dataset, split_name='val')
                self._create_split_dirs(dataset=self.test_dataset, split_name='test')
                
            self._delete_raw_data()
            logging.info(f"Pipeline complete. Organized data at: {os.path.abspath(self.output_root)}")
        except Exception as e:
            raise CustomException(e, sys)