import yaml
import sys
import os
import torch
import matplotlib.pyplot as plt
from PIL import Image

from src.exception import CustomException
from src.logger import logging

def read_yaml(file_path: str) -> dict:
    """
    Reads a YAML file and returns the content as a dictionary.
    """
    try:
        with open(file_path, "rb") as yaml_file:
            content = yaml.safe_load(yaml_file)
            logging.info(f"YAML file: {file_path} loaded successfully")
            return content
    except Exception as e:
        raise CustomException(e, sys)

def save_object(file_path, obj):
    """
    Saves a python object to a file.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            import pickle
            pickle.dump(obj, file_obj)
        logging.info(f"Object saved at: {file_path}")
    except Exception as e:
        raise CustomException(e, sys)

def predict_custom_image(image_path, model, class_names, transform, device, rf_model=None):
    """
    Predicts the emotion of a custom image using the CNN (and optionally a Random Forest model)
    and plots the result.
    """
    try:
        logging.info(f"Predicting on custom image: {image_path}")
        # Ensure image exists and is loaded
        img = Image.open(image_path).convert('RGB')
        transformed_image = transform(img).unsqueeze(0).to(device)

        model.eval()
        with torch.inference_mode():
            # 1. Get CNN Prediction
            logits = model(transformed_image)
            cnn_probs = torch.softmax(logits, dim=1)
            cnn_label = torch.argmax(cnn_probs, dim=1).item()

            # 2. Get Random Forest Prediction if provided
            rf_label_name = "N/A"
            if rf_model is not None:
                # Extract features using the model as a feature extractor
                feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
                features = feature_extractor(transformed_image).view(1, -1).cpu().numpy()
                rf_pred = rf_model.predict(features)[0]
                rf_label_name = class_names[rf_pred]

        # Plot the result
        plt.figure(figsize=(6, 4))
        plt.imshow(img)
        plt.title(f"CNN: {class_names[cnn_label]} ({cnn_probs.max():.2f})\nRF: {rf_label_name}")
        plt.axis('off')
        plt.show()
        
        logging.info("Prediction plotted successfully.")
        
    except Exception as e:
        raise CustomException(e, sys)
