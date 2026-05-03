import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm

from ignite.engine import create_supervised_trainer, create_supervised_evaluator, Events
from ignite.handlers import EarlyStopping
from ignite.metrics import Accuracy as IgniteAccuracy, Loss
from torchmetrics import Accuracy

from src.logger import logging
from src.exception import CustomException
from src.utils import read_yaml

class ModelTrainer:
    """
    Class for handling the training and validation of the facial emotion detection model,
    integrating PyTorch Ignite for Early Stopping and TorchMetrics for evaluation.
    """
    def __init__(self, config_path="config.yml"):
        try:
            self.config_all = read_yaml(config_path)
            self.config = self.config_all['model_trainer']
            
            self.epochs = self.config.get('epochs', 15)
            self.lr = self.config.get('learning_rate', 1e-3)
            self.weight_decay = self.config.get('weight_decay', 1e-2)
            self.model_name = self.config.get('model_name', 'resnet50')
            self.patience = self.config.get('patience', 3)
            self.resume = self.config.get('resume', False)
            self.checkpoint_path = os.path.join("artifacts", "checkpoint.pth")
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logging.info(f"ModelTrainer initialized. Device: {self.device} | Resume: {self.resume}")
        except Exception as e:
            raise CustomException(e, sys)

    def _get_model(self, num_classes):
        """
        Initializes the model architecture, freezing all layers except layer4 and fc for fine-tuning.
        """
        try:
            logging.info(f"Initializing {self.model_name} for {num_classes} classes")
            
            if self.model_name == 'resnet50':
                model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            elif self.model_name == 'resnet18':
                model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            else:
                logging.warning(f"Model {self.model_name} not recognized. Falling back to ResNet50.")
                model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

            # Replace the final fully connected layer with Dropout for regularization
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(p=0.4),
                nn.Linear(num_ftrs, num_classes)
            )
            
            # Unfreeze specific layers (layer3, layer4 and fc)
            for name, param in model.named_parameters():
                if "layer4" in name or "layer3" in name or "fc" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            
            logging.info("Model prepared with layer3, layer4 and fc unfrozen + Dropout for fine-tuning.")
            return model.to(self.device)
        except Exception as e:
            raise CustomException(e, sys)

    def _train_step(self, model, dataloader, loss_fn, optimizer, scheduler, accuracy_fn, position=None):
        model.train()
        train_loss = 0
        accuracy_fn.reset()
        for X, y in tqdm(dataloader, desc="Training", leave=False, position=position):
            X, y = X.to(self.device), y.to(self.device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Step the OneCycleLR scheduler after every batch
            scheduler.step()
            
            accuracy_fn.update(y_pred.argmax(dim=1), y)
        train_acc = accuracy_fn.compute().item()
        return train_loss / len(dataloader), train_acc

    def _test_step(self, model, dataloader, loss_fn, accuracy_fn, position=None):
        model.eval()
        test_loss = 0
        accuracy_fn.reset()
        with torch.inference_mode():
            for X, y in tqdm(dataloader, desc="Testing", leave=False, position=position):
                X, y = X.to(self.device), y.to(self.device)
                test_pred = model(X)
                test_loss += loss_fn(test_pred, y).item()
                accuracy_fn.update(test_pred.argmax(dim=1), y)
        test_acc = accuracy_fn.compute().item()
        return test_loss / len(dataloader), test_acc

    def initiate_model_trainer(self, train_loader, test_loader):
        """
        Runs the training loop with Early Stopping, OneCycleLR, and saves the best model.
        """
        try:
            num_classes = len(train_loader.dataset.classes)
            model = self._get_model(num_classes)
            
            loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
            
            # Optimizer with Differential Learning Rates
            # Backbone layers (layer3, layer4) get a lower LR, while the FC head gets the full LR.
            optimizer = optim.AdamW([
                {'params': model.layer3.parameters(), 'lr': self.lr / 10},
                {'params': model.layer4.parameters(), 'lr': self.lr / 10},
                {'params': model.fc.parameters(), 'lr': self.lr}
            ], weight_decay=self.weight_decay)
            
            # OneCycleLR Scheduler
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, 
                max_lr=self.lr,
                steps_per_epoch=len(train_loader),
                epochs=self.epochs
            )

            accuracy_fn = Accuracy(task="multiclass", num_classes=num_classes).to(self.device)
            
            best_test_loss = float("inf")
            start_epoch = 0
            results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

            # Resume from checkpoint if requested
            if self.resume and os.path.exists(self.checkpoint_path):
                logging.info(f"Resuming from checkpoint: {self.checkpoint_path}")
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                best_test_loss = checkpoint['best_test_loss']
                results = checkpoint.get('results', results)
                logging.info(f"Resumed from epoch {start_epoch} with best test loss {best_test_loss:.4f}")
            
            # Setup Ignite objects for EarlyStopping
            trainer = create_supervised_trainer(model, optimizer, loss_fn, device=self.device)
            evaluator = create_supervised_evaluator(model, metrics={"accuracy": IgniteAccuracy(), "loss": Loss(loss_fn)}, device=self.device)

            def score_function(engine):
                return -engine.state.metrics["loss"]

            handler = EarlyStopping(patience=self.patience, score_function=score_function, trainer=trainer)
            evaluator.add_event_handler(Events.COMPLETED, handler)

            logging.info(f"Starting training loop from epoch {start_epoch+1}")
            for epoch in tqdm(range(start_epoch, self.epochs), desc="Epochs", position=0, leave=True):
                train_loss, train_acc = self._train_step(model, train_loader, loss_fn, optimizer, scheduler, accuracy_fn, position=1)
                test_loss, test_acc = self._test_step(model, test_loader, loss_fn, accuracy_fn, position=1)

                msg = f"Epoch: {epoch+1} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}"
                print(msg)
                logging.info(msg)
                
                results["train_loss"].append(train_loss)
                results["train_acc"].append(train_acc)
                results["test_loss"].append(test_loss)
                results["test_acc"].append(test_acc)

                os.makedirs("artifacts", exist_ok=True)
                
                # Save Checkpoint based on best test loss
                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    best_model_path = os.path.join("artifacts", "best_model_checkpoint.pth")
                    torch.save(model.state_dict(), best_model_path)
                    msg_cp = f"New best test loss {best_test_loss:.4f}. Best model saved at {best_model_path}"
                    print(msg_cp)
                    logging.info(msg_cp)

                # Save general checkpoint for resuming
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_test_loss': best_test_loss,
                    'results': results
                }
                torch.save(checkpoint, self.checkpoint_path)
                logging.info(f"Checkpoint saved at epoch {epoch+1} to {self.checkpoint_path}")

                # Use Ignite evaluator to check for early stopping trigger
                evaluator.run(test_loader)
                if trainer.should_terminate:
                    msg_es = "Early stopping triggered."
                    print(msg_es)
                    logging.info(msg_es)
                    break

            logging.info("Training complete")
            
            # Load best model weights for feature extraction
            best_model_path = os.path.join("artifacts", "best_model_checkpoint.pth")
            if os.path.exists(best_model_path):
                model.load_state_dict(torch.load(best_model_path))
                
            # Train Random Forest
            self._train_random_forest(model, train_loader)
            
            return results
            
        except Exception as e:
            raise CustomException(e, sys)

    def _train_random_forest(self, model, dataloader):
        """
        Trains a Random Forest classifier using features extracted by the trained CNN.
        """
        try:
            import numpy as np
            import joblib
            from sklearn.ensemble import RandomForestClassifier
            
            logging.info("Starting Random Forest training on extracted features...")
            
            # Create feature extractor (all layers except the last fc)
            feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
            feature_extractor.eval()
            feature_extractor = feature_extractor.to(self.device)

            X_train, y_train = [], []
            with torch.inference_mode():
                for images, labels in tqdm(dataloader, desc="Extracting features for RF", leave=False):
                    images = images.to(self.device)
                    features = feature_extractor(images).view(images.size(0), -1).cpu().numpy()
                    X_train.append(features)
                    y_train.append(labels.numpy())

            X_train = np.vstack(X_train)
            y_train = np.concatenate(y_train)

            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)

            rf_model_path = os.path.join("artifacts", "rf_model.joblib")
            joblib.dump(rf_model, rf_model_path)
            logging.info(f"Random Forest model trained and saved at {rf_model_path}")
            
        except Exception as e:
            raise CustomException(e, sys)
