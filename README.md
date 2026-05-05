# 🎭 Facial Emotion Recognition

### 🛠️ Software & Tools Requirements

1. [GitHub Account](https://github.com)
2. [VS Code](https://code.visualstudio.com)
3. [Python 3.10+](https://www.python.org)
4. [CUDA-compatible GPU](https://developer.nvidia.com/cuda-downloads) *(recommended)*

---

## 📌 Overview

This project classifies **human facial emotions** from images into 7 categories — **Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise** — using a fine-tuned **ResNet-50** deep learning model combined with a **Random Forest** classifier for ensemble-based prediction.

The project follows a **modular ML pipeline architecture** covering data ingestion, transformation, model training, evaluation, and a **live Flask web interface** for real-time prediction.

---

## 🚀 Project Objectives

- Classify facial expressions into 7 emotion categories
- Build a robust, modular ML pipeline with checkpointing
- Fine-tune a pre-trained ResNet-50 via transfer learning
- Train a Random Forest on CNN-extracted features for ensemble prediction
- Deploy a web interface for real-time emotion detection

---

## 🧠 Problem Statement

Given a face image, predict the **expressed human emotion** from:
`angry` · `disgust` · `fear` · `happy` · `neutral` · `sad` · `surprise`

---

## 🗂️ Project Structure

```
Facial Emotion Detection/
│
├── artifacts/                        # Generated files
│   ├── archive.zip                   # Raw dataset zip
│   ├── final_dataset/                # Processed train/val/test splits
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── best_model_checkpoint.pth     # Best CNN weights
│   ├── checkpoint.pth                # Full training checkpoint (resume support)
│   ├── rf_model.joblib               # Trained Random Forest model
│   ├── confusion_matrix.png          # Evaluation plot
│   └── training_metrics.png          # Loss & accuracy curves
│
├── notebooks/                        # Jupyter notebooks for EDA & experiments
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py         # Unzips & splits dataset
│   │   ├── data_transformation.py    # Augmentation & DataLoaders
│   │   ├── model_trainer.py          # ResNet-50 fine-tuning + RF training
│   │   └── visualization.py          # Plots metrics & confusion matrix
│   │
│   ├── pipeline/
│   │   ├── training_pipeline.py      # End-to-end training orchestration
│   │   └── evaluation_pipeline.py    # Evaluation & custom image testing
│   │
│   ├── exception.py                  # Custom exception handler
│   ├── logger.py                     # Custom logger
│   └── utils.py                      # YAML reader, image predictor, helpers
│
├── templates/
│   └── index.html                    # Flask web UI
│
├── app.py                            # Flask web application
├── config.yml                        # Project configuration
├── requirements.txt
├── setup.py
└── README.md
```

---

## ⚙️ Tech Stack

| Category | Tools / Libraries |
|---|---|
| **Deep Learning** | PyTorch, TorchVision, PyTorch Ignite, TorchMetrics |
| **Machine Learning** | Scikit-learn (Random Forest) |
| **Data Processing** | NumPy, Pandas, Pillow |
| **Visualization** | Matplotlib, Seaborn |
| **Web Framework** | Flask |
| **Config & Logging** | PyYAML, Python logging |
| **Serialization** | Joblib, Pickle |
| **GPU Acceleration** | CUDA 12.4 (PyTorch cu124) |

---

## 🔄 ML Pipeline Workflow

### 1. Data Ingestion
- Extracts raw zip dataset
- Splits images into `train / val / test` directories
- Saves all splits under `artifacts/final_dataset/`

### 2. Data Transformation
- Applies augmentations (RandomCrop, Flip, Rotation, ColorJitter, GaussianBlur)
- Normalizes using ImageNet mean & std
- Converts all images to grayscale (3-channel) to match training distribution
- Returns PyTorch `DataLoader` objects

### 3. Model Training
- Loads **ResNet-50** with pre-trained ImageNet weights
- Freezes early layers; fine-tunes **layer3**, **layer4**, and **fc head**
- Adds **Dropout (0.4)** for regularization
- Uses **AdamW** optimizer with differential learning rates
- Uses **OneCycleLR** scheduler per batch
- Implements **Early Stopping** via PyTorch Ignite
- Supports **checkpoint resumption** (`config.yml → resume: true`)
- Saves best model based on lowest validation loss

### 4. Random Forest (Ensemble)
- Extracts CNN feature vectors from the best ResNet-50 model
- Trains a `RandomForestClassifier(n_estimators=100)` on those features
- Saved as `artifacts/rf_model.joblib`

### 5. Evaluation
- Plots training/validation loss & accuracy curves
- Generates confusion matrix
- Prints full classification report
- Supports single custom image inference

### 6. Web Interface
- Drag & drop image upload
- Real-time CNN + RF prediction
- Animated probability bars for all 7 emotions
- Emotion emoji, confidence %, and RF secondary label

---

## 📊 Emotion Classes

| Class | Label |
|---|---|
| 😡 | angry |
| 🤢 | disgust |
| 😨 | fear |
| 😄 | happy |
| 😐 | neutral |
| 😢 | sad |
| 😲 | surprise |

---

## 🧪 How to Run the Project

### Step 1: Clone the repository

```bash
git clone https://github.com/program-nayan/FacialEmotionRecognition.git
cd FacialEmotionRecognition
```

### Step 2: Create & activate virtual environment

```bash
python -m venv myenv
myenv\Scripts\activate      # Windows
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

###  Step 4: Download  the data in artifacts directory

Download the data from kaggle (https://www.kaggle.com/datasets/fahadullaha/facial-emotion-recognition-dataset) in artifacts folder

### Step 5: Run the training pipeline

```bash
python -m src.pipeline.training_pipeline
```

### Step 6: Run the evaluation pipeline

```bash
python -m src.pipeline.evaluation_pipeline
```

### Step 7: Launch the web interface

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

---

## 🌐 Web Interface Preview

The web app provides a clean dark-mode UI where you can:
- Drag & drop or browse a face image
- Click **Detect Emotion** to get the prediction
- View CNN confidence score + Random Forest secondary prediction
- See animated probability bars for all 7 emotions

---

## 📈 Generated Artifacts

After running the full pipeline:

```
artifacts/
 ├── final_dataset/train/       # Training images (per class)
 ├── final_dataset/val/         # Validation images
 ├── final_dataset/test/        # Test images
 ├── best_model_checkpoint.pth  # Best CNN weights
 ├── checkpoint.pth             # Full resumable checkpoint
 ├── rf_model.joblib            # Random Forest model
 ├── training_metrics.png       # Loss & accuracy plot
 └── confusion_matrix.png       # Confusion matrix
```

---

## ⚠️ Common Issues & Fixes

| Issue | Solution |
|---|---|
| `ModuleNotFoundError` | Run scripts using `python -m src.pipeline.training_pipeline` |
| `FileNotFoundError` on dataset | Ensure `archive.zip` is in the `artifacts/` folder |
| CUDA not available | Install PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu124` |
| Port 5000 already in use | Change port in `app.py`: `app.run(port=5001)` |
| Model checkpoint not found | Run the training pipeline first |

---

## ⚙️ Configuration (`config.yml`)

```yaml
data_ingestion:
  zip_path: "artifacts/archive.zip"
  extract_path: "artifacts/"
  output_root: "artifacts/final_dataset"

data_transformation:
  batch_size: 32
  img_size: 224
  num_workers: 4

model_trainer:
  epochs: 20
  learning_rate: 0.0005
  weight_decay: 0.001
  model_name: "resnet50"
  patience: 3
  resume: false        # Set to true to resume from checkpoint
```

---

## 📌 Key Highlights

- ✅ Transfer learning with ResNet-50 (layer3 + layer4 + fc fine-tuned)
- ✅ Ensemble: CNN + Random Forest
- ✅ Checkpoint resumption support
- ✅ Early stopping via PyTorch Ignite
- ✅ Production-grade modular architecture
- ✅ Custom logging & exception handling
- ✅ Live Flask web interface with drag & drop

---

## 📚 Future Improvements

- Real-time webcam feed prediction
- Docker containerization
- CI/CD integration with GitHub Actions
- REST API with FastAPI
- Model monitoring & drift detection
- Hyperparameter tuning with Optuna

---

## 👨‍💻 Author

**Nayan Badgujar**

---

## 🤝 Collaborators

- Chayan Sinam
- Abhinav Pandey
- Aryan Singh
- Gaurav Mali

Developed for learning purposes at **Chhatrapati Shivaji Maharaj University**.

---

## ⭐ Acknowledgements

- [PyTorch](https://pytorch.org) & [TorchVision](https://pytorch.org/vision)
- [PyTorch Ignite](https://pytorch-ignite.ai) for Early Stopping
- [Scikit-learn](https://scikit-learn.org) for Random Forest
- Kaggle FER dataset inspiration

---

## 📬 Contact

Feel free to connect for collaboration or queries.
