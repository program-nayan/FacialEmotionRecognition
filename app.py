import os
import sys
import io
import base64
import torch
import torch.nn as nn
import joblib
from PIL import Image
from torchvision import models, transforms
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

# ── Project imports ──────────────────────────────────────────────────────────
from src.utils import read_yaml
from src.logger import logging

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB upload limit
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH       = os.path.join(BASE_DIR, "config.yml")
BEST_MODEL_PATH   = os.path.join(BASE_DIR, "artifacts", "best_model_checkpoint.pth")
RF_MODEL_PATH     = os.path.join(BASE_DIR, "artifacts", "rf_model.joblib")
DATASET_TRAIN_DIR = None   # resolved after reading config

# ── Emotion metadata ──────────────────────────────────────────────────────────
# Fallback class names if we cannot read them from the dataset folder
FALLBACK_CLASSES  = ["angry", "fear", "happy", "neutral", "sad"]

EMOTION_EMOJI = {
    "angry":   "😡",
    "anger":   "😡",
    "fear":    "😨",
    "happy":   "😄",
    "happiness":"😄",
    "neutral": "😐",
    "sad":     "😢",
    "sadness": "😢",
    "disgust": "🤢",
    "surprise":"😲",
}

# ── Model globals ─────────────────────────────────────────────────────────────
_model       = None
_rf_model    = None
_transform   = None
_class_names = None
_device      = None


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _build_model(num_classes: int, device):
    """Reconstruct the ResNet50 architecture with the same head used during training."""
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(num_ftrs, num_classes)
    )
    return model.to(device)


def _load_resources():
    """Load model, RF model, class names, and transform once at startup."""
    global _model, _rf_model, _transform, _class_names, _device

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"[app] Using device: {_device}")

    # ── Resolve class names ────────────────────────────────────────────────────
    try:
        config = read_yaml(CONFIG_PATH)
        output_root = config["data_ingestion"]["output_root"]
        train_dir   = os.path.join(BASE_DIR, output_root, "train")
        if os.path.isdir(train_dir):
            _class_names = sorted([
                d for d in os.listdir(train_dir)
                if os.path.isdir(os.path.join(train_dir, d))
            ])
            logging.info(f"[app] Classes from dataset: {_class_names}")
        else:
            raise FileNotFoundError(train_dir)
    except Exception:
        _class_names = FALLBACK_CLASSES
        logging.warning(f"[app] Using fallback class names: {_class_names}")

    num_classes = len(_class_names)

    # ── Load CNN ───────────────────────────────────────────────────────────────
    _model = _build_model(num_classes, _device)
    if os.path.exists(BEST_MODEL_PATH):
        _model.load_state_dict(
            torch.load(BEST_MODEL_PATH, map_location=_device)
        )
        logging.info(f"[app] Best model loaded from {BEST_MODEL_PATH}")
    else:
        logging.warning(f"[app] best_model_checkpoint.pth not found — model weights are random!")
    _model.eval()

    # ── Load Random Forest ─────────────────────────────────────────────────────
    if os.path.exists(RF_MODEL_PATH):
        _rf_model = joblib.load(RF_MODEL_PATH)
        logging.info("[app] RF model loaded.")

    # ── Build inference transform (mirrors test_transform in DataTransformation) ──
    img_size   = config.get("data_transformation", {}).get("img_size", 224)
    _transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    logging.info("[app] Inference transform ready.")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400
    if not _allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Use PNG/JPG/WEBP."}), 400

    try:
        img_bytes = file.read()
        img       = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # ── CNN inference ──────────────────────────────────────────────────────
        tensor = _transform(img).unsqueeze(0).to(_device)

        with torch.inference_mode():
            logits    = _model(tensor)
            probs     = torch.softmax(logits, dim=1).squeeze()
            cnn_idx   = int(torch.argmax(probs).item())
            cnn_conf  = float(probs[cnn_idx].item())
            cnn_label = _class_names[cnn_idx]

        # Build sorted probability list for the bar chart
        all_probs = [
            {"label": _class_names[i], "prob": float(probs[i].item())}
            for i in range(len(_class_names))
        ]
        all_probs.sort(key=lambda x: x["prob"], reverse=True)

        # ── Optional RF inference ──────────────────────────────────────────────
        rf_label = None
        if _rf_model is not None:
            with torch.inference_mode():
                feature_extractor = nn.Sequential(*list(_model.children())[:-1])
                feature_extractor.eval().to(_device)
                feats  = feature_extractor(tensor).view(1, -1).cpu().numpy()
            rf_idx   = int(_rf_model.predict(feats)[0])
            rf_label = _class_names[rf_idx]

        emoji = EMOTION_EMOJI.get(cnn_label.lower(), "🙂")

        return jsonify({
            "cnn_label":  cnn_label,
            "cnn_conf":   round(cnn_conf * 100, 2),
            "rf_label":   rf_label,
            "all_probs":  all_probs,
            "emoji":      emoji,
        })

    except Exception as exc:
        logging.error(f"[app] Prediction error: {exc}")
        return jsonify({"error": str(exc)}), 500


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _load_resources()
    app.run(host="0.0.0.0", port=5000, debug=False)
