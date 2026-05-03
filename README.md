# Facial Emotion Detection

A deep learning project to classify human facial emotions (happiness, anger, fear, sadness, neutrality, etc.) using image data.

## Project Structure

This repository follows a structured ML project layout containing:
- `artifacts/`: Generated datasets and model objects.
- `notebooks/`: Jupyter notebooks for EDA and experiments.
- `src/`: Core pipeline and components (data ingestion, transformation, model trainer).
- `processed_data/`: Image data for facial emotions.

## How to run

1. Install requirements:
`pip install -r requirements.txt`

2. Run the pipeline:
`python -m src.pipeline.training_pipeline`
