
# Neural Financial Risk Prediction System

An end-to-end financial risk prediction system comparing
conventional machine learning with deep learning using
PyTorch.

## Business Problem

Can a neural network provide meaningful improvements over
Logistic Regression for financial fraud-risk prediction?

This project evaluates predictive performance, training
efficiency, inference latency, calibration, generalization,
and operational trade-offs.

## Models

- Logistic Regression (baseline)
- PyTorch Feedforward Neural Network

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- PyTorch
- Matplotlib and Seaborn
- Plotly and Streamlit
- Pytest

## Key Engineering Features

- Temporal data splitting
- Leakage-aware feature engineering
- Reproducible preprocessing
- Neural network training and checkpointing
- Threshold optimization
- Probability calibration analysis
- Model performance benchmarking
- Automated testing
- Interactive experiment dashboard

## Project Structure

- data/: Dataset storage
- src/: ML engineering source code
- models/: Model artifacts
- configs/: Experiment configurations
- tests/: Automated tests
- reports/: Evaluation reports
- dashboard/: Interactive dashboard
- docs/: Technical documentation

## Installation

Clone the repository:

    git clone YOUR_GITHUB_REPOSITORY_URL

Navigate into the project:

    cd neural-financial-risk-prediction

Create virtual environment:

    py -3.12 -m venv .venv

Activate:

    .venv\Scripts\Activate.ps1

Install dependencies:

    pip install -r requirements.txt

## Reproducibility

All experiments will use controlled random seeds,
documented configurations, and temporal evaluation.

## Dataset

PaySim Synthetic Financial Dataset.

Dataset attribution and licensing information will
be included in the dataset documentation.

## Results

Results will be added after model training and
evaluation.

## Limitations

The dataset is synthetic and does not fully represent
real-world banking fraud patterns.

## Author

Software Engineering Graduate | Machine Learning |
Financial Data Science
