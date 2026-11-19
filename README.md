# Mental Health Diagnostic System

An experimental Machine Learning and Deep Learning-based text classification system that explores the identification of mental-health-related patterns in textual data.

## Overview

This project explores how Natural Language Processing (NLP) and machine learning techniques can be used to classify text into different categories.

It includes data generation, text preprocessing, model training, and evaluation using traditional machine learning and deep learning approaches.

## Features

- Synthetic text dataset generation for experimentation.
- Text preprocessing and feature extraction using TF-IDF.
- Machine learning models: Random Forest and Linear SVC.
- Deep learning model: LSTM.
- Evaluation using accuracy, precision, recall, and F1-score.
- Results saved in CSV format.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- TensorFlow / Keras
- Natural Language Processing (NLP)
- TF-IDF Vectorization

## Project Structure

```text
Mental-Health-Diagnostic-System/
│
├── configs/
├── data/
│   ├── raw/
│   └── processed/
├── logs/
├── models/
├── notebooks/
├── results/
├── src/
│   ├── api/
│   ├── data/
│   ├── evaluation/
│   ├── experiments/
│   ├── features/
│   ├── models/
│   ├── training/
│   └── utils/
│
├── tests/
├── main.py
├── train.py
├── evaluate.py
├── requirements.txt
└── README.md
```

## Installation and Setup

1. Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

2. Navigate to the project directory:

```bash
cd mental-health-diagnostic-system
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Generate the experimental datasets:

```bash
python main.py --generate-data
```

Train and evaluate the models:

```bash
python main.py --train-models
```

You can also run both operations:

```bash
python main.py --generate-data --train-models
```

## Model Evaluation

The system evaluates models using:

- Accuracy
- Precision
- Recall
- F1-score

Evaluation results are saved in:

```text
results/tables/final_results.csv
```

## Important Disclaimer

This project is an academic and experimental prototype developed for learning and exploring machine learning techniques in text classification.

It is not a clinically validated diagnostic tool and should not be used for medical diagnosis, treatment decisions, or mental-health assessments.

## Future Improvements

- Integration of larger, validated datasets.
- Improved text preprocessing and feature engineering.
- Hyperparameter tuning and model optimization.
- Development of a web-based interface.
- Further evaluation on real-world, ethically sourced datasets.
