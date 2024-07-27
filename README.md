# SurvivalEdge-Adaptive-Titanic-Prediction-System
SurvivalEdge offers a robust and adaptive system for predicting Titanic survival rates. By leveraging state-of-the-art algorithms and dynamic feature engineering, this tool provides accurate and insightful predictions.

Here's a GitHub README file for your Titanic Survival Prediction project:

---

# Titanic Survival Prediction

This project aims to predict the survival of passengers on the Titanic using various machine learning models, including Logistic Regression, Decision Tree, Random Forest, and XGBoost. Additionally, a deep learning model is implemented for further analysis. The project also includes clustering analysis, advanced visualizations, and a detailed GUI for predicting the survival of passengers based on their IDs.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Data Preprocessing](#data-preprocessing)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Deep Learning Model](#deep-learning-model)
- [Clustering Analysis](#clustering-analysis)
- [Advanced Visualizations](#advanced-visualizations)
- [GUI for Detailed Predictions](#gui-for-detailed-predictions)
- [Usage](#usage)
- [Results](#results)
- [Files](#files)

## Introduction

The Titanic Survival Prediction project uses machine learning and deep learning techniques to predict whether a passenger survived the Titanic disaster. The project also provides detailed predictions through a GUI based on passenger IDs.

## Requirements

- Python 3.6+
- pandas
- scikit-learn
- xgboost
- torch
- matplotlib
- seaborn
- tkinter
- joblib

Install the required packages using the following command:

```bash
pip install pandas scikit-learn xgboost torch matplotlib seaborn tkinter joblib
```

## Data Preprocessing

The dataset is loaded and preprocessed to fill missing values, convert categorical features, and engineer new features. The features are then scaled for model training.

```python
def load_and_preprocess_data(file_path):
    ...
```

## Model Training and Evaluation

Four machine learning models are trained and evaluated using GridSearchCV for hyperparameter tuning. The results are saved, and the best models are stored using `joblib`.

```python
def run_models(X_train, y_train, X_test, y_test):
    ...
```

## Deep Learning Model

A deep learning model is implemented using PyTorch to predict the survival of passengers.

```python
class TitanicNN(nn.Module):
    ...
```

## Clustering Analysis

KMeans clustering is performed on the scaled features, and the results are visualized using PCA.

```python
def clustering_analysis(X_scaled):
    ...
```

## Advanced Visualizations

Advanced visualizations, such as age and fare distribution by survival and survival rate by class, are created using seaborn and matplotlib.

```python
def advanced_visualizations(df):
    ...
```

## GUI for Detailed Predictions

A GUI is created using Tkinter to provide detailed predictions for passengers based on their IDs.

```python
def create_detailed_gui(models, scaler):
    ...
```

## Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/titanic-survival-prediction.git
    cd titanic-survival-prediction
    ```

2. Place the Titanic dataset (`titanic.csv.xlsx`) in the appropriate directory.

3. Run the main script:
    ```bash
    python main.py
    ```

## Results

The project provides the following results:
- Accuracy, Precision, Recall, F1-Score, and ROC AUC for each model.
- Detailed predictions for each passenger based on their ID.
- Clustering analysis and advanced visualizations.

## Files

- `main.py`: Main script to run the project.
- `titanic.csv.xlsx`: Titanic dataset.
- `detailed_predictions.txt`: File to store detailed predictions.
- `clustering_analysis.png`: Clustering analysis plot.
- `age_distribution.png`: Age distribution plot.
- `fare_distribution.png`: Fare distribution plot.
- `survival_rate_by_class.png`: Survival rate by class plot.

---

Feel free to customize the README as per your specific requirements or preferences.
