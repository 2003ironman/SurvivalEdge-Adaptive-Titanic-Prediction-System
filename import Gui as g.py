import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import tkinter as tk
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
import numpy as np
import os

# Data loading and preprocessing
def load_and_preprocess_data(file_path):
    if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a .xlsx or .xls file.")

    required_columns = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Survived']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing columns: {', '.join(missing_columns)}")

    # Fill missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # Convert categorical features
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

    # Feature engineering
    df['Family_Size'] = df['SibSp'] + df['Parch']
    df['Is_Alone'] = (df['Family_Size'] == 0).astype(int)

    # Feature and target split
    X = df[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Family_Size', 'Is_Alone']]
    y = df['Survived']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return df, X_scaled, y, scaler

# Model training and evaluation
def run_models(X_train, y_train, X_test, y_test):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'XGBoost': XGBClassifier()
    }

    # Hyperparameters for tuning
    param_grids = {
        'Logistic Regression': {'C': [0.1, 1, 10]},
        'Decision Tree': {'max_depth': [3, 5, 7]},
        'Random Forest': {'n_estimators': [50, 100, 200]},
        'XGBoost': {'learning_rate': [0.01, 0.1, 0.2]}
    }

    results = {}
    for name, model in models.items():
        grid_search = GridSearchCV(model, param_grid=param_grids[name], cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test)
        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'ROC AUC': roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
        }

        joblib.dump(best_model, f'{name.lower().replace(" ", "_")}_model.pkl')

    return results

# Deep learning model
class TitanicNN(nn.Module):
    def __init__(self, input_size):
        super(TitanicNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, 25)
        self.fc3 = nn.Linear(25, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

def deep_learning_model(X_train, y_train, X_test, y_test):
    X_train_torch = torch.tensor(X_train, dtype=torch.float32)
    y_train_torch = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_test_torch = torch.tensor(X_test, dtype=torch.float32)
    y_test_torch = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

    model = TitanicNN(X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_torch)
        loss = criterion(outputs, y_train_torch)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        outputs = model(X_test_torch)
        predictions = (outputs > 0.5).float()
        accuracy_dl = (predictions.eq(y_test_torch).sum() / y_test_torch.shape[0]).item()
    
    torch.save(model.state_dict(), 'deep_learning_model.pth')
    
    return accuracy_dl

# Clustering analysis
def clustering_analysis(X_scaled):
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X_scaled)
    labels = kmeans.labels_

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='viridis')
    plt.title('KMeans Clustering with PCA')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(title='Cluster')
    plt.grid(True)
    plt.savefig('clustering_analysis.png')
    plt.show()

# Advanced visualizations
def advanced_visualizations(df):
    plt.figure(figsize=(12, 6))
    sns.histplot(df[df['Survived'] == 1]['Age'], kde=False, bins=30, color='green', label='Survived')
    sns.histplot(df[df['Survived'] == 0]['Age'], kde=False, bins=30, color='red', label='Not Survived')
    plt.title('Age Distribution by Survival')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    plt.savefig('age_distribution.png')
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.histplot(df[df['Survived'] == 1]['Fare'], kde=False, bins=30, color='green', label='Survived')
    sns.histplot(df[df['Survived'] == 0]['Fare'], kde=False, bins=30, color='red', label='Not Survived')
    plt.title('Fare Distribution by Survival')
    plt.xlabel('Fare')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    plt.savefig('fare_distribution.png')
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.barplot(x='Pclass', y='Survived', data=df, palette='viridis', estimator='mean')
    plt.title('Survival Rate by Class')
    plt.xlabel('Pclass')
    plt.ylabel('Survival Rate')
    plt.grid(True)
    plt.savefig('survival_rate_by_class.png')
    plt.show()

# Prediction function
def predict_survival_from_excel(passenger_id, df, models, scaler):
    passenger_data = df[df['PassengerId'] == passenger_id]
    if passenger_data.empty:
        return "Passenger ID not found.", None

    passenger_data = passenger_data[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Family_Size', 'Is_Alone']]
    X_passenger = scaler.transform(passenger_data)
    
    results = {}
    for model_name, model in models.items():
        prediction = model.predict(X_passenger)
        results[model_name] = "Survived" if prediction[0] == 1 else "Not Survived"
    
    return results, passenger_data

# GUI for detailed predictions
def create_detailed_gui(models, scaler):
    def predict():
        passenger_id = entry_passenger_id.get()
        try:
            passenger_id = int(passenger_id)
        except ValueError:
            result_label.config(text="Invalid Passenger ID. Please enter a number.")
            return
        
        results, passenger_data = predict_survival_from_excel(passenger_id, df, models, scaler)
        if passenger_data is None:
            result_label.config(text="Passenger ID not found.")
        else:
            result_text = f"Detailed Predictions for Passenger ID {passenger_id}:\n"
            for model_name, prediction in results.items():
                result_text += f"{model_name}: {prediction}\n"

            # Auto-save results
            with open('detailed_predictions.txt', 'a') as f:
                f.write(result_text + '\n' + '-'*40 + '\n')
            
            result_label.config(text=result_text)
    
    root = tk.Tk()
    root.title("Titanic Survival Detailed Predictions")

    tk.Label(root, text="Enter Passenger ID:").pack(pady=10)
    entry_passenger_id = tk.Entry(root)
    entry_passenger_id.pack(pady=5)
    
    predict_button = tk.Button(root, text="Predict", command=predict)
    predict_button.pack(pady=10)
    
    result_label = tk.Label(root, text="", wraplength=400)
    result_label.pack(pady=10)
    
    root.mainloop()

# Main execution
if __name__ == "__main__":
    file_path = "C:\\Users\\Aditya Singh\\Desktop\\churn detection model\\titanic.csv.xlsx"
    df, X_scaled, y, scaler = load_and_preprocess_data(file_path)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model_results = run_models(X_train, y_train, X_test, y_test)
    deep_learning_accuracy = deep_learning_model(X_train, y_train, X_test, y_test)

    clustering_analysis(X_scaled)
    advanced_visualizations(df)
    
    models = {name.lower().replace(" ", "_"): joblib.load(f'{name.lower().replace(" ", "_")}_model.pkl') for name in ['Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost']}
    create_detailed_gui(models, scaler)
