import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
import pickle
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np


# Load data
data = pd.read_csv("data/heart_lab3.csv")

X = data.drop('HeartDisease', axis=1)
y = data['HeartDisease']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

numeric_features = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
categorical_features = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ],
    remainder='passthrough'
)

X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

with open('models/preprocessor.pkl', 'wb') as file:
    pickle.dump(preprocessor, file)

X_train.to_csv("data/processed_train_data_heart.csv")
X_val.to_csv("data/processed_val_data_heart.csv")
X_test.to_csv("data/processed_test_data_heart.csv")
