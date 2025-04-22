from metaflow import FlowSpec, step
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

class HeartDiseaseTrainingFlow(FlowSpec):
    
    @step
    def start(self):
        """
        Load and preprocess the data
        """
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Load data
        self.data = pd.read_csv("data/heart.csv")
        
        # Split features and target
        self.X = self.data.drop('HeartDisease', axis=1)
        self.y = self.data['HeartDisease']
        
        # Split data into train, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(self.X, self.y, test_size=0.4, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        # Define feature groups
        self.numeric_features = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
        self.categorical_features = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
        
        # Create preprocessor
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numeric_features),
                ('cat', OneHotEncoder(drop='first'), self.categorical_features)
            ],
            remainder='passthrough'
        )
        
        # Preprocess data
        self.X_train = X_train
        self.X_val = X_val 
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        
        self.X_train_processed = self.preprocessor.fit_transform(X_train)
        self.X_val_processed = self.preprocessor.transform(X_val)
        self.X_test_processed = self.preprocessor.transform(X_test)
        
        # Save preprocessor
        with open('models/preprocessor.pkl', 'wb') as f:
            pickle.dump(self.preprocessor, f)
        
        self.next(self.train_models)
        
    @step
    def train_models(self):
        """
        Train multiple models with different hyperparameters
        """
        # Define models and their hyperparameters
        self.models = {
            'Logistic Regression': (LogisticRegression(random_state=42, max_iter=1000),
                                {'C': [0.01, 0.1, 1, 10]}),
            'Decision Tree': (DecisionTreeClassifier(random_state=42),
                            {'max_depth': [3, 5, 7, None]}),
            'Random Forest': (RandomForestClassifier(random_state=42),
                            {'n_estimators': [50, 100, 200],
                            'max_depth': [5, 10, None]}),
            'SVM': (SVC(random_state=42, probability=True),
                    {'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf']})
        }
        
        self.results = []
        
        # Train and evaluate each model
        for name, (model, param_grid) in self.models.items():
            print(f"\nTraining {name}...")
            
            # Perform grid search
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
            grid_search.fit(self.X_train_processed, self.y_train)
            
            # Get best model
            best_model = grid_search.best_estimator_
            
            # Evaluate on validation set
            y_val_pred = best_model.predict(self.X_val_processed)
            accuracy = accuracy_score(self.y_val, y_val_pred)
            report = classification_report(self.y_val, y_val_pred)
            
            print(f"{name} - Best parameters: {grid_search.best_params_}")
            print(f"{name} - Validation accuracy: {accuracy:.4f}")
            print(f"{name} - Classification Report:\n{report}")
            
            self.results.append((name, best_model, accuracy))
        
        self.next(self.select_best_model)
    
    @step
    def select_best_model(self):
        """
        Select the best performing model and evaluate on test set
        """
        # Get best model
        self.best_model_name, self.best_model, best_val_accuracy = max(self.results, key=lambda x: x[2])
        
        # Evaluate on test set
        y_test_pred = self.best_model.predict(self.X_test_processed)
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        test_report = classification_report(self.y_test, y_test_pred)
        
        print(f"\nBest Model: {self.best_model_name}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print("\nClassification Report:")
        print(test_report)
        
        # Save the best model
        with open('models/best_model.pkl', 'wb') as f:
            pickle.dump(self.best_model, f)
        
        self.next(self.end)
    
    @step
    def end(self):
        """
        Final step
        """
        print(f"\nTraining completed successfully!")
        print(f"Best model ({self.best_model_name}) saved to models/best_model.pkl")
        print("Preprocessor saved to models/preprocessor.pkl")

if __name__ == '__main__':
    HeartDiseaseTrainingFlow() 