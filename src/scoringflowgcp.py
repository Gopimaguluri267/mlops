from metaflow import FlowSpec, step, kubernetes, resources, retry, timeout, catch, conda_base
import pandas as pd
import numpy as np
import pickle

@conda_base(libraries={
    'pandas': '1.5.3',
    'numpy': '1.24.3',
    'scikit-learn': '1.2.2'
})
class HeartDiseaseScoringFlow(FlowSpec):
    
    @kubernetes
    @retry(times=3)
    @timeout(minutes=5)
    @step
    def start(self):
        """
        Load model and preprocessor
        """
        # Load the best model
        with open('models/best_model.pkl', 'rb') as f:
            self.model = pickle.load(f)
            
        # Load the preprocessor
        with open('models/preprocessor.pkl', 'rb') as f:
            self.preprocessor = pickle.load(f)
            
        self.next(self.preprocess_data)
    
    @kubernetes
    @retry(times=2)
    @timeout(minutes=5)
    @catch(var='preprocess_error', print_exception=True)
    @step
    def preprocess_data(self):
        """
        Load and preprocess input data
        """
        # Fixed path for test data
        test_data_path = "data/heart.csv"
        
        # Load data
        self.data = pd.read_csv(test_data_path)
        
        # Ensure all required columns are present
        required_columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
                          'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 
                          'Oldpeak', 'ST_Slope']
        
        missing_cols = set(required_columns) - set(self.data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Preprocess data
        self.processed_data = self.preprocessor.transform(self.data)
        
        self.next(self.make_predictions)
    
    @kubernetes
    @resources(cpu=1, memory=2000)
    @retry(times=2)
    @timeout(minutes=10)
    @step
    def make_predictions(self):
        """
        Make predictions using the loaded model
        """
        # Get predictions
        self.predictions = self.model.predict(self.processed_data)
        self.probabilities = self.model.predict_proba(self.processed_data)
        
        # Create results dataframe
        self.results = self.data.copy()
        self.results['predicted_heart_disease'] = self.predictions
        self.results['probability_heart_disease'] = self.probabilities[:, 1]
        
        # Save predictions
        output_path = 'predictions.csv'
        self.results.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
        
        self.next(self.end)
    
    @kubernetes
    @step
    def end(self):
        """
        Final step
        """
        print("Scoring completed successfully!")
        print(f"Processed {len(self.predictions)} records")
        print(f"Predicted heart disease cases: {sum(self.predictions)}")

if __name__ == '__main__':
    HeartDiseaseScoringFlow()