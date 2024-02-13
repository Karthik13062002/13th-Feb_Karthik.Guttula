import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

class RandomForestModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = {}
    
    def load(self, file_path):
        self.data = pd.read_csv(file_path)
        
    def preprocess(self):
        # To drop unnecessary columns
        self.data.drop(['item_no'], axis=1, inplace=True)
        
        # To encode categorical features
        categorical_features = ['category', 'main_promotion', 'color']
        for feature in categorical_features:
            self.label_encoders[feature] = LabelEncoder()
            self.data[feature + '_encoded'] = self.label_encoders[feature].fit_transform(self.data[feature])
            self.data.drop(feature, axis=1, inplace=True)
        
        # To convert stars to binary (0 or 1)
        self.data['stars'] = np.where(self.data['stars'] <= 3, 0, 1)
        
        # Split data into features and target
        X = self.data.drop('success_indicator', axis=1)
        y = self.data['success_indicator']
        
        # Scale numerical features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    def train(self):
        self.model = RandomForestClassifier()
        self.model.fit(self.X_train, self.y_train)
    
    def test(self):
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        
        evaluation_summary = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        return evaluation_summary
    
    def predict(self, data):
        # For Preprocessing new data
        processed_data = data.copy()
        for feature in ['category', 'main_promotion', 'color']:
            processed_data[feature + '_encoded'] = self.label_encoders[feature].transform(processed_data[feature])
            processed_data.drop(feature, axis=1, inplace=True)
        
        processed_data['stars'] = np.where(processed_data['stars'] <= 3, 0, 1)
        processed_data = self.scaler.transform(processed_data)
        
        # For making predictions
        predictions = self.model.predict(processed_data)
        return predictions

