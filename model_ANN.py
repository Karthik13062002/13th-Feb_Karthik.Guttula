import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

class ANNClassifier:
    def __init__(self):
        self.pipeline = None
        self.label_encoder1 = None
        self.label_encoder2 = None
        self.label_encoder3 = None
        self.scaler = None
        self.x_train = None
        self.x_test = None
        self.y_encoded_train = None
        self.y_encoded_test = None

    def load(self, file_path):
        self.data = pd.read_csv(file_path)

    def preprocess(self):
        x = self.data.drop(['success_indicator', 'item_no'], axis=1)
        y = self.data['success_indicator']

        self.label_encoder1 = LabelEncoder()
        x['category_encoded'] = self.label_encoder1.fit_transform(x['category'])
        x.drop('category', axis=1, inplace=True)

        self.label_encoder2 = LabelEncoder()
        x['main_promotion_encoded'] = self.label_encoder2.fit_transform(x['main_promotion'])
        x.drop('main_promotion', axis=1, inplace=True)

        self.label_encoder3 = LabelEncoder()
        x['color_encoded'] = self.label_encoder3.fit_transform(x['color'])
        x.drop('color', axis=1, inplace=True)

        x['stars'] = np.where(x['stars'] <= 3, 0, x['stars'])
        x['stars'] = np.where(x['stars'] > 3, 1, x['stars'])

        label_encoder4 = LabelEncoder()
        y_encoded = label_encoder4.fit_transform(y)
        y_encoded = np.where(y_encoded == label_encoder4.classes_.tolist().index('flop'), 0, y_encoded)
        y_encoded = np.where(y_encoded == label_encoder4.classes_.tolist().index('top'), 1, y_encoded)

        self.scaler = StandardScaler()
        x_train = self.scaler.fit_transform(x)

        self.x_train, self.x_test, self.y_encoded_train, self.y_encoded_test = train_test_split(x_train, y_encoded, test_size=0.2, random_state=77)

    def create_model(self):
        model = Sequential()
        model.add(Dense(10, input_dim=4, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def fit(self):
        keras_model = KerasClassifier(build_fn=self.create_model, epochs=10, batch_size=32, verbose=0)

        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', keras_model)
        ])

        self.pipeline.fit(self.x_train, self.y_encoded_train)

    def predict(self):
        return self.pipeline.predict(self.x_test)
    
    def evaluate(self):
        y_pred = self.predict()
        accuracy = accuracy_score(self.y_encoded_test, y_pred)
        precision = precision_score(self.y_encoded_test, y_pred)
        recall = recall_score(self.y_encoded_test, y_pred)
        f1 = f1_score(self.y_encoded_test, y_pred)
        
        print("Accuracy with ANN classifier is:", accuracy)
        print("Precision with ANN classifier model is :", precision)
        print("Recall with ANN classifier model is:", recall)
        print("F1 Score with ANN classifier model is :", f1)

    def load_test_file(self, file_path):
        self.input_data = pd.read_csv(file_path)
        return self.input_data
        
    def test_data_preprocessor(self):
        self.input_data_processed = self.input_data.drop(['item_no'], axis=1)
        self.input_data_processed['category_encoded'] = self.label_encoder1.transform(self.input_data_processed['category'])
        self.input_data_processed.drop('category', axis=1, inplace=True)
        self.input_data_processed['main_promotion_encoded'] = self.label_encoder2.transform(self.input_data_processed['main_promotion'])
        self.input_data_processed.drop('main_promotion', axis=1, inplace=True)
        self.input_data_processed['color_encoded'] = self.label_encoder3.transform(self.input_data_processed['color'])
        self.input_data_processed.drop('color', axis=1, inplace=True)
        self.input_data_processed['stars'] = np.where(self.input_data_processed['stars'] <= 3, 0, self.input_data_processed['stars'])
        self.input_data_processed['stars'] = np.where(self.input_data_processed['stars'] > 3, 1, self.input_data_processed['stars'])
        self.input_data_processed = self.scaler.transform(self.input_data_processed)
        return self.input_data_processed
        
    def predict_for_test_data(self):
        output = self.pipeline.predict(self.input_data_processed)
        return output

