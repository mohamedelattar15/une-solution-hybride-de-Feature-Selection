import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataLoader:
    def __init__(self):
        pass

    def load_wine_data(self, path):
        # wine.data: class, 13 features
        df = pd.read_csv(path, header=None)
        X = df.iloc[:, 1:].values
        y = df.iloc[:, 0].values
        return self.prepare_data(X, y)

    def load_zoo_data(self, path):
        # zoo.data (skip name col): name, 16 features, class
        df = pd.read_csv(path, header=None)
        X = df.iloc[:, 1:-1].values
        y = df.iloc[:, -1].values
        return self.prepare_data(X, y)

    def load_krvskp_data(self, path):
        # krvskp.data: 36 features, class (categorical)
        df = pd.read_csv(path, header=None)
        X_cat = df.iloc[:, :-1]
        X = pd.get_dummies(X_cat).values # Simple encoding
        y_cat = df.iloc[:, -1].values
        y = np.where(y_cat == 'won', 1, 0)
        return self.prepare_data(X, y)

    def load_breast_data(self, path):
        # diagnosis: M=1, B=0, 30 features, drop 'id'
        df = pd.read_csv(path)
        if 'id' in df.columns:
            df = df.drop(columns=['id'])
        # Handle trailing comma if present (Unnamed columns)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        X = df.drop(columns=['diagnosis']).values
        y = df['diagnosis'].map({'M': 1, 'B': 0}).values
        return self.prepare_data(X, y)

    def prepare_data(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return {
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test,
            'n_features': X.shape[1]
        }

    def load_all_datasets(self, data_dir='../datasets'):
        datasets = {}
        
        wine_path = os.path.join(data_dir, 'wine.data')
        if os.path.exists(wine_path):
            datasets['wine'] = self.load_wine_data(wine_path)
            
        zoo_path = os.path.join(data_dir, 'zoo.data')
        if os.path.exists(zoo_path):
            try:
                datasets['zoo'] = self.load_zoo_data(zoo_path)
            except Exception as e:
                print(f"Failed to load Zoo: {e}")
                
        krvskp_path = os.path.join(data_dir, 'krvskp.data')
        if os.path.exists(krvskp_path):
            try:
                datasets['krvskp'] = self.load_krvskp_data(krvskp_path)
            except Exception as e:
                print(f"Failed to load Krvskp: {e}")

        breast_path = os.path.join(data_dir, 'breast_dataset.csv')
        if os.path.exists(breast_path):
            try:
                datasets['breast'] = self.load_breast_data(breast_path)
            except Exception as e:
                print(f"Failed to load Breast: {e}")
                
        return datasets
