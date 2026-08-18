import os
import joblib
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class BasePipeline(ABC):
    def __init__(self, script_dir: str):
        self.script_dir = script_dir
        self.standard_scaler = StandardScaler()
        self.min_max_scaler = MinMaxScaler()
        self.best_models: dict = {}

    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def preprocess(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def train(self, *args) -> None:
        pass

    @abstractmethod
    def predict(self) -> None:
        pass

    def run(self) -> None:
        df = self.load_data()
        preprocessed = self.preprocess(df)
        self.train(*preprocessed)
        self.predict()

    def _scale_data(self, x_train, x_test):
        x_train_std = self.standard_scaler.fit_transform(x_train)
        x_test_std = self.standard_scaler.transform(x_test)
        x_train_mm = self.min_max_scaler.fit_transform(x_train)
        x_test_mm = self.min_max_scaler.transform(x_test)
        return x_train_std, x_test_std, x_train_mm, x_test_mm

    def _save_models(self) -> None:
        save_dir = os.path.join(self.script_dir, 'models')
        os.makedirs(save_dir, exist_ok=True)
        for name, model in self.best_models.items():
            filename = os.path.join(save_dir, f'{name}.joblib')
            joblib.dump(model, filename)
            status = '成功' if os.path.exists(filename) else '失敗'
            print(f'模型儲存{status}: {filename}')
