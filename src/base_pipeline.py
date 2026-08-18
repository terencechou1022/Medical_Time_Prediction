import os
import joblib
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class BasePipeline(ABC):
    def __init__(self, department: str, data_dir: str, output_dir: str, figure_dir: str):
        self.department = department
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.figure_dir = figure_dir
        self.standard_scaler = StandardScaler()
        self.min_max_scaler = MinMaxScaler()
        self.trained_models: dict = {}
        self.model_scalers: dict = {}
        self.metrics: list = []
        self.best_model_name: str = ''

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

    def _figure_path(self, name: str) -> str:
        slug = name.replace(' ', '_').replace('(', '').replace(')', '')
        return os.path.join(self.figure_dir, f'{self.department}_{slug}.png')

    def _scale_data(self, x_train, x_val):
        x_train_std = self.standard_scaler.fit_transform(x_train)
        x_val_std = self.standard_scaler.transform(x_val)
        x_train_mm = self.min_max_scaler.fit_transform(x_train)
        x_val_mm = self.min_max_scaler.transform(x_val)
        return x_train_std, x_val_std, x_train_mm, x_val_mm

    def _scaler_for(self, model_name: str):
        kind = self.model_scalers.get(model_name)
        if kind == 'std':
            return self.standard_scaler
        if kind == 'mm':
            return self.min_max_scaler
        return None

    def _select_best_model(self, metric: str, higher_is_better: bool) -> None:
        scores = {row['model']: row[metric] for row in self.metrics}
        pick = max if higher_is_better else min
        self.best_model_name = pick(scores, key=scores.get)
        print(f'\n[{self.department}] 依驗證集 {metric.upper()} 選出最佳模型: '
              f'{self.best_model_name}（{metric.upper()} = {scores[self.best_model_name]}）')
        for row in self.metrics:
            row['selected'] = row['model'] == self.best_model_name

    def _save_models(self) -> None:
        save_dir = os.path.join(self.output_dir, 'models', self.department)
        os.makedirs(save_dir, exist_ok=True)
        for name, model in self.trained_models.items():
            slug = name.replace(' ', '_').replace('(', '').replace(')', '')
            filename = os.path.join(save_dir, f'{slug}.joblib')
            joblib.dump(model, filename)
            status = '成功' if os.path.exists(filename) else '失敗'
            print(f'模型儲存{status}: {filename}')
