import os
import copy
import warnings
warnings.filterwarnings('ignore')
import joblib
import numpy as np
import pandas as pd
from typing import Any
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .base_pipeline import BasePipeline
from .visualizer import Visualizer


class SurgeryPipeline(BasePipeline):
    LABEL_COLUMNS = ['性別(H)', '身份(I)', '分類(J)', '麻醉(K)', '手術名稱(L)', '主治醫師(AF)', '分類(AZ)']
    TREE_MODELS = {'Decision Tree', 'Random Forest', 'XGBoost'}

    def __init__(self, department: str, script_dir: str):
        super().__init__(script_dir)
        self.department = department
        self.label_encoders: dict = {}
        self.x_columns = None

    def load_data(self) -> pd.DataFrame:
        file_path = os.path.join(self.script_dir, f'{self.department}_Training.csv')
        df = pd.read_csv(file_path, encoding='big5')
        df['手術時間_log'] = np.log1p(df['手術時間（分）(BQ)'])
        return df

    def _filter_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        col = '手術時間（分）(BQ)'
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        return df[(df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr)]

    def preprocess(self, df: pd.DataFrame):
        df = self._filter_outliers(df)
        Visualizer.plot_distribution(
            df['手術時間（分）(BQ)'],
            f'{self.department} - Operation Time Distribution Chart',
            'Operation Time (min)'
        )

        null_cols = df.columns[df.isnull().any()]
        print(df[null_cols].isnull().sum())
        df = df.dropna(subset=null_cols)

        for col in self.LABEL_COLUMNS:
            df[col] = df[col].astype(str)
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le

        x = df.drop(columns=['手術時間（分）(BQ)', '手術時間_log'])
        y = df['手術時間_log']
        self.x_columns = x.columns

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        x_train_std, x_test_std, x_train_mm, x_test_mm = self._scale_data(x_train, x_test)
        return x_train, x_test, y_train, y_test, x_train_std, x_test_std, x_train_mm, x_test_mm

    def _print_metrics(self, name: str, y_test, y_pred) -> None:
        y_test_actual, y_pred_actual = np.expm1(y_test), np.expm1(y_pred)
        print(f'\n{name} Validation Index:')
        print(f'  MAE: {mean_absolute_error(y_test_actual, y_pred_actual):.2f}')
        print(f'  MSE: {mean_squared_error(y_test_actual, y_pred_actual):.2f}')
        print(f'  R2: {r2_score(y_test_actual, y_pred_actual):.2f}')

    def train(self, x_train, x_test, y_train, y_test, x_train_std, x_test_std, x_train_mm, x_test_mm) -> None:
        all_models = {
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'Random Forest': RandomForestRegressor(random_state=42),
            'SVM': SVR(),
            'KNN': KNeighborsRegressor(),
            'XGBoost': XGBRegressor(random_state=42)
        }

        for name, model in all_models.items():
            if name in ['SVM', 'KNN']:
                model_std = copy.deepcopy(model)
                model_std.fit(x_train_std, y_train)
                self._print_metrics(f'{name} (StandardScaler)', y_test, model_std.predict(x_test_std))
                self.best_models[f'{name} StandardScaler'] = model_std

                model_mm = copy.deepcopy(model)
                model_mm.fit(x_train_mm, y_train)
                self._print_metrics(f'{name} (MinMaxScaler)', y_test, model_mm.predict(x_test_mm))
                self.best_models[f'{name} MinMaxScaler'] = model_mm
            else:
                model.fit(x_train, y_train)
                self._print_metrics(name, y_test, model.predict(x_test))
                self.best_models[name] = model

        self._save_models()
        for name, model in self.best_models.items():
            if name in self.TREE_MODELS:
                Visualizer.plot_feature_importance(model, self.x_columns, f'{name} Feature Importance')

    def predict(self) -> None:
        file_path = os.path.join(self.script_dir, f'{self.department}_Testing.csv')
        df = pd.read_csv(file_path, encoding='big5').replace('?', np.nan)

        null_cols = df.columns[df.isnull().any()]
        print(df[null_cols].isnull().sum())

        for col in self.LABEL_COLUMNS:
            df[col] = df[col].astype(str)
            known_labels = set[Any](self.label_encoders[col].classes_)
            unknown_labels = set[Any](df[col]) - known_labels
            if unknown_labels:
                print(f'{col} 出現未知標籤: {unknown_labels}')
            df[col] = df[col].apply(lambda v: v if v in known_labels else 'unknown')
            new_le = LabelEncoder()
            new_le.fit(sorted(list[Any](self.label_encoders[col].classes_) + ['unknown']))
            df[col] = new_le.transform(df[col])

        x = df.drop(columns=['手術時間（分）(BQ)'])
        model = joblib.load(os.path.join(self.script_dir, 'models', 'XGBoost.joblib'))
        df['手術時間（分）(BQ)'] = np.round(np.expm1(model.predict(x))).astype(int)
        df.to_excel(os.path.join(self.script_dir, 'output_result.xlsx'), index=False)
