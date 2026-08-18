import os
import copy
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from typing import Optional
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from .base_pipeline import BasePipeline
from .visualizer import Visualizer


class ClinicPipeline(BasePipeline):
    LABEL_MAPPING = {
        0: -120,  # 大提前
        1: -60,   # 中提前
        2: -30,   # 小提前
        3: 15,    # 準時
        4: 30,    # 小延後
        5: 60,    # 中延後
        6: 120    # 大延後
    }

    def __init__(
        self,
        department: str,
        filename: str,
        script_dir: str,
        wait_min: Optional[float] = None,
        wait_max: Optional[float] = None,
        diag_max: float = 100.0,
    ):
        super().__init__(script_dir)
        self.department = department
        self.filename = filename
        self.wait_min = wait_min
        self.wait_max = wait_max
        self.diag_max = diag_max
        self.x_columns = None

    def load_data(self) -> pd.DataFrame:
        file_path = os.path.join(self.script_dir, self.filename)
        df = pd.read_excel(file_path)

        df['門診日期'] = pd.to_datetime(df['門診日期'])
        df['是否取消掛號'] = np.where(df['是否取消掛號'] == 'Y', 1, 0)
        df['預估看診時間'] = pd.to_datetime(df['門診日期'].astype(str) + ' ' + df['預估看診時間'].astype(str))
        df['實際看診時間'] = pd.to_datetime(df['實際看診時間'], format='%H:%M:%S', errors='coerce')
        df = df.dropna(subset=['實際看診時間'])
        df = df.sort_values(['門診日期', '實際看診時間'])
        df['等待時間(分)'] = ((df['實際看診時間'] - df['預估看診時間']).dt.total_seconds() / 60).round(1)
        df['看診時間(分)'] = (
            (df.groupby(['門診日期'])['實際看診時間'].shift(-1) - df['實際看診時間']).dt.total_seconds() / 60
        ).round(1)
        df['看診人數累計'] = df.groupby(['門診日期']).cumcount()
        df['掛號人數總計'] = df.groupby(['門診日期'])['掛號序號'].transform('count')
        df = df.dropna(subset=['等待時間(分)', '看診時間(分)'])
        df = df.drop(columns=['科別代碼', '科別名稱', '診間', '醫師', '病歷號'])
        return df

    def _print_distribution(self, df: pd.DataFrame) -> None:
        delay_time = df['等待時間(分)'].mean()
        print(f'{self.department} - Average Delay Time: {delay_time:.2f} minutes')
        Visualizer.plot_distribution(
            df['等待時間(分)'],
            f'{self.department} - Delay Time Distribution Chart',
            'Delay Time (min)'
        )
        diagnosis_time = df['看診時間(分)'].mean()
        print(f'{self.department} - Average Diagnosis Time: {diagnosis_time:.2f} minutes')
        Visualizer.plot_distribution(
            df['看診時間(分)'],
            f'{self.department} - Diagnosis Time Distribution Chart',
            'Diagnosis Time (min)'
        )

    def _filter_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.wait_min is not None:
            df = df[df['等待時間(分)'] >= self.wait_min]
        if self.wait_max is not None:
            df = df[df['等待時間(分)'] <= self.wait_max]
        return df[df['看診時間(分)'] <= self.diag_max]

    def preprocess(self, df: pd.DataFrame):
        self._print_distribution(df)
        df = self._filter_outliers(df)

        bins = [-float('inf'), -100, -50, -25, 25, 50, 100, float('inf')]
        labels = ['大提前', '中提前', '小提前', '準時', '小延後', '中延後', '大延後']
        df['類別'] = pd.cut(df['等待時間(分)'], bins=bins, labels=labels)

        x = df[['掛號序號', '預估看診時間', '看診人數累計', '掛號人數總計']].copy()
        x['預估看診時間(時)'] = x['預估看診時間'].dt.hour
        x['預估看診時間(分)'] = x['預估看診時間'].dt.minute
        x = x.drop(columns=['預估看診時間'])
        self.x_columns = x.columns

        le = LabelEncoder()
        y_encoded = le.fit_transform(df['類別'])

        x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42)
        x_train_std, x_test_std, x_train_mm, x_test_mm = self._scale_data(x_train, x_test)
        return x_train, x_test, y_train, y_test, x_train_std, x_test_std, x_train_mm, x_test_mm

    def _print_metrics(self, name: str, y_test, y_pred) -> None:
        cm = confusion_matrix(y_test, y_pred)
        print(f'\n{name} Validation Index:')
        print(f'  Accuracy: {accuracy_score(y_test, y_pred):.2f}')
        print(f'  Precision: {precision_score(y_test, y_pred, average="weighted"):.2f}')
        print(f'  Recall: {recall_score(y_test, y_pred, average="weighted"):.2f}')
        print(f'  F1: {f1_score(y_test, y_pred, average="weighted"):.2f}')
        Visualizer.plot_confusion_matrix(cm, f'{name} Confusion Matrix')

    def train(self, x_train, x_test, y_train, y_test, x_train_std, x_test_std, x_train_mm, x_test_mm) -> None:
        all_models = {
            'Decision Tree': (DecisionTreeClassifier(random_state=42), {
                'max_depth': [10, 20, 50],
                'min_samples_split': [5, 10, 20],
                'min_samples_leaf': [1, 2, 4]
            }),
            'Random Forest': (RandomForestClassifier(random_state=42), {
                'n_estimators': [100, 500, 1000],
                'max_depth': [10, 20, 50],
                'min_samples_split': [5, 10, 20],
                'min_samples_leaf': [1, 2, 4]
            }),
            'SVM': (SVC(), {
                'C': [0.1, 1],
                'kernel': ['rbf', 'poly', 'sigmoid'],
                'gamma': ['scale', 'auto']
            }),
            'KNN': (KNeighborsClassifier(), {
                'n_neighbors': [5, 10, 20],
                'weights': ['uniform', 'distance'],
                'p': [1, 2]
            }),
            'XGBoost': (XGBClassifier(random_state=42), {
                'n_estimators': [100, 500, 1000],
                'max_depth': [10, 20, 50],
                'learning_rate': [0.01, 0.1],
            })
        }

        for name, (model, param_grid) in all_models.items():
            if name in ['SVM', 'KNN']:
                grid_std = GridSearchCV(copy.deepcopy(model), param_grid, cv=5, scoring='f1_weighted')
                grid_std.fit(x_train_std, y_train)
                self._print_metrics(f'{name} (StandardScaler)', y_test, grid_std.predict(x_test_std))
                self.best_models[f'{name} (StandardScaler)'] = grid_std.best_estimator_

                grid_mm = GridSearchCV(copy.deepcopy(model), param_grid, cv=5, scoring='f1_weighted')
                grid_mm.fit(x_train_mm, y_train)
                self._print_metrics(f'{name} (MinMaxScaler)', y_test, grid_mm.predict(x_test_mm))
                self.best_models[f'{name} (MinMaxScaler)'] = grid_mm.best_estimator_
            else:
                grid = GridSearchCV(model, param_grid, cv=5, scoring='f1_weighted')
                grid.fit(x_train, y_train)
                self._print_metrics(name, y_test, grid.predict(x_test))
                self.best_models[name] = grid.best_estimator_

    def predict(self) -> None:
        print(f'========== {self.department} ==========')
        sample = pd.DataFrame(
            [[10, 25, 50, 15, 30]],
            columns=['掛號序號', '看診人數累計', '掛號人數總計', '預估看診時間(時)', '預估看診時間(分)']
        )

        for name, model in self.best_models.items():
            if 'StandardScaler' in name:
                pred_encoded = model.predict(self.standard_scaler.transform(sample))[0]
            elif 'MinMaxScaler' in name:
                pred_encoded = model.predict(self.min_max_scaler.transform(sample))[0]
            else:
                pred_encoded = model.predict(sample)[0]

            pred_minutes = self.LABEL_MAPPING[pred_encoded]
            actual_time = pd.to_datetime('15:30:00') + pd.Timedelta(minutes=float(pred_minutes))
            print(f"\n{name} 預測實際看診時間: {actual_time.strftime('%H:%M:%S')}")

            if name in ['Random Forest', 'XGBoost']:
                Visualizer.plot_feature_importance(model, self.x_columns, f'{name} Feature Importance')
