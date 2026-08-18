import os
import copy
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from typing import Optional
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
    BIN_EDGES = [-float('inf'), -100, -50, -25, 25, 50, 100, float('inf')]
    BIN_LABELS = ['大提前', '中提前', '小提前', '準時', '小延後', '中延後', '大延後']
    LABEL_TO_MINUTES = {
        '大提前': -120, '中提前': -60, '小提前': -30, '準時': 15,
        '小延後': 30, '中延後': 60, '大延後': 120,
    }
    TREE_MODELS = {'Decision Tree', 'Random Forest', 'XGBoost'}

    def __init__(
        self,
        department: str,
        filename: str,
        data_dir: str,
        output_dir: str,
        figure_dir: str,
        wait_min: Optional[float] = None,
        wait_max: Optional[float] = None,
        diag_max: float = 100.0,
    ):
        super().__init__(department, data_dir, output_dir, figure_dir)
        self.filename = filename
        self.wait_min = wait_min
        self.wait_max = wait_max
        self.diag_max = diag_max
        self.x_columns = None
        self.class_labels: list = []

    def load_data(self) -> pd.DataFrame:
        file_path = os.path.join(self.data_dir, self.filename)
        df = pd.read_excel(file_path)

        df['門診日期'] = pd.to_datetime(df['門診日期'])
        df['是否取消掛號'] = (df['是否取消掛號'] == 'Y').astype(int)
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
            f'{self.department} - Delay Time Distribution',
            'Delay Time (min)',
            self._figure_path('delay time distribution'),
        )
        diagnosis_time = df['看診時間(分)'].mean()
        print(f'{self.department} - Average Diagnosis Time: {diagnosis_time:.2f} minutes')
        Visualizer.plot_distribution(
            df['看診時間(分)'],
            f'{self.department} - Diagnosis Time Distribution',
            'Diagnosis Time (min)',
            self._figure_path('diagnosis time distribution'),
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

        # 以 pd.cut 的有序類別直接編碼（0..k-1 依語意順序），
        # 取代 LabelEncoder 的 Unicode 排序，確保編碼與分鐘映射一致
        df['類別'] = pd.cut(df['等待時間(分)'], bins=self.BIN_EDGES, labels=self.BIN_LABELS)
        df['類別'] = df['類別'].cat.remove_unused_categories()
        self.class_labels = list(df['類別'].cat.categories)
        print(f'{self.department} 實際出現的類別（依語意順序編碼 0..{len(self.class_labels) - 1}）: {self.class_labels}')
        y = df['類別'].cat.codes.to_numpy()

        x = df[['掛號序號', '預估看診時間', '看診人數累計', '掛號人數總計']].copy()
        x['預估看診時間(時)'] = x['預估看診時間'].dt.hour
        x['預估看診時間(分)'] = x['預估看診時間'].dt.minute
        x = x.drop(columns=['預估看診時間'])
        self.x_columns = x.columns

        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
        x_train_std, x_val_std, x_train_mm, x_val_mm = self._scale_data(x_train, x_val)
        return x_train, x_val, y_train, y_val, x_train_std, x_val_std, x_train_mm, x_val_mm

    def _print_metrics(self, name: str, y_val, y_pred) -> None:
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_val, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
        print(f'\n{name} Validation Index:')
        print(f'  Accuracy: {acc:.2f}')
        print(f'  Precision: {prec:.2f}')
        print(f'  Recall: {rec:.2f}')
        print(f'  F1: {f1:.2f}')
        self.metrics.append({
            'department': self.department, 'model': name,
            'accuracy': round(acc, 3), 'precision': round(prec, 3),
            'recall': round(rec, 3), 'f1': round(f1, 3),
        })
        cm = confusion_matrix(y_val, y_pred, labels=list(range(len(self.class_labels))))
        Visualizer.plot_confusion_matrix(
            cm,
            f'{self.department} - {name} Confusion Matrix',
            self._figure_path(f'{name} confusion matrix'),
            labels=self.class_labels,
        )

    def train(self, x_train, x_val, y_train, y_val, x_train_std, x_val_std, x_train_mm, x_val_mm) -> None:
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
                grid_std = GridSearchCV(copy.deepcopy(model), param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
                grid_std.fit(x_train_std, y_train)
                self._print_metrics(f'{name} (StandardScaler)', y_val, grid_std.predict(x_val_std))
                self.trained_models[f'{name} (StandardScaler)'] = grid_std.best_estimator_
                self.model_scalers[f'{name} (StandardScaler)'] = 'std'

                grid_mm = GridSearchCV(copy.deepcopy(model), param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
                grid_mm.fit(x_train_mm, y_train)
                self._print_metrics(f'{name} (MinMaxScaler)', y_val, grid_mm.predict(x_val_mm))
                self.trained_models[f'{name} (MinMaxScaler)'] = grid_mm.best_estimator_
                self.model_scalers[f'{name} (MinMaxScaler)'] = 'mm'
            else:
                grid = GridSearchCV(model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
                grid.fit(x_train, y_train)
                self._print_metrics(name, y_val, grid.predict(x_val))
                self.trained_models[name] = grid.best_estimator_

        self._select_best_model('f1', higher_is_better=True)
        for name, model in self.trained_models.items():
            if name in self.TREE_MODELS:
                Visualizer.plot_feature_importance(
                    model, self.x_columns,
                    f'{self.department} - {name} Feature Importance',
                    self._figure_path(f'{name} feature importance'),
                )

    def predict(self) -> None:
        print(f'\n========== {self.department} 範例推論 ==========')
        sample = pd.DataFrame([[10, 25, 50, 15, 30]], columns=self.x_columns)
        hour = int(sample['預估看診時間(時)'].iloc[0])
        minute = int(sample['預估看診時間(分)'].iloc[0])

        model = self.trained_models[self.best_model_name]
        scaler = self._scaler_for(self.best_model_name)
        features = scaler.transform(sample) if scaler is not None else sample
        pred_code = int(model.predict(features)[0])
        pred_label = self.class_labels[pred_code]
        pred_minutes = self.LABEL_TO_MINUTES[pred_label]
        estimated = pd.Timestamp(f'{hour:02d}:{minute:02d}:00')
        actual = estimated + pd.Timedelta(minutes=pred_minutes)

        print(f'範例輸入: 掛號序號 {int(sample["掛號序號"].iloc[0])}、預約 {hour:02d}:{minute:02d}、'
              f'現場進度 {int(sample["看診人數累計"].iloc[0])}/{int(sample["掛號人數總計"].iloc[0])} 人')
        print(f'{self.best_model_name} 預測類別: {pred_label}（{pred_minutes:+d} 分）'
              f'，預估實際看診時間: {actual.strftime("%H:%M")}')
