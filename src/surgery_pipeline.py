import os
import copy
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
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
    # 各科 CSV 的欄位字母後綴不一致（如分類欄 OPH 為 (AY)、其他科為 (AZ)），依語意名稱解析
    LABEL_PREFIXES = ['性別', '身份', '分類', '麻醉', '手術名稱', '主治醫師']
    TREE_MODELS = {'Decision Tree', 'Random Forest', 'XGBoost'}
    TARGET = '手術時間（分）(BQ)'
    # 對外服務固定用 XGBoost：Random Forest 在部分科別雖然 MAE 略低（ENT 28.04 vs 28.35，
    # 差 0.31 分），但未壓縮模型檔達 50 MB，而 XGBoost 僅 0.39 MB（約 128 倍）。
    # 以 0.3 分鐘的誤差換取檔案縮減，對需要隨 repo 部署的線上服務是合理取捨。
    SERVING_MODEL = 'XGBoost'

    def __init__(self, department: str, data_dir: str, output_dir: str, figure_dir: str,
                 model_dir: str = 'models'):
        super().__init__(department, data_dir, output_dir, figure_dir, model_dir)
        self.label_encoders: dict = {}
        self.label_columns: list = []
        self.x_columns = None
        self.numeric_medians = None
        self.feature_defaults: dict = {}

    def _serving_artifacts(self) -> dict:
        return {
            'task': 'regression',
            'x_columns': list(self.x_columns),
            'label_columns': list(self.label_columns),
            # 類別→編碼的映射，同時作為 UI 下拉選單的選項來源
            'label_classes': {c: [str(v) for v in le.classes_]
                              for c, le in self.label_encoders.items()},
            'feature_defaults': self.feature_defaults,
            'target_transform': 'log1p',
        }

    def load_data(self) -> pd.DataFrame:
        file_path = os.path.join(self.data_dir, f'{self.department}_Training.csv')
        df = pd.read_csv(file_path, encoding='big5')
        self.label_columns = [c for c in df.columns if c.split('(')[0] in self.LABEL_PREFIXES]
        df['手術時間_log'] = np.log1p(df[self.TARGET])
        return df

    def _filter_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        q1, q3 = df[self.TARGET].quantile([0.25, 0.75])
        iqr = q3 - q1
        return df[(df[self.TARGET] >= q1 - 1.5 * iqr) & (df[self.TARGET] <= q3 + 1.5 * iqr)]

    def preprocess(self, df: pd.DataFrame):
        df = self._filter_outliers(df)
        Visualizer.plot_distribution(
            df[self.TARGET],
            f'{self.department} - Operation Time Distribution',
            'Operation Time (min)',
            self._figure_path('operation time distribution'),
        )

        null_cols = df.columns[df.isnull().any()]
        print(df[null_cols].isnull().sum())
        df = df.dropna(subset=null_cols)

        for col in self.label_columns:
            df[col] = df[col].astype(str)
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le

        x = df.drop(columns=[self.TARGET, '手術時間_log'])
        y = df['手術時間_log']
        self.x_columns = x.columns
        self.numeric_medians = x.median()
        # 服務時 UI 只詢問主要欄位，其餘以類別眾數／數值中位數填補
        self.feature_defaults = {
            c: int(x[c].mode().iloc[0]) if c in self.label_columns else float(x[c].median())
            for c in x.columns
        }

        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
        x_train_std, x_val_std, x_train_mm, x_val_mm = self._scale_data(x_train, x_val)
        return x_train, x_val, y_train, y_val, x_train_std, x_val_std, x_train_mm, x_val_mm

    def _print_metrics(self, name: str, y_val, y_pred) -> None:
        y_val_actual, y_pred_actual = np.expm1(y_val), np.expm1(y_pred)
        mae = mean_absolute_error(y_val_actual, y_pred_actual)
        rmse = float(np.sqrt(mean_squared_error(y_val_actual, y_pred_actual)))
        r2 = r2_score(y_val_actual, y_pred_actual)
        print(f'\n{name} Validation Index:')
        print(f'  MAE: {mae:.2f}')
        print(f'  RMSE: {rmse:.2f}')
        print(f'  R2: {r2:.2f}')
        self.metrics.append({
            'department': self.department, 'model': name,
            'mae': round(mae, 2), 'rmse': round(rmse, 2), 'r2': round(r2, 3),
        })

    def train(self, x_train, x_val, y_train, y_val, x_train_std, x_val_std, x_train_mm, x_val_mm) -> None:
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
                self._print_metrics(f'{name} (StandardScaler)', y_val, model_std.predict(x_val_std))
                self.trained_models[f'{name} (StandardScaler)'] = model_std
                self.model_scalers[f'{name} (StandardScaler)'] = 'std'

                model_mm = copy.deepcopy(model)
                model_mm.fit(x_train_mm, y_train)
                self._print_metrics(f'{name} (MinMaxScaler)', y_val, model_mm.predict(x_val_mm))
                self.trained_models[f'{name} (MinMaxScaler)'] = model_mm
                self.model_scalers[f'{name} (MinMaxScaler)'] = 'mm'
            else:
                model.fit(x_train, y_train)
                self._print_metrics(name, y_val, model.predict(x_val))
                self.trained_models[name] = model

        self._select_best_model('mae', higher_is_better=False)
        self._save_models()
        for name, model in self.trained_models.items():
            if name in self.TREE_MODELS:
                Visualizer.plot_feature_importance(
                    model, self.x_columns,
                    f'{self.department} - {name} Feature Importance',
                    self._figure_path(f'{name} feature importance'),
                )

    def predict(self) -> None:
        file_path = os.path.join(self.data_dir, f'{self.department}_Testing.csv')
        df = pd.read_csv(file_path, encoding='big5').replace('?', np.nan)

        for col in self.label_columns:
            df[col] = df[col].astype(str)
            classes = self.label_encoders[col].classes_
            # 沿用訓練期的類別→編碼映射；未知標籤給保留值，不位移既有編碼
            mapping = {c: i for i, c in enumerate(classes)}
            unknown_code = len(classes)
            unknown_labels = set(df[col]) - set(classes)
            if unknown_labels:
                print(f'{col} 出現未知標籤（統一編碼為 {unknown_code}）: {unknown_labels}')
            df[col] = df[col].map(lambda v: mapping.get(v, unknown_code))

        x = df[self.x_columns].apply(pd.to_numeric, errors='coerce')
        n_missing = int(x.isnull().sum().sum())
        if n_missing:
            print(f'測試資料共 {n_missing} 格缺值，以訓練資料中位數補值')
            x = x.fillna(self.numeric_medians)

        best_model = self.trained_models[self.best_model_name]
        scaler = self._scaler_for(self.best_model_name)
        features = scaler.transform(x) if scaler is not None else x
        df[self.TARGET] = np.round(np.expm1(best_model.predict(features))).astype(int)

        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, f'{self.department}_prediction.xlsx')
        df.to_excel(output_path, index=False)
        print(f'\n[{self.department}] 以 {self.best_model_name} 產出預測: {output_path}')
