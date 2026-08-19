r"""服務用 bundle 的契約測試：models/*.joblib 能載入，且推論路徑能跑通。

這批 bundle 由 `python main.py all` 產生、隨 repo 一起版控，是別人 clone 下來
唯一拿得到的模型檔。它們是用當時的 scikit-learn / XGBoost 版本 pickle 出來的，
而跨版本載入是最容易無聲失效的地方——升級套件後 joblib.load 可能直接拋錯，
也可能載入成功卻讓 predict 壞掉。所以這裡不只檢查檔案存在，而是照 bundle 自己
記錄的特徵順序與縮放器真的推論一次，把完整那條路走過（含 Rheumatology 的
MinMaxScaler 分支）。

執行方式（於專案根目錄）：
    .venv\Scripts\python.exe -m pytest -q
"""
import os

import joblib
import numpy as np
import pandas as pd
import pytest

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')

SURGERY = ['ENT', 'GS', 'GU', 'OPH', 'ORTH']
CLINIC = ['Cardiology', 'Neurology', 'Rheumatology']

# 門診的五個特徵；數值取訓練時的表單預設值，讓推論走在合理範圍內。
CLINIC_SAMPLE = {
    '掛號序號': 10,
    '看診人數累計': 25,
    '掛號人數總計': 50,
    '預估看診時間(時)': 15,
    '預估看診時間(分)': 30,
}


def load(dept):
    return joblib.load(os.path.join(MODEL_DIR, f'{dept}.joblib'))


def predict(bundle, features: dict):
    """依 bundle 記錄的特徵順序組成輸入，必要時套用當初的縮放器"""
    row = pd.DataFrame([[features[c] for c in bundle['x_columns']]],
                       columns=bundle['x_columns'])
    scaler = bundle.get('scaler')
    return bundle['model'].predict(scaler.transform(row) if scaler else row)[0]


def test_bundle_files_match_expected_departments():
    """models/ 的內容必須與八個科別完全一致。

    多一個檔案代表有孤兒模型，少一個代表該科別的 bundle 沒被產出來。
    """
    on_disk = sorted(f[:-len('.joblib')] for f in os.listdir(MODEL_DIR)
                     if f.endswith('.joblib'))
    assert on_disk == sorted(SURGERY + CLINIC)


@pytest.mark.parametrize('dept', SURGERY + CLINIC)
def test_bundle_loads_with_common_contract(dept):
    bundle = load(dept)
    for key in ('department', 'model_name', 'model', 'metrics', 'task', 'x_columns'):
        assert key in bundle, f'{dept} 的 bundle 缺少 {key}'
    assert bundle['department'] == dept
    assert bundle['x_columns'], f'{dept} 的 x_columns 為空'
    assert hasattr(bundle['model'], 'predict')


@pytest.mark.parametrize('dept', SURGERY)
def test_surgery_bundle_contract(dept):
    bundle = load(dept)
    assert bundle['task'] == 'regression'
    # 預測值要用 np.expm1 還原，所以這個欄位是反轉換的依據，不能變。
    assert bundle['target_transform'] == 'log1p'
    assert bundle['label_classes'], f'{dept} 缺少類別選項'
    assert set(bundle['label_columns']) <= set(bundle['x_columns'])
    assert set(bundle['feature_defaults']) <= set(bundle['x_columns'])
    for key in ('mae', 'rmse', 'r2'):
        assert key in bundle['metrics'], f'{dept} 的 metrics 缺少 {key}'


@pytest.mark.parametrize('dept', CLINIC)
def test_clinic_bundle_contract(dept):
    bundle = load(dept)
    assert bundle['task'] == 'classification'
    assert list(bundle['x_columns']) == list(CLINIC_SAMPLE), '門診特徵與樣本不一致'
    assert bundle['class_labels'], f'{dept} 的 class_labels 為空'
    # 先用 class_labels 取標籤、再用 label_to_minutes 換分鐘，缺一個就 KeyError。
    missing = [c for c in bundle['class_labels'] if c not in bundle['label_to_minutes']]
    assert not missing, f'{dept} 的 label_to_minutes 缺少 {missing}'
    assert sorted(bundle['class_counts']) == sorted(bundle['class_labels'])
    for key in ('accuracy', 'f1'):
        assert key in bundle['metrics'], f'{dept} 的 metrics 缺少 {key}'


@pytest.mark.parametrize('dept', SURGERY)
def test_surgery_prediction_gives_plausible_minutes(dept):
    """走完完整推論路徑，並確認還原後的分鐘數落在合理區間。"""
    bundle = load(dept)
    features = {c: bundle['feature_defaults'].get(c, 0) for c in bundle['x_columns']}
    minutes = float(np.expm1(predict(bundle, features)))
    assert np.isfinite(minutes)
    assert 0 < minutes < 24 * 60, f'{dept} 預測 {minutes:.1f} 分鐘，超出合理範圍'


@pytest.mark.parametrize('dept', CLINIC)
def test_clinic_prediction_maps_to_known_label(dept):
    """預測值必須是 class_labels 的合法索引，且該標籤在 label_to_minutes 內。"""
    bundle = load(dept)
    code = int(predict(bundle, CLINIC_SAMPLE))
    assert 0 <= code < len(bundle['class_labels']), \
        f'{dept} 預測索引 {code} 超出 class_labels 範圍'
    label = bundle['class_labels'][code]
    assert isinstance(bundle['label_to_minutes'][label], int)
