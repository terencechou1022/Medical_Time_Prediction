import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 圖內含中文標籤（特徵名、類別名），需指定 CJK 字型避免缺字
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


class Visualizer:
    @staticmethod
    def _save(save_path: str) -> None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'圖表已儲存: {save_path}')

    @staticmethod
    def plot_distribution(data: pd.Series, title: str, xlabel: str, save_path: str, ylabel: str = 'Frequency') -> None:
        plt.figure(figsize=(8, 5))
        plt.hist(data, bins=50)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        Visualizer._save(save_path)

    @staticmethod
    def plot_feature_importance(model, features, title: str, save_path: str, top_n: int = 15) -> None:
        plt.figure(figsize=(8, 6))
        importance = pd.Series(model.feature_importances_, index=features).sort_values()
        importance.tail(top_n).plot(kind='barh')
        plt.title(title)
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        Visualizer._save(save_path)

    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray, title: str, save_path: str, labels='auto') -> None:
        plt.figure(figsize=(6.5, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('Actual Label')
        Visualizer._save(save_path)
