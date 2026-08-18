import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class Visualizer:
    @staticmethod
    def plot_distribution(data: pd.Series, title: str, xlabel: str, ylabel: str = 'Frequency') -> None:
        plt.hist(data, bins=50)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    @staticmethod
    def plot_feature_importance(model, features, title: str) -> None:
        plt.figure(figsize=(6, 4))
        pd.Series(model.feature_importances_, index=features).sort_values().plot(kind='barh')
        plt.title(title)
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.show()

    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray, title: str) -> None:
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('Actual Label')
        plt.show()
