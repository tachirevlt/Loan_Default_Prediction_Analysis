import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, pointbiserialr

def analyze_features(X, y, max_plots=5):
    """
    Phân tích đặc trưng (numeric & categorical) và sự tương quan với target (nhị phân).
    
    Parameters
    ----------
    X : DataFrame
        Feature DataFrame (train set)
    y : Series
        Target (0/1)
    max_plots : int
        Số lượng plot tối đa mỗi loại (tránh vẽ quá nhiều nếu dữ liệu nhiều cột)
    """
    
    # Tách numeric & categorical
    numeric_features = X.select_dtypes(include=['int64', 'float64'])
    categorical_features = X.select_dtypes(include=['object', 'category'])
    
    print("======== 1. Numeric Features vs Target ========")
    for i, col in enumerate(numeric_features.columns[:max_plots]):
        # Boxplot
        plt.figure(figsize=(6,4))
        sns.boxplot(x=y, y=X[col])
        plt.title(f"Boxplot {col} theo Target")
        plt.show()
        
        # Correlation
        corr, p = pointbiserialr(X[col], y)
        print(f"{col}: Point Biserial Corr = {corr:.5f}, p-value = {p:.10f}")
    
    print("\n======== 2. Categorical Features vs Target ========")
    for i, col in enumerate(categorical_features.columns[:max_plots]):
        # Countplot
        plt.figure(figsize=(6,4))
        sns.countplot(x=X[col], hue=y)
        plt.title(f"Phân phối {col} theo Target")
        plt.xticks(rotation=30)
        plt.show()
        
        # Chi-square
        ct = pd.crosstab(X[col], y)
        chi2, p, dof, ex = chi2_contingency(ct)
        print(f"{col}: Chi-square test p-value = {p:.10f}")
    
    print("\n======== 3. Correlation Heatmap (Numeric) ========")
    if len(numeric_features.columns) > 1:
        plt.figure(figsize=(10,8))
        sns.heatmap(numeric_features.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap giữa các Numeric features")
        plt.show()
    else:
        print("Không đủ numeric features để vẽ heatmap.")
