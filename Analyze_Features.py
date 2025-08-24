import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, pointbiserialr

def analyze_features(X, y, max_plots=10):
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
    

    numeric_features = X.select_dtypes(include=[np.number]).columns
    categorical_features = X.select_dtypes(include=[object]).columns
    
    print("======== 1. Numeric Features vs Target ========")
    numeric_correlations = []
    for col in numeric_features[:max_plots]:
        corr, p = pointbiserialr(X[col], y)
        numeric_correlations.append([col, f"{corr:.5f}", f"{p:.10f}" if p < 1e-308 else f"< 1e-308 (underflow)"])
    print(pd.DataFrame(numeric_correlations, columns=["", "corr", "p-value"]).to_string(index=False))
    
    print("\n======== 2. Categorical Features vs Target ========")
    categorical_correlations = []
    for col in categorical_features[:max_plots]:
        ct = pd.crosstab(X[col], y)
        chi2, p, dof, ex = chi2_contingency(ct)
        categorical_correlations.append([col, f"{p:.10f}"])
    print(pd.DataFrame(categorical_correlations, columns=["", "p-value"]).to_string(index=False))
    

    print("\n======== 3. Visualization ========")
    n_numeric = min(len(numeric_features), max_plots)
    n_categorical = min(len(categorical_features), max_plots)
    n_plots = n_numeric + n_categorical
    
    fig, axes = plt.subplots(nrows=4, ncols=max(1, (n_plots + 3) // 4), figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, col in enumerate(numeric_features[:max_plots]):
        sns.histplot(data=X, x=col, ax=axes[idx])
        axes[idx].set_title(col)
    
    for idx, col in enumerate(categorical_features[:max_plots]):
        sns.histplot(data=X, x=col, ax=axes[n_numeric + idx] if n_numeric + idx < len(axes) else axes[0])
        axes[n_numeric + idx].set_title(col) if n_numeric + idx < len(axes) else axes[0].set_title(col)
    
    for idx in range(n_plots, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.show()