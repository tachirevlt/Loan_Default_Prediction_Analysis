import pandas as pd
import numpy as np
from scipy.stats import pearsonr, chi2_contingency, f_oneway, kruskal
from itertools import combinations

def cross_correlation_table(df, target=None):
    """
    Tạo bảng tương quan chéo giữa các biến trong DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame chứa dữ liệu.
    target : str, optional
        Tên cột target (nếu có) để so sánh với các biến khác.
        
    Returns:
    --------
    result_df : pandas.DataFrame
        Bảng chứa cặp biến, loại kiểm định, giá trị thống kê, và p-value.
    """

    numeric_features = df.select_dtypes(include=[np.number]).columns
    categorical_features = df.select_dtypes(include=[object]).columns

    results = []
    
    # 1. Tương quan chéo giữa numeric (Pearson)
    for col1, col2 in combinations(numeric_features, 2):
        corr, p_value = pearsonr(df[col1], df[col2])
        results.append([f"{col1} vs {col2}", "Pearson", corr, p_value])
    
    # 2. Tương quan chéo giữa categorical (Chi-square)
    for col1, col2 in combinations(categorical_features, 2):
        contingency_table = pd.crosstab(df[col1], df[col2])
        chi2, p_value, dof, ex = chi2_contingency(contingency_table)
        results.append([f"{col1} vs {col2}", "Chi-square", chi2, p_value])
    
    # 3. Tương quan chéo giữa numeric và categorical (ANOVA hoặc Kruskal)
    for num_col in numeric_features:
        for cat_col in categorical_features:
            groups = [df[num_col][df[cat_col] == val].dropna() for val in df[cat_col].unique()]
            if len(groups) > 1:  # Đảm bảo có đủ nhóm
                # Kiểm tra phân phối normal (có thể bỏ qua để đơn giản)
                stat, p_value = f_oneway(*groups) if len(groups) < 10 else kruskal(*groups)
                test_type = "ANOVA" if len(groups) < 10 else "Kruskal"
                results.append([f"{num_col} vs {cat_col}", test_type, stat, p_value])
    
    # 4. Tương quan với target (nếu có)
    if target and target in df.columns:
        if target in numeric_features:
            for col in numeric_features.drop(target):
                corr, p_value = pearsonr(df[col], df[target])
                results.append([f"{col} vs {target}", "Pearson", corr, p_value])
        elif target in categorical_features:
            for col in categorical_features.drop(target):
                contingency_table = pd.crosstab(df[col], df[target])
                chi2, p_value, dof, ex = chi2_contingency(contingency_table)
                results.append([f"{col} vs {target}", "Chi-square", chi2, p_value])
            for col in numeric_features:
                groups = [df[col][df[target] == val].dropna() for val in df[target].unique()]
                stat, p_value = f_oneway(*groups) if len(groups) < 10 else kruskal(*groups)
                test_type = "ANOVA" if len(groups) < 10 else "Kruskal"
                results.append([f"{col} vs {target}", test_type, stat, p_value])
    
    result_df = pd.DataFrame(results, columns=["Pair", "Test Type", "Statistic", "p-value"])
    result_df['p-value'] = result_df['p-value'].apply(lambda x: f"{x:.4f}" if x >= 1e-10 else "< 1e-10")
    
    return result_df
