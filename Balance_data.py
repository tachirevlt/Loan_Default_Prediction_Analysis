import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

def balance_data_smote(X_train, y_train, minority_class, majority_class):

    # Chuyển thành DataFrame nếu là numpy array
    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train)
    if isinstance(y_train, np.ndarray):
        y_train = pd.Series(y_train)
    
    # Tính số lượng mẫu của lớp ít ban đầu
    min_samples = len(y_train[y_train == minority_class])
    
    # Áp dụng SMOTE cho lớp ít
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    # Undersampling lớp nhiều (giảm xuống bằng số lượng lớp ít ban đầu)
    rus = RandomUnderSampler(sampling_strategy={majority_class: min_samples}, random_state=42)
    X_balanced, y_balanced = rus.fit_resample(X_train_smote, y_train_smote)
    
    # Trộn ngẫu nhiên
    combined = pd.concat([pd.DataFrame(X_balanced), pd.Series(y_balanced, name='target')], axis=1)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Tách lại X và y
    X_balanced = combined.drop(columns=['target'])
    y_balanced = combined['target']
    
    return X_balanced, y_balanced