import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from joblib import Parallel, delayed
import streamlit as st

@st.cache_data(show_spinner=False)
def serrf_normalization(df_data, df_meta, run_order_col, sample_type_col, qc_label, n_trees=100):
    """SERRF Batch Correction"""
    # 确保索引对齐
    common_idx = df_data.index.intersection(df_meta.index)
    if len(common_idx) < len(df_data): return None, "索引不匹配"
    
    X_raw = df_data.loc[common_idx].copy()
    meta = df_meta.loc[common_idx].copy()
    
    # 校验 Order 列
    try:
        meta[run_order_col] = pd.to_numeric(meta[run_order_col], errors='coerce')
        valid = meta[run_order_col].notna()
        X_raw = X_raw[valid]
        meta = meta[valid]
    except: return None, "Order列包含非数字"
    
    # 校验 QC 数量
    qc_mask = meta[sample_type_col] == qc_label
    if qc_mask.sum() < 3: return None, f"QC样本不足 ({qc_mask.sum()}<3)"

    run_orders = meta[[run_order_col]].values
    
    # 并行处理函数
    def process_metabolite(y_all, is_qc, run_orders):
        valid_val = (y_all > 0) & (~np.isnan(y_all))
        valid_qc = is_qc & valid_val
        if valid_qc.sum() < 3: return y_all
        
        y_qc = y_all[valid_qc]
        X_qc = run_orders[valid_qc]
        # 对数空间拟合
        y_qc_log = np.log1p(y_qc)
        
        rf = RandomForestRegressor(n_estimators=n_trees, min_samples_leaf=3, max_features='sqrt', random_state=42, n_jobs=1)
        rf.fit(X_qc, y_qc_log)
        
        y_pred = np.expm1(rf.predict(run_orders))
        qc_target = np.median(y_qc)
        min_pred = qc_target * 0.1
        y_pred = np.maximum(y_pred, min_pred)
        
        return y_all * (qc_target / y_pred)

    # 4核并行
    results = Parallel(n_jobs=4, backend='threading')(
        delayed(process_metabolite)(X_raw[col].values, qc_mask.values, run_orders)
        for col in X_raw.columns
    )
    
    df_corrected = pd.DataFrame(np.array(results).T, index=X_raw.index, columns=X_raw.columns)
    
    def get_rsd(df, mask):
        qc = df.loc[mask]
        qc = qc.loc[:, (qc != 0).any(axis=0)]
        if qc.shape[1] == 0: return 0.0
        return (qc.std() / qc.mean() * 100).median()
    
    return df_corrected, {
        "RSD_Before": get_rsd(X_raw, qc_mask), 
        "RSD_After": get_rsd(df_corrected, qc_mask)
    }
