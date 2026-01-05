import pandas as pd
import numpy as np
import re
import os
import streamlit as st
from sklearn.impute import KNNImputer
# ====================
# 辅助工具
# ====================
def make_unique(series):
    """处理重名ID"""
    seen = set()
    result = []
    for item in series:
        new_item = item
        counter = 1
        while new_item in seen:
            new_item = f"{item}_{counter}"
            counter += 1
        seen.add(new_item)
        result.append(new_item)
    return result

# ====================
# 解析 MetDNA 文件
# ====================
def parse_metdna_file(file_buffer, file_name, file_type='csv'):
    try:
        if file_type == 'csv':
            try:
                df = pd.read_csv(file_buffer, engine='pyarrow')
            except:
                file_buffer.seek(0)
                df = pd.read_csv(file_buffer)
        else:
            df = pd.read_excel(file_buffer)
    except Exception as e:
        return None, None, f"读取失败: {str(e)}"

    # 1. 识别样本数据列
    known_meta_cols = {
        'peak_name', 'mz', 'rt', 'id', 'id_zhulab', 'name', 'formula', 
        'confidence_level', 'smiles', 'inchikey', 'isotope', 'adduct', 
        'total_score', 'mz_error', 'rt_error_abs', 'rt_error_rela', 
        'ms2_score', 'iden_score', 'iden_type', 'peak_group_id', 
        'base_peak', 'num_peaks', 'cons_formula_pred', 'id_kegg', 
        'id_hmdb', 'id_metacyc', 'stereo_isomer_id', 'stereo_isomer_name'
    }
    
    potential_cols = [c for c in df.columns if c not in known_meta_cols]
    sample_cols = []
    if potential_cols:
        subset = df[potential_cols].head(5)
        is_numeric = subset.apply(lambda x: pd.to_numeric(x, errors='coerce').notna().all())
        sample_cols = is_numeric[is_numeric].index.tolist()
            
    if not sample_cols:
        return None, None, "未找到样本数据列"

    # 2. 构建元数据
    file_tag = os.path.splitext(os.path.basename(file_name))[0]
    clean_tag = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', file_tag)
    
    if 'name' not in df.columns: df['name'] = ""
    if 'confidence_level' not in df.columns: df['confidence_level'] = 'Unknown'
    
    df['name'] = df['name'].fillna("").astype(str)
    mask_annotated = (df['name'] != "") & (df['name'].str.lower() != "nan")
    clean_names = df['name'].str.split(';', expand=True)[0]
    
    mz_str = df['mz'].map('{:.4f}'.format).astype(str) if 'mz' in df.columns else ""
    rt_str = df['rt'].map('{:.2f}'.format).astype(str) if 'rt' in df.columns else ""
    unannotated_ids = "m/z" + mz_str + "_RT" + rt_str + "_" + clean_tag
    
    final_ids = np.where(mask_annotated, clean_names + "_" + clean_tag, unannotated_ids)
    final_ids = make_unique(final_ids)

    meta_df = pd.DataFrame({
        "Metabolite_ID": final_ids,
        "Original_Name": df['name'],
        "Clean_Name": np.where(mask_annotated, clean_names, final_ids),
        "Confidence_Level": df['confidence_level'],
        "Is_Annotated": mask_annotated,
        "Source_File": clean_tag
    })
    meta_df.set_index('Metabolite_ID', inplace=True)
    
    # 3. 提取数据矩阵
    df_data = df[sample_cols].copy()
    df_data.index = meta_df.index
    df_transposed = df_data.T
    
    df_transposed.reset_index(inplace=True)
    df_transposed.rename(columns={'index': 'SampleID'}, inplace=True)
    
    # 写入来源标记
    df_transposed['Source_Files'] = clean_tag
    
    # 默认Group (占位)
    df_transposed['Group'] = df_transposed['SampleID'].astype(str).str.extract(r'([^\d]+)')[0].str.strip('._-').fillna("Unknown")
    
    return df_transposed, meta_df, None

# ====================
# 合并函数
# ====================
def merge_multiple_dfs(results_list):
    if not results_list: return None, None, "无数据"
    
    best_features = {}
    sample_source_map = {}
    
    # 1. 记录来源 & 选最佳特征
    for file_idx, (df, meta, fname) in enumerate(results_list):
        if 'SampleID' in df.columns and 'Source_Files' in df.columns:
            current_tag = df['Source_Files'].iloc[0]
            for sid in df['SampleID']:
                if sid not in sample_source_map: sample_source_map[sid] = set()
                sample_source_map[sid].add(current_tag)
        
        numeric_df = df.select_dtypes(include=[np.number])
        intensities = numeric_df.sum(axis=0)
        
        for feat_id in numeric_df.columns:
            try:
                clean_name = meta.loc[feat_id, 'Clean_Name']
            except KeyError: continue
            curr_score = intensities.get(feat_id, 0)
            
            if clean_name not in best_features:
                best_features[clean_name] = (file_idx, feat_id, curr_score)
            else:
                prev_idx, prev_id, prev_score = best_features[clean_name]
                if curr_score > prev_score:
                    best_features[clean_name] = (file_idx, feat_id, curr_score)
    
    # 2. 拼接
    files_features_to_keep = {i: [] for i in range(len(results_list))}
    for c_name, (f_idx, f_id, score) in best_features.items():
        files_features_to_keep[f_idx].append(f_id)
        
    dfs_to_concat = []
    base_group_series = None
    
    for i, (df, meta, fname) in enumerate(results_list):
        if 'SampleID' in df.columns: df = df.set_index('SampleID')
        # 移除辅助列防止冲突
        cols_to_drop = [c for c in ['Group', 'Source_Files'] if c in df.columns]
        
        if 'Group' in df.columns and base_group_series is None:
            base_group_series = df['Group']
            
        df_clean = df.drop(columns=cols_to_drop, errors='ignore')
        cols_to_keep = files_features_to_keep[i]
        valid_cols = [c for c in cols_to_keep if c in df_clean.columns]
        dfs_to_concat.append(df_clean[valid_cols])
        
    try:
        full_df = pd.concat(dfs_to_concat, axis=1, join='outer')
    except Exception as e:
        return None, None, f"合并出错: {str(e)}"
    
    full_df.fillna(0, inplace=True)
    
    # 3. 还原辅助列
    if base_group_series is not None:
        aligned_group = base_group_series.reindex(full_df.index).fillna('Unknown')
        full_df.insert(0, 'Group', aligned_group)
    else:
        full_df.insert(0, 'Group', 'Unknown')
        
    full_df.reset_index(inplace=True)
    full_df.rename(columns={'index': 'SampleID'}, inplace=True)
    
    def get_combined_source(sid):
        sources = sample_source_map.get(sid, set())
        return "; ".join(sorted(list(sources)))
    
    full_df['Source_Files'] = full_df['SampleID'].apply(get_combined_source)
    
    final_ids = [fid for f_list in files_features_to_keep.values() for fid in f_list]
    all_meta = pd.concat([res[1] for res in results_list])
    merged_meta = all_meta.loc[final_ids]
    
    return full_df, merged_meta, None

# ====================
# 信息对齐 (支持指定列)
# ====================
def align_sample_info(data_df, info_df, sample_col_name=None):
    """根据指定列名对齐样本信息"""
    target_col = None
    
    # 确定连接键
    if sample_col_name and sample_col_name in info_df.columns:
        target_col = sample_col_name
    else:
        # 自动猜
        cols_lower = [c.lower() for c in info_df.columns]
        candidates = ['sample', 'sampleid', 'sample.name', 'name', 'id']
        for cand in candidates:
            if cand in cols_lower:
                target_col = info_df.columns[cols_lower.index(cand)]
                break
        if not target_col: target_col = info_df.columns[0]
        
    def normalize(s): 
        return re.sub(r'[^a-zA-Z0-9]', '', str(s)).lower()
    
    info_map = {}
    for idx, row in info_df.iterrows():
        key = normalize(row[target_col])
        info_map[key] = row
        
    aligned_data = []
    for sid in data_df['SampleID']:
        key = normalize(sid)
        if key in info_map: aligned_data.append(info_map[key])
        else: aligned_data.append(pd.Series([np.nan]*len(info_df.columns), index=info_df.columns))
            
    aligned_df = pd.DataFrame(aligned_data)
    aligned_df.index = data_df.index 
    return aligned_df

# ====================
# 清洗与归一化
# ====================
def pqn_normalization(df):
    reference = df.median(axis=0)
    reference[reference <= 0] = 1e-6
    quotients = df.div(reference, axis=1)
    dilution_factors = quotients.median(axis=1)
    return df.div(dilution_factors, axis=0)

@st.cache_data(show_spinner=False)
def data_cleaning_pipeline(df, group_col, missing_thresh=0.5, impute_method='min', 
                           norm_method='None', log_transform=True, scale_method='None'):
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # 保护非特征列
    exclude_cols = [group_col, 'SampleID', 'Source_Files']
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
    
    meta_cols = [c for c in df.columns if c not in numeric_cols]
    data_df = df[numeric_cols].copy()
    meta_df = df[meta_cols].copy()
    
    # 1. 过滤
    missing_ratio = data_df.isnull().mean()
    cols_to_keep = missing_ratio[missing_ratio <= missing_thresh].index
    data_df = data_df[cols_to_keep]
    
    # 2. 填充
    if data_df.isnull().sum().sum() > 0:
        if impute_method == 'min': data_df = data_df.fillna(data_df.min() * 0.5)
        elif impute_method == 'mean': data_df = data_df.fillna(data_df.mean())
        elif impute_method == 'median': data_df = data_df.fillna(data_df.median())
        elif impute_method == 'KNN':
            imputer = KNNImputer(n_neighbors=5)
            filled_vals = imputer.fit_transform(data_df)
            data_df = pd.DataFrame(filled_vals, columns=data_df.columns, index=data_df.index)
        elif impute_method == 'zero': data_df = data_df.fillna(0)
        data_df = data_df.fillna(0)

    # 3. 归一化
    if norm_method == 'Sum':
        data_df = data_df.div(data_df.sum(axis=1), axis=0) * data_df.sum(axis=1).mean()
    elif norm_method == 'Median':
        data_df = data_df.div(data_df.median(axis=1), axis=0) * data_df.median(axis=1).mean()
    elif norm_method == 'PQN':
        data_df = pqn_normalization(data_df)

    # 4. Log
    if log_transform:
        if (data_df <= 0).any().any(): data_df = np.log2(data_df + 1)
        else: data_df = np.log2(data_df)

    # 5. Scale
    if scale_method != 'None':
        mean = data_df.mean()
        std = data_df.std()
        if scale_method == 'Auto': data_df = (data_df - mean) / std
        elif scale_method == 'Pareto': data_df = (data_df - mean) / np.sqrt(std)

    # 6. Var
    var_mask = data_df.var() > 1e-9
    data_df = data_df.loc[:, var_mask]
    
    return pd.concat([meta_df, data_df], axis=1), data_df.columns.tolist()
