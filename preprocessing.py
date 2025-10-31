import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
import streamlit as st

def event_columns(df):
    df = df.copy()
    if 'Side' in df.columns:
        df.loc[df['Side'] == 2, 'LT_event'] = 1
        df.loc[df['LT_event'].isnull(), 'LT_event'] = 0
        df.loc[df['Side'] == 1, 'RT_event'] = 1
        df.loc[df['RT_event'].isnull(), 'RT_event'] = 0
    return df

def side_columns(df):
    df = df.copy()
    frequencies = ['250', '500', '1000', '2000', '3000', '4000', '8000']
    for freq in frequencies:
        lt_col = f'PTA_LT_AC_{freq}'
        rt_col = f'PTA_RT_AC_{freq}'
        affected_col = f'affected_side_{freq}'
        normal_col = f'normal_side_{freq}'
        if lt_col in df.columns and rt_col in df.columns:
            df.loc[df['LT_event'] == 1, affected_col] = df[lt_col]
            df.loc[(df[affected_col].isnull()) & (df['RT_event'] == 1), affected_col] = df[rt_col]
            df.loc[df['LT_event'] == 0, normal_col] = df[lt_col]
            df.loc[(df[normal_col].isnull()) & (df['RT_event'] == 0), normal_col] = df[rt_col]
    return df

def mean_columns(df):
    df = df.copy()
    req_aff = ['affected_side_500', 'affected_side_1000', 'affected_side_2000', 'affected_side_4000']
    req_norm = ['normal_side_500', 'normal_side_1000', 'normal_side_2000', 'normal_side_4000']
    if all(col in df.columns for col in req_aff):
        df['mean_affected_four'] = (df['affected_side_500'] + df['affected_side_1000'] +
                                    df['affected_side_2000'] + df['affected_side_4000']) / 4
    if all(col in df.columns for col in req_norm):
        df['mean_normal_four'] = (df['normal_side_500'] + df['normal_side_1000'] +
                                  df['normal_side_2000'] + df['normal_side_4000']) / 4
    return df

def assign_hl_severity(df):
    df = df.copy()
    df['HL_severity'] = np.nan
    df.loc[df['mean_affected_four'] <= 25, 'HL_severity'] = 1
    df.loc[(df['HL_severity'].isnull()) & (df['mean_affected_four'] > 25) & (df['mean_affected_four'] <= 40), 'HL_severity'] = 2
    df.loc[(df['HL_severity'].isnull()) & (df['mean_affected_four'] > 40) & (df['mean_affected_four'] <= 60), 'HL_severity'] = 3
    df.loc[(df['HL_severity'].isnull()) & (df['mean_affected_four'] > 60) & (df['mean_affected_four'] <= 80), 'HL_severity'] = 4
    df.loc[(df['HL_severity'].isnull()) & (df['mean_affected_four'] > 80), 'HL_severity'] = 5
    return df

def assign_normal_severity(df):
    df = df.copy()
    df['Normal_severity'] = np.nan
    df.loc[df['mean_normal_four'] <= 25, 'Normal_severity'] = 1
    df.loc[(df['Normal_severity'].isnull()) & (df['mean_normal_four'] > 25) & (df['mean_normal_four'] <= 40), 'Normal_severity'] = 2
    df.loc[(df['Normal_severity'].isnull()) & (df['mean_normal_four'] > 40) & (df['mean_normal_four'] <= 60), 'Normal_severity'] = 3
    df.loc[(df['Normal_severity'].isnull()) & (df['mean_normal_four'] > 60) & (df['mean_normal_four'] <= 80), 'Normal_severity'] = 4
    df.loc[(df['Normal_severity'].isnull()) & (df['mean_normal_four'] > 80), 'Normal_severity'] = 5
    return df

def assign_lab_flags(df):
    df = df.copy()
    required_cols = ["AST", "K", "Na", "Glucose", "Hb", "Cr", "ALT", "WBC", "PLT", "Total_Protein", "Cl"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0
    df["Hyper_AST"]      = np.where(df["AST"]      >= 40, 1, 0)
    df["Hypo_K"]         = np.where(df["K"]        < 3.5, 1, 0)
    df["Hypo_natremia"]  = np.where(df["Na"]       >= 145, 1, 0)
    df["Hyper_natremia"] = np.where(df["Na"]       <= 135, 1, 0)
    df["Hypo_Glucose"]   = np.where(df["Glucose"]  <= 80, 1, 0)
    df["Hyper_Glucose"]  = np.where(df["Glucose"]  >= 200, 1, 0)
    df["Hypo_Hb"]        = np.where(df["Hb"]       < 11, 1, 0)
    df["Hyper_Hb"]       = np.where(df["Hb"]       >= 17, 1, 0)
    df["Hyper_Cr"]       = np.where(df["Cr"]       > 1.3, 1, 0)
    df["Hypo_Cr"]        = np.where(df["Cr"]       < 0.2, 1, 0)
    df["Hyper_ALT"]      = np.where(df["ALT"]      >= 40, 1, 0)
    df["Hyper_WBC"]      = np.where(df["WBC"]      > 10, 1, 0)
    df["Hypo_WBC"]       = np.where(df["WBC"]      < 4, 1, 0)
    df["Hyper_PLT"]      = np.where(df["PLT"]      > 360, 1, 0)
    df["Hypo_PLT"]       = np.where(df["PLT"]      < 165, 1, 0)
    df["Hyper_Protein"]  = np.where(df["Total_Protein"] > 8.1, 1, 0)
    df["Hypo_Protein"]   = np.where(df["Total_Protein"] < 6.2, 1, 0)
    df["Hyper_Cl"]       = np.where(df["Cl"]       > 107, 1, 0)
    df["Hypo_Cl"]        = np.where(df["Cl"]       < 98, 1, 0)
    return df

def load_and_process_data(df):
    try:
        df = df.groupby('ID').head(1)
        df_ids = df[['ID']].copy()
    
        df['ID'] = df['ID'].astype(str).apply(lambda x: '00' + x if len(x) in [5,6] else x)
    
        if 'test_date' in df.columns:
            df = df.dropna(subset=['test_date'])
            df = df[~df['test_date'].str.contains('-00')]
    
        if 'Birth' in df.columns and 'test_date' in df.columns:
            df['Birth'] = df['Birth'].astype(str).str.slice(stop=4).astype(float)
            df['Date'] = df['test_date'].astype(str).str.slice(stop=4).astype(float)
            df['Age'] = df['Date'] - df['Birth']
            df.drop(['Birth','Date','test_date'], axis=1, inplace=True)
    
        df.drop(columns=['HSPTCD', 'ID', 'Name', 'Clinic_date'], inplace=True, errors='ignore')
    
        if 'HL_type' not in df.columns:
            df['HL_type'] = 'irregular'
        df = pd.get_dummies(df, columns=['HL_type'], drop_first=False)
        for col in ['HL_type_descending', 'HL_type_flat', 'HL_type_irregular', 'HL_type_profound']:
            if col not in df.columns:
                df[col] = False
    
        if 'Sex' not in df.columns:
            df['Sex'] = 1
        df['Sex_2.0'] = (df['Sex'] == 2.0).astype(bool)
        df.drop(columns=['Sex'], inplace=True)
    
        df = event_columns(df)
        df = side_columns(df)
        df = mean_columns(df)
        df = assign_hl_severity(df)
        df = assign_normal_severity(df)
    
        pta_cols = [col for col in df.columns if col.startswith("PTA_")]
        extra_cols = ['LT_event', 'RT_event']
        df.drop(columns=pta_cols + extra_cols, inplace=True, errors='ignore')
    
        return df, df_ids
    except Exception as e:
        st.error("Error in load_and_process_data:")
        st.exception(e)
        st.write("Current DataFrame state (columns and head):")
        st.write(df.columns.tolist())
        st.dataframe(df.head())
        raise

def impute_data(df):
    try:
        mice = IterativeImputer(random_state=42)
        df_imputed = mice.fit_transform(df)
        return pd.DataFrame(df_imputed, columns=df.columns)
    except Exception as e:
        st.error("Error in impute_data:")
        st.exception(e)
        st.write("DataFrame before imputation:")
        st.dataframe(df.head())
        raise

def finalize_data(df):
    final_columns = [
        'HL_duration', 'Steroid', 'IT_dexa', 'HBOT', 'HL_severity', 'Age',
        'WBC', 'RBC', 'Hb', 'PLT', 'Neutrophil', 'Lymphocyte', 'AST', 'ALT',
        'BUN', 'Cr', 'Glucose', 'Total_Protein', 'Na', 'K', 'Cl', 'Dx_COM',
        'Dx_SSNHL', 'Dx_Dizziness', 'Dx_Tinnitus', 'Hx_HTN', 'Hx_DM', 'Hx_CRF',
        'Hx_MI', 'Hx_stroke', 'Hx_cancer', 'Hx_others',
        'affected_side_250', 'affected_side_500', 'affected_side_1000', 'affected_side_2000',
        'affected_side_3000', 'affected_side_4000', 'affected_side_8000',
        'normal_side_250', 'normal_side_500', 'normal_side_1000', 'normal_side_2000',
        'normal_side_3000', 'normal_side_4000', 'normal_side_8000',
        'mean_affected_four', 'mean_normal_four', 'Normal_severity',
        'Hyper_Hb', 'Hypo_Hb', 'Hyper_natremia', 'Hypo_natremia',
        'Hyper_WBC', 'Hypo_WBC', 'Hyper_Cr', 'Hypo_Cr', 'Hyper_ALT', 'Hyper_AST',
        'Hyper_K', 'Hypo_K', 'Hyper_PLT', 'Hypo_PLT', 'Hyper_Protein', 'Hypo_Protein',
        'Hyper_Glucose', 'Hypo_Glucose', 'Hyper_Cl', 'Hypo_Cl',
        'HL_type_descending', 'HL_type_flat', 'HL_type_irregular', 'HL_type_profound',
        'Sex_2.0'
    ]
    try:
        df = df.reindex(columns=final_columns, fill_value=0)
    except Exception as e:
        st.error("Error reindexing DataFrame to final_columns:")
        st.exception(e)
        st.write("Columns in current DataFrame:")
        st.write(df.columns.tolist())
        raise

    bool_cols = ['HL_type_descending', 'HL_type_flat', 'HL_type_irregular', 'HL_type_profound', 'Sex_2.0']
    df[bool_cols] = df[bool_cols].astype(bool)
    
    try:
        df = assign_lab_flags(df)
    except Exception as e:
        st.error("Error in assign_lab_flags:")
        st.exception(e)
        st.write("Current DataFrame state:")
        st.dataframe(df.head())
        raise

    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.fillna(0)
    
    return df
