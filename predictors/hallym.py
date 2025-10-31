import streamlit as st
import pandas as pd
import numpy as np
import joblib
from preprocessing import load_and_process_data, impute_data, finalize_data

class HallymPredictor:
    def __init__(self):
        # main.py에서 Google Drive를 통해 모델, 스케일러, 정확도가 주입됨
        self.lgbm_model = None
        self.xgb_model = None
        self.scaler = None
        self.lgbm_acc = None
        self.xgb_acc = None

    def predict_outcome(self, df):
        try:
            # ===== 데이터 전처리 =====
            df_processed, df_ids = load_and_process_data(df)
            df_imputed = impute_data(df_processed)
            df_final = finalize_data(df_imputed)

            # ===== 연속형 변수 목록 =====
            num = [
                'HL_duration', 'WBC', 'RBC', 'Hb', 'PLT', 'Neutrophil', 'Lymphocyte',
                'AST', 'ALT', 'BUN', 'Cr', 'Glucose', 'Total_Protein', 'Na', 'K', 'Cl',
                'Age', 'HL_severity', 'Normal_severity',
                'affected_side_250', 'affected_side_500', 'affected_side_1000',
                'affected_side_2000', 'affected_side_3000', 'affected_side_4000',
                'affected_side_8000', 'normal_side_250', 'normal_side_500',
                'normal_side_1000', 'normal_side_2000', 'normal_side_3000',
                'normal_side_4000', 'normal_side_8000', 'mean_affected_four',
                'mean_normal_four'
            ]

            df_cont = df_final[num]
            df_other = df_final.drop(columns=num)

            # ===== 스케일러 적용 =====
            if self.scaler:
                df_cont_scaled = self.scaler.transform(df_cont)
                df_cont_scaled = pd.DataFrame(df_cont_scaled, columns=num, index=df_other.index)
                df_final_scaled = df_other.join(df_cont_scaled)
            else:
                df_final_scaled = df_final

            # ===== 모델 주입 확인 =====
            if self.lgbm_model is None or self.xgb_model is None:
                st.error("모델이 로드되지 않았습니다. main.py에서 모델을 주입해야 합니다.")
                return [None] * 10

            # ===== 입력 열 정렬 =====
            lgbm_columns = self.lgbm_model.feature_name_
            xgb_columns = self.xgb_model.feature_names_in_

            df_lgbm = df_final_scaled.reindex(columns=lgbm_columns, fill_value=0)
            df_xgb = df_final_scaled.reindex(columns=xgb_columns, fill_value=0)

            # ===== 예측 =====
            lgbm_pred = self.lgbm_model.predict(df_lgbm)
            lgbm_pred_prob = 1 - self.lgbm_model.predict_proba(df_lgbm)[:, 1]

            xgb_pred = self.xgb_model.predict(df_xgb)
            xgb_pred_prob = 1 - self.xgb_model.predict_proba(df_xgb)[:, 1]

            # ===== 결과 반환 =====
            return (
                lgbm_pred, lgbm_pred_prob, xgb_pred, xgb_pred_prob,
                df_lgbm, df_xgb, df_ids,
                self.lgbm_model, self.xgb_model, self.lgbm_acc, self.xgb_acc
            )

        except Exception as e:
            st.error("예측 중 오류 발생")
            st.exception(e)
            return [None] * 10


def get_predictor():
    return HallymPredictor()
