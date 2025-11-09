import streamlit as st
# 🚨 캐시 완전 초기화
st.cache_resource.clear()

# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os, sys, io, re, joblib, tempfile, json, datetime, traceback, shap, importlib
import matplotlib
matplotlib.rc('font', family='Malgun Gothic')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# ======================
# 🔹 Google Drive 설정
# ======================
FOLDER_ID = '1rTMoyzj1qxc8ET5648XvF0E-3oN46lel'

# ======================
# 🔹 Google Sheets 설정
# ======================
SPREADSHEET_ID = '17Y24_hFUJSXXTdHVbL6doo6A2pnvmsRDCG98Ihvenlw'

@st.cache_resource
def get_sheets_client():
    """Google Sheets 클라이언트 생성"""
    try:
        if 'google' in st.secrets:
            service_account_info = dict(st.secrets['google'])
        else:
            st.error("Google Sheets 서비스 계정 정보가 설정되지 않았습니다.")
            return None
        
        scope = [
            'https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive'
        ]
        credentials = ServiceAccountCredentials.from_json_keyfile_dict(
            service_account_info, scope
        )
        client = gspread.authorize(credentials)
        return client
    except Exception as e:
        st.error(f"Google Sheets 클라이언트 초기화 실패: {str(e)}")
        return None

def save_to_sheets(user_data):
    """사용자 입력 데이터를 Google Sheets에 저장"""
    try:
        client = get_sheets_client()
        if not client:
            return False
        
        # 스프레드시트 열기
        sheet = client.open_by_key(SPREADSHEET_ID).sheet1
        
        # 데이터를 리스트로 변환 (헤더 순서와 동일하게)
        row_data = [
            user_data.get('timestamp', ''),
            user_data.get('language', ''),
            user_data.get('hospital', ''),
            user_data.get('id', ''),
            user_data.get('birth', ''),
            user_data.get('sex', ''),
            user_data.get('name', ''),
            user_data.get('hsptcd', ''),
            user_data.get('side', ''),
            user_data.get('hl_duration', ''),
            user_data.get('clinic_date', ''),
            user_data.get('steroid_treatment', ''),
            user_data.get('it_dexa_treatment', ''),
            user_data.get('hyperbaric_treatment', ''),
            user_data.get('pta_rt_ac_250', ''),
            user_data.get('pta_rt_ac_500', ''),
            user_data.get('pta_rt_ac_1000', ''),
            user_data.get('pta_rt_ac_2000', ''),
            user_data.get('pta_rt_ac_3000', ''),
            user_data.get('pta_rt_ac_4000', ''),
            user_data.get('pta_rt_ac_8000', ''),
            user_data.get('pta_lt_ac_250', ''),
            user_data.get('pta_lt_ac_500', ''),
            user_data.get('pta_lt_ac_1000', ''),
            user_data.get('pta_lt_ac_2000', ''),
            user_data.get('pta_lt_ac_3000', ''),
            user_data.get('pta_lt_ac_4000', ''),
            user_data.get('pta_lt_ac_8000', ''),
            user_data.get('wbc', ''),
            user_data.get('rbc', ''),
            user_data.get('hb', ''),
            user_data.get('plt', ''),
            user_data.get('neutrophil', ''),
            user_data.get('lymphocyte', ''),
            user_data.get('ast', ''),
            user_data.get('alt', ''),
            user_data.get('bun', ''),
            user_data.get('cr', ''),
            user_data.get('glucose', ''),
            user_data.get('total_protein', ''),
            user_data.get('na', ''),
            user_data.get('k', ''),
            user_data.get('cl', ''),
            user_data.get('dx_com', ''),
            user_data.get('dx_ssnhl', ''),
            user_data.get('dx_dizziness', ''),
            user_data.get('dx_tinnitus', ''),
            user_data.get('hx_htn', ''),
            user_data.get('hx_dm', ''),
            user_data.get('hx_crf', ''),
            user_data.get('hx_mi', ''),
            user_data.get('hx_stroke', ''),
            user_data.get('hx_cancer', ''),
            user_data.get('hx_others', ''),
            user_data.get('prediction', ''),
            user_data.get('probability', '')
        ]
        
        # 시트에 행 추가
        sheet.append_row(row_data)
        return True
        
    except Exception as e:
        st.error(f"❌ 데이터 저장 실패: {str(e)}")
        import traceback
        st.error(f"상세 에러: {traceback.format_exc()}")
        return False

@st.cache_resource
def get_drive_service():
    """Google Drive 서비스 객체 생성"""
    try:
        if 'google' in st.secrets:
            service_account_info = st.secrets['google']
        else:
            st.error("Google Drive 서비스 계정 정보가 설정되지 않았습니다.")
            return None
        
        credentials = service_account.Credentials.from_service_account_info(
            service_account_info,
            scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        return build('drive', 'v3', credentials=credentials)
    except Exception as e:
        st.error(f"Google Drive 서비스 초기화 실패: {str(e)}")
        return None

@st.cache_data
def download_file_from_drive(file_name):
    """Google Drive에서 지정된 파일을 다운로드 (predictors, models 폴더 포함)"""
    service = get_drive_service()
    if not service:
        return None

    def find_file_recursive(folder_id, target_name):
        """폴더 전체를 재귀 탐색 (predictors/, models/ 모두 지원)"""
        try:
            # 현재 폴더에서 파일 검색
            query = f"'{folder_id}' in parents and name='{target_name}' and trashed=false"
            results = service.files().list(q=query, fields="files(id, name, mimeType)").execute()
            files = results.get("files", [])
            if files:
                return files[0]["id"]

            # 하위 폴더 검색
            subfolders_query = f"'{folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
            subfolders = service.files().list(q=subfolders_query, fields="files(id, name)").execute().get("files", [])

            for sub in subfolders:
                found = find_file_recursive(sub["id"], target_name)
                if found:
                    return found

            return None
        except Exception as e:
            st.warning(f"폴더 탐색 중 오류 발생 ({target_name}): {e}")
            return None

    # 🔍 실제 탐색 시작
    file_id = find_file_recursive(FOLDER_ID, os.path.basename(file_name))

    if not file_id:
        st.warning(f"❌ Google Drive에서 {file_name}을(를) 찾을 수 없습니다.")
        return None

    # ✅ 파일 다운로드
    try:
        request = service.files().get_media(fileId=file_id)
        file_data = io.BytesIO()
        downloader = MediaIoBaseDownload(file_data, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        file_data.seek(0)
        return file_data
    except Exception as e:
        st.error(f"📁 {file_name} 다운로드 실패: {str(e)}")
        return None


# ======================
# 🔹 predictor 모듈 로드
# ======================
@st.cache_resource
def load_predictor_modules():
    predictor_files = [
        'predictors/all.py', 'predictors/wonju.py', 'predictors/sev.py',
        'predictors/hallym.py', 'predictors/jeju.py', 'predictors/hagen.py'
    ]
    temp_dir = tempfile.mkdtemp()
    predictors_dir = os.path.join(temp_dir, 'predictors')
    os.makedirs(predictors_dir, exist_ok=True)

    with open(os.path.join(predictors_dir, '__init__.py'), 'w') as f:
        f.write('')

    for file_path in predictor_files:
        content = download_file_from_drive(file_path)
        if content:
            with open(os.path.join(predictors_dir, os.path.basename(file_path)), 'wb') as f:
                f.write(content.read())
        else:
            st.warning(f"⚠️ {file_path} 다운로드 실패")

    if 'predictors' in sys.modules:
        del sys.modules['predictors']

    sys.path = [p for p in sys.path if 'predictors' not in p]
    parent_dir = os.path.dirname(predictors_dir)
    sys.path.insert(0, parent_dir)
    sys.path.insert(0, predictors_dir)

    return predictors_dir

# ======================
# 🔹 preprocessing / translation
# ======================
@st.cache_resource
def load_preprocessing_and_translation():
    files = ['preprocessing.py', 'translate_texts.py']
    temp_dir = tempfile.mkdtemp()
    if temp_dir not in sys.path:
        sys.path.insert(0, temp_dir)

    for f in files:
        content = download_file_from_drive(f)
        if content:
            with open(os.path.join(temp_dir, f), 'wb') as w:
                w.write(content.read())

    try:
        from preprocessing import load_and_process_data, impute_data, finalize_data
        from translate_texts import texts_ko, texts_en_us, texts_ja, texts_zh, texts_es, texts_de, texts_hi, texts_ar
        return True
    except Exception as e:
        st.error(f"모듈 로드 실패: {str(e)}")
        return False

# ======================
# 🔹 모델 파일 로드 함수
# ======================
@st.cache_resource
def load_models_from_drive():
    """Google Drive에서 모델 파일 로드"""
    model_files = {
        "all": {
            "lgbm": "models/all_lightgbm_model.joblib",
            "xgb": "models/all_xgboost_model.joblib",
            "scaler": "models/all_minmax_scaler.joblib"
        },
        "wonju": {
            "lgbm": "models/ys_lightgbm_model.joblib",
            "xgb": "models/ys_xgboost_model.joblib",
            "scaler": "models/ys_minmax_scaler.joblib"
        },
        "sev": {
            "lgbm": "models/sev_lightgbm_model.joblib",
            "xgb": "models/sev_xgboost_model.joblib",
            "scaler": "models/sev_minmax_scaler.joblib"
        },
        "hallym": {
            "lgbm": "models/hallym_lightgbm_model.joblib",
            "xgb": "models/hallym_xgboost_model.joblib",
            "scaler": "models/hallym_minmax_scaler.joblib"
        },
        "jeju": {
            "lgbm": "models/jeju_lightgbm_model.joblib",
            "xgb": "models/jeju_xgboost_model.joblib",
            "scaler": "models/jeju_minmax_scaler.joblib"
        },
        "hagen": {
            "lgbm": "models/hagen_lightgbm_model.joblib",
            "xgb": "models/hagen_xgboost_model.joblib",
            "scaler": "models/hagen_minmax_scaler.joblib"
        }
    }

    loaded_models = {}

    for hospital, paths in model_files.items():
        loaded_models[hospital] = {}
        for model_type, path in paths.items():
            try:
                content = download_file_from_drive(path)
                if content:
                    tmp = tempfile.NamedTemporaryFile(delete=False)
                    tmp.write(content.read())
                    tmp.close()
                    loaded_models[hospital][model_type] = joblib.load(tmp.name)
                else:
                    st.warning(f"⚠️ {hospital} {model_type} 모델 없음")
            except Exception as e:
                st.error(f"❌ {hospital} {model_type} 로드 실패: {e}")

    return loaded_models

# ======================
# 🔹 메인 실행 (파일 로드)
# ======================
with st.spinner("Google Drive에서 파일을 로드하는 중..."):
    modules_loaded = load_preprocessing_and_translation()
    if not modules_loaded:
        st.error("필수 모듈 로드 실패")
        st.stop()

    predictor_dir = load_predictor_modules()

    # ✅ predictors import (강제 캐시 초기화 + 디버그)
    if 'predictors' in sys.modules:
        del sys.modules['predictors']
    sys.path.insert(0, predictor_dir)

    try:
        predictors_all = importlib.import_module("predictors.all")
    except ModuleNotFoundError as e:
        st.warning(f"⚠️ Predictor 모듈 로드 실패: {e}")
        predictors_all = None

    # ✅ 모델 로드
    models = load_models_from_drive()

# 이후의 UI / 예측 파트는 그대로 유지


# 이제 import 가능
from preprocessing import load_and_process_data, impute_data, finalize_data
from translate_texts import texts_ko, texts_en_us, texts_ja, texts_zh, texts_es, texts_de, texts_hi, texts_ar

# ==== 메인 UI 디자인
st.markdown("""
<style>
/* ===== 1. 메인 제목 스타일 ===== */
.main-title {
    text-align: center;
    font-size: 4em;
    color: #0077B6;
    font-weight: 700;
    margin-bottom: 0.5em;
    white-space: nowrap;
}

/* ===== 2. 본문 최대 폭 넓히기 ===== */
.block-container {
    max-width: 1200px;
    padding-left: 5rem;
    padding-right: 5rem;
}

/* ===== 3. 예측 버튼 스타일 ===== */
div.stButton > button:first-child {
    background-color: #0077B6;
    color: white;
    font-size: 18px;
    padding: 0.6em 1.2em;
    border-radius: 6px;
}

/* ===== 4. 사이드바 이미지 여백 제거 ===== */
[data-testid="stSidebar"] img {
    margin-top: 0px;
}

/* ===== 5. 사이드바 전체 위 여백 제거 ===== */
[data-testid="stSidebar"] .block-container {
    padding-top: 1rem;
    padding-left: 1rem;
    padding-right: 1rem;
}
[data-testid="stSidebar"] {
    overflow-x: hidden;
}
            
            
</style>

<!-- 메인 제목 -->
<div class="main-title">Prediction Model for Prognosis of SSNHL</div>
""", unsafe_allow_html=True)

# ===== 경로 설정
BASE_DIR = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
sys.path.append(BASE_DIR)

lang_options = {
    "한국어": "ko",
    "English": "en-us",
    "日本語" : "ja",
    "中文": "zh",
    "Español": "es",
    "Deutsch": "de",
    "हिन्दी": "hi",
    "العربية": "ar",
}

with st.sidebar:
    # 🌐 언어 선택 (사이드바 최상단)
    lang_choice_label = st.sidebar.selectbox("🌐 Translate", list(lang_options.keys()), index=0)
    lang_code = lang_options[lang_choice_label]

    # 언어에 따라 텍스트 사전 선택
    if lang_code == "ko":
        texts = texts_ko
    elif lang_code == "en-us":
        texts = texts_en_us
    elif lang_code == "ja":
        texts = texts_ja
    elif lang_code == "zh":
        texts = texts_zh
    elif lang_code == "es":
        texts = texts_es
    elif lang_code == "de":
        texts = texts_de
    elif lang_code == "hi":
        texts = texts_hi
    elif lang_code == "ar":
        texts = texts_ar
    else:
        texts = texts_ko  # 기본값

    # 로고 - Google Drive에서 로드
    logo_content = download_file_from_drive("ON AIR.jpg")
    if logo_content:
        st.image(logo_content, use_column_width=True)

# ===== 병원 선택
hospital_modules = {
    texts["전체 병원"]: "predictors.all",
    texts["원주세브란스기독병원"]: "predictors.wonju",
    texts["신촌-강남세브란스병원"]: "predictors.sev",
    texts["한림대학교 강남성심병원"]: "predictors.hallym",
    texts["제주대학병원"]: "predictors.jeju",
    texts["독일하겐병원"]: "predictors.hagen",
}

st.sidebar.title(f"📋 {texts['병원 선택']}")
selected_hospital = st.sidebar.selectbox("", list(hospital_modules.keys()))

# predictor 모듈 import 및 모델 설정
try:
    predictor = importlib.import_module(hospital_modules[selected_hospital]).get_predictor()
    
    # 해당 병원의 모델 설정
    hospital_key = hospital_modules[selected_hospital].split('.')[-1]
    if hospital_key in models and 'lgbm' in models[hospital_key] and 'xgb' in models[hospital_key]:
        predictor.lgbm_model = models[hospital_key]['lgbm']
        predictor.xgb_model = models[hospital_key]['xgb']
except Exception as e:
    st.error(f"Predictor 로드 실패: {str(e)}")
    st.stop()

# 입력값 수집
pta_values = {}
pta_frequencies = ["250", "500", "1000", "2000", "3000", "4000", "8000"]

with st.sidebar:
    with st.expander(f"🧍 {texts['기본 정보 입력']}"):
        id_value = st.text_input("ID")
        birth_date = st.date_input(
            texts["생년월일"],
            min_value=datetime.date(1900, 1, 1),
            max_value=datetime.date.today()
        )
        gender = st.selectbox(texts["성별"], ["Male", "Female"])
        name = st.text_input(texts["이름"])
        hsptcd = st.text_input(texts["병원코드 (HSPTCD)"])

    with st.expander(f"🧠 {texts['PTA 검사']}"):
        for freq in pta_frequencies:
            rt = st.text_input(f"PTA_RT_AC_ {freq}", key=f"rt_{freq}")
            lt = st.text_input(f"PTA_LT_AC_ {freq}", key=f"lt_{freq}")
            pta_values[f"PTA_RT_AC_{freq}"] = float(rt) if rt.strip() else None
            pta_values[f"PTA_LT_AC_{freq}"] = float(lt) if lt.strip() else None

    with st.expander(f"🧬 {texts['의료 정보']}"):
        side = st.selectbox(texts["측면 (Side)"], ["Right", "Left"])
        hl_duration = st.text_input(texts["HL_duration (일)"])
        clinic_date = st.date_input("Clinic_date")
        steroid = st.checkbox(texts["스테로이드 치료"])
        it_dexa = st.checkbox(texts["IT_dexa 치료"])
        hbot = st.checkbox(texts["고압산소 치료"])

    with st.expander(f"🧪 {texts['혈액 검사']}"):
        blood_tests = ["WBC", "RBC", "Hb", "PLT", "Neutrophil", "Lymphocyte",
                       "AST", "ALT", "BUN", "Cr", "Glucose", "Total_Protein",
                       "Na", "K", "Cl"]
        blood_values = {}
        for test in blood_tests:
            val = st.text_input(test)
            blood_values[test] = float(val) if val.strip() != "" else None

    with st.expander(f"📄 {texts['진단 및 병력']}"):
        diagnosis = ["Dx_COM", "Dx_SSNHL", "Dx_Dizziness", "Dx_Tinnitus"]
        diagnosis_values = {dx: int(st.checkbox(f"{dx} {texts['여부']}" )) for dx in diagnosis}

        history = ["Hx_HTN", "Hx_DM", "Hx_CRF", "Hx_MI", "Hx_stroke", "Hx_cancer"]
        history_values = {hx: int(st.checkbox(f"{hx} {texts['여부']}")) for hx in history}

        hx_others_text = st.text_input(f"{texts['기타 병력']} (Hx_others)")
        hx_others = 1 if hx_others_text.strip() != "" else 0

    predict_button = st.button(f"\U0001F50D {texts['예측 결과 보기']}")

# 매핑
side_mapping = {"Right": 1, "Left": 2}
sex_mapping = {"Male": 1, "Female": 2}

def create_combined_image(result_df, shap_fig, title="모델 결과 요약", summary_text=None):
    
    fig = plt.figure(figsize=(10, 12))

    ax1 = fig.add_axes([0,0.6,1,0.35]) # [left, bottom, width, height]
    ax1.axis('off')
    ax1.set_title(title, fontsize=30, fontweight='bold', pad=30)

    table_data = [result_df.columns.tolist()] + result_df.values.tolist()
    table = Table(ax1, bbox=[0.2, 0.25, 0.6, 0.6])
    n_cols = len(table_data[0])
    for i, row in enumerate(table_data):
        for j, val in enumerate(row):
            cell = table.add_cell(i, j, width=1.0 / n_cols, height=0.25,
                           text=str(val), loc='center',
                           facecolor='#cce5ff' if i == 0 else 'white')
            cell.get_text().set_fontsize(16)
    ax1.add_table(table)

     # 중단: 텍스트 설명 넣기
    if summary_text:
        ax_text = fig.add_axes([0, 0.52, 1, 0.25])
        ax_text.axis('off')
        ax_text.text(0.5, 0.5, summary_text, fontsize=15, ha='center', va='center', wrap=True)

    shap_buf = io.BytesIO()
    shap_fig.savefig(shap_buf, format='png', bbox_inches='tight')
    shap_buf.seek(0)
    shap_img = plt.imread(shap_buf)

    ax2 = fig.add_axes([0, 0, 1, 0.6])
    ax2.imshow(shap_img)
    ax2.axis('off')

    return fig

# 예측 버튼
if predict_button:
    with st.spinner(f"⏳ {texts['예측 진행 중...']}"):
        df_input = pd.DataFrame([{
            "ID": id_value,
            "Birth": birth_date.strftime("%Y-%m-%d"),
            "test_date": clinic_date.strftime("%Y-%m-%d"),
            "Sex": sex_mapping.get(gender, 1),
            "Side": side_mapping.get(side, 1),
            "HL_duration": float(hl_duration) if hl_duration.strip() else None,
            "Steroid": int(steroid),
            "IT_dexa": int(it_dexa),
            "HBOT": int(hbot),
            **pta_values,
            **blood_values,
            **diagnosis_values,
            **history_values,
            "Hx_others": hx_others
        }])

        lgbm_result, lgbm_prob, xgb_result, xgb_prob, df_lgbm, df_xgb, df_ids, lgbm_model, xgb_model, lgbm_acc, xgb_acc = predictor.predict_outcome(df_input)

        if all(v is not None for v in [lgbm_result, lgbm_prob, xgb_result, xgb_prob]):            

            # 정확도 가져오기 - Google Drive에서 txt 파일 로드
            def get_accuracy_from_drive(hospital_key, model_type):
                """Google Drive에서 정확도 txt 파일 로드"""
                try:
                    file_name = f"txt/{hospital_key}_{model_type}_accuracy.txt"
                    file_content = download_file_from_drive(file_name)
                    if file_content:
                        content = file_content.read().decode('utf-8')
                        # 정확도 값 추출 (예: "0.8523" 형태의 숫자)
                        import re
                        numbers = re.findall(r"\d+\.\d+", content)
                        if numbers:
                            return float(numbers[0])
                    return 0.75  # 기본값
                except Exception as e:
                    return 0.75  # 기본값

            # 병원 키 가져오기
            hospital_key = hospital_modules[selected_hospital].split('.')[-1]
            
            # 정확도 값 가져오기 (캐시 처리)
            if not hasattr(predictor, 'lgbm_acc') or predictor.lgbm_acc is None:
                predictor.lgbm_acc = get_accuracy_from_drive(hospital_key, 'lgbm')
            if not hasattr(predictor, 'xgb_acc') or predictor.xgb_acc is None:
                predictor.xgb_acc = get_accuracy_from_drive(hospital_key, 'xgb')

            # LightGBM 결과
            result_df_lgbm = pd.DataFrame({
                "ID": df_ids["ID"].values,
                "LightGBM 회복 판단": ["회복" if p >= 0.5 else "비회복" for p in lgbm_prob],
                "LightGBM 회복 확률": [f"{(p * 100):.1f}%" for p in lgbm_prob],
                "예측 정확도": [f"{predictor.lgbm_acc * 100:.1f}%" for _ in lgbm_result]
            })

            # XGBoost 결과
            result_df_xgb = pd.DataFrame({
                "ID": df_ids["ID"].values,
                "XGBoost 회복 판단": ["회복" if p >= 0.5 else "비회복" for p in xgb_prob],
                "XGBoost 회복 확률": [f"{(p * 100):.1f}%" for p in xgb_prob],
                "예측 정확도": [f"{predictor.xgb_acc * 100:.1f}%" for _ in xgb_result]
            })

            st.markdown(f"### 📋 {texts['summary_title']}")
        
            # LightGBM 회복 확률 (첫 번째 샘플 기준)
            lgbm_prob_val = lgbm_prob[0] * 100
            xgb_prob_val = xgb_prob[0] * 100

            # 통합 예측 요약 테이블 (표 스타일로)
            
            st.markdown(f"""
            <style>
            .result-table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 1rem;
            }}
            .result-table th, .result-table td {{
                border: 1px solid #ccc;
                padding: 0.6rem 1rem;
                text-align: center;
                font-size: 1.05rem;
            }}
            .result-comment {{
                font-size: 1.1rem;
                line-height: 1.6;
                background-color: #e7f5ff;
                border-radius: 10px;
                padding: 1.2rem;
                border-left: 5px solid #0077B6;
                margin-bottom: 2rem;
            }}
            </style>

            <table class="result-table">
                <tr>
                    <th>{texts["모델"]}</th>
                    <th>{texts["회복 판단"]}</th>
                    <th>{texts["회복 확률"]}</th>
                    <th>{texts["예측 정확도"]}</th>
                </tr>
                <tr>
                    <td><b>LightGBM</b></td>
                    <td style="color: {'green' if lgbm_prob[0] >= 0.5 else 'red'}; font-weight: bold;">
                        {texts['회복'] if lgbm_prob[0] >= 0.5 else texts['비회복']}
                    </td>
                    <td><b>{lgbm_prob[0]*100:.1f}%</b></td>
                    <td>{predictor.lgbm_acc*100:.1f}%</td>
                </tr>
                <tr>
                    <td><b>XGBoost</b></td>
                    <td style="color: {'green' if xgb_prob[0] >= 0.5 else 'red'}; font-weight: bold;">
                        {texts['회복'] if xgb_prob[0] >= 0.5 else texts['비회복']}
                    </td>
                    <td><b>{xgb_prob[0]*100:.1f}%</b></td>
                    <td>{predictor.xgb_acc*100:.1f}%</td>
                </tr>
            </table>

            <div class="result-comment">
                <b>{name}</b>&nbsp;{texts["님의 예측 결과는 다음과 같습니다."]}<br><br>
                🔵 <b>LightGBM</b> {texts["기준"]} : {texts["회복 확률"]} <b>{lgbm_prob[0]*100:.1f}%</b>, 
                 {texts["예측 정확도"]} <b>{predictor.lgbm_acc*100:.1f}%<br></b>
                🟢 <b>XGBoost</b> {texts["기준"]} : {texts["회복 확률"]} <b>{xgb_prob[0]*100:.1f}%</b>, 
                 {texts["예측 정확도"]} <b>{predictor.xgb_acc*100:.1f}%<br></b>
            </div>
            """, unsafe_allow_html=True)
      
            # 🎯 SHAP explainer 및 계산
            explainer_lgbm = shap.TreeExplainer(predictor.lgbm_model)
            shap_values_lgbm_raw = explainer_lgbm.shap_values(df_lgbm)  # 원본 저장

            explainer_xgb = shap.TreeExplainer(predictor.xgb_model)
            shap_values_xgb_raw = explainer_xgb.shap_values(df_xgb)

            # ⚠️ multiclass 대응 (보통 binary이면 list로 반환됨)
            shap_values_lgbm = shap_values_lgbm_raw[1] if isinstance(shap_values_lgbm_raw, list) else shap_values_lgbm_raw
            shap_values_xgb = shap_values_xgb_raw[1] if isinstance(shap_values_xgb_raw, list) else shap_values_xgb_raw
            
            target_features = [
                "WBC", "RBC", "Hb", "PLT", "Neutrophil", "Lymphocyte",
                "AST", "ALT", "BUN", "Cr", "Glucose", "Total_Protein",
                "Na", "K", "Cl"
            ]

            # ✅ 생화학 변수 중 실제 존재하는 것만 필터링
            filtered_features_lgbm = [f for f in target_features if f in df_lgbm.columns]
            filtered_features_xgb = [f for f in target_features if f in df_xgb.columns]

            # ✅ 해당 feature 인덱스 추출
            feature_indices_lgbm = [df_lgbm.columns.get_loc(col) for col in filtered_features_lgbm]
            feature_indices_xgb = [df_xgb.columns.get_loc(col) for col in filtered_features_xgb]

            st.markdown(f"### {texts['변수 중요도']}")

            # 전체 변수 중요도 보기
            with st.expander(f"📊 {texts['전체 변수 중요도 보기']}"):
                col1, col2 = st.columns(2)
                with col1:
                    fig_lgbm = plt.figure()
                    st.subheader(f"🔍 LightGBM {texts['변수 중요도']}")
                    shap.summary_plot(shap_values_lgbm, df_lgbm, plot_type='bar', show=False)
                    plt.gcf().subplots_adjust(top=0.88)
                    st.pyplot(plt.gcf())

                with col2:
                    st.subheader(f"🔍 XGBoost {texts['변수 중요도']}")
                    fig_xgb = plt.figure()
                    shap.summary_plot(shap_values_xgb, df_xgb, plot_type="bar", show=False)
                    plt.gcf().subplots_adjust(top=0.88)
                    st.pyplot(plt.gcf())

            normal_ranges = {
                "WBC": (4.0, 10.0), "RBC": (3.8, 5.2), "Hb": (12.0, 16.0), "PLT": (165, 360),
                "Neutrophil": (35, 75), "Lymphocyte": (25, 40), "AST": (0, 35), "ALT": (0, 40),
                "BUN": (5, 19), "Cr": (0.20, 1.10), "Glucose": (70, 110), "Total_Protein": (6.4, 8.3),
                "Na": (136, 145), "K": (3.5, 5.1), "Cl": (98, 107)
            }
            
            def plot_shap_dot_with_ranges(features, shap_values, feature_values, normal_ranges, title="SHAP Dot Plot"):
                import matplotlib.pyplot as plt
                import numpy as np

                sorted_idx = np.argsort(shap_values)[::-1]
                features_sorted = [features[i] for i in sorted_idx if features[i]]
                shap_sorted = shap_values[sorted_idx]

                fig, ax = plt.subplots(figsize=(6, 6))
                y = np.arange(len(features_sorted))

                normal_range_plotted = False
                patient_point_plotted = False

                # 정상범위 박스: 정확한 y 위치로 박스 그리기
                for i, feat in enumerate(features_sorted):
                    if feat in normal_ranges:
                        low, high = normal_ranges[feat]
                        width = high - low
                        label = texts["정상범위"] if not normal_range_plotted else ""
                        ax.barh(i, width, left=low, height=0.6, color='green', alpha=0.2, label=label)
                        normal_range_plotted = True

                    if feat in feature_values:
                        label = texts["환자수치"] if not patient_point_plotted else ""
                        ax.scatter(feature_values[feat], i, color='red', marker='.', label=label)
                        patient_point_plotted = True

                ax.set_yticks(y)
                ax.set_yticklabels(features_sorted)
                ax.set_xlabel(texts["수치"])
                ax.set_title(title)
                ax.invert_yaxis()
                ax.legend(loc="lower right")
                plt.tight_layout()
                return fig
                        
            def plot_single_variable_graph(
                feature,
                value,
                normal_ranges,
                xlim_range=(0, 400),
                title_fontsize=9,
                tick_fontsize=8
            ):
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(4.5, 1.0))  # 적당히 작게
                ax.set_xlim(*xlim_range)
                ax.set_yticks([])
                ax.set_title(feature, fontsize=title_fontsize, pad=2)
                ax.set_xlabel(texts["수치"], fontsize=tick_fontsize)

                if feature in normal_ranges:
                    low, high = normal_ranges[feature]
                    width = high - low
                    ax.barh(0, width, left=low, height=0.3, color='green', alpha=0.2, label="정상범위")

                if value is not None:
                    ax.scatter(value, 0, color='red', s=25, label="환자 수치")

                ax.set_frame_on(False)
                ax.tick_params(axis='x', labelsize=tick_fontsize)
                plt.tight_layout()
                return fig

            # 조정 가능한 변수 중요도 보기
            with st.expander(f"🛠️ {texts['조정가능한 변수 중요도 보기']}"):
                col3, col4 = st.columns(2)
                with col3:
                    st.subheader(f"🧪 LightGBM {texts['조정가능 변수']}")
                    plt.clf()
                    shap.summary_plot(
                        shap_values_lgbm[:, feature_indices_lgbm],
                        df_lgbm[filtered_features_lgbm],
                        plot_type="bar", show=False
                    )
                    plt.gcf().subplots_adjust(top=0.90)
                    st.pyplot(plt.gcf())

                with col4:
                    st.subheader(f"🧪 XGBoost {texts['조정가능 변수']}")
                    shap.summary_plot(
                        shap_values_xgb[:, feature_indices_xgb],
                        df_xgb[filtered_features_xgb],
                        plot_type="bar", show=False
                    )
                    plt.gcf().subplots_adjust(top=0.90)
                    st.pyplot(plt.gcf())

            # 변수별 x축 범위 설정 (없으면 기본값 사용)
            custom_xlims = {
                "WBC": (0, 20), "RBC": (0, 8), "Hb": (0, 20), "PLT": (0, 500),
                "Neutrophil": (0, 100), "Lymphocyte": (0, 100), "AST": (0, 100), "ALT": (0, 100),
                "BUN": (0, 50), "Cr": (0, 3), "Glucose": (0, 200), "Total_Protein": (0, 10),
                "Na": (120, 160), "K": (2, 7), "Cl": (80, 120)
            }

            # --- 조정 가능한 변수 중요도 순 정렬 (LightGBM 기준) ---
            shap_mean_lgbm = np.abs(shap_values_lgbm[:, feature_indices_lgbm]).mean(axis=0)
            feature_importance_lgbm = list(zip(filtered_features_lgbm, shap_mean_lgbm))
            sorted_features_lgbm = [
                feat for feat, val in sorted(feature_importance_lgbm, key=lambda x: x[1], reverse=True)
                if feat in blood_values and feat in normal_ranges
            ]

            # --- 조정 가능한 변수 중요도 순 정렬 (XGBoost 기준) ---
            shap_mean_xgb = np.abs(shap_values_xgb[:, feature_indices_xgb]).mean(axis=0)
            feature_importance_xgb = list(zip(filtered_features_xgb, shap_mean_xgb))
            sorted_features_xgb = [
                feat for feat, val in sorted(feature_importance_xgb, key=lambda x: x[1], reverse=True)
                if feat in blood_values and feat in normal_ranges
            ]

            # ----------------- Streamlit 출력 -------------------
            with st.expander(f"📊 {texts['변수중요도 기반 환자 수치 확인']}"):

                # 좌/우 박스 생성
                col_lgbm, col_xgb = st.columns(2)

                with col_lgbm:
                    st.markdown(
                            f"### 🔵 LightGBM {texts['기준']}&nbsp;&nbsp;&nbsp; "
                            f"<span style='font-size:12px;'><span style='color:red;'>●</span> {texts['환자수치']}</span>, "
                            f"<span style='font-size:12px; background-color:#a4d4a4; padding:1px 5px; border-radius:2px;'>{texts['정상범위']}</span>",
                            unsafe_allow_html=True
                        )
                    lgbm_container = st.container()
                    for feature in sorted_features_lgbm:
                        value = blood_values.get(feature)
                        fig = plot_single_variable_graph(
                            feature=feature,
                            value=value,
                            normal_ranges=normal_ranges,
                            xlim_range=custom_xlims.get(feature, (0, 400)),
                            title_fontsize=8,
                            tick_fontsize=6
                        )
                        lgbm_container.pyplot(fig)

                with col_xgb:
                    st.markdown(
                        f"### 🟢 XGBoost {texts['기준']}&nbsp;&nbsp;&nbsp; "
                        f"<span style='font-size:12px;'><span style='color:red;'>●</span> {texts['환자수치']}</span>, "
                        f"<span style='font-size:12px; background-color:#a4d4a4; padding:1px 5px; border-radius:2px;'>{texts['정상범위']}</span>",
                        unsafe_allow_html=True
                    )
                    xgb_container = st.container()
                    for feature in sorted_features_xgb:
                        value = blood_values.get(feature)
                        fig = plot_single_variable_graph(
                            feature=feature,
                            value=value,
                            normal_ranges=normal_ranges,
                            xlim_range=custom_xlims.get(feature, (0, 400)),
                            title_fontsize=8,
                            tick_fontsize=6
                        )
                        xgb_container.pyplot(fig)

                for var, val in blood_values.items():
                    if var in normal_ranges:
                        low, high = normal_ranges[var]
                        if val < low:
                            direction = f"<span style='color:#d62728;'>{texts['낮아']}</span>, <b>{texts['증가시켜야 함.']}</b>"
                        elif val > high:
                            direction = f"<span style='color:#1f77b4;'>{texts['높아']}</span>, <b>{texts['감소시켜야 함.']}</b>"
                        else:
                            continue  # 정상범위 내 수치는 표시하지 않음

                        # 변수 수치 해석 문장 구성
                        st.markdown(
                            f"<p style='text-align: center;'>📍 <b>{var}</b> {texts['수치는']} <b>{val}</b>{texts['로']} "
                            f"{texts['정상범위인']} <b>{low}~{high}</b>&nbsp;{texts['보다']} {direction}</p>",
                            unsafe_allow_html=True
                        )


            # ✅ Google Sheets에 데이터 저장
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 저장할 데이터 준비
            save_data = {
                'timestamp': timestamp,
                'language': lang_choice_label,
                'hospital': selected_hospital,
                'id': id_value,
                'birth': birth_date.strftime("%Y-%m-%d"),
                'sex': gender,
                'name': name,
                'hsptcd': hsptcd,
                'side': side,
                'hl_duration': hl_duration.strip() if hl_duration else '',
                'clinic_date': str(clinic_date),
                'steroid_treatment': int(steroid),
                'it_dexa_treatment': int(it_dexa),
                'hyperbaric_treatment': int(hbot),
                'pta_rt_ac_250': pta_values.get('PTA_RT_AC_250', ''),
                'pta_rt_ac_500': pta_values.get('PTA_RT_AC_500', ''),
                'pta_rt_ac_1000': pta_values.get('PTA_RT_AC_1000', ''),
                'pta_rt_ac_2000': pta_values.get('PTA_RT_AC_2000', ''),
                'pta_rt_ac_3000': pta_values.get('PTA_RT_AC_3000', ''),
                'pta_rt_ac_4000': pta_values.get('PTA_RT_AC_4000', ''),
                'pta_rt_ac_8000': pta_values.get('PTA_RT_AC_8000', ''),
                'pta_lt_ac_250': pta_values.get('PTA_LT_AC_250', ''),
                'pta_lt_ac_500': pta_values.get('PTA_LT_AC_500', ''),
                'pta_lt_ac_1000': pta_values.get('PTA_LT_AC_1000', ''),
                'pta_lt_ac_2000': pta_values.get('PTA_LT_AC_2000', ''),
                'pta_lt_ac_3000': pta_values.get('PTA_LT_AC_3000', ''),
                'pta_lt_ac_4000': pta_values.get('PTA_LT_AC_4000', ''),
                'pta_lt_ac_8000': pta_values.get('PTA_LT_AC_8000', ''),
                'wbc': blood_values.get('WBC', ''),
                'rbc': blood_values.get('RBC', ''),
                'hb': blood_values.get('Hb', ''),
                'plt': blood_values.get('PLT', ''),
                'neutrophil': blood_values.get('Neutrophil', ''),
                'lymphocyte': blood_values.get('Lymphocyte', ''),
                'ast': blood_values.get('AST', ''),
                'alt': blood_values.get('ALT', ''),
                'bun': blood_values.get('BUN', ''),
                'cr': blood_values.get('Cr', ''),
                'glucose': blood_values.get('Glucose', ''),
                'total_protein': blood_values.get('Total_Protein', ''),
                'na': blood_values.get('Na', ''),
                'k': blood_values.get('K', ''),
                'cl': blood_values.get('Cl', ''),
                'dx_com': diagnosis_values.get('Dx_COM', 0),
                'dx_ssnhl': diagnosis_values.get('Dx_SSNHL', 0),
                'dx_dizziness': diagnosis_values.get('Dx_Dizziness', 0),
                'dx_tinnitus': diagnosis_values.get('Dx_Tinnitus', 0),
                'hx_htn': history_values.get('Hx_HTN', 0),
                'hx_dm': history_values.get('Hx_DM', 0),
                'hx_crf': history_values.get('Hx_CRF', 0),
                'hx_mi': history_values.get('Hx_MI', 0),
                'hx_stroke': history_values.get('Hx_stroke', 0),
                'hx_cancer': history_values.get('Hx_cancer', 0),
                'hx_others': hx_others_text,
                'prediction': f"LightGBM: {lgbm_prob[0]*100:.1f}%, XGBoost: {xgb_prob[0]*100:.1f}%",
                'probability': f"LightGBM: {'회복' if lgbm_prob[0] >= 0.5 else '비회복'}, XGBoost: {'회복' if xgb_prob[0] >= 0.5 else '비회복'}"
            }
            
            # Google Sheets에 저장 (조용히 실행)
            save_to_sheets(save_data)

            # 결과 정리 텍스트
            summary_lgbm = f"회복 확률 {lgbm_prob_val:.1f}%, 예측정확도 {predictor.lgbm_acc * 100:.1f}%."
            summary_xgb = f"회복 확률 {xgb_prob_val:.1f}%, 예측정확도 {predictor.xgb_acc * 100:.1f}%."

            # 📸 결과 요약 이미지 생성
            def create_summary_image(
                name,
                hospital,
                clinic_date,
                summary_lgbm,
                summary_xgb,
                fig_lgbm,
                fig_xgb,
                font_path=None  # 폰트 경로를 None으로 설정
            ):
                # A4 이미지 사이즈 설정 (단위: 픽셀)
                a4_width, a4_height = 595, 842  # 72dpi 기준 A4: 595 x 842
                margin = 40
                gap = 30
                table_height = 120
                text_height = 80
                graph_width = (a4_width - margin * 2 - gap) // 2

                # SHAP 그래프 저장 및 리사이징
                buf_lgbm, buf_xgb = io.BytesIO(), io.BytesIO()
                fig_lgbm.savefig(buf_lgbm, format='png', bbox_inches='tight')
                fig_xgb.savefig(buf_xgb, format='png', bbox_inches='tight')
                buf_lgbm.seek(0)
                buf_xgb.seek(0)
                img_lgbm = Image.open(buf_lgbm)
                img_xgb = Image.open(buf_xgb)

                # 그래프 크기 조정
                img_lgbm = img_lgbm.resize((graph_width, int(graph_width * img_lgbm.height / img_lgbm.width)))
                img_xgb = img_xgb.resize((graph_width, int(graph_width * img_xgb.height / img_xgb.width)))
                graph_height = max(img_lgbm.height, img_xgb.height)

                # 전체 이미지 캔버스 생성
                total_height = margin + 30 + table_height + text_height + graph_height + margin
                img = Image.new("RGB", (a4_width, a4_height), (255, 255, 255))
                draw = ImageDraw.Draw(img)

                # 폰트 설정 - 기본 폰트 사용 또는 시스템 폰트 시도
                try:
                    # Windows 환경인 경우 맑은 고딕 시도
                    if os.name == 'nt' and os.path.exists("C:/Windows/Fonts/malgun.ttf"):
                        font_title = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 30)
                        font_main = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 15)
                        font_bold = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 15)
                    else:
                        # 기본 폰트 사용
                        font_title = ImageFont.load_default()
                        font_main = ImageFont.load_default()
                        font_bold = ImageFont.load_default()
                except:
                    # 폰트 로드 실패 시 기본 폰트 사용
                    font_title = ImageFont.load_default()
                    font_main = ImageFont.load_default()
                    font_bold = ImageFont.load_default()

                # 제목 출력
                title = "SSNHL 예측 결과 요약"
                try:
                    title_width = draw.textlength(title, font=font_title)
                except:
                    title_width = len(title) * 10  # 대략적인 계산
                draw.text(((a4_width - title_width) // 2, margin), title, fill=(0, 0, 0), font=font_title)

                # 기본 정보 출력
                base_y = margin + 80
                info_lines = [
                    f"예측일자: {clinic_date}",
                    f"병원명: {hospital}",
                    f"환자명: {name}",
                ]
                for i, line in enumerate(info_lines):
                    draw.text((margin, base_y + i * 20), line, fill=(0, 0, 0), font=font_main)

                # 예측 결과 요약 텍스트 출력
                result_y = base_y + 1 * 10 + 200
                result_texts = [
                    f"🔵 LightGBM 기준 : {summary_lgbm}",
                    f"🟢 XGBoost 기준 : {summary_xgb}"
                ]
                for i, line in enumerate(result_texts):
                    try:
                        text_width = draw.textlength(line, font=font_main)
                    except:
                        text_width = len(line) * 8
                    draw.text(((a4_width - text_width) // 2, result_y + i * 22), line, fill=(0, 0, 0), font=font_main)

                # SHAP 그래프 삽입
                graph_y = result_y + len(result_texts) * 22 + 50
                img.paste(img_lgbm, (margin, graph_y))
                img.paste(img_xgb, (margin + graph_width + gap, graph_y))

                # 그래프 아래 모델명 라벨 출력
                draw.text((margin + graph_width // 2 - 70, graph_y - 20), "🔍 LightGBM 변수 중요도", font=font_bold, fill=(0, 0, 0))
                draw.text((margin + graph_width + gap + graph_width // 2 - 70, graph_y - 20), "🔍 XGBoost 변수 중요도", font=font_bold, fill=(0, 0, 0))

                # 이미지 반환 (BytesIO 형태로 반환)
                result_buf = io.BytesIO()
                img.save(result_buf, format="PNG")
                result_buf.seek(0)
                return result_buf

            # 이미지 생성
            img_buf = create_summary_image(
                name=name,
                hospital=selected_hospital,
                clinic_date=clinic_date,
                summary_lgbm=summary_lgbm,
                summary_xgb=summary_xgb,
                fig_lgbm=fig_lgbm,
                fig_xgb=fig_xgb
            )

            def convert_image_to_pdf(image_bytes):
                buffer = io.BytesIO()
                c = canvas.Canvas(buffer, pagesize=A4)
                width, height = A4
                image = ImageReader(image_bytes)
                c.drawImage(image, 0, 0, width=width, height=height)
                c.showPage()
                c.save()
                buffer.seek(0)
                return buffer
            
            pdf_buf = convert_image_to_pdf(img_buf)

            # 🔽 예측 결과 다운로드
            col_result, col_button = st.columns([5, 1])

            with col_button:
                with st.expander("💾"+texts["결과 저장"], expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button("🖼 PNG"+texts["저장"], data=img_buf, file_name="result.png", mime="image/png")
                    with col2:
                        st.download_button("📄 PDF"+ texts["저장"], data=pdf_buf, file_name="result.pdf", mime="application/pdf")
