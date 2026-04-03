import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import joblib
import altair as alt
import shap

# 1. 頁面配置
st.set_page_config(page_title="食安風險預測系統", layout="wide")
st.title("🛡️ 國際食安風險：趨勢比較與回測預測模式")

# 翻譯字典：RASFF 常見類別與風險
CATEGORY_DICT = {
    'bivalve molluscs and products thereof': '雙殼貝類及其產品',
    'fruits and vegetables': '生鮮蔬果',
    'cereals and bakery products': '穀物與烘焙食品',
    'nuts, nut products and seeds': '堅果與種子類',
    'dietetic foods, food supplements, fortified foods': '營養補充品與強化食品',
    'dietetic foods, food supplements and fortified foods': '營養補充品與強化食品',
    'fish and fish products': '魚類及其產品',
    'meat and meat products (other than poultry)': '肉類及其產品(非家禽)',
    'poultry meat and poultry meat products': '家禽肉類及其產品',
    'herbs and spices': '香草與香料',
    'fats and oils': '油脂類',
    'confectionery': '糖果糕點',
    'milk and milk products': '乳製品',
    'beverages': '飲料類',
    'crustaceans and products thereof': '甲殼類及其產品',
    'food contact materials': '食品接觸材質',
    'other food product / mixed': '其他食品/混合食品',
    'pet food': '寵物食品',
    'prepared dishes and snacks': '調理食品與零食',
    'soups, broths, sauces and condiments': '湯品、醬料與調味料',
    'water for human consumption (other)': '飲用水(其他)',
    'wine': '葡萄酒',
    'alcoholic beverages': '酒精飲料',
    'animal by-products': '動物副產品',
    'cephalopods and products thereof': '頭足類及其產品',
    'cocoa and cocoa preparations, coffee and tea': '可可及其製品、咖啡與茶',
    'compound feeds': '配合飼料',
    'eggs and egg products': '蛋類及其產品',
    'feed additives': '飼料添加物',
    'feed materials': '飼料原料',
    'feed premixtures': '飼料預拌物',
    'food additives and flavourings': '食品添加物與香料',
    'gastropods': '腹足類(如螺類)',
    'honey and royal jelly': '蜂蜜與蜂王乳',
    'ices and desserts': '冰品與甜點',
    'natural mineral waters': '天然礦泉水',
    'non-alcoholic beverages': '非酒精飲料',
    'live animals': '活體動物',
    'plant protection products': '植物保護產品',
}

HAZARD_DICT = {
    'adulteration / fraud': '摻偽與詐欺',
    'mycotoxins': '真菌毒素',
    'pesticide residues': '農藥殘留',
    'heavy metals': '重金屬',
    'pathogenic micro-organisms': '病原微生物',
    'composition': '成分標示不符',
    'allergens': '過敏原',
    'food additives and flavourings': '食品添加物與香料',
    'foreign bodies': '異物',
    'novel food': '未經核准新興食品',
    'biocontaminants': '生物污染',
    'industrial contaminants': '工業污染',
    'residues of veterinary medicinal products': '動物用藥殘留',
    'other/mixed': '其他/混合',
    'natural toxins (other)': '天然毒素(其他)',
    'non-pathogenic micro-organisms': '非致病微生物',
    'not determined (other)': '未確認(其他)',
    'organoleptic aspects': '感官異常(異味或變質)',
    'packaging defective / incorrect': '包裝瑕疵或錯誤',
    'parasitic infestation': '寄生蟲感染',
    'radiation': '輻射污染',
    'tses': '傳染性海綿狀腦病(TSEs)',
    'biological contaminants': '生物性污染',
    'chemical contamination (other)': '化學污染(其他)',
    'environmental pollutants': '環境污染物',
    'feed additives': '飼料添加物',
    'genetically modified': '基因改造',
    'labelling absent/incomplete/incorrect': '標示缺失/不完整/錯誤',
    'migration': '包裝材質物質遷移',
    'gmo / novel food': '基因改造/新興食品',
    'poor or insufficient controls': '品管不良或管制不足'
}

# 2. 真實資料載入與 ETL 處理


@st.cache_data
def load_and_preprocess_data(file_path):
    try:
        df_raw = pd.read_excel(file_path)

        req_cols = ['date', 'category', 'hazards', 'risk_decision']
        if not all(col in df_raw.columns for col in req_cols):
            st.error("資料欄位不符，請確認是否為標準 RASFF 格式。")
            return pd.DataFrame()

        df_raw = df_raw.dropna(subset=req_cols).copy()

        df_raw['date'] = pd.to_datetime(
            df_raw['date'], format='%d-%m-%Y %H:%M:%S', errors='coerce')
        if df_raw['date'].isnull().any():
            df_raw['date'] = df_raw['date'].fillna(pd.to_datetime(
                df_raw['date'], dayfirst=True, errors='coerce'))

        df_raw = df_raw.dropna(subset=['date'])
        df_raw['年份'] = df_raw['date'].dt.year.astype(int)
        df_raw['月份'] = df_raw['date'].dt.month.astype(int)

        df_raw['產品類別_英'] = df_raw['category'].astype(
            str).str.lower().str.strip()
        df_raw['產品類別'] = df_raw['產品類別_英'].map(
            CATEGORY_DICT).fillna(df_raw['產品類別_英'])

        df_raw['風險原因_英'] = df_raw['hazards'].astype(str).str.extract(
            r'\{(.*?)\}')[0].fillna('other/mixed').str.lower().str.strip()
        df_raw['風險原因'] = df_raw['風險原因_英'].map(
            HAZARD_DICT).fillna(df_raw['風險原因_英'])

        high_risk_labels = ['serious', 'potentially serious']
        df_raw['是否為高風險'] = df_raw['risk_decision'].astype(
            str).str.lower().isin(high_risk_labels).astype(int)

        return df_raw[['年份', '月份', '產品類別', '風險原因', '是否為高風險']]

    except Exception as e:
        st.error(f"資料處理失敗: {e}")
        return pd.DataFrame()


FILE_PATH = "RASFF 202001-202512.xlsx"

with st.spinner("正在讀取並解析大型資料庫，請稍候..."):
    df = load_and_preprocess_data(FILE_PATH)

# 3. 模組化訓練與模型存取機制


@st.cache_resource
def load_or_train_models(_df):
    MODEL_PATH = "rasff_trained_models.joblib"

    if os.path.exists(MODEL_PATH):
        saved_data = joblib.load(MODEL_PATH)
        le_cat = saved_data['le_cat']
        le_risk = saved_data['le_risk']

        df_recent = _df[_df['年份'] == 2025].copy(
        ) if not _df.empty else pd.DataFrame()
        if not df_recent.empty:
            df_recent['cat_encoded'] = df_recent['產品類別'].apply(
                lambda x: le_cat.transform([x])[0] if x in le_cat.classes_ else 0)
            df_recent['risk_encoded'] = df_recent['風險原因'].apply(
                lambda x: le_risk.transform([x])[0] if x in le_risk.classes_ else 0)

        return (
            saved_data['model_all'],
            saved_data['model_past'],
            saved_data['model_recent'],
            df_recent,
            le_cat,
            le_risk
        )

    if _df.empty:
        return None, None, None, None, None, None

    le_cat = LabelEncoder()
    le_risk = LabelEncoder()
    _df['cat_encoded'] = le_cat.fit_transform(_df['產品類別'])
    _df['risk_encoded'] = le_risk.fit_transform(_df['風險原因'])

    df_all = _df.copy()
    df_past = _df[_df['年份'] < 2025].copy()
    df_recent = _df[_df['年份'] == 2025].copy()

    def train_xgb(train_data):
        if train_data.empty:
            return None
        X = train_data[['月份', 'cat_encoded', 'risk_encoded']]
        y = train_data['是否為高風險']
        pos_weight = (len(y) - y.sum()) / y.sum() if y.sum() > 0 else 1

        model = xgb.XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            scale_pos_weight=pos_weight, random_state=42, eval_metric='logloss'
        )
        model.fit(X, y)
        return model

    model_all = train_xgb(df_all)
    model_past = train_xgb(df_past)
    model_recent = train_xgb(df_recent)

    joblib.dump({
        'model_all': model_all,
        'model_past': model_past,
        'model_recent': model_recent,
        'le_cat': le_cat,
        'le_risk': le_risk
    }, MODEL_PATH)

    return model_all, model_past, model_recent, df_recent, le_cat, le_risk


with st.spinner("正在載入或訓練模型..."):
    model_all, model_past, model_recent, df_recent, le_cat, le_risk = load_or_train_models(
        df)

# 繪製 SHAP 動態圖表的共用函數


def plot_shap_altair(model, input_df, target_month, target_cat, target_risk):
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(input_df)[0]

    # 建立特徵名稱對應
    feature_names = [f'月份 ({target_month}月)',
                     f'類別 ({target_cat})', f'風險 ({target_risk})']

    shap_df = pd.DataFrame({
        '特徵': feature_names,
        'SHAP值 (影響力)': shap_vals
    })

    # 區分正負向影響
    shap_df['影響方向'] = shap_df['SHAP值 (影響力)'].apply(
        lambda x: '推高風險' if x > 0 else '降低風險')

    chart = alt.Chart(shap_df).mark_bar().encode(
        x=alt.X('SHAP值 (影響力):Q', title='對預測機率的影響幅度 (Log Odds)'),
        y=alt.Y('特徵:N', sort='-x', axis=alt.Axis(labelAngle=0, title=None)),
        color=alt.Color('影響方向:N',
                        scale=alt.Scale(domain=['推高風險', '降低風險'], range=[
                                        '#d62728', '#1f77b4']),
                        legend=alt.Legend(title="影響方向"))
    ).properties(height=250)

    return chart


# 4. 介面呈現
if model_all:
    st.sidebar.header("🔮 預測情境設定")
    target_month = st.sidebar.slider("目標月份", 1, 12, 4)
    target_cat = st.sidebar.selectbox("待測產品類別", options=le_cat.classes_)
    target_risk = st.sidebar.selectbox("待測風險因子", options=le_risk.classes_)

    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ 模型參數調校")
    decision_threshold = st.sidebar.slider(
        "預警觸發門檻 (Threshold)",
        min_value=0.1, max_value=0.9, value=0.5, step=0.05,
        help="調低門檻可提升攔截率(Recall)，但可能增加誤報；調高則反之。"
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("📥 維護工具")
    cat_csv = pd.DataFrame({'產品類別清單': le_cat.classes_}).to_csv(
        index=False).encode('utf-8-sig')
    risk_csv = pd.DataFrame({'風險因子清單': le_risk.classes_}).to_csv(
        index=False).encode('utf-8-sig')
    st.sidebar.download_button(
        "⬇️ 下載產品類別", data=cat_csv, file_name="rasff_categories.csv", mime="text/csv")
    st.sidebar.download_button(
        "⬇️ 下載風險因子", data=risk_csv, file_name="rasff_hazards.csv", mime="text/csv")

    input_data = pd.DataFrame([[
        target_month, le_cat.transform([target_cat])[
            0], le_risk.transform([target_risk])[0]
    ]], columns=['月份', 'cat_encoded', 'risk_encoded'])

    # 主畫面 1：模型回測驗證
    st.header(f"📈 模型回測驗證 (閾值: {decision_threshold:.2f})")
    if model_past and df_recent is not None and not df_recent.empty:
        X_test = df_recent[['月份', 'cat_encoded', 'risk_encoded']]
        y_test = df_recent['是否為高風險']

        y_prob_test = model_past.predict_proba(X_test)[:, 1]
        y_pred_dynamic = (y_prob_test >= decision_threshold).astype(int)

        acc = accuracy_score(y_test, y_pred_dynamic)
        prec = precision_score(y_test, y_pred_dynamic, zero_division=0)
        rec = recall_score(y_test, y_pred_dynamic, zero_division=0)
        f1 = f1_score(y_test, y_pred_dynamic, zero_division=0)

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("整體準確率 (Accuracy)", f"{acc:.1%}")
        col_m2.metric("高風險精準率 (Precision)", f"{prec:.1%}")
        col_m3.metric("高風險召回率 (Recall)", f"{rec:.1%}")
        col_m4.metric("F1-Score", f"{f1:.1%}")
    else:
        st.warning("⚠️ 2025 年可用資料筆數為 0，無法執行回測驗證。")

    st.markdown("---")

    # 主畫面 2：多重模型預測與特徵比較 (導入動態 SHAP 解析)
    st.header("🔍 動態情境解析：影響力追蹤")
    st.caption("以下圖表顯示「當下設定條件」中，哪個因素是推高或降低風險的關鍵主因。")
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("長線趨勢 (2020-2025 全資料訓練)")
        prob_all = model_all.predict_proba(input_data)[0][1]
        st.metric(label="高風險發生機率", value=f"{prob_all:.1%}")

        if prob_all >= decision_threshold:
            st.warning(f"⚠️ 預警：此組合長線風險達標 ({prob_all:.1%})。")
        else:
            st.success("✅ 長線趨勢處於安全範圍。")

        chart_all_shap = plot_shap_altair(
            model_all, input_data, target_month, target_cat, target_risk)
        st.altair_chart(chart_all_shap, use_container_width=True)

    with c2:
        st.subheader("近期趨勢 (僅 2025 資料訓練)")
        if model_recent:
            prob_recent = model_recent.predict_proba(input_data)[0][1]
            st.metric(label="高風險發生機率", value=f"{prob_recent:.1%}")

            if prob_recent >= decision_threshold:
                st.warning(f"⚠️ 預警：此組合近期高發 ({prob_recent:.1%})，強烈建議監控。")
            else:
                st.success("✅ 近期趨勢處於安全範圍。")

            chart_recent_shap = plot_shap_altair(
                model_recent, input_data, target_month, target_cat, target_risk)
            st.altair_chart(chart_recent_shap, use_container_width=True)

        else:
            st.error("❌ 缺乏 2025 年資料，無法訓練近期模型。")

else:
    st.error("❌ 模型訓練失敗，請確認檔案存在且內容格式正確。")
