import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import re

# 1. 頁面配置
st.set_page_config(page_title="食安風險預測系統", layout="wide")
st.title("🛡️ 國際食安風險：RASFF 真實資料預測模式")

# 翻譯字典：RASFF 常見類別與風險
CATEGORY_DICT = {
    'bivalve molluscs and products thereof': '雙殼貝類及其產品',
    'fruits and vegetables': '生鮮蔬果',
    'cereals and bakery products': '穀物與烘焙食品',
    'nuts, nut products and seeds': '堅果與種子類',
    'dietetic foods, food supplements, fortified foods': '營養補充品與強化食品',
    'dietetic foods, food supplements and fortified foods': '營養補充品與強化食品',  # 處理逗號與and的差異
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
    # 根據最新匯出清單新增
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
    # 根據最新匯出清單新增
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
        # 讀取 Excel 檔案
        df_raw = pd.read_excel(file_path)

        req_cols = ['date', 'category', 'hazards', 'risk_decision']
        if not all(col in df_raw.columns for col in req_cols):
            st.error("資料欄位不符，請確認是否為標準 RASFF 格式。")
            return pd.DataFrame()

        df_raw = df_raw.dropna(subset=req_cols).copy()

        # (1) 時間特徵
        df_raw['date'] = pd.to_datetime(df_raw['date'], errors='coerce')
        df_raw = df_raw.dropna(subset=['date'])
        df_raw['月份'] = df_raw['date'].dt.month.astype(int)

        # (2) 產品類別 (轉小寫後進行字典比對翻譯)
        df_raw['產品類別_英'] = df_raw['category'].astype(
            str).str.lower().str.strip()
        df_raw['產品類別'] = df_raw['產品類別_英'].map(
            CATEGORY_DICT).fillna(df_raw['產品類別_英'])

        # (3) 風險原因 (萃取大類後轉小寫，進行字典比對翻譯)
        df_raw['風險原因_英'] = df_raw['hazards'].astype(str).str.extract(
            r'\{(.*?)\}')[0].fillna('other/mixed').str.lower().str.strip()
        df_raw['風險原因'] = df_raw['風險原因_英'].map(
            HAZARD_DICT).fillna(df_raw['風險原因_英'])

        # (4) 預測目標：定義高風險
        high_risk_labels = ['serious', 'potentially serious']
        df_raw['是否為高風險'] = df_raw['risk_decision'].astype(
            str).str.lower().isin(high_risk_labels).astype(int)

        return df_raw[['月份', '產品類別', '風險原因', '是否為高風險']]

    except Exception as e:
        st.error(f"資料處理失敗: {e}")
        return pd.DataFrame()


# 檔案路徑設定
FILE_PATH = "RASFF 202001-202512.xlsx"
df = load_and_preprocess_data(FILE_PATH)

# 3. 模型訓練


@st.cache_resource
def train_xgboost_model(_df):
    if _df.empty:
        return None, None, None

    le_cat = LabelEncoder()
    le_risk = LabelEncoder()

    _df['cat_encoded'] = le_cat.fit_transform(_df['產品類別'])
    _df['risk_encoded'] = le_risk.fit_transform(_df['風險原因'])

    X = _df[['月份', 'cat_encoded', 'risk_encoded']]
    y = _df['是否為高風險']

    pos_weight = (len(y) - y.sum()) / y.sum() if y.sum() > 0 else 1

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=pos_weight,
        random_state=42
    )
    model.fit(X, y)
    return model, le_cat, le_risk


model, le_cat, le_risk = train_xgboost_model(df)

# 4. 預測介面
if model:
    st.sidebar.header("🔮 預測情境設定")
    target_month = st.sidebar.slider("目標月份", 1, 12, 4)
    target_cat = st.sidebar.selectbox("待測產品類別", options=le_cat.classes_)
    target_risk = st.sidebar.selectbox("待測風險因子", options=le_risk.classes_)

    # 匯出資料集功能 (維護工具)
    st.sidebar.markdown("---")
    st.sidebar.subheader("📥 維護工具：匯出當前欄位清單")
    st.sidebar.caption("若清單中出現英文，可將其加入程式碼的翻譯字典中。")

    cat_csv = pd.DataFrame({'產品類別清單': le_cat.classes_}).to_csv(
        index=False).encode('utf-8-sig')
    risk_csv = pd.DataFrame({'風險因子清單': le_risk.classes_}).to_csv(
        index=False).encode('utf-8-sig')

    st.sidebar.download_button(
        label="⬇️ 下載產品類別清單",
        data=cat_csv,
        file_name="rasff_categories_list.csv",
        mime="text/csv",
    )
    st.sidebar.download_button(
        label="⬇️ 下載風險因子清單",
        data=risk_csv,
        file_name="rasff_hazards_list.csv",
        mime="text/csv",
    )

    input_data = pd.DataFrame([[
        target_month,
        le_cat.transform([target_cat])[0],
        le_risk.transform([target_risk])[0]
    ]], columns=['月份', 'cat_encoded', 'risk_encoded'])

    prob = model.predict_proba(input_data)[0][1]

    # 5. 畫面呈現
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🔍 模型預測結果")
        st.metric(
            label=f"{target_month}月【{target_cat}】出現【{target_risk}】之高風險機率",
            value=f"{prob:.1%}"
        )
        if prob > 0.6:
            st.warning("⚠️ 預警：此組合在 RASFF 歷史通報中屬於高嚴重度，建議加強抽驗或監控。")
        else:
            st.success("✅ 趨勢分析：目前該組合歷史嚴重程度預期較低。")

    with col2:
        st.subheader("📊 特徵影響力")
        importance = pd.DataFrame({
            '特徵': ['季節月份', '產品類別', '風險原因'],
            '權重': model.feature_importances_
        }).sort_values(by='權重', ascending=False)
        st.bar_chart(importance.set_index('特徵'))

else:
    st.warning("模型訓練失敗，請確認資料集內容。")
