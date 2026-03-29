import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import requests

# 1. 頁面配置
st.set_page_config(page_title="AI 數據即時爬取演示", layout="wide")
st.title("🌐 全球地震數據：即時爬取與 AI 風險預測")

# 2. 【真爬蟲實作】直接抓取維基百科表格


@st.cache_data
def fetch_wiki_data():
    # 目標：維基百科「21世紀地震列表」
    url = "https://zh.wikipedia.org/wiki/21%E4%B8%96%E7%B4%80%E5%9C%B0%E9%9C%87%E5%88%97%E8%A1%A8"

    # 執行爬取 (使用 pandas 內建的 read_html，底層是 BeautifulSoup)
    with st.spinner("正在連線維基百科，即時解析網頁表格..."):
        tables = pd.read_html(url)
        # 通常表格在第 2 或第 3 個，我們抓取具備「震級」欄位的表
        df = tables[1]

    # 資料清洗 (ETL)
    # 簡化欄位，確保模型能跑
    df = df[['日期', '震級', '地點']].dropna().head(100)
    # 將震級轉為浮點數
    df['震級'] = pd.to_numeric(df['震級'].astype(
        str).str.extract(r'(\d+\.\d+)')[0], errors='coerce')
    df = df.dropna()

    # 建立目標變數：震級是否大於 7.5 (高風險)
    df['高風險地震'] = (df['震級'] > 7.5).astype(int)
    return df


# 3. 側邊欄控制
st.sidebar.header("🛠️ 數據自動化控制")
if st.sidebar.button("🔍 啟動即時網頁爬蟲"):
    df = fetch_wiki_data()
    st.session_state['wiki_df'] = df
    st.sidebar.success("✅ 爬取成功！已獲取最新地震數據")
elif 'wiki_df' in st.session_state:
    df = st.session_state['wiki_df']
else:
    st.info("請點擊左側按鈕，現場執行網頁爬蟲演示。")
    st.stop()

# 4. 【AI 建模】XGBoost 訓練


@st.cache_resource
def train_earthquake_model(_df):
    le_loc = LabelEncoder()
    _df['loc_encoded'] = le_loc.fit_transform(_df['地點'])

    X = _df[['loc_encoded']]
    y = _df['高風險地震']

    model = xgb.XGBClassifier(n_estimators=50)
    model.fit(X, y)
    return model, le_loc


model, le_loc = train_earthquake_model(df)

# 5. 畫面呈現
st.subheader("📋 現場爬取之原始數據 (維基百科)")
st.write(df[['日期', '地點', '震級']].head(10))

col1, col2 = st.columns(2)
with col1:
    st.subheader("🔮 區域風險模擬預測")
    selected_loc = st.selectbox("選擇預測地點", options=le_loc.classes_)

    # 推理預測
    input_val = pd.DataFrame(
        [[le_loc.transform([selected_loc])[0]]], columns=['loc_encoded'])
    prob = model.predict_proba(input_val)[0][1]

    st.metric(f"{selected_loc} 發生強震機率", f"{prob:.1%}")
    st.progress(prob)

with col2:
    st.subheader("📈 震級分佈趨勢")
    st.line_chart(df['震級'])

st.markdown("---")
st.caption("技術亮點：本 Demo 演示了如何從公開網頁直接獲取非結構化數據，並即時轉化為機器學習模型的訓練特徵。")
