import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# 1. 頁面配置
st.set_page_config(page_title="食研院-AI風險預測系統", layout="wide")
st.title("🛡️ 國際食安風險：AI 預測與趨勢探索模式")

# 2. 模擬數據


@st.cache_data
def get_historical_data():
    np.random.seed(42)
    data = {
        '月份': np.random.randint(1, 13, 100),
        '產品類別': np.random.choice(['冷凍蔬菜', '堅果', '水果', '香料', '海鮮'], 100),
        '風險原因': np.random.choice(['農藥', '毒素', '微生物', '添加物'], 100),
        '通報強度': np.random.randint(1, 10, 100)
    }
    df = pd.DataFrame(data)
    df['是否為高風險'] = (df['通報強度'] > 7).astype(int)
    return df


df = get_historical_data()

# --- 修改處：新增快取訓練函數 ---


@st.cache_resource
def train_xgboost_model(_df):
    # 轉換類別資料
    le_cat = LabelEncoder()
    le_risk = LabelEncoder()
    _df['cat_encoded'] = le_cat.fit_transform(_df['產品類別'])
    _df['risk_encoded'] = le_risk.fit_transform(_df['風險原因'])

    # 定義特徵與目標
    X = _df[['月份', 'cat_encoded', 'risk_encoded']]
    y = _df['是否為高風險']

    # 建立模型
    model = xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1)
    model.fit(X, y)
    return model, le_cat, le_risk


# 執行訓練並獲取快取結果
model, le_cat, le_risk = train_xgboost_model(df)
# ----------------------------

# 4. 預測介面
st.sidebar.header("🔮 預測情境設定")
target_month = st.sidebar.slider("目標月份", 1, 12, 4)
target_cat = st.sidebar.selectbox("待測產品類別", options=le_cat.classes_)
target_risk = st.sidebar.selectbox("待測風險因子", options=le_risk.classes_)

# 執行預測
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
        label=f"{target_month}月 {target_cat} 出現 {target_risk} 之風險機率", value=f"{prob:.1%}")
    if prob > 0.6:
        st.warning("⚠️ 預警：該組合歷史通報頻率偏高，建議加強抽驗。")
    else:
        st.success("✅ 趨勢分析：目前風險處於安全監控範圍。")

with col2:
    st.subheader("📊 特徵影響力 (Feature Importance)")
    importance = pd.DataFrame({
        '特徵': ['季節月份', '產品品項', '風險因子'],
        '權重': model.feature_importances_
    }).sort_values(by='權重', ascending=False)
    st.bar_chart(importance.set_index('特徵'))

st.markdown("---")
st.caption("技術註解：本系統採用 XGBoost 分類器，分析國際食安通報之非線性關聯，達成主動式預警。")
