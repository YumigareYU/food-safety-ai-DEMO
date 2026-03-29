import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# 1. 頁面配置
st.set_page_config(page_title="食研院-AI風險預測系統", layout="wide")
st.title("🛡️ 國際食安風險：AI 預測與趨勢探索模式")

# 2. 模擬數據 (代表從歐盟 RASFF 或 FDA 爬取後的結構化資料)


@st.cache_data
def get_historical_data():
    np.random.seed(42)
    # 模擬 100 筆歷史通報紀錄
    data = {
        '月份': np.random.randint(1, 13, 100),
        '產品類別': np.random.choice(['冷凍蔬菜', '堅果', '水果', '香料', '海鮮'], 100),
        '風險原因': np.random.choice(['農藥', '毒素', '微生物', '添加物'], 100),
        '通報強度': np.random.randint(1, 10, 100)  # 1-10 代表嚴重程度
    }
    df = pd.DataFrame(data)
    # 建立目標變數：下個月是否會發生「嚴重通報」(強度 > 7)
    df['是否為高風險'] = (df['通報強度'] > 7).astype(int)
    return df


df = get_historical_data()

# 3. 模型訓練 (XGBoost)
# 轉換類別資料為數值
le_cat = LabelEncoder()
le_risk = LabelEncoder()
df['cat_encoded'] = le_cat.fit_transform(df['產品類別'])
df['risk_encoded'] = le_risk.fit_transform(df['風險原因'])

# 定義特徵與目標
X = df[['月份', 'cat_encoded', 'risk_encoded']]
y = df['是否為高風險']

model = xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1)
model.fit(X, y)

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

prob = model.predict_proba(input_data)[0][1]  # 取得高風險機率

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
    # 展現 XGBoost 的解釋力
    importance = pd.DataFrame({
        '特徵': ['季節月份', '產品品項', '風險因子'],
        '權重': model.feature_importances_
    }).sort_values(by='權重', ascending=False)
    st.bar_chart(importance.set_index('特徵'))

st.markdown("---")
st.caption("技術註解：本系統採用 XGBoost 分類器，分析國際食安通報之非線性關聯，達成主動式預警。")
