import streamlit as st
import pandas as pd
import plotly.express as px
import requests

st.set_page_config(page_title="進口食品邊境查驗儀表板", layout="wide", page_icon="🛡️")


@st.cache_data(ttl=3600)
def load_real_data():
    api_url = "https://data.fda.gov.tw/opendata/exportDataList.do?method=ExportData&InfoId=52&logType=5"
    try:
        resp = requests.get(api_url, timeout=15)
        df = pd.DataFrame(resp.json())

        if '發布日期' in df.columns:
            df['發布日期'] = pd.to_datetime(df['發布日期'], errors='coerce')
            # 新增「年月」欄位供時間序列分析
            df['年月'] = df['發布日期'].dt.to_period('M').astype(str)

        df = df.fillna('未提供')
        return df
    except Exception as e:
        st.error(f"API 載入失敗: {e}")
        return pd.DataFrame()


# --- 資料載入 ---
with st.spinner('同步政府最新數據中...'):
    df = load_real_data()

if df.empty:
    st.stop()

# --- 側邊欄：篩選與匯出 ---
st.sidebar.header("🔍 條件篩選")

# 產地篩選
if '產地' in df.columns:
    default_countries = df['產地'].value_counts().head(5).index.tolist()
    selected_country = st.sidebar.multiselect(
        "選擇產地", df['產地'].unique(), default=default_countries)
else:
    selected_country = []

# 關鍵字篩選
search_term = st.sidebar.text_input("搜尋 (產品/進口商/原因)")

# 套用篩選
filtered_df = df.copy()
if selected_country and '產地' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['產地'].isin(selected_country)]

if search_term:
    filtered_df = filtered_df[
        filtered_df['主旨'].str.contains(search_term, na=False) |
        filtered_df['進口商名稱'].str.contains(search_term, na=False) |
        filtered_df['原因'].str.contains(search_term, na=False)
    ]

# 匯出按鈕
st.sidebar.markdown("---")
st.sidebar.subheader("📥 資料匯出")
csv = filtered_df.to_csv(index=False).encode(
    'utf-8-sig')  # 使用 utf-8-sig 避免 Excel 中文亂碼
st.sidebar.download_button(
    label="下載當前篩選資料 (CSV)",
    data=csv,
    file_name='food_safety_filtered.csv',
    mime='text/csv',
)

# --- 頂部 KPI ---
st.title("🚢 進口食品邊境查驗違規監控")
col1, col2, col3, col4 = st.columns(4)
col1.metric("總違規件數", len(filtered_df))
col2.metric("主要違規產地", filtered_df['產地'].mode()[0] if (
    '產地' in filtered_df.columns and not filtered_df.empty) else "無")
col3.metric("最大違規進口商", filtered_df['進口商名稱'].mode()[0] if (
    '進口商名稱' in filtered_df.columns and not filtered_df.empty) else "無")
col4.metric("最常見原因", filtered_df['原因'].mode()[0] if (
    '原因' in filtered_df.columns and not filtered_df.empty) else "無")

st.markdown("---")

# --- 視覺化圖表區塊 ---
if not filtered_df.empty:
    # 第一排：時間趨勢 & 產地佔比
    row1_col1, row1_col2 = st.columns([2, 1])

    with row1_col1:
        st.subheader("📈 歷月違規通報趨勢")
        if '年月' in filtered_df.columns:
            trend_df = filtered_df.groupby('年月').size(
            ).reset_index(name='件數').sort_values('年月')
            fig_trend = px.line(trend_df, x='年月', y='件數', markers=True)
            fig_trend.update_layout(xaxis_title="時間", yaxis_title="違規件數")
            st.plotly_chart(fig_trend, use_container_width=True)

    with row1_col2:
        st.subheader("📍 各國違規佔比")
        if '產地' in filtered_df.columns:
            country_counts = filtered_df['產地'].value_counts().reset_index()
            country_counts.columns = ['產地', '件數']
            fig_pie = px.pie(country_counts, names='產地', values='件數', hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)

    # 第二排：違規原因 & 黑名單排行
    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        st.subheader("⚠️ 主要不合格原因 (Top 10)")
        if '原因' in filtered_df.columns:
            reason_counts = filtered_df['原因'].value_counts().head(
                10).reset_index()
            reason_counts.columns = ['原因', '件數']
            fig_bar_reason = px.bar(
                reason_counts, x='件數', y='原因', orientation='h')
            fig_bar_reason.update_layout(
                yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_bar_reason, use_container_width=True)

    with row2_col2:
        st.subheader("🏢 進口商累犯排行 (Top 10)")
        if '進口商名稱' in filtered_df.columns:
            importer_counts = filtered_df['進口商名稱'].value_counts().head(
                10).reset_index()
            importer_counts.columns = ['進口商名稱', '件數']
            fig_bar_importer = px.bar(
                importer_counts, x='件數', y='進口商名稱', orientation='h', color_discrete_sequence=['#ef553b'])
            fig_bar_importer.update_layout(
                yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_bar_importer, use_container_width=True)

# --- 數據表格 ---
st.markdown("---")
st.subheader("📄 原始明細資料")
display_cols = [col for col in ['發布日期', '產地', '主旨',
                                '原因', '進口商名稱', '處置情形'] if col in filtered_df.columns]
st.dataframe(filtered_df[display_cols].sort_values(by='發布日期', ascending=False)
             if '發布日期' in filtered_df.columns else filtered_df[display_cols], use_container_width=True)
