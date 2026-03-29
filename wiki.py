import streamlit as st
import pandas as pd
import re
import requests
from bs4 import BeautifulSoup

# 1. 頁面配置
st.set_page_config(page_title="非結構化數據爬取與分析演示", layout="wide")
st.title("🌐 維基百科列表爬取：從文字到數據分析")

# 2. 爬蟲函式


@st.cache_data
def fetch_bullet_data():
    url = "https://zh.wikipedia.org/wiki/21%E4%B8%96%E7%B4%80%E5%9C%B0%E9%9C%87%E5%88%97%E8%A1%A8"
    headers = {'User-Agent': 'Mozilla/5.0'}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')

        content = soup.find('div', id='mw-content-text')
        main_output = content.find('div', class_='mw-parser-output')
        li_items = main_output.find_all('li')

        extracted_data = []
        for item in li_items:
            full_text = item.get_text().strip()

            # 1. 抓日期
            date_match = re.search(r"(\d{4}年\d{1,2}月\d{1,2}日)", full_text)
            if not date_match:
                continue

            # 2. 抓震級
            mag_matches = re.findall(r"(\d\.\d)", full_text)
            if not mag_matches:
                continue

            # 3. 抓地點並拼接 <a> 標籤內容
            a_tags = item.find_all('a')
            location_parts = []
            for a in a_tags:
                a_text = a.get_text().strip()
                if not re.match(r"^\[.*\]$", a_text) and a_text.lower() not in ['usgs', 'jma']:
                    # 清理「大地震」與「地震」
                    clean_part = a_text.replace("大地震", "").replace("地震", "")
                    location_parts.append(clean_part)

            clean_location = "".join(location_parts).strip()

            # 兜底清理
            if len(clean_location) < 2:
                after_date = full_text.split(date_match.group(1))[-1].strip()
                raw_loc = re.split(
                    r"[,，\s\(\（\[]|規模|地震", after_date)[0].strip()
                clean_location = raw_loc.replace("大地震", "").replace("地震", "")

            if len(clean_location) >= 2:
                extracted_data.append({
                    '日期': date_match.group(1),
                    '地點': clean_location,
                    '震級': float(mag_matches[-1]),
                    '年份': int(date_match.group(1)[:4])  # 額外提取年份做分析
                })

        df = pd.DataFrame(
            extracted_data).drop_duplicates().reset_index(drop=True)
        return df

    except Exception as e:
        st.error(f"連線或解析發生錯誤: {e}")
        return pd.DataFrame()


# 3. 側邊欄控制
st.sidebar.header("🛠️ 數據自動化控制")
if st.sidebar.button("🔍 啟動非結構化文字爬取"):
    df = fetch_bullet_data()
    st.session_state['bullet_df'] = df
    st.sidebar.success(f"✅ 爬取成功！已獲獲取 {len(df)} 筆數據")
elif 'bullet_df' in st.session_state:
    df = st.session_state['bullet_df']
else:
    st.info("請點擊左側按鈕，現場執行網頁爬蟲演示。")
    st.stop()

# 4. 數據分析展示
st.subheader("📋 現場爬取之原始數據 (維基百科)")
st.write(f"目前共獲取 {len(df)} 筆有效數據：")
st.dataframe(df[['日期', '地點', '震級']], use_container_width=True)

st.markdown("---")
st.subheader("📊 數據統計與分析結果")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("最高震級紀錄", f"M {df['震級'].max()}")
with col2:
    st.metric("平均震級", f"M {df['震級'].mean():.2f}")
with col3:
    st.metric("總計爬取筆數", f"{len(df)} 筆")

col_left, col_right = st.columns(2)

with col_left:
    st.write("📍 **地震熱點區域排行 (次數)**")
    # 簡單清理地名（取前兩個字，例如：印尼、台灣、日本）做聚合分析
    df['國家/大區域'] = df['地點'].str[:2]
    area_counts = df['國家/大區域'].value_counts()
    st.bar_chart(area_counts)

with col_right:
    st.write("📅 **各年度發生頻率**")
    year_counts = df['年份'].value_counts().sort_index()
    st.line_chart(year_counts)

st.markdown("---")
st.caption(
    "技術亮點：本 Demo 展示了如何透過正則表達式與 BeautifulSoup 從雜亂的網頁列表提取結構化數據，並進行即時 ETL 清理與初步統計分析。")
