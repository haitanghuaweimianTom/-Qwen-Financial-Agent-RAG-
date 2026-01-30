import streamlit as st
import akshare as ak
import json
import pandas as pd
from openai import OpenAI
from duckduckgo_search import DDGS  # 记得 pip install duckduckgo-search

# ================= 1. 页面配置 =================
st.set_page_config(page_title="AI全能投研助手", page_icon="🌍", layout="wide")

@st.cache_data(ttl=180)
def get_market_data_cache():
    try:
        df_stock = ak.stock_zh_a_spot_em()
        return df_stock[['代码', '名称', '最新价', '涨跌幅']]
    except: return pd.DataFrame()

# ================= 2. 三大核心工具 =================

def get_stock_price_pro(symbol):
    """查A股"""
    df = get_market_data_cache()
    if df.empty: return "行情接口异常"
    
    res = df[df['代码'] == symbol]
    if res.empty: res = df[df['名称'].str.contains(symbol)]
    
    if res.empty: return "未找到该A股标的"
    
    data = res.iloc[0]
    return json.dumps({
        "名称": data['名称'],
        "价格": data['最新价'],
        "涨幅": f"{data['涨跌幅']}%"
    }, ensure_ascii=False)

def get_stock_news(symbol):
    """查公告"""
    try:
        news = ak.stock_news_em(symbol=symbol).head(2)
        return json.dumps([f"{row['发布时间']} {row['新闻标题']}" for _, row in news.iterrows()], ensure_ascii=False)
    except: return "无最新公告"

def search_web(query):
    """🌍 查外网/美股"""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if not results: return "无搜索结果"
            return "\n\n".join([f"标题: {r['title']}\n内容: {r['body']}" for r in results])
    except Exception as e:
        return f"搜索报错: {e}"

# 工具列表
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price_pro",
            "description": "查A股、ETF价格。参数 symbol 为代码或名称。",
            "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_news",
            "description": "查A股个股新闻。参数 symbol 为代码。",
            "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "查美股、宏观经济、黄金走势、外盘数据。参数 query 为搜索词。",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
        }
    }
]

# ================= 3. 主逻辑 =================

st.title("🌍 AI全能投研助手 (联网版)")

with st.sidebar:
    api_key = st.text_input("API Key", type="password")
    if st.button("清空"): st.session_state.messages = []; st.rerun()

if not api_key: st.stop()
client = OpenAI(api_key=api_key, base_url="https://api.siliconflow.cn/v1")

if "messages" not in st.session_state: st.session_state.messages = []

for m in st.session_state.messages:
    role = m.get("role") if isinstance(m, dict) else m.role
    content = m.get("content") if isinstance(m, dict) else m.content
    if content:
        with st.chat_message(role):
            if isinstance(m, dict) and "tool_call_id" in m: st.code(content)
            else: st.write(content)

if prompt := st.chat_input("试试问：'美股黄金最近怎么走？'"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        # 构造消息
        msgs = [{"role": "system", "content": "你是专业投研助手。A股问题调get_stock_price，美股/宏观问题调search_web。"}]
        for m in st.session_state.messages:
            if isinstance(m, dict): msgs.append(m)
            else: msgs.append(m.model_dump())

        # 调用
        resp = client.chat.completions.create(
            model="Qwen/Qwen2.5-72B-Instruct",
            messages=msgs,
            tools=tools_schema
        )
        msg = resp.choices[0].message

        if msg.tool_calls:
            st.session_state.messages.append(msg)
            
            with st.status("🔍 联网检索中...") as s:
                call = msg.tool_calls[0]
                fname = call.function.name
                args = json.loads(call.function.arguments)
                
                # 统一参数提取
                val = args.get("symbol") or args.get("query")
                st.write(f"调取工具: {fname} | 关键词: {val}")
                
                if fname == "search_web": res = search_web(val)
                elif fname == "get_stock_price_pro": res = get_stock_price_pro(val)
                elif fname == "get_stock_news": res = get_stock_news(val)
                else: res = "Error"
                
                s.update(label="数据已获取", state="complete")
            
            st.session_state.messages.append({"role": "tool", "content": res, "tool_call_id": call.id})
            
            # 最终回答
            msgs.append(msg.model_dump())
            msgs.append({"role": "tool", "content": res, "tool_call_id": call.id})
            
            final = client.chat.completions.create(model="Qwen/Qwen2.5-72B-Instruct", messages=msgs)
            reply = final.choices[0].message.content
            st.write(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
        else:
            st.write(msg.content)
            st.session_state.messages.append({"role": "assistant", "content": msg.content})