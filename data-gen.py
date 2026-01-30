# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 05:26:02 2026

@author: hhh
"""

import json
import time
from openai import OpenAI
from tqdm import tqdm

# ================= 配置区 =================
# 请务必替换为你重新生成的 Key！
API_KEY = "sk-fayesrhyhwqgavlwoynrzkoxkkiduatbywjvqkmcffjdlvyw" 
BASE_URL = "https://api.siliconflow.cn/v1"
# 硅基流动上的免费/便宜模型，用来造数据
TEACHER_MODEL = "Qwen/Qwen2.5-7B-Instruct" 

# 模拟的金融话题，为了让数据多样化

TOPICS = [
    # --- 宏观经济 ---
    "美联储加息路径预测", "CPI数据对A股影响", "人民币汇率波动逻辑", "LPR利率调整解读", "M2增速与通胀关系",
    # --- 行业分析 ---
    "新能源车企价格战", "光伏行业产能过剩", "AI算力芯片需求", "房地产政策放松", "白酒库存周期",
    "创新药出海逻辑", "半导体国产替代", "低空经济概念股", "猪周期拐点判断", "跨境电商物流成本",
    # --- 个股/财务 ---
    "贵州茅台现金流分析", "宁德时代毛利率变化", "腾讯控股回购计划", "中芯国际资本开支", "比亚迪研发投入",
    "ROE与股价关系", "商誉减值风险", "应收账款周转率", "经营性现金流为负", "非经常性损益解读",
    # --- 量化/交易 ---
    "多因子选股策略", "网格交易法优缺点", "R-Breaker策略", "MACD背离形态", "布林带突破策略",
    "高频交易延迟优化", "雪球产品敲入风险", "股指期货基差套利", "融资融券杠杆风险", "量化私募回撤控制"
]
# ==================================================

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def generate_qa(topic):
    """让大模型针对特定话题生成微调数据"""
    prompt = f"""
    你是一位资深的金融行业分析师。请针对话题“{topic}”，编写 3 个专业的问答对。
    
    要求：
    1. 问题(instruction)要具体，模拟投资人或初级研究员的提问。
    2. 回答(output)要专业、详尽，包含逻辑推导，字数在100-200字之间。
    3. 格式必须是标准的 JSON 列表，不要包含 Markdown 标记。
    
    格式示例：
    [
        {{"instruction": "如何看待某公司库存周转率上升？", "input": "", "output": "库存周转率上升通常意味着..."}},
        {{"instruction": "解释一下Alpha和Beta的区别", "input": "", "output": "Alpha代表超额收益..."}}
    ]
    """
    
    try:
        response = client.chat.completions.create(
            model=TEACHER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        content = response.choices[0].message.content
        # 清洗可能存在的 markdown 符号
        content = content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)
    except Exception as e:
        print(f"Error generating data for {topic}: {e}")
        return []

def main():
    dataset = []
    print(f"开始生成数据，共 {len(TOPICS)} 个话题...")
    
    # 循环生成
    for topic in tqdm(TOPICS):
        qa_list = generate_qa(topic)
        dataset.extend(qa_list)
        time.sleep(1) # 防止速率限制
        
    # 保存为 LLaMA-Factory 支持的 Alpaca 格式
    output_file = "financial_fine_tuning_data.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
        
    print(f"\n✅ 数据生成完毕！共 {len(dataset)} 条数据，已保存至 {output_file}")
    print("请打开文件检查一下格式是否正确。")

if __name__ == "__main__":
    main()