# 🏦 Financial-Agent-Qwen: 基于 Qwen 大模型的金融投研智能体

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LLM](https://img.shields.io/badge/LLM-Qwen2.5--72B-green)
![Framework](https://img.shields.io/badge/Framework-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📖 项目简介 (Introduction)

**Financial-Agent-Qwen** 是一个面向金融垂直领域的智能投研助手，旨在解决通用大模型在金融场景下“知识滞后”与“幻觉”的痛点。

本项目采用了 **"Base Model + RAG + Agent"** 的复合架构：
1.  **微调阶段**：使用 LLaMA-Factory 对 Qwen2.5-1.5B 进行指令微调 (SFT)，验证了 LoRA 在金融垂直语料上的有效性。
2.  **应用阶段**：基于 Qwen2.5-72B (via SiliconFlow API) 构建 Agent，结合 **Function Calling** 技术，实现了 A 股实时行情查询、个股新闻检索及互联网宏观资讯搜索。

## ✨ 核心功能 (Key Features)

- **📈 A 股实时行情 (Real-time A-Share Data)**
    - 集成 `AkShare` 开源财经数据接口。
    - 支持股票代码 (如 `600519`) 及名称模糊搜索。
    - 覆盖个股、ETF、基金等多种标的。

- **🌍 联网 RAG 能力 (Web Search RAG)**
    - 集成 `DuckDuckGo` 搜索引擎。
    - 当用户询问美股、黄金、宏观经济等非 A 股数据时，自动联网检索最新资讯，打破模型知识截止日期的限制。

- **🤖 鲁棒的 Agent 架构 (Robust Agentic Workflow)**
    - 基于 **ReAct** 范式，模型自主决策工具调用。
    - **自适应解析层**：针对 LLM 输出不稳定的 JSON 格式问题，设计了中间件进行自动纠错与类型强制转换，系统稳定性极高。

- **💻 交互式 Web 终端 (Interactive Web UI)**
    - 使用 `Streamlit` 打造现代化聊天界面。
    - 支持多轮对话记忆、流式输出与工具调用结果的可视化渲染。

## 🛠️ 技术栈 (Tech Stack)

- **LLM Core**: Qwen/Qwen2.5-72B-Instruct (SiliconFlow API)
- **Fine-tuning**: LLaMA-Factory, LoRA, PyTorch
- **Agent Framework**: OpenAI SDK (Compatible), Function Calling
- **Data Providers**: AkShare (A-Share), DuckDuckGo (Web Search)
- **Frontend**: Streamlit, Pandas

## 🚀 快速开始 (Quick Start)

### 1. 环境准备
确保你的本地环境已安装 Python 3.10+。

```bash
git clone https://github.com/YourUsername/Financial-Agent-Qwen.git
cd Financial-Agent-Qwen
```

### 2. 安装依赖
建议使用 Conda 创建虚拟环境：

```bash
conda create -n finance_agent python=3.10
conda activate finance_agent
pip install -r requirements.txt
```

*`requirements.txt` 内容参考:*
```text
streamlit
akshare
openai
pandas
duckduckgo-search
```

### 3. 运行应用
启动 Streamlit 服务：

```bash
streamlit run app.py
```

### 4. 配置 API Key
在打开的网页侧边栏中，输入你的 **SiliconFlow API Key** 即可开始对话。

## 📊 效果展示 (Demo)

### 1. 智能体对话界面
> 自动识别意图，调用 AkShare 查询股价数据

### 2. 联网搜索能力

### 3. 微调训练监控
> Qwen-1.5B LoRA 微调过程中的 Loss 下降曲线如图。

## 🤝 致谢 (Acknowledgements)

- [Qwen (通义千问)](https://github.com/QwenLM/Qwen) 提供强大的基座模型。
- [AkShare](https://github.com/akfamily/akshare) 提供免费且稳定的金融数据接口。
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 提供高效的微调框架。
- [SiliconFlow](https://siliconflow.cn/) 提供稳定的 API 服务。

---
*Created by [你的名字] | 2026*
