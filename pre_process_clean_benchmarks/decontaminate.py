import pandas as pd
import json
import os
import re
import requests
from tqdm import tqdm
from glob import glob

# --- 配置区 ---
# 1. Benchmark 路径
BENCHMARK_PATHS = {
    "math500": "/mdr5/user/quantaalpha/zhouyan/cyh/benchmarks/HuggingFaceH4_MATH-500/test.jsonl",
    "amc23": "/mdr5/user/quantaalpha/zhouyan/cyh/benchmarks/math-ai_amc23/test-00000-of-00001.parquet",
    "aime2025": "/mdr5/user/quantaalpha/zhouyan/cyh/benchmarks/opencompass_AIME2025/aime2025.jsonl"
}

# 2. 待清洗数据路径 (10个分片)
DATASET_SHARDS = glob("/mdr5/user/quantaalpha/zhouyan/cyh/datasets/raw_data/OpenR1-Math-220k/data/train-*.parquet")
OUTPUT_DIR = "/mdr5/user/quantaalpha/zhouyan/cyh/datasets/cleaned_data/OpenR1-Math-220k"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 3. 本地小模型 API 配置 (假设你用 vLLM 或 Ollama 启动了 OpenAI 兼容接口)
API_URL = "http://127.0.0.1:8001/v1/chat/completions"
MODEL_NAME = "/nfsdata-117/model/Qwen2.5-3B-Instruct" # 替换为你本地部署的模型名

# --- 工具函数 ---

def normalize_answer(ans):
    """简单清洗答案字符串，统一格式以便比对"""
    if ans is None: return ""
    ans = str(ans).strip().lower()
    ans = re.sub(r'\\boxed\{(.*?)\}', r'\1', ans) # 去掉 \boxed
    ans = ans.replace('$', '').replace(' ', '') # 去掉符号和空格
    return ans

def is_problem_same_llm(p1, p2):
    """调用本地小模型判断两个数学问题是否语义相同"""
    prompt = f"""判断以下两个数学问题是否是同一个题目（可能叙述方式不同，但逻辑和数值完全一致）。
只需要回答 "YES" 或 "NO"。

问题 A: {p1}
问题 B: {p2}

是否相同？"""
    
    try:
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0
        }
        response = requests.post(API_URL, json=payload, timeout=10)
        res_text = response.json()['choices'][0]['message']['content'].strip().upper()
        return "YES" in res_text
    except Exception as e:
        print(f"LLM 调用失败: {e}")
        return False

# --- 核心逻辑 ---

# 1. 加载 Benchmark 所有的答案和问题，建立索引
print("🔍 正在加载 Benchmark 库并建立答案索引...")
benchmark_lookup = {} # { normalized_answer: [problem1, problem2, ...] }

for name, path in BENCHMARK_PATHS.items():
    if path.endswith('.jsonl'):
        df_bench = pd.read_json(path, lines=True)
    else:
        df_bench = pd.read_parquet(path)
    
    # 统一列名映射
    q_col = 'problem' if 'problem' in df_bench.columns else 'question'
    
    for _, row in df_bench.iterrows():
        ans = normalize_answer(row['answer'])
        prob = row[q_col]
        if ans not in benchmark_lookup:
            benchmark_lookup[ans] = []
        benchmark_lookup[ans].append(prob)

# 2. 处理 Dataset 分片
print(f"🚀 开始清洗 Dataset (共 {len(DATASET_SHARDS)} 个分片)...")

for shard_path in DATASET_SHARDS:
    shard_name = os.path.basename(shard_path)
    save_path = os.path.join(OUTPUT_DIR, shard_name)
    
    df_shard = pd.read_parquet(shard_path)
    initial_count = len(df_shard)
    drop_indices = []

    for idx, row in tqdm(df_shard.iterrows(), total=initial_count, desc=f"处理 {shard_name}"):
        train_ans = normalize_answer(row['answer'])
        train_prob = row['problem']
        
        # 步骤 1: 比较答案
        if train_ans in benchmark_lookup:
            # 步骤 2: 答案相同时，调用 LLM 判断问题语义
            for bench_prob in benchmark_lookup[train_ans]:
                if is_problem_same_llm(train_prob, bench_prob):
                    drop_indices.append(idx)
                    break # 只要匹配到一个 Benchmark 题目就删除
    
    # 执行删除操作
    df_cleaned = df_shard.drop(drop_indices)
    df_cleaned.to_parquet(save_path)
    
    print(f"✅ 分片 {shard_name} 清洗完成: 原始 {initial_count} 条 -> 剩余 {len(df_cleaned)} 条 (删除了 {len(drop_indices)} 条)")

print(f"✨ 所有数据清洗完毕！存至: {OUTPUT_DIR}")