import json
import requests
import re
import os
import hashlib
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= 配置区 =================
API_URL = "http://localhost:8001/v1/chat/completions"
INPUT_PATH = "/home/zhouyan/share/cyh/CoT/3k_test_data/data_sample_3k.jsonl"
OUTPUT_PATH = "/home/zhouyan/share/cyh/CoT/3k_test_data/judge_data_3k.jsonl" 

# --- 防爆显存与并发控制 ---
MAX_CHAR_LIMIT = 12000  # 单个 CoT 字符数上限，超过该值判定为死循环，直接丢弃
MAX_WORKERS = 4         # 配合 vLLM 的 --max-num-seqs 2，设置 2-4 最稳，防止请求大量堆积超时
REQUEST_TIMEOUT = 300   # 长文本打分耗时较长，增加超时容忍度 (5分钟)
# ==========================================

def get_expert_compare_prompt(problem, cot1, cot2):
    return f"""You are an expert evaluator in mathematical reasoning and Chain-of-Thought (CoT) processes.
You will be given a mathematical problem and two AI-generated reasoning traces (wrapped in <thinking> tags).

Your task is to compare the two traces and decide which one has higher learning value and quality based on the critical dimensions:
1. Problem Difficulty (Easy vs Hard)
2. Reasoning Depth and Detail (Shallow vs Deep)
3. Self-Verification and Reflection (Absent vs Present)
4. Exploratory Approach (Linear vs Exploratory)
5. Logical Cohesion and Adaptive Granularity (Disjointed vs Cohesive)

Problem: {problem}

---
[Reasoning Trace 1]:
{cot1}

---
[Reasoning Trace 2]:
{cot2}

---
Final Output Requirement (Strictly JSON format):
{{
  "better_index": 1 or 2,
  "difficulty_level": "Easy" or "Hard",
  "quality_tags": ["Deep", "Present", "Exploratory", "Cohesive"],
  "brief_reason": "Why is the chosen one better?"
}}"""

def get_hash_id(problem):
    """通过对 problem 求 hash，作为这条数据的唯一标识，用于断点续跑"""
    return hashlib.md5(problem.encode('utf-8')).hexdigest()

def process_item(item):
    try:
        problem = item['problem']
        cots = item['generations']
        verifications = item.get('correctness_math_verify', [False, False])
        
        # 【新增】安全提取 Ground Truth 字段
        solution = item.get('solution', '')
        answer = item.get('answer', '')
        
        # 🛑 拦截器：检测超长死循环数据
        # 暂时保持原样：只取前两条进行简单暴力的限长拦截
        if len(cots) < 2 or len(cots[0]) > MAX_CHAR_LIMIT or len(cots[1]) > MAX_CHAR_LIMIT:
            return None # 默默丢弃，不计入最终结果
        
        prompt = get_expert_compare_prompt(problem, cots[0], cots[1])
        
        payload = {
            "model": "Qwen2.5-32B-Instruct",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1, 
            "response_format": {"type": "json_object"} 
        }
        
        r = requests.post(API_URL, json=payload, timeout=REQUEST_TIMEOUT)
        r.raise_for_status() # 如果返回 400/500 等错误码，直接抛出异常被 except 捕获
        
        response_json = r.json()
        result_text = response_json['choices'][0]['message']['content']
        result = json.loads(result_text)
        
        idx = int(result.get('better_index', 1)) - 1
        # 防止模型胡言乱语返回除了 1 和 2 之外的索引
        if idx not in [0, 1]: 
            idx = 0 
        
        # 整合结果，增加 solution 和 answer 字段
        return {
            "problem_hash": get_hash_id(problem), 
            "instruction": problem,
            "output": cots[idx],
            "solution": solution,  # 【新增】
            "answer": answer,      # 【新增】
            "metadata": {
                "source_idx": idx,
                "is_correct": verifications[idx] if idx < len(verifications) else False,
                "other_is_correct": verifications[1-idx] if (1-idx) < len(verifications) else False,
                "difficulty": result.get('difficulty_level', 'Unknown'),
                "quality_tags": result.get('quality_tags', []),
                "judge_reason": result.get('brief_reason', '')
            }
        }
    except Exception as e:
        # 如果报错（如 JSON 解析失败、请求超时），返回 None 跳过
        # print(f"Error processing item: {e}") 
        return None

def main():
    print(f"📦 正在读取基础数据...")
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        all_data = [json.loads(line) for line in f]

    # --- 🔄 断点续跑逻辑 ---
    processed_hashes = set()
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        processed_hashes.add(json.loads(line)['problem_hash'])
                    except KeyError:
                        pass
        print(f"🔄 检测到本地已有 {len(processed_hashes)} 条进度，正在剔除已处理数据...")

    # 过滤出还没处理过的数据
    data_to_process = [item for item in all_data if get_hash_id(item['problem']) not in processed_hashes]
    
    if not data_to_process:
        print("✅ 所有数据均已处理完毕，无需重复执行！")
        return

    print(f"⚖️ 正在进行全量二选一 + 深度画像 (剩余任务数: {len(data_to_process)})...")
    
    # 采用 as_completed 实现实时写入，出一根写一根，杜绝断电数据丢失
    with open(OUTPUT_PATH, 'a', encoding='utf-8') as f_out:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 提交所有任务
            futures = {executor.submit(process_item, item): item for item in data_to_process}
            
            # 使用 tqdm 包装 as_completed 呈现动态进度条
            for future in tqdm(as_completed(futures), total=len(futures), desc="🚀 Evaluating"):
                result = future.result()
                if result is not None:
                    # 实时安全写入单行 JSON
                    f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                    f_out.flush() # 强制刷新缓冲，确保落盘

    print(f"\n🎉 评测及清洗全部完成！高质量 SFT 数据已存至:\n📂 {OUTPUT_PATH}")

if __name__ == "__main__":
    main()