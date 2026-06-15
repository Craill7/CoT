import json
import requests
import re
import os
import hashlib
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= 配置区 =================
API_URL = "http://localhost:8001/v1/chat/completions"
# 替换为你的真实路径
INPUT_PATH = "/home/zhouyan/share/cyh/CoT/sample_data/train_sampled_3k.jsonl"
OUTPUT_PATH = "/home/zhouyan/share/cyh/CoT/evaluation/profiled_data_3k.jsonl"

# --- 防爆与并发控制 ---
MAX_CHAR_LIMIT = 12000  # 超长死循环物理拦截线
MAX_WORKERS = 4         # A800 16K 上下文建议维持在 4 左右，太高容易排队超时
REQUEST_TIMEOUT = 300   # 长文本打分超时容忍度 (5分钟)
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
    result_text = ""  # 初始化，防止报错时变量未定义
    try:
        problem = item['problem']
        cots = item['generations']
        verifications = item.get('correctness_math_verify', [False, False])
        
        # 🛑 死因 1：超长死循环数据拦截
        if len(cots[0]) > MAX_CHAR_LIMIT or len(cots[1]) > MAX_CHAR_LIMIT:
            print(f"\n✂️ [物理拦截] 发现超长畸形数据 (len: {len(cots[0])} / {len(cots[1])})，直接丢弃！")
            return None
        
        prompt = get_expert_compare_prompt(problem, cots[0], cots[1])
        
        payload = {
            "model": "Qwen2.5-32B-Instruct",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1, 
            "response_format": {"type": "json_object"} 
        }
        
        # 发送请求
        r = requests.post(API_URL, json=payload, timeout=REQUEST_TIMEOUT)
        r.raise_for_status() 
        
        response_json = r.json()
        result_text = response_json['choices'][0]['message']['content']
        
        # 🛠️ 抢救方案：强制正则抠出 JSON（无视 Markdown 和开头废话）
        match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if match:
            clean_json_str = match.group(0)
        else:
            # 🛑 死因 2：完全没有 JSON 结构
            raise ValueError("Model output does not contain any JSON structure.")

        result = json.loads(clean_json_str)
        
        # 安全获取 index
        try:
            idx = int(result.get('better_index', 1)) - 1
            if idx not in [0, 1]: 
                idx = 0 
        except (ValueError, TypeError):
            idx = 0
        
        # 整合结果
        return {
            "problem_hash": get_hash_id(problem),
            "instruction": problem,
            "output": cots[idx],
            "metadata": {
                "source_idx": idx,
                "is_correct": verifications[idx],
                "other_is_correct": verifications[1-idx],
                "difficulty": result.get('difficulty_level', 'Unknown'),
                "quality_tags": result.get('quality_tags', []),
                "judge_reason": result.get('brief_reason', '')
            }
        }
        
    except requests.exceptions.RequestException as e:
        # 🛑 死因 3：并发太高导致 API 响应超时或拒绝连接
        print(f"\n⚠️ [网络/超时错误]: {e}")
        return None
    except json.JSONDecodeError as e:
        # 🛑 死因 4：提取出了花括号，但里面的 JSON 格式依然是碎的
        print(f"\n❌ [JSON 碎裂] 无法解析，原始输出:\n{result_text}")
        return None
    except Exception as e:
        # 其他未知死因
        print(f"\n🚨 [未知异常]: {e}")
        return None

def main():
    print(f"📦 正在读取基础数据: {INPUT_PATH}")
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        all_data = [json.loads(line) for line in f]

    # --- 🔄 智能断点续跑 ---
    processed_hashes = set()
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        processed_hashes.add(json.loads(line)['problem_hash'])
                    except KeyError:
                        pass
        print(f"🔄 检测到本地已有 {len(processed_hashes)} 条打分结果！")

    # 过滤出还没处理过的数据 (1796条)
    data_to_process = [item for item in all_data if get_hash_id(item['problem']) not in processed_hashes]
    
    if not data_to_process:
        print("✅ 所有数据均已处理完毕，无需重复执行！")
        return

    print(f"⚖️ 开始抢救剩余任务... (剩余数: {len(data_to_process)})")
    
    # 追加模式 'a'，实时落盘
    with open(OUTPUT_PATH, 'a', encoding='utf-8') as f_out:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_item, item): item for item in data_to_process}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="🚀 Evaluating"):
                result = future.result()
                if result is not None:
                    # 单条写入并强制刷盘，即使被 Kill 也不会丢数据
                    f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                    f_out.flush()

    print(f"\n🎉 评测及清洗全部完成！高质量 SFT 数据已存至:\n📂 {OUTPUT_PATH}")

if __name__ == "__main__":
    main()