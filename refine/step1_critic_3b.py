import json
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= 全局配置区 =================
INPUT_FILE = "/home/zhouyan/share/cyh/CoT/select/refined_results/low_quality_math.json"
OUTPUT_FILE = "/home/zhouyan/share/cyh/CoT/refine/refined_results/critic_tagged_math.jsonl"

API_URL_3B = "http://localhost:8001/v1/chat/completions"
MODEL_NAME_3B = "Qwen2.5-3B-Instruct"

MAX_WORKERS = 32   # 3B 模型，高并发
REQUEST_TIMEOUT = 120
# ==========================================

def call_critic_3b(instruction, old_cot):
    # 【保留】全英文硬核诊断 Prompt，保证诊断逻辑准确
    prompt = f"""You are an expert mathematical logic diagnostician. Please read the following math problem and a low-quality reasoning trace. 
This reasoning trace contains redundancies, dead loops, or logical jumps. Identify its core issue.

Problem: 
{instruction}

Reasoning Trace: 
{old_cot[:3000]} ... (truncated)

Strictly output a JSON report containing the following fields:
- "error_type": A short categorization (e.g., "Dead Loop", "Logic Jump", "Over-complication", "Calculation Error").
- "error_span": Briefly describe where the error begins or which step derailed the logic.
- "advice": Specific suggestions for rewriting and avoiding the previous dead ends."""

    payload = {
        "model": MODEL_NAME_3B,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "response_format": {"type": "json_object"}
    }
    
    r = requests.post(API_URL_3B, json=payload, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return json.loads(r.json()['choices'][0]['message']['content'])

def process_critic(item):
    try:
        instruction = item.get('instruction', '')
        old_cot = item.get('output', '')
        
        critic_report = call_critic_3b(instruction, old_cot)
        
        if 'metadata' not in item:
            item['metadata'] = {}
        item['metadata']['critic_report'] = critic_report
        
        return item
    except Exception as e:
        return None

def main():
    print(f"📦 正在加载低质量样本库: {INPUT_FILE}")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        low_quality_data = json.load(f)

    print(f"🩺 开始全量病理诊断 (总任务数: {len(low_quality_data)})...")
    
    success_count = 0
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_critic, item): item for item in low_quality_data}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="🚀 诊断扫描中"):
                result = future.result()
                if result is not None:
                    f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                    f_out.flush()
                    success_count += 1

    print("\n" + "="*50)
    print("📋 阶段一：病理诊断任务完成！")
    print(f"✅ 成功打标并保存: {success_count} / {len(low_quality_data)} 条")
    print(f"📁 诊断结果已存至: {OUTPUT_FILE}")
    print("="*50)

if __name__ == "__main__":
    main()