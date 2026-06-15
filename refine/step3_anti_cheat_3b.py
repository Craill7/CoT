import json
import requests
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= 全局配置区 =================
CANDIDATES_FILE = "/home/zhouyan/share/cyh/CoT/refine/refined_results/step2_success_candidates.jsonl"
HIGH_QUALITY_BASE = "/home/zhouyan/share/cyh/CoT/select/refined_results/high_quality_math.json"

FINAL_SFT_OUTPUT = "/home/zhouyan/share/cyh/CoT/refine/refined_results/all_high_quality_math.jsonl"
CHEATING_BADCASE_OUTPUT = "/home/zhouyan/share/cyh/CoT/refine/refined_results/cheating_badcase.jsonl"

API_URL_3B = "http://localhost:8001/v1/chat/completions" # 确保此时后台跑的是 3B
MODEL_NAME_3B = "Qwen2.5-3B-Instruct"

MAX_WORKERS = 32   # 3B模型，尽情拉高并发
REQUEST_TIMEOUT = 120
# ==========================================

def check_logic_jump(instruction, cot, gt):
    prompt = f"""You are a strict math reasoning inspector. Evaluate whether the following mathematical Chain of Thought (CoT) demonstrates genuine, complete reasoning, or if it abruptly jumps to the Ground Truth answer without sufficient logical steps (a phenomenon known as "Reward Hacking" or "Cheating").

Problem: 
{instruction}

Ground Truth: 
{gt}

Chain of Thought to evaluate:
{cot}

Respond strictly in JSON format with the following fields:
- "is_cheating": true (if it skips crucial steps and forces the final answer) or false (if the reasoning is rigorous and complete).
- "reason": A brief explanation of your judgment."""

    payload = {
        "model": MODEL_NAME_3B,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "response_format": {"type": "json_object"}
    }
    
    r = requests.post(API_URL_3B, json=payload, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return json.loads(r.json()['choices'][0]['message']['content'])

def process_audit(item):
    try:
        instruction = item.get('instruction', '')
        cot = item.get('output', '')
        gt = item.get('answer', '')
        
        audit_report = check_logic_jump(instruction, cot, gt)
        
        if audit_report.get('is_cheating', False):
            item['metadata']['anti_cheat_report'] = audit_report
            return {"status": "cheat", "item": item}
        else:
            return {"status": "pass", "item": item}
    except Exception as e:
        return {"status": "error", "item": None}

def main():
    print(f"📦 正在加载待审候选数据: {CANDIDATES_FILE}")
    candidates = [json.loads(line) for line in open(CANDIDATES_FILE, 'r', encoding='utf-8')]

    # 1. 准备大本营数据
    if os.path.exists(HIGH_QUALITY_BASE):
        print(f"🔄 正在合并基础高质量池: {HIGH_QUALITY_BASE}")
        base_data = json.load(open(HIGH_QUALITY_BASE, 'r', encoding='utf-8'))
        with open(FINAL_SFT_OUTPUT, 'w', encoding='utf-8') as f_out:
            for item in base_data:
                f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
    else:
        open(FINAL_SFT_OUTPUT, 'w').close()
        
    open(CHEATING_BADCASE_OUTPUT, 'w').close()

    print(f"🕵️ 开始执行防作弊审核 (总任务数: {len(candidates)})...")
    
    report = {"pass": 0, "cheat": 0, "error": 0, "Total": len(candidates)}

    with open(FINAL_SFT_OUTPUT, 'a', encoding='utf-8') as f_pass, \
         open(CHEATING_BADCASE_OUTPUT, 'a', encoding='utf-8') as f_cheat:
             
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_audit, item): item for item in candidates}
            for future in tqdm(as_completed(futures), total=len(futures), desc="🚀 逻辑审查中"):
                res = future.result()
                status = res['status']
                report[status] += 1
                
                if status == 'pass':
                    f_pass.write(json.dumps(res['item'], ensure_ascii=False) + '\n')
                    f_pass.flush()
                elif status == 'cheat':
                    f_cheat.write(json.dumps(res['item'], ensure_ascii=False) + '\n')
                    f_cheat.flush()

    print("\n" + "="*55)
    print("🛡️ 阶段三：防作弊审查完成！")
    print("-" * 55)
    print(f"🔸 审查总数: {report['Total']} 条")
    print(f"✅ 真金不怕火炼 (审核通过): {report['pass']} 条 (已并入 SFT 终极池)")
    print(f"🚨 抓获逻辑作弊者 (强行凑答案): {report['cheat']} 条 (已打入 Badcase 专栏)")
    print(f"⚠️ API 报错: {report['error']} 条")
    print("="*55)

if __name__ == "__main__":
    main()