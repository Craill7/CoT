import json
import requests
import re
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# 确保安装了 math_verify: pip install math_verify
try:
    from math_verify import parse, verify
except ImportError:
    raise ImportError("请先安装 math_verify 库: pip install math_verify")

# ================= 全局配置区 =================
# 输入：Step 1 产出的带诊断标签的数据
TAGGED_INPUT_FILE = "/home/zhouyan/share/cyh/CoT/refine/refined_results/critic_tagged_math.jsonl"
# 仅作为合流参考，Step 2 后的成功项会进入 Candidate 供 Step 3 最终合并
SUCCESS_CANDIDATES_OUTPUT = "/home/zhouyan/share/cyh/CoT/refine/refined_results/step2_success_candidates.jsonl"
HARD_NEGATIVE_OUTPUT = "/home/zhouyan/share/cyh/CoT/refine/refined_results/hard_negative.jsonl"

API_URL_32B = "http://localhost:8001/v1/chat/completions" # 32B 端口
MODEL_NAME_32B = "Qwen2.5-32B-Instruct"

MAX_WORKERS = 6    # 32B 模型并发，视显存情况可调至 4-8
REQUEST_TIMEOUT = 300 
# ==========================================

def extract_boxed_answer(text):
    """
    使用栈匹配算法精准提取 \boxed{} 内容，完美处理嵌套括号
    """
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return text[-200:] # 兜底：取最后 200 字符
    
    stack = 0
    start = idx + 7
    for i in range(start, len(text)):
        if text[i] == '{':
            stack += 1
        elif text[i] == '}':
            if stack == 0:
                return text[start:i]
            stack -= 1
    return text[start:]

def llm_equivalence_judge(instruction, expected, predicted):
    """
    【语义裁判】32B 模拟老师阅卷，解决 Case 4 这种格式差异但语义正确的问题
    """
    prompt = f"""You are a tolerant and intelligent math teacher grading a student's exam. 
Given the original problem, the absolute Ground Truth, and the Student's Answer, determine if the Student's Answer is effectively correct.

Problem:
{instruction}

Ground Truth: 
{expected}

Student Answer: 
{predicted}

Grading Guidelines:
1. Ignore formatting differences, LaTeX syntax variations, or variable naming (e.g., t vs x).
2. If the problem asks for a list/sequence and the student provides the correct items, but the Ground Truth uses a sum/series notation (or vice versa), consider it correct.
3. As long as the core mathematical meaning matches the Ground Truth in the context of the problem, mark it as True.

Respond strictly in JSON format:
{{"is_equivalent": true or false}}"""

    payload = {
        "model": MODEL_NAME_32B,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0, 
        "response_format": {"type": "json_object"}
    }
    
    try:
        r = requests.post(API_URL_32B, json=payload, timeout=60)
        r.raise_for_status()
        res = json.loads(r.json()['choices'][0]['message']['content'])
        return res.get('is_equivalent', False)
    except Exception:
        return False

def verify_correctness(instruction, new_cot, ground_truth):
    """
    多级校验流水线：Math_Verify -> 字符串匹配 -> LLM 语义裁判
    """
    ans_str = extract_boxed_answer(new_cot)
    gt_str = str(ground_truth)
    
    # 1. 自动解析校验 (math_verify)
    try:
        predicted = parse(ans_str)
        expected = parse(gt_str)
        if verify(expected, predicted):
            return True
    except Exception:
        pass
        
    # 2. 字符串硬匹配兜底
    clean_gt = gt_str.replace(" ", "").replace("$", "")
    clean_ans = ans_str.replace(" ", "").replace("$", "")
    if clean_gt == clean_ans or (len(clean_gt) > 0 and clean_gt in clean_ans):
        return True
        
    # 3. LLM 语义平反 (终审)
    return llm_equivalence_judge(instruction, gt_str, ans_str)

def call_refiner_32b(instruction, critic_report, ground_truth):
    """
    32B 全英文强约束重写
    """
    prompt = f"""You are a top-tier mathematician. A previous reasoning trace for the problem below contained errors.
    
Diagnostic Report of the Previous Trace:
- Error Type: [{critic_report.get('error_type', 'Logical Error')}]
- Error Span: [{critic_report.get('error_span', 'Mid-way divergence')}]
- Advice: [{critic_report.get('advice', 'Replan the logic')}]

The absolute Ground Truth for this problem is: {ground_truth}.

Your Task:
From a global perspective, replan the reasoning path. Avoid the previous dead ends and derive the correct answer directly.
- Ensure your logic is concise, cohesive, and rigorous.
- Strictly use a step-by-step format (e.g., Step 1, Step 2...).
- You MUST enclose your final answer in \\boxed{{}}.

Problem:
{instruction}

Please output the refined Chain of Thought (enclosed within <think> and </think> tags) and the final answer directly:"""

    payload = {
        "model": MODEL_NAME_32B,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }
    
    r = requests.post(API_URL_32B, json=payload, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()['choices'][0]['message']['content'].strip()

def process_refine(item):
    stats = {"path": None, "item": None}
    try:
        instruction = item.get('instruction', '')
        old_cot = item.get('output', '')
        ground_truth = item.get('answer', '')
        old_is_correct = item.get('metadata', {}).get('is_correct', False)
        critic_report = item.get('metadata', {}).get('critic_report', {})
        
        # 1. 重写
        new_cot = call_refiner_32b(instruction, critic_report, ground_truth)
        
        # 2. 增强型校验
        new_is_correct = verify_correctness(instruction, new_cot, ground_truth)
        
        # 3. 判决分流
        if not old_is_correct and new_is_correct:
            # 路径 A: 逆袭成功
            stats['path'] = 'A'
            item['output'] = new_cot
            item['metadata']['refine_path'] = 'A'
            stats['item'] = item
            
        elif old_is_correct and not new_is_correct:
            # 路径 B: 能力退化
            stats['path'] = 'B'
            
        elif old_is_correct and new_is_correct:
            # 路径 C: 提纯评估
            compression_ratio = len(new_cot) / len(old_cot) if len(old_cot) > 0 else 1.0
            has_clear_steps = bool(re.search(r'Step \d', new_cot, re.IGNORECASE))
            
            if compression_ratio < 0.7 and has_clear_steps:
                stats['path'] = 'C_Success'
                item['output'] = new_cot
                item['metadata']['refine_path'] = 'C'
                item['metadata']['compression_ratio'] = compression_ratio
                stats['item'] = item
            else:
                stats['path'] = 'C_Fail'
        else:
            # 路径 D: 顽固毒瘤
            stats['path'] = 'D'
            item['output'] = new_cot
            item['metadata']['refine_path'] = 'D'
            stats['item'] = item
            
        return stats
    except Exception:
        stats['path'] = 'Error'
        return stats

def main():
    print(f"📦 正在加载待修复数据: {TAGGED_INPUT_FILE}")
    tagged_data = []
    with open(TAGGED_INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            tagged_data.append(json.loads(line))

    # 预准备输出文件
    open(SUCCESS_CANDIDATES_OUTPUT, 'w').close()
    open(HARD_NEGATIVE_OUTPUT, 'w').close()

    print(f"⚖️ 执行增强型判决流水线 (总任务数: {len(tagged_data)})...")
    report = {"A": 0, "B": 0, "C_Success": 0, "C_Fail": 0, "D": 0, "Error": 0, "Total": len(tagged_data)}

    with open(SUCCESS_CANDIDATES_OUTPUT, 'a', encoding='utf-8') as f_cand, \
         open(HARD_NEGATIVE_OUTPUT, 'a', encoding='utf-8') as f_hard:
             
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_refine, item): item for item in tagged_data}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="🚀 深度重写与判决"):
                res = future.result()
                path = res['path']
                report[path] += 1
                
                if path in ['A', 'C_Success']:
                    f_cand.write(json.dumps(res['item'], ensure_ascii=False) + '\n')
                    f_cand.flush()
                elif path == 'D':
                    f_hard.write(json.dumps(res['item'], ensure_ascii=False) + '\n')
                    f_hard.flush()

    print("\n" + "="*55)
    print("🏆 Step 2 最终战报（已启用 LLM 语义平反）")
    print("-" * 55)
    print(f"✅ 进入待审区的优质候选 (A + C): {report['A'] + report['C_Success']} 条")
    print(f"💀 顽固毒瘤 (D): {report['D']} 条")
    print(f"❌ 能力退化 (B): {report['B']} 条")
    print(f"⚠️ 报错数: {report['Error']} 条")
    print("="*55)
    print(f"💡 接下来请切换至 3B 模型，运行 Step 3 脚本进行防作弊审核。")

if __name__ == "__main__":
    main()