import json
import random

HARD_NEGATIVE_PATH = "/home/zhouyan/share/cyh/CoT/refine/refined_results/cheating_badcase.jsonl"

def extract_boxed_answer(text):
    """精准提取 \boxed{} 内容，用于对比 GT"""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return "⚠️ 未提取到 boxed (模型可能崩溃或未遵守指令)"
    
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

def main():
    badcases = []
    with open(HARD_NEGATIVE_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                badcases.append(json.loads(line))
                
    if not badcases:
        print("未找到 Hard Negative 数据！")
        return
        
    sample_size = min(5, len(badcases))
    samples = random.sample(badcases, sample_size)
    
    print(f"🔬 从 {len(badcases)} 条顽固毒瘤中随机抽取了 {sample_size} 条进行开箱验尸：\n")
    
    for i, item in enumerate(samples, 1):
        instruction = item.get('instruction', 'N/A')
        gt = item.get('answer', 'N/A')
        critic_report = item.get('metadata', {}).get('critic_report', {})
        new_cot = item.get('output', '')
        
        generated_ans = extract_boxed_answer(new_cot)
        
        print("="*80)
        print(f"🩸 Case {i}")
        print(f"❓ 【原题】: {instruction}")
        print(f"🎯 【绝对真理 (GT)】: {gt}")
        print(f"🤖 【32B 强填答案】: {generated_ans}")
        print(f"🩺 【3B 诊断意见】: [{critic_report.get('error_type')}] -> {critic_report.get('advice')}")
        print(f"📝 【32B 临终推导 (最后 600 字符)】:\n...\n{new_cot[-600:]}")
        print("="*80 + "\n")

if __name__ == "__main__":
    main()